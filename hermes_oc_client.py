"""Hermes-facing client for the private native OpenCode runtime.

The client owns model policy, the OpenAI-shaped facade and canonical history
translation. It never executes a Hermes tool: every model tool call is returned
to the Hermes agent loop, and the next completion replays the canonical history
as the native snapshot. Stock Hermes, no core patch, no execution bridge.
"""
from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import oc_affinity
import oc_catalog
import oc_translate
from oc_runtime import (
    ControlRefused,
    NativeRuntime,
    OpenCodeRuntimeError,
    RuntimeClosed,
    SUPPORTED_PACKAGE,
    SnapshotRefused,
    StepResult,
    ToolCall,
)

LOGICAL_BASE_URL = "opencode-native://local"
LOGGER = logging.getLogger(__name__)
PROVIDER_NAME = "opencode-free"
EXECUTE_UNSUPPORTED = (
    "native JavaScript execute is unsupported; use the enabled Hermes Python execute_code tool"
)
EXECUTE_UNSUPPORTED_NO_CODE = (
    "native JavaScript execute is unsupported and was not run"
)


def _authoritative_profile_home() -> str:
    try:
        from hermes_constants import get_hermes_home  # type: ignore[import-not-found]

        return str(get_hermes_home())
    except Exception:  # noqa: BLE001 - standalone provider import
        return os.environ.get("HERMES_HOME", "")


class ModelRefused(OpenCodeRuntimeError):
    code = "runtime.model"


class TransportRefused(OpenCodeRuntimeError):
    code = "runtime.transport"


@dataclass
class Usage:
    prompt_tokens: int | float | None = None
    completion_tokens: int | float | None = None
    total_tokens: int | float | None = None
    prompt_tokens_details: Any = None
    completion_tokens_details: Any = None
    raw: dict | None = None


@dataclass
class AssistantMessage:
    role: str = "assistant"
    content: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    # Opaque native part metadata, preserved by core across turns.
    reasoning_details: Any = None
    tool_calls: list = field(default_factory=list)


class _CallDict(dict):
    """OpenAI call shape with both mapping and attribute access."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _call_view(call: ToolCall) -> _CallDict:
    function = _CallDict(call.as_openai()["function"])
    return _CallDict(
        id=call.call_id,
        call_id=call.call_id,
        response_item_id=None,
        type="function",
        function=function,
    )


@dataclass
class Choice:
    message: AssistantMessage
    finish_reason: str


@dataclass
class Completion:
    choices: list[Choice]
    usage: Usage | None
    model: str
    finish_reason: str
    text: str
    # True only when a persisted native session was missing and a fresh
    # official ID replaced the mapping. Never a wire identifier.
    affinity_reset: bool = False
    reasoning: str | None = None
    tool_calls: list = field(default_factory=list)
    native_tool_calls: list = field(default_factory=list)
    native_usage: dict | None = None

    def model_dump(self, **kwargs: Any) -> dict:
        """SDK-shaped serialization for Hermes middleware and tracing."""
        return asdict(self)


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict) or not raw:
        return None
    prompt = _number(raw.get("prompt_tokens", raw.get("input")))
    completion = _number(raw.get("completion_tokens", raw.get("output")))
    total = _number(raw.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=raw.get("prompt_tokens_details"),
        completion_tokens_details=raw.get("completion_tokens_details"),
        raw=dict(raw),
    )


class RuntimeOwner:
    """A runtime shared by facades of one profile/session conversation."""

    def __init__(self, *, factory: Callable[[], NativeRuntime] | None = None, **runtime_kwargs: Any):
        self._factory = factory
        self._kwargs = dict(runtime_kwargs)
        self._runtime: NativeRuntime | None = None
        self._refs = 0
        self._used = False
        self._closed = False
        self.last_used = time.monotonic()
        self._lock = threading.RLock()
        # Retained lane state. The generation lock serialises same-scope calls
        # in this process; a cross-process lock is taken per request too.
        self.generation_lock = threading.Lock()
        self.scope_key: str | None = None
        self.session: Any = None
        self.native_session_id: str | None = None
        # Cross-process ownership is held for the runtime lifetime (including
        # idle), so another process cannot start a second server on this root
        # or prune/reset its data.
        self.ownership_lock: oc_affinity.ScopeLock | None = None

    def _build_runtime(self) -> NativeRuntime:
        return self._factory() if self._factory is not None else NativeRuntime(**self._kwargs)

    def _refresh_locked(self) -> NativeRuntime:
        runtime = self._runtime
        if runtime is not None and not runtime.alive:
            started = self._used or getattr(runtime, "_started", False) or getattr(runtime, "_stopped", False)
            if started:
                try:
                    runtime.stop()
                finally:
                    self._runtime = None
                    self.session = None
        if self._runtime is None:
            self._runtime = self._build_runtime()
        runtime = self._runtime
        if not runtime.alive:
            runtime.start()
        self._used = True
        return runtime

    @property
    def runtime(self) -> NativeRuntime:
        with self._lock:
            if self._runtime is None:
                self._runtime = self._build_runtime()
            return self._runtime

    def acquire(self) -> NativeRuntime:
        with self._lock:
            if self._closed:
                raise RuntimeClosed("runtime owner is closed")
            runtime = self._refresh_locked()
            self._refs += 1
            self.last_used = time.monotonic()
            return runtime

    def ensure(self) -> NativeRuntime:
        """Renew an already acquired facade, following a replaced runtime."""
        with self._lock:
            if self._closed:
                raise RuntimeClosed("runtime owner is closed")
            runtime = self._refresh_locked()
            self.last_used = time.monotonic()
            return runtime

    def release(self) -> None:
        with self._lock:
            if self._refs > 0:
                self._refs -= 1
            self.last_used = time.monotonic()

    def ensure_ownership(self, profile_home: str, key: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeClosed("runtime owner is closed")
            if self.ownership_lock is None:
                self.ownership_lock = oc_affinity.acquire_lock(profile_home, key)

    def close(self) -> None:
        lock = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._refs = 0
            try:
                if self._runtime is not None:
                    self._runtime.stop()
            finally:
                self.session = None
                lock = self.ownership_lock
                self.ownership_lock = None
        if lock is not None:
            lock.release()

    def cancel(self) -> None:
        """Poison this process lane; durable native state and mapping survive."""
        with self._lock:
            if self._runtime is not None and self._runtime.alive:
                self._runtime.stop()
            self.session = None

    @property
    def pid(self) -> int | None:
        with self._lock:
            return self._runtime.pid if self._runtime is not None else None


class RuntimePool:
    """Profile/session keyed owners; auxiliary keys stay isolated."""

    _owners: dict[str, RuntimeOwner] = {}
    _lock = threading.RLock()
    TTL_SECONDS = 300.0
    REAP_INTERVAL = 1.0
    MAX_OWNERS = 32
    _reaper: threading.Thread | None = None

    @staticmethod
    def key_for(profile_home: str, session_id: str) -> str:
        # JSON pair: neither path nor session text can create the other's key.
        return json.dumps([str(profile_home), str(session_id)], ensure_ascii=False)

    @staticmethod
    def key_for_scope(profile_home: str, scope: str, provider_id: str, model_id: str) -> str:
        return oc_affinity.scope_key(profile_home, scope, provider_id, model_id)

    @classmethod
    def _ensure_reaper(cls) -> None:
        with cls._lock:
            if cls._reaper is not None and cls._reaper.is_alive():
                return
            thread = threading.Thread(target=cls._reap_loop, name="oc-runtime-reaper", daemon=True)
            cls._reaper = thread
            thread.start()

    @classmethod
    def _reap_loop(cls) -> None:
        while True:
            time.sleep(max(0.01, float(cls.REAP_INTERVAL)))
            try:
                cls._prune()
            except Exception:  # noqa: BLE001 - the reaper must survive one bad owner
                continue

    @classmethod
    def _prune(cls, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        expired: list[RuntimeOwner] = []
        with cls._lock:
            for key, owner in list(cls._owners.items()):
                if owner._refs == 0 and now - owner.last_used >= cls.TTL_SECONDS:
                    expired.append(cls._owners.pop(key))
        for owner in expired:
            owner.close()

    @classmethod
    def acquire(cls, key: str, **runtime_kwargs: Any) -> RuntimeOwner:
        if not isinstance(key, str) or not key:
            raise ControlRefused("runtime owner key must be a non-empty string")
        cls._ensure_reaper()
        cls._prune()
        signature = (
            str(runtime_kwargs.get("binary") or ""),
            tuple(runtime_kwargs.get("binary_hints") or ()),
            str(runtime_kwargs.get("companion") or ""),
            str(runtime_kwargs.get("root") or ""),
        )
        evicted: RuntimeOwner | None = None
        with cls._lock:
            owner = cls._owners.get(key)
            if owner is None:
                if len(cls._owners) >= cls.MAX_OWNERS:
                    idle = sorted(
                        (item for item in cls._owners.values() if item._refs == 0),
                        key=lambda item: item.last_used,
                    )
                    if not idle:
                        raise ControlRefused(
                            "native runtime pool is at capacity with live scopes; retry when one finishes"
                        )
                    evicted = idle[0]
                    cls._owners.pop(next(name for name, item in cls._owners.items() if item is evicted))
                owner = RuntimeOwner(**runtime_kwargs)
                owner.signature = signature  # type: ignore[attr-defined]
                cls._owners[key] = owner
            elif getattr(owner, "signature", None) != signature:
                raise RuntimeClosed(f"runtime owner key {key!r} was reused with a different configuration")
            owner.last_used = time.monotonic()
        if evicted is not None:
            evicted.close()
        return owner

    @classmethod
    def prune(cls, now: float | None = None) -> None:
        cls._prune(now)

    @classmethod
    def active_keys(cls) -> set[str]:
        with cls._lock:
            return set(cls._owners)

    @classmethod
    def close(cls, key: str) -> None:
        with cls._lock:
            owner = cls._owners.pop(key, None)
        if owner is not None:
            owner.close()

    @classmethod
    def close_for_session(cls, profile_home: str, session_id: str) -> None:
        cls.close(cls.key_for(profile_home, session_id))

    @classmethod
    def close_all(cls) -> None:
        with cls._lock:
            owners = list(cls._owners.values())
            cls._owners.clear()
        for owner in owners:
            owner.close()


atexit.register(RuntimePool.close_all)


class _LazyCompletion:
    """A sync-or-awaitable completion, so the same client is async-safe."""

    def __init__(self, factory: Callable[[], Completion]):
        self._factory = factory
        self._value: Completion | None = None
        self._lock = threading.Lock()

    def _resolve(self) -> Completion:
        with self._lock:
            if self._value is None:
                self._value = self._factory()
            return self._value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __await__(self):
        async def resolve_in_thread() -> Completion:
            return await asyncio.to_thread(self._resolve)

        return resolve_in_thread().__await__()

    def __iter__(self):
        # A caller expecting a stream must get an explicit refusal, not the
        # TypeError iterating a completion object produces.
        raise ControlRefused("native streaming is not advertised; a completion is not iterable")

    def model_dump(self, **kwargs: Any) -> dict:
        # Middleware inspects the type, not __getattr__ on the lazy facade.
        return self._resolve().model_dump(**kwargs)

    def __repr__(self) -> str:
        return repr(self._resolve())


class OpenCodeClient:
    """OpenAI-shaped Hermes client backed by the pinned native runtime."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True
    supports_streaming = False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        binary: str | None = None,
        companion: str | None = None,
        root: str | None = None,
        startup_timeout: float = 60.0,
        cleanup: bool = True,
        runtime: NativeRuntime | None = None,
        runtime_owner: RuntimeOwner | None = None,
        owner_key: str | None = None,
        command: str | None = None,
        args: Any = None,
        **kwargs: Any,
    ):
        # Credentials/endpoints are accepted only for profile-call compatibility.
        # `command` is the executable core resolved for its eligibility gate: an
        # execution hint, validated before use, never a substitute for our own
        # PATH/config policy. Nothing here is sent anywhere.
        del api_key, base_url, args
        hint = command if isinstance(command, str) and os.path.isabs(command) and os.path.isfile(command) else None
        self.ignored_profile_options = tuple(sorted(kwargs))
        self.model = model
        self.timeout = float(timeout) if timeout else 120.0
        self.api_key = ""
        self.base_url = LOGICAL_BASE_URL
        self.is_closed = False
        self._owned_session_key = owner_key
        if self._owned_session_key is None:
            session_id = kwargs.get("session_id")
            profile_home = kwargs.get("profile_home") or _authoritative_profile_home()
            if isinstance(session_id, str) and session_id and profile_home:
                self._owned_session_key = RuntimePool.key_for(str(profile_home), session_id)
        self._runtime = runtime
        self._owner: RuntimeOwner | None = runtime_owner
        self._active_owner: RuntimeOwner | None = None
        self._owns_runtime = runtime is None and runtime_owner is None and self._owned_session_key is None
        self._runtime_acquired = False
        self._runtime_kwargs = {
            "binary": binary,
            "binary_hints": (hint,) if hint else None,
            "companion": companion,
            "root": root,
            "startup_timeout": startup_timeout,
            "rpc_timeout": self.timeout,
            "cleanup": cleanup,
        }
        self._runtime_kwargs = {key: value for key, value in self._runtime_kwargs.items() if value is not None}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))

    # -- runtime / discovery ----------------------------------------------
    @property
    def runtime(self) -> NativeRuntime:
        return self._ensure_runtime()

    def _ensure_runtime(self) -> NativeRuntime:
        if self.is_closed:
            raise ControlRefused("OpenCode client is closed")
        if self._owner is None and self._owned_session_key:
            self._owner = RuntimePool.acquire(self._owned_session_key, **self._runtime_kwargs)
        if self._owner is not None:
            if not self._runtime_acquired:
                self._runtime = self._owner.acquire()
                self._runtime_acquired = True
            else:
                self._runtime = self._owner.ensure()
            return self._runtime
        if self._runtime is None:
            self._runtime = NativeRuntime(**self._runtime_kwargs)
        if not self._runtime.alive:
            if getattr(self._runtime, "_started", False):
                if not self._owns_runtime:
                    raise RuntimeClosed("OpenCode runtime is no longer running; refusing an automatic restart")
                self._runtime.stop()
                self._runtime = NativeRuntime(**self._runtime_kwargs)
            elif getattr(self._runtime, "_stopped", False):
                raise RuntimeClosed("OpenCode runtime was stopped; refusing an automatic restart")
            self._runtime.start()
        return self._runtime

    def catalog(self, *, timeout: float | None = None) -> list[dict]:
        return self._ensure_runtime().catalog(timeout=timeout)

    def list_models(self, *, timeout: float = 15.0) -> list[str]:
        records = self.catalog(timeout=timeout)
        try:
            return oc_catalog.free_model_ids(records)
        except oc_catalog.CatalogError as error:
            raise ModelRefused(str(error)) from None

    def list_model_details(self, *, timeout: float = 15.0) -> list[dict]:
        """Raw sanitized native records; the profile applies `oc_catalog` policy."""
        return self.catalog(timeout=timeout)

    def select_model(self, model: str, *, records: list[dict] | None = None) -> dict:
        if not isinstance(model, str) or not model:
            raise ModelRefused("an exact native model id is required")
        rows = records if records is not None else self.catalog()
        try:
            chosen = oc_catalog.select_model(rows, model)
        except oc_catalog.CatalogError as error:
            raise ModelRefused(str(error)) from None
        if not isinstance(chosen, dict):
            raise ModelRefused("catalog policy returned a non-object model record")
        if not chosen.get("providerID") or not chosen.get("id"):
            raise ModelRefused("catalog policy returned a record without providerID/id")
        return chosen

    @staticmethod
    def _require_transport(record: dict) -> None:
        if record.get("supported") is False:
            raise TransportRefused(str(record.get("unsupported_reason") or "unsupported native transport"))
        package = record.get("package") or record.get("providerPackage")
        api = record.get("api")
        if package is None and isinstance(api, dict):
            package = api.get("npm")
        if package is not None and package != SUPPORTED_PACKAGE:
            raise TransportRefused(f"unsupported native transport package: {package!r}")

    # -- request validation ------------------------------------------------
    @staticmethod
    def _validate_request(
        *,
        stream: Any = False,
        tool_choice: Any = None,
        max_tokens: Any = None,
        max_completion_tokens: Any = None,
        temperature: Any = None,
        top_p: Any = None,
        stop: Any = None,
        response_format: Any = None,
        reasoning_effort: Any = None,
        extra_body: Any = None,
        **kwargs: Any,
    ) -> None:
        if stream not in (False, None):
            raise ControlRefused("streaming is not advertised; set stream=False")
        if tool_choice not in (None, "auto"):
            raise ControlRefused(f"tool_choice {tool_choice!r} is unsupported before inference")
        for name, value in (
            ("max_tokens", max_tokens),
            ("max_completion_tokens", max_completion_tokens),
            ("temperature", temperature),
            ("top_p", top_p),
            ("stop", stop),
            ("response_format", response_format),
            ("reasoning_effort", reasoning_effort),
            ("extra_body", extra_body),
        ):
            if value is not None:
                raise ControlRefused(f"{name} is unsupported before inference")
        # Relay may attach tracing to the OpenAI-shaped facade. It is local
        # metadata, not a native vendor header; never forward it or credentials.
        headers = kwargs.pop("extra_headers", None)
        if headers is not None and (
            not isinstance(headers, dict)
            or any(not isinstance(key, str) or key.lower() != "traceparent"
                   or not isinstance(value, str) for key, value in headers.items())
        ):
            raise ControlRefused("extra_headers supports only local traceparent metadata")
        if kwargs:
            raise ControlRefused(f"unsupported request controls: {', '.join(sorted(kwargs))}")

    # -- affinity ----------------------------------------------------------
    @staticmethod
    def _session_scope(context: Any) -> dict | None:
        """Consume the provider-only per-request marker; never forward it."""
        if context is None:
            return None
        if not isinstance(context, dict):
            raise ControlRefused("hermes_session_context must be an object")
        if context.get("version") != 2:
            raise ControlRefused("hermes_session_context has an unsupported version")
        session_id = context.get("session_id") or ""
        cache_scope_id = context.get("cache_scope_id") or ""
        for name, value in (("session_id", session_id), ("cache_scope_id", cache_scope_id)):
            if not isinstance(value, str) or len(value) > 1024 or "\x00" in value:
                raise ControlRefused(
                    f"hermes_session_context.{name} must be a bounded string"
                )
        if not session_id and not cache_scope_id:
            return None
        if cache_scope_id.startswith("sub_"):
            # Delegate siblings share one sub_<sha> cache bucket. The physical
            # session is the only sibling-unique value; compression rotates it,
            # so delegate affinity resets at compression by design.
            if not session_id:
                return None
            lane = f"sub:{cache_scope_id}:{session_id}"
        else:
            # Normal/compression/gateway/branch: the scope is stable across
            # physical-id rotation and per-response gateway ids.
            lane = cache_scope_id or session_id
        profile_home = _authoritative_profile_home()
        if not profile_home:
            # No profile identity means no safe shared bucket. Stay ephemeral.
            return None
        return {
            "lane": lane,
            "session_id": session_id,
            "cache_scope_id": cache_scope_id,
            "profile_home": profile_home,
        }

    @staticmethod
    def _prepare_generation(
        runtime: NativeRuntime, messages: list[dict], tools: list[dict] | None
    ) -> tuple[oc_translate.WirePlan, Any]:
        # Every native registry id is reserved: a Hermes tool with the same name
        # gets a reversible wire alias instead of being dropped.
        native_ids = {
            str(tool.get("id"))
            for tool in (runtime.info or {}).get("tools", [])
            if isinstance(tool, dict) and isinstance(tool.get("id"), str)
            and tool.get("hermes") is not True
        }
        try:
            plan = oc_translate.plan_tools(tools, native_ids)
        except oc_translate.CompatibilityUnavailable as error:
            raise ControlRefused(str(error)) from None
        except oc_translate.NativeTranslationError as error:
            raise ControlRefused(f"Hermes tool catalog cannot be advertised: {error}") from None
        try:
            snapshot = oc_translate.snapshot_from_openai(messages, plan=plan)
        except oc_translate.SnapshotError as error:
            raise SnapshotRefused(str(error)) from None
        return plan, snapshot

    @staticmethod
    def _select_record(runtime: NativeRuntime, selected_model: str) -> dict:
        if not isinstance(selected_model, str) or not selected_model:
            raise ModelRefused("an explicit native model id is required")
        try:
            chosen = oc_catalog.select_model(runtime.catalog(), selected_model)
        except oc_catalog.CatalogError as error:
            raise ModelRefused(str(error)) from None
        if not isinstance(chosen, dict):
            raise ModelRefused("catalog policy returned a non-object model record")
        if not chosen.get("providerID") or not chosen.get("id"):
            raise ModelRefused("catalog policy returned a record without providerID/id")
        return chosen

    def _handle_result(
        self, runtime: NativeRuntime, result: StepResult, record: dict, plan: oc_translate.WirePlan,
        *, affinity_reset: bool = False,
    ) -> Completion:
        if any(call.name == "execute" for call in result.calls):
            # ponytail: explicit terminal refusal, not a tool call. Handing JS
            # to Hermes would only produce an unknown-tool turn and a second
            # vendor inference; the companion refuses any execute attempt too.
            message = EXECUTE_UNSUPPORTED if "execute_code" in plan.host_to_wire else EXECUTE_UNSUPPORTED_NO_CODE
            return self._refusal(result, record, message, affinity_reset=affinity_reset)
        native_ids = {
            str(tool.get("id"))
            for tool in (runtime.info or {}).get("tools", [])
            if isinstance(tool, dict) and isinstance(tool.get("id"), str)
            and tool.get("hermes") is not True
        }
        unsolicited = sorted({call.name for call in result.calls if call.name in native_ids
                              and call.name not in oc_translate.NATIVE_KEEP})
        if unsolicited:
            # Native descriptors outside the catalog are never advertised and
            # never dispatched. Refuse the turn instead of returning a tool
            # Hermes does not (and must not) run.
            message = (
                f"native tool(s) {', '.join(unsolicited)} are not exposed by this provider; "
                "use the enabled Hermes tools instead"
            )
            return self._refusal(result, record, message, affinity_reset=affinity_reset)
        calls = self._translate_calls(result, plan)
        return self._completion(result, record, calls, affinity_reset=affinity_reset)

    # -- inference ---------------------------------------------------------
    def complete(
        self,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        *,
        model: str | None = None,
        stream: bool = False,
        tool_choice: Any = None,
        max_tokens: Any = None,
        max_completion_tokens: Any = None,
        temperature: Any = None,
        top_p: Any = None,
        stop: Any = None,
        response_format: Any = None,
        reasoning_effort: Any = None,
        extra_body: Any = None,
        timeout: float | None = None,
        hermes_session_context: Any = None,
        **kwargs: Any,
    ) -> Completion:
        self._validate_request(
            stream=stream,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            **kwargs,
        )
        if messages is None:
            raise ControlRefused("messages are required")
        selected_model = model or self.model
        if not selected_model:
            raise ModelRefused("an explicit native model id is required")
        scope = self._session_scope(hermes_session_context)
        if scope is None:
            return self._complete_ephemeral(messages, tools, selected_model, timeout)
        return self._complete_retained(scope, messages, tools, selected_model, timeout)

    def _complete_ephemeral(
        self, messages: list[dict], tools: list[dict] | None, selected_model: str, timeout: float | None
    ) -> Completion:
        runtime = self._ensure_runtime()
        plan, snapshot = self._prepare_generation(runtime, messages, tools)
        record = self._select_record(runtime, selected_model)
        self._require_transport(record)
        session = runtime.new_session(str(record["providerID"]), str(record["id"]), plan.register)
        try:
            result = session.generate(
                snapshot=snapshot.messages,
                system_parts=snapshot.system_parts,
                user_text=snapshot.user_text,
                timeout=timeout or self.timeout,
                hermes_tools=plan.register,
            )
        finally:
            session.close()
        return self._handle_result(runtime, result, record, plan)

    def _owner_kwargs(self, root: Path) -> dict:
        kwargs = dict(self._runtime_kwargs)
        kwargs.update(root=str(root), durable_root=True, cleanup=False)
        return kwargs

    def _acquire_owner(self, scope: dict, selected_model: str) -> RuntimeOwner:
        key = oc_affinity.scope_key(
            scope["profile_home"], scope["lane"], PROVIDER_NAME, selected_model
        )
        root = oc_affinity.native_root(scope["profile_home"], key)
        oc_affinity.prune_once(
            scope["profile_home"], active_keys=RuntimePool.active_keys() | {key}
        )
        owner = RuntimePool.acquire(key, **self._owner_kwargs(root))
        owner.scope_key = key
        try:
            owner.ensure_ownership(scope["profile_home"], key)
        except oc_affinity.AffinityBusy as error:
            raise ControlRefused(str(error)) from None
        return owner

    @staticmethod
    def _corroborate(data: dict, runtime: NativeRuntime, session_id: str) -> None:
        if str(data.get("id") or "") != session_id:
            raise oc_affinity.AffinityRefused(
                "native session get returned a different id than the mapping"
            )
        location = data.get("location")
        if isinstance(location, dict):
            location = location.get("directory")
        if not location:
            location = data.get("directory")
        if not isinstance(location, str) or not location:
            raise oc_affinity.AffinityRefused(
                "native session get returned no workspace; refusing to reuse it"
            )
        if runtime.workspace is None or (
            Path(location).expanduser().resolve() != Path(runtime.workspace).resolve()
        ):
            raise oc_affinity.AffinityRefused(
                "native session belongs to a different workspace; refusing to reuse it"
            )

    def _open_retained(
        self,
        owner: RuntimeOwner,
        runtime: NativeRuntime,
        scope: dict,
        record: dict,
        plan: oc_translate.WirePlan,
    ) -> tuple[Any, bool]:
        key = owner.scope_key
        assert key is not None
        provider_id = str(record["providerID"])
        model_id = str(record["id"])
        mapping = oc_affinity.load(scope["profile_home"], key, provider_id, model_id)
        reset = False
        session_id = None
        if mapping is not None:
            session_id = mapping["native_session_id"]
            data = runtime.get_session(session_id)
            if data is None:
                # A real 404 is the only evidence of absence. Allocate a fresh
                # official ID and replace the mapping atomically.
                reset = True
                session_id = None
            else:
                self._corroborate(data, runtime, session_id)
        if session_id is None:
            stale = owner.session
            owner.session = None
            owner.native_session_id = None
            if stale is not None:
                try:
                    stale.close()
                except Exception:  # noqa: BLE001 - the native session is already absent
                    LOGGER.debug("stale native session cleanup failed", exc_info=True)
            session = runtime.new_session(provider_id, model_id, plan.register)
            try:
                oc_affinity.save(
                    scope["profile_home"], key, session.session_id, provider_id, model_id
                )
            except Exception:
                try:
                    session.close()
                finally:
                    raise
            owner.native_session_id = session.session_id
            owner.session = session
            if reset:
                LOGGER.warning(
                    "OpenCode native affinity reset for scope %s…: previous session was missing",
                    scope["lane"][:16],
                )
            return session, reset
        existing = owner.session
        if (
            existing is not None
            and existing.runtime is runtime
            and existing.session_id == session_id
        ):
            session = existing
        else:
            session = runtime.new_session(
                provider_id, model_id, plan.register, session_id=session_id
            )
            owner.session = session
        owner.native_session_id = session_id
        return session, reset

    def _complete_retained(
        self,
        scope: dict,
        messages: list[dict],
        tools: list[dict] | None,
        selected_model: str,
        timeout: float | None,
    ) -> Completion:
        for _attempt in range(2):
            owner = self._acquire_owner(scope, selected_model)
            if not owner.generation_lock.acquire(blocking=False):
                raise ControlRefused(
                    "this OpenCode conversation is already running; wait for it to finish"
                )
            self._active_owner = owner
            try:
                try:
                    runtime = owner.acquire()
                except RuntimeClosed:
                    # Reaper evicted this idle owner between lookup and
                    # acquire; retry registers a fresh one.
                    continue
                try:
                    return self._run_retained(
                        owner, runtime, scope, messages, tools, selected_model, timeout
                    )
                finally:
                    owner.release()
            finally:
                self._active_owner = None
                owner.generation_lock.release()
        raise RuntimeClosed("native affinity lane could not be acquired")

    def _run_retained(
        self,
        owner: RuntimeOwner,
        runtime: NativeRuntime,
        scope: dict,
        messages: list[dict],
        tools: list[dict] | None,
        selected_model: str,
        timeout: float | None,
    ) -> Completion:
        plan, snapshot = self._prepare_generation(runtime, messages, tools)
        record = self._select_record(runtime, selected_model)
        self._require_transport(record)
        session, reset = self._open_retained(owner, runtime, scope, record, plan)
        if session.generation > 0:
            settled = session.finish(timeout=min(20.0, float(timeout or self.timeout)))
            if settled.get("settled") is False:
                # The previous generation could not be recovered. Remove that
                # native session and allocate a fresh official ID; never reuse
                # an unknown state and never replay a Hermes effect.
                try:
                    session.close()
                except Exception:  # noqa: BLE001 - poison the process lane instead
                    LOGGER.warning("OpenCode failed session cleanup after unsafe finish", exc_info=True)
                    owner.cancel()
                    # `ensure()` follows the replacement without taking a new
                    # lease; the caller already holds exactly one.
                    runtime = owner.ensure()
                owner.session = None
                owner.native_session_id = None
                session = runtime.new_session(
                    str(record["providerID"]), str(record["id"]), plan.register
                )
                oc_affinity.save(
                    scope["profile_home"], owner.scope_key or "", session.session_id,
                    str(record["providerID"]), str(record["id"]),
                )
                owner.session = session
                owner.native_session_id = session.session_id
                reset = True
                LOGGER.warning(
                    "OpenCode native affinity reset for scope %s…: previous generation was unrecoverable",
                    scope["lane"][:16],
                )
        result = session.generate(
            snapshot=snapshot.messages,
            system_parts=snapshot.system_parts,
            user_text=snapshot.user_text,
            timeout=timeout or self.timeout,
            hermes_tools=plan.register,
        )
        return self._handle_result(runtime, result, record, plan, affinity_reset=reset)

    create = complete

    @staticmethod
    def _translate_calls(result: StepResult, plan: oc_translate.WirePlan) -> list[ToolCall]:
        translated: list[ToolCall] = []
        for call in result.calls:
            try:
                host_name, host_arguments = oc_translate.translate_call(
                    call.name, call.arguments, plan=plan, enabled=set(plan.host_to_wire)
                )
            except oc_translate.NativeTranslationError as error:
                raise ControlRefused(f"native tool {call.name!r} cannot run: {error}") from None
            call_id = oc_translate.encode_call_id(
                call.name, call.arguments, host_arguments, call.call_id
            )
            translated.append(ToolCall(call_id=call_id, name=host_name, arguments=host_arguments))
        return translated

    def _refusal(
        self, result: StepResult, record: dict, message: str, *, affinity_reset: bool = False
    ) -> Completion:
        """A provider-authored terminal refusal: no tool call, no effect, no extra inference."""
        return Completion(
            choices=[Choice(message=AssistantMessage(content=message), finish_reason="stop")],
            usage=_usage(result.usage),
            model=str(record.get("id") or ""),
            finish_reason="stop",
            text=message,
            affinity_reset=affinity_reset,
            native_tool_calls=[dict(call.as_openai()) for call in result.calls],
            native_usage=result.usage,
        )

    def _completion(
        self,
        result: StepResult,
        record: dict,
        calls: list[ToolCall],
        *,
        affinity_reset: bool = False,
    ) -> Completion:
        tool_calls = [_call_view(call) for call in calls]
        carrier = oc_translate.native_carrier(result.native_parts)
        message = AssistantMessage(
            content=(result.text if not tool_calls else (result.text or None)),
            reasoning=result.reasoning,
            reasoning_content=result.reasoning,
            reasoning_details=[carrier] if carrier else None,
            tool_calls=tool_calls,
        )
        choice = Choice(message=message, finish_reason=result.finish)
        return Completion(
            choices=[choice],
            usage=_usage(result.usage),
            model=str(record.get("id") or ""),
            finish_reason=result.finish,
            text=result.text,
            affinity_reset=affinity_reset,
            reasoning=result.reasoning,
            tool_calls=tool_calls,
            native_tool_calls=[dict(call.as_openai()) for call in result.calls],
            native_usage=result.usage,
        )

    # -- async facade ------------------------------------------------------
    async def acomplete(self, *args, **kwargs) -> Completion:
        return await asyncio.to_thread(self.complete, *args, **kwargs)

    def create_lazy(self, **kwargs: Any) -> _LazyCompletion:
        """Return a value that resolves on attribute access or `await`."""
        return _LazyCompletion(lambda: self.complete(**kwargs))

    def _create_chat_completion(self, **kwargs: Any) -> _LazyCompletion:
        # Unsupported controls (including stream=True) fail here, at create
        # time, so auxiliary/Relay callers never receive an object they cannot
        # iterate and never fall back as if the provider had answered.
        allowed = {
            "messages", "tools", "model", "stream", "tool_choice", "max_tokens",
            "max_completion_tokens", "temperature", "top_p", "stop",
            "response_format", "reasoning_effort", "extra_body", "extra_headers",
            "timeout", "hermes_session_context",
        }
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise ControlRefused(f"unsupported request controls: {', '.join(unknown)}")
        controls = {
            key: kwargs[key]
            for key in ("stream", "tool_choice", "max_tokens", "max_completion_tokens",
                        "temperature", "top_p", "stop", "response_format",
                        "reasoning_effort", "extra_body", "extra_headers")
            if key in kwargs
        }
        self._validate_request(**controls)
        return self.create_lazy(**kwargs)

    def stream(self, **_kwargs: Any):
        raise ControlRefused("native streaming is not advertised")

    # -- lifecycle ---------------------------------------------------------
    def cancel(self) -> None:
        """Abort this lane's process; retained native data and mapping survive."""
        owner = self._active_owner or self._owner
        if owner is not None:
            owner.cancel()

    def close(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        if self._owner is not None:
            if self._runtime_acquired:
                self._owner.release()
                self._runtime_acquired = False
            RuntimePool.prune()
        elif self._owns_runtime and self._runtime is not None:
            self._runtime.stop()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def __enter__(self) -> "OpenCodeClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
