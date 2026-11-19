"""Private, pinned, loopback OpenCode v2 runtime owned by Hermes.

This module owns the official process, its authentication boundary, the
one-generation RPC handshake and the response completion gate. It does not
translate native tools or dispatch Hermes tools; those belong to the client
and `oc_translate.py`.

The official binary is never modified. A companion plugin is loaded from the
private runtime configuration and must report the pinned protocol.
"""
from __future__ import annotations

import base64
import collections
import json
import os
import queue
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

if __package__:  # one profile copy's package; see __init__.py
    from .oc_binary import MIN_VERSION, BinaryUnavailable, _probe_env, probe_version, resolve as resolve_binary
    from .oc_watchdog import ConversationFailed, Watchdog, WatchdogRefused
else:
    from oc_binary import MIN_VERSION, BinaryUnavailable, _probe_env, probe_version, resolve as resolve_binary
    from oc_watchdog import ConversationFailed, Watchdog, WatchdogRefused

# Minimum official version; the handshake verifies the loaded companion and
# binary at runtime. A newer release is accepted but never assumed compatible.
SUPPORTED_BINARY_VERSION = ".".join(str(part) for part in MIN_VERSION)
SUPPORTED_COMPANION_PROTOCOL = 5
COMPANION_ID = "hermes-oc-native"
SUPPORTED_PACKAGE = "@opencode/ai/providers/openai-compatible"

COMPANION_ENV = "HERMES_OPENCODE_COMPANION"
ROOT_ENV = "HERMES_OPENCODE_RUNTIME_ROOT"

MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
MAX_STDERR_LINES = 400
MAX_STDOUT_LINES = 200
CATALOG_KEYS = (
    "id",
    "modelID",
    "providerID",
    "name",
    "type",
    "family",
    "package",
    "providerPackage",
    "api",
    "capabilities",
    "compatibility",
    "cost",
    "limit",
    "time",
    "status",
    "enabled",
    "variants",
)


class OpenCodeRuntimeError(RuntimeError):
    """Base for every explicit native-runtime refusal."""

    code = "runtime.error"

    def __init__(self, message: str, *, code: str | None = None, details: dict | None = None,
                 status_code: int | None = None):
        super().__init__(message)
        self.code = code or type(self).code
        self.details = details or {}
        # Upstream HTTP status reported by the official runtime, never inferred
        # from text. Hermes' error classifier reads `.status_code` first.
        self.status_code = status_code


def _upstream_status(event: Any) -> int | None:
    """HTTP error status the official runtime/companion attached to an event.

    `completion.status` (companion outcome), `step_failed.error.status`
    (OpenCode `Session.StructuredError`) and `response_unverified.outcome.status`.
    Only 4xx/5xx count: a 200 stream that failed mid-way has no error status.
    """
    if not isinstance(event, dict):
        return None
    for holder in (event, event.get("error"), event.get("outcome")):
        status = holder.get("status") if isinstance(holder, dict) else None
        if type(status) is int and 400 <= status <= 599:
            return status
    return None


class BinaryRefused(OpenCodeRuntimeError):
    code = "runtime.binary"


class StartupTimeout(OpenCodeRuntimeError):
    code = "runtime.startup"


class HandshakeRefused(OpenCodeRuntimeError):
    code = "runtime.handshake"


class RuntimeClosed(OpenCodeRuntimeError):
    code = "runtime.closed"


class RpcRefused(OpenCodeRuntimeError):
    code = "runtime.rpc"

    def __init__(self, message: str, *, method: str | None = None, status: int | None = None, code: str | None = None, details: dict | None = None):
        super().__init__(message, code=code, details=details)
        self.method = method
        self.status = status


class CatalogRefused(OpenCodeRuntimeError):
    code = "runtime.catalog"


class SessionRefused(OpenCodeRuntimeError):
    code = "runtime.session"


class EventGap(OpenCodeRuntimeError):
    code = "runtime.event_gap"


class CompletionRefused(OpenCodeRuntimeError):
    code = "runtime.completion"


class SnapshotRefused(OpenCodeRuntimeError):
    code = "runtime.snapshot"


class ControlRefused(OpenCodeRuntimeError):
    code = "runtime.control"


class ExecutionRefused(OpenCodeRuntimeError):
    code = "runtime.execute"


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict
    # Opaque per-call native metadata from `session.tool.called`; echoed back on
    # replay, never interpreted.
    native: Any = None

    def as_openai(self) -> dict:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)},
        }


@dataclass(frozen=True)
class Snapshot:
    system_parts: list[str]
    messages: list[dict]
    user_text: str


@dataclass
class StepResult:
    text: str
    reasoning: str | None
    calls: list[ToolCall]
    finish: str
    usage: dict | None
    generation: int
    assistant_message_id: str
    session: "Session"
    step_usage: dict | None = None
    completion: dict | None = None
    # Opaque per-part native metadata, in native order. Preserved through the
    # canonical history; never reconstructed from visible text.
    native_parts: dict = field(default_factory=dict)

    @property
    def has_execute(self) -> bool:
        return any(call.name == "execute" for call in self.calls)


def _valid_usage(usage: Any) -> bool:
    """Prompt/completion token usage that can be reported without inventing values.

    Zero is valid; missing, negative, non-integer or internally inconsistent
    totals are not. Optional cache/reasoning breakdowns are never required.
    """
    if not isinstance(usage, dict):
        return False
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(prompt, int) or isinstance(prompt, bool) or prompt < 0:
        return False
    if not isinstance(completion, int) or isinstance(completion, bool) or completion < 0:
        return False
    total = usage.get("total_tokens")
    if total is None:
        return True
    return isinstance(total, int) and not isinstance(total, bool) and total == prompt + completion


def _as_text(value: Any, *, field_name: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if not isinstance(part, dict):
                raise SnapshotRefused(f"{field_name} content part is not an object")
            kind = part.get("type")
            if kind in {"text", "input_text", "output_text"}:
                text = part.get("text")
                if not isinstance(text, str):
                    raise SnapshotRefused(f"{field_name} text part has no string text")
                parts.append(text)
                continue
            if kind in {"image_url", "image", "input_image", "audio", "input_audio", "file"}:
                raise SnapshotRefused(
                    f"{field_name} {kind} content is not supported by protocol {SUPPORTED_COMPANION_PROTOCOL}; "
                    "refusing rather than silently dropping media"
                )
            raise SnapshotRefused(f"{field_name} content part {kind!r} is unsupported")
        return "".join(parts)
    raise SnapshotRefused(f"{field_name} content must be a string or a list of text parts")


def _arguments(value: Any, *, call_id: str) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError as error:
            raise SnapshotRefused(f"tool call {call_id!r} arguments are not valid JSON") from error
    else:
        raise SnapshotRefused(f"tool call {call_id!r} arguments must be an object")
    if not isinstance(parsed, dict):
        raise SnapshotRefused(f"tool call {call_id!r} arguments must be a JSON object")
    try:
        json.dumps(parsed, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise SnapshotRefused(f"tool call {call_id!r} arguments are not JSON-serializable") from error
    return parsed


def canonical_messages(messages: Iterable[dict]) -> Snapshot:
    """Convert Hermes/OpenAI messages into the native canonical transcript.

    System and developer messages are appended to OpenCode's own system parts,
    never folded into the transcript. Native call ids and exact JSON arguments
    are preserved. Malformed or orphan tool results fail before any RPC.
    """
    if not isinstance(messages, (list, tuple)):
        raise SnapshotRefused("messages must be a list")
    system_parts: list[str] = []
    transcript: list[dict] = []
    names: dict[str, str] = {}
    user_text: str | None = None
    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            raise SnapshotRefused("message must be an object")
        role = message.get("role")
        if role in {"system", "developer"}:
            text = _as_text(message.get("content"), field_name=str(role))
            if text:
                system_parts.append(text)
            continue
        if role == "user":
            text = _as_text(message.get("content"), field_name="user")
            transcript.append({"role": "user", "content": [{"type": "text", "text": text}]})
            user_text = text
            continue
        if role == "assistant":
            parts: list[dict] = []
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                if not isinstance(reasoning, str):
                    raise SnapshotRefused("assistant reasoning must be a string")
                parts.append({"type": "reasoning", "text": reasoning})
            text = _as_text(message.get("content"), field_name="assistant")
            if text:
                parts.append({"type": "text", "text": text})
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise SnapshotRefused("assistant tool_calls must be a list")
            for call in calls:
                if not isinstance(call, dict):
                    raise SnapshotRefused("assistant tool call must be an object")
                call_id = str(call.get("id") or "")
                if not call_id:
                    raise SnapshotRefused("assistant tool call has no id")
                if call_id in seen_calls:
                    raise SnapshotRefused(f"duplicate tool call id {call_id!r}")
                function = call.get("function") or {}
                if not isinstance(function, dict):
                    raise SnapshotRefused(f"tool call {call_id!r} function must be an object")
                name = str(function.get("name") or "")
                if not name:
                    raise SnapshotRefused(f"tool call {call_id!r} has no name")
                seen_calls.add(call_id)
                names[call_id] = name
                parts.append(
                    {
                        "type": "tool-call",
                        "id": call_id,
                        "name": name,
                        "input": _arguments(function.get("arguments"), call_id=call_id),
                    }
                )
            if parts:
                transcript.append({"role": "assistant", "content": parts})
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if not call_id:
                raise SnapshotRefused("tool result has no tool_call_id")
            if call_id not in names:
                raise SnapshotRefused(f"tool result references unknown call {call_id!r}")
            if call_id in seen_results:
                raise SnapshotRefused(f"duplicate tool result for call {call_id!r}")
            seen_results.add(call_id)
            text = _as_text(message.get("content"), field_name="tool")
            explicit_name = message.get("name")
            name = str(explicit_name or names[call_id])
            if explicit_name is not None and name != names[call_id]:
                raise SnapshotRefused(
                    f"tool result name {name!r} does not match call {call_id!r} name {names[call_id]!r}"
                )
            if not name:
                raise SnapshotRefused(f"tool result {call_id!r} has no tool name")
            if "native_output" in message and not text:
                result = {"type": "json", "value": message["native_output"]}
            elif message.get("is_error"):
                result = {"type": "error", "value": text or "tool failed"}
            else:
                result = {"type": "text", "value": text}
            transcript.append(
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool-result",
                            "id": call_id,
                            "name": name,
                            "result": result,
                        }
                    ],
                }
            )
            continue
        raise SnapshotRefused(f"unsupported message role {role!r}")
    if not user_text:
        raise SnapshotRefused("a non-empty user message is required as the native prompt anchor")
    return Snapshot(system_parts=system_parts, messages=transcript, user_text=user_text)


def sanitize_catalog(records: Iterable[dict]) -> list[dict]:
    """Keep native discovery metadata; never return provider settings/secrets."""
    if not isinstance(records, (list, tuple)):
        raise CatalogRefused("native catalog is not a list")
    cleaned: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            raise CatalogRefused("native catalog contains a non-object record")
        row: dict[str, Any] = {}
        for key in CATALOG_KEYS:
            if key not in record:
                continue
            value = record[key]
            if key == "api":
                if not isinstance(value, dict):
                    continue
                api = {name: value[name] for name in ("npm", "id", "name", "type") if name in value}
                if api:
                    row["api"] = api
                continue
            row[key] = json.loads(json.dumps(value, ensure_ascii=False))
        cleaned.append(row)
    return cleaned


_POSIX = os.name == "posix"


def _spawn_kwargs() -> dict:
    """New process group per runtime: portable argv launch, no shell.

    POSIX uses a session leader so the whole tree can be signalled; Windows
    uses a new process group (Popen itself owns the child handle).
    """
    if _POSIX:
        return {"start_new_session": True}
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return {"creationflags": flags} if flags else {}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001 - urllib signature
        return None


class NativeRuntime:
    """One private official OpenCode server and its companion handshake."""

    def __init__(
        self,
        *,
        binary: str | Path | None = None,
        companion: str | Path | None = None,
        root: str | Path | None = None,
        workspace: str | Path | None = None,
        durable_root: bool = False,
        startup_timeout: float = 60.0,
        rpc_timeout: float = 120.0,
        execute_deadline: float = 120.0,
        cleanup: bool = True,
        env: dict | None = None,
        binary_hints: Iterable[str] | None = None,
    ):
        self._binary_arg = binary
        self._binary_hints = tuple(hint for hint in (binary_hints or ()) if isinstance(hint, str) and hint)
        self._companion_arg = companion
        self._root_arg = root
        self._workspace_arg = workspace
        # Durable runtimes reuse an owned root across process eviction. The
        # caller owns retention/removal; cleanup=True would delete it.
        self._durable_root = bool(durable_root)
        # Test subclasses may set these before start; production keeps them
        # empty so no fixture provider or fixture option ever loads.
        self.plugin_options: dict = {}
        self.config_overrides: dict = {}
        self.startup_timeout = float(startup_timeout)
        self.rpc_timeout = float(rpc_timeout)
        self.execute_deadline = float(execute_deadline)
        self.cleanup = cleanup
        self._env_overrides = dict(env or {})
        self.binary: Path | None = None
        self.companion: Path | None = None
        self.root: Path | None = None
        self.workspace: Path | None = None
        self.home: Path | None = None
        self.process: subprocess.Popen | None = None
        self.url: str | None = None
        self.password: str | None = None
        self.info: dict | None = None
        self.watchdog: Watchdog | None = None
        self._stdout: collections.deque[str] = collections.deque(maxlen=MAX_STDOUT_LINES)
        self._stderr: collections.deque[str] = collections.deque(maxlen=MAX_STDERR_LINES)
        self._ready: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._threads: list[threading.Thread] = []
        self._prepared = False
        self._started = False
        self._stopped = False
        self._lock = threading.RLock()

    # -- identity ----------------------------------------------------------
    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    @property
    def alive(self) -> bool:
        return bool(self.process is not None and self.process.poll() is None and not self._stopped)

    def _resolve_binary(self) -> Path:
        try:
            return Path(resolve_binary(self._binary_arg, hints=self._binary_hints)).resolve()
        except (BinaryUnavailable, OSError, ValueError) as error:
            raise BinaryRefused(str(error)) from None

    def _resolve_companion(self) -> Path:
        raw = self._companion_arg or os.environ.get(COMPANION_ENV) or Path(__file__).resolve().parent / "companion"
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            if not (path / "index.ts").is_file():
                raise BinaryRefused(f"bundled OpenCode companion has no index.ts: {path}")
            return path
        if not path.is_file():
            raise BinaryRefused(f"bundled OpenCode companion not found: {path}")
        return path

    def _check_version(self, version: str) -> None:
        parsed: tuple[int, ...] = ()
        try:
            parsed = tuple(int(part) for part in version.split("."))
        except ValueError:
            raise BinaryRefused(f"OpenCode version {version!r} is not a semantic version") from None
        if parsed < MIN_VERSION:
            raise BinaryRefused(
                f"OpenCode v{version} is unsupported; v{SUPPORTED_BINARY_VERSION} or newer is required"
            )

    def _version_of_binary(self, path: Path, timeout: float = 10.0) -> str:
        try:
            version, _parsed = probe_version(path, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise StartupTimeout(f"OpenCode version probe exceeded the startup budget: {error}") from None
        except BinaryUnavailable as error:
            raise BinaryRefused(str(error)) from None
        return version

    def _prepare(self, deadline: float | None = None) -> None:
        if self._prepared:
            return
        self.binary = self._resolve_binary()
        budget = 10.0 if deadline is None else max(1.0, deadline - time.monotonic())
        version = self._version_of_binary(self.binary, timeout=budget)
        self._check_version(version)
        self.companion = self._resolve_companion()
        # Own only a fresh mkdtemp child. The caller's root/workspace are parent
        # directories: never write, chmod, or delete their existing contents.
        base_arg = self._workspace_arg or self._root_arg or os.environ.get(ROOT_ENV)
        base = Path(base_arg).expanduser().resolve() if base_arg else Path(tempfile.gettempdir()).resolve()
        if not self._durable_root and not base.is_dir():
            raise RuntimeClosed(f"runtime parent directory does not exist: {base}")
        if self._durable_root:
            if not base_arg:
                raise RuntimeClosed("a durable runtime requires an explicit owned root")
            self.root = base.resolve()
            try:
                self.root.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise RuntimeClosed(f"durable runtime root is unusable: {error}") from None
        else:
            self.root = Path(tempfile.mkdtemp(prefix="hermes-oc-native-", dir=str(base))).resolve()
        self.workspace = self.root / "project"
        self.home = self.root / "home"
        for directory in (self.root, self.workspace, self.home, self.root / "tmp", self.root / "state", self.root / "cache", self.root / "data", self.root / "config"):
            directory.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(directory, 0o700)
            except OSError:
                pass
        (self.workspace / "AGENTS.md").write_text("Hermes OpenCode native runtime workspace\n")
        plugins: list[dict] = [{"package": str(self.companion)}]
        if self.plugin_options:
            plugins[0]["options"] = dict(self.plugin_options)
        config = {
            "$schema": "https://opencode.ai/config.json",
            "plugins": plugins,
            "compaction": {"auto": False},
            "snapshots": False,
            # Registers native `todowrite` (translated to Hermes todo_list and
            # advertised only while that tool is enabled).
            "experimental": {"todo": True},
        }
        for key, value in self.config_overrides.items():
            config[key] = value
        capture = os.environ.get("OC_CAPTURE_URL")
        if capture:
            # Experiment capture: route the provider through a local logging proxy.
            config["provider"] = {"opencode": {"options": {"baseURL": capture}}}
        config_path = self.workspace / "opencode.json"
        config_path.write_text(json.dumps(config, indent=2))
        try:
            os.chmod(config_path, 0o600)
        except OSError:
            pass
        try:
            subprocess.run(
                ["git", "init", "-q"],
                cwd=str(self.workspace),
                capture_output=True,
                timeout=10,
                env={"PATH": os.environ.get("PATH", ""), "HOME": str(self.home), "LANG": "C"},
            )
        except (OSError, subprocess.SubprocessError):
            pass
        self._prepared = True

    def _child_env(self) -> dict[str, str]:
        # Same allowlist as the version probe: Windows needs its loader vars to start.
        env = _probe_env()
        assert self.root is not None and self.home is not None
        env.update(
            HOME=str(self.home),
            XDG_CONFIG_HOME=str(self.root / "config"),
            XDG_DATA_HOME=str(self.root / "data"),
            XDG_CACHE_HOME=str(self.root / "cache"),
            XDG_STATE_HOME=str(self.root / "state"),
            TMPDIR=str(self.root / "tmp"),
            OPENCODE_PASSWORD=self.password or "",
            OPENCODE_PRINT_LOGS="0",
            NO_COLOR="1",
        )
        for key, value in self._env_overrides.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ControlRefused("runtime env overrides must be string keys and values")
            env[key] = value
        return env

    def start(self, timeout: float | None = None) -> "NativeRuntime":
        with self._lock:
            if self._started:
                if self.alive:
                    return self
                raise RuntimeClosed(f"OpenCode runtime exited: {self.stderr_tail()}")
            if self._stopped:
                raise RuntimeClosed("runtime was stopped")
            budget = timeout or self.startup_timeout
            deadline = time.monotonic() + budget
            self._prepare(deadline)
            assert self.binary and self.workspace and self.root
            self.password = secrets.token_urlsafe(32)
            cmd = [
                str(self.binary),
                "serve",
                "--stdio",
                "--port",
                "0",
                "--hostname",
                "127.0.0.1",
                "--log-level",
                "warning",
            ]
            try:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=str(self.workspace),
                    env=self._child_env(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    **_spawn_kwargs(),
                )
            except OSError as error:
                raise RuntimeClosed(f"could not start OpenCode: {error}") from None
            self._threads = [
                threading.Thread(target=self._drain_stdout, daemon=True, name="oc-stdout"),
                threading.Thread(target=self._drain_stderr, daemon=True, name="oc-stderr"),
            ]
            for thread in self._threads:
                thread.start()
            try:
                try:
                    line = self._ready.get(timeout=max(0.1, deadline - time.monotonic()))
                except queue.Empty:
                    raise StartupTimeout(
                        f"OpenCode startup timed out after {timeout or self.startup_timeout:.1f}s: {self.stderr_tail()}"
                    ) from None
                if not line:
                    raise StartupTimeout(f"OpenCode exited before readiness: {self.stderr_tail()}")
                info = json.loads(line)
                url = str(info.get("url") or "")
                self._validate_url(url)
                self.url = url.rstrip("/")
                self.watchdog = Watchdog(self.process, binary=self.binary, execute_deadline=self.execute_deadline)
                self.hello()
            except BaseException:
                self._terminate()
                raise
            self._started = True
            self._stopped = False
            return self

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "http":
            raise HandshakeRefused(f"runtime readiness URL is not loopback HTTP: {url!r}")
        if parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise HandshakeRefused(f"runtime readiness URL is not loopback: {url!r}")
        if parsed.username or parsed.password or parsed.path not in {"", "/"} or not parsed.port:
            raise HandshakeRefused(f"runtime readiness URL has an unexpected authority/path: {url!r}")

    def _drain_stdout(self) -> None:
        assert self.process and self.process.stdout
        first = True
        for line in self.process.stdout:
            text = line.rstrip("\n")
            self._stdout.append(text)
            if first:
                first = False
                try:
                    self._ready.put_nowait(text.strip())
                except queue.Full:
                    pass
        try:
            self._ready.put_nowait(None)
        except queue.Full:
            pass

    def _drain_stderr(self) -> None:
        assert self.process and self.process.stderr
        for line in self.process.stderr:
            self._stderr.append(line.rstrip("\n"))

    def stderr_tail(self) -> str:
        return "\n".join(list(self._stderr)[-40:])

    # -- RPC ---------------------------------------------------------------
    def _endpoint(self, method: str | None = None) -> str:
        if not self.url or not self.password:
            raise RuntimeClosed("runtime is not started")
        query = urllib.parse.urlencode({"location[directory]": str(self.workspace)})
        if method is None:
            raise RuntimeClosed("RPC method is required")
        return f"{self.url}/api/rpc/{COMPANION_ID}/{method}?{query}"

    def _request(self, method: str, path: str, payload: dict | None, timeout: float) -> dict:
        if not self.url or not self.password:
            raise RuntimeClosed("runtime is not started")
        if not path.startswith("/api/"):
            raise RpcRefused(f"refusing non-API path {path!r}", method=method)
        data = None if payload is None else json.dumps({"input": payload}, ensure_ascii=False).encode()
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers={
                "content-type": "application/json",
                "authorization": "Basic " + base64.b64encode(f"opencode:{self.password}".encode()).decode(),
            },
            method="POST" if data is not None else "GET",
        )
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise RpcRefused(f"{method} response exceeded {MAX_RESPONSE_BYTES} bytes", method=method)
        except urllib.error.HTTPError as error:
            try:
                raw = error.read(MAX_ERROR_BYTES).decode(errors="replace")
            finally:
                error.close()
            code = f"http.{error.code}"
            message = raw
            try:
                parsed = json.loads(raw)
                detail = parsed.get("error") if isinstance(parsed, dict) and parsed.get("error") else parsed
                if isinstance(detail, dict):
                    code = str(detail.get("code") or detail.get("type") or code)
                    message = str(detail.get("message") or raw)
                elif isinstance(detail, str):
                    message = detail
            except ValueError:
                pass
            raise RpcRefused(f"{method} HTTP {error.code}: {message[:1000]}", method=method, status=error.code, code=code) from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RpcRefused(f"{method} transport failed: {error}", method=method, code="runtime.transport") from None
        try:
            body = json.loads(raw or b"{}")
        except ValueError as error:
            raise RpcRefused(f"{method} returned non-JSON", method=method) from error
        if not isinstance(body, dict):
            raise RpcRefused(f"{method} returned a non-object envelope", method=method)
        if body.get("error"):
            detail = body["error"]
            if isinstance(detail, dict):
                raise RpcRefused(
                    str(detail.get("message") or detail),
                    method=method,
                    code=str(detail.get("code") or detail.get("type") or "runtime.rpc"),
                    details=detail,
                )
            raise RpcRefused(str(detail), method=method)
        return body.get("output", body)

    def rpc(self, method: str, payload: dict | None = None, *, timeout: float | None = None) -> dict:
        return self._request(method, f"/api/rpc/{COMPANION_ID}/{method}?{urllib.parse.urlencode({'location[directory]': str(self.workspace)})}", payload, timeout or self.rpc_timeout)

    def get(self, path: str, *, timeout: float | None = None) -> dict:
        if not path.startswith("/api/"):
            raise RpcRefused(f"refusing non-API GET {path!r}")
        query = urllib.parse.urlencode({"location[directory]": str(self.workspace)})
        separator = "&" if "?" in path else "?"
        return self._request("GET", f"{path}{separator}{query}", None, timeout or self.rpc_timeout)

    # -- handshake ---------------------------------------------------------
    def hello(self, timeout: float | None = None, *, refresh: bool = False) -> dict:
        if self.info is not None and not refresh:
            return self.info
        deadline = time.monotonic() + (timeout or self.startup_timeout)
        last: str | None = None
        while time.monotonic() < deadline:
            if not self.alive:
                raise HandshakeRefused(f"OpenCode exited during handshake: {self.stderr_tail()}")
            try:
                output = self.rpc("hello", {}, timeout=min(5.0, max(1.0, deadline - time.monotonic())))
            except RpcRefused as error:
                last = str(error)
                if error.code not in {"rpc.unavailable", "rpc.internal", "runtime.transport"}:
                    raise HandshakeRefused(f"companion handshake failed: {error}") from None
                time.sleep(0.15)
                continue
            self.validate_handshake(output)
            self.info = output
            return output
        raise HandshakeRefused(f"companion RPC never became available: {last}")

    def reconcile(self, timeout: float | None = None) -> dict:
        """Ask the companion to wrap late native tools before one generation."""
        return self.hello(timeout=timeout, refresh=True)

    @staticmethod
    def validate_handshake(output: dict) -> None:
        if not isinstance(output, dict):
            raise HandshakeRefused("companion handshake is not an object")
        if output.get("protocol") != SUPPORTED_COMPANION_PROTOCOL:
            raise HandshakeRefused(
                f"companion protocol {output.get('protocol')!r} is unsupported; expected {SUPPORTED_COMPANION_PROTOCOL}"
            )
        if output.get("companion") != COMPANION_ID:
            raise HandshakeRefused(f"unexpected companion id {output.get('companion')!r}")
        version = str(output.get("binaryVersion") or "")
        try:
            parsed = tuple(int(part) for part in version.split("."))
        except ValueError:
            parsed = ()
        if parsed < MIN_VERSION:
            raise HandshakeRefused(f"OpenCode version {version!r} is unsupported")
        tools = output.get("tools")
        if not isinstance(tools, list):
            raise HandshakeRefused("companion handshake has no tool inventory")
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("forwarded") is not True:
                raise HandshakeRefused("companion exposes an unwrapped native tool")

    # -- catalog -----------------------------------------------------------
    def catalog(self, *, timeout: float | None = None) -> list[dict]:
        records: Any = None
        try:
            output = self.rpc("catalog", {}, timeout=timeout or self.rpc_timeout)
            if isinstance(output, dict):
                records = output.get("models", output.get("data"))
        except RpcRefused as error:
            if error.code not in {"rpc.unavailable", "rpc.internal", "runtime.transport", "http.404"}:
                raise CatalogRefused(f"native catalog RPC failed: {error}") from None
        if not isinstance(records, list):
            payload = self.get("/api/model", timeout=timeout or self.rpc_timeout)
            records = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(records, list) and isinstance(payload, dict):
                records = payload.get("data")
        if not isinstance(records, list):
            raise CatalogRefused("native model discovery returned no catalog list")
        return sanitize_catalog(records)

    def list_models(self, *, timeout: float | None = None) -> list[str]:
        return [str(row["id"]) for row in self.catalog(timeout=timeout) if isinstance(row.get("id"), str)]

    # -- sessions ----------------------------------------------------------
    def _guard_alive(self) -> None:
        if self._stopped or not self.alive:
            raise RuntimeClosed(f"OpenCode runtime is not running: {self.stderr_tail()}")
        if self.watchdog is not None and self.watchdog.failing:
            reason = (self.watchdog.failure or {}).get("reason", "runtime teardown in progress")
            raise RuntimeClosed(f"OpenCode runtime failed: {reason}")

    def new_session(
        self,
        provider_id: str,
        model_id: str,
        hermes_tools: list[dict] | None = None,
        *,
        session_id: str | None = None,
        native_tools: Iterable[str] | None = None,
    ) -> "Session":
        self._guard_alive()
        if not isinstance(provider_id, str) or not provider_id or not isinstance(model_id, str) or not model_id:
            raise SessionRefused("provider_id and model_id must be non-empty strings")
        payload: dict[str, Any] = {"providerID": provider_id, "modelID": model_id}
        if hermes_tools is not None:
            payload["hermesTools"] = hermes_tools
        if native_tools is not None:
            payload["nativeTools"] = sorted(native_tools)
        if session_id is not None:
            if not isinstance(session_id, str) or not session_id:
                raise SessionRefused("session_id must be a non-empty string when supplied")
            payload["sessionID"] = session_id
        deadline = time.monotonic() + 15.0
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                output = self.rpc("open", payload, timeout=min(30.0, max(2.0, deadline - time.monotonic())))
                break
            except RpcRefused as error:
                last = error
                if error.code not in {"rpc.unavailable", "rpc.internal", "runtime.transport"}:
                    raise SessionRefused(f"OpenCode session creation failed: {error}") from None
                if not self.alive:
                    raise RuntimeClosed(f"OpenCode died during session creation: {self.stderr_tail()}") from None
                time.sleep(0.2)
        else:
            raise SessionRefused(f"OpenCode session creation timed out: {last}")
        session_id = str((output or {}).get("sessionID") or "")
        if not session_id:
            raise SessionRefused("companion returned no session id")
        return Session(self, session_id)

    def get_session(self, session_id: str, *, timeout: float | None = None) -> dict | None:
        """Corroborate one persisted native session; None only on a real 404."""
        self._guard_alive()
        if not isinstance(session_id, str) or not session_id:
            raise SessionRefused("a native session id is required")
        path = f"/api/session/{urllib.parse.quote(session_id, safe='')}"
        try:
            payload = self.get(path, timeout=timeout or min(15.0, self.rpc_timeout))
        except RpcRefused as error:
            if error.status == 404 or error.code == "http.404":
                return None
            raise
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise SessionRefused("native session get returned no session object")
        if str(data.get("id") or "") != session_id:
            raise SessionRefused("native session get returned a different session id")
        return data

    # -- ownership / teardown ---------------------------------------------
    def verify_ownership(self) -> int:
        if self.watchdog is not None:
            return self.watchdog.verify_ownership()
        if self.process is None:
            raise WatchdogRefused("no owned process")
        pid = self.process.pid
        if self.process.poll() is not None:
            raise WatchdogRefused(f"pid {pid} already exited")
        if not _POSIX:
            # Popen owns the live child handle; there is no /proc identity to read.
            return pid
        if os.getpgid(pid) != pid:
            raise WatchdogRefused(f"pid {pid} is not its own process-group leader")
        stat_path = Path(f"/proc/{pid}/stat")
        if stat_path.is_file():
            fields = stat_path.read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) != os.getpid():
                raise WatchdogRefused(f"pid {pid} is not our child")
        return pid

    def _signal_group(self, sig: int) -> bool:
        """Signal the owned group when ownership is proven; False otherwise."""
        try:
            pid = self.verify_ownership()
        except WatchdogRefused:
            return False
        try:
            if _POSIX:
                os.killpg(pid, sig)
            elif sig == getattr(signal, "SIGKILL", signal.SIGTERM):
                self.process.kill()  # type: ignore[union-attr]
            else:
                self.process.terminate()  # type: ignore[union-attr]
        except (ProcessLookupError, OSError):
            return False
        return True

    def _shutdown(self, process: subprocess.Popen, timeout: float) -> int:
        """Stop one owned process tree: graceful, then proven signal, then handle kill."""
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=max(0.1, timeout))
        except subprocess.TimeoutExpired:
            if not self._signal_group(signal.SIGTERM):
                if process.poll() is None:
                    try:
                        process.kill()
                    except OSError:
                        pass
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._signal_group(getattr(signal, "SIGKILL", signal.SIGTERM))
                if process.poll() is None:
                    try:
                        process.kill()
                    except OSError:
                        pass
                try:
                    process.wait(timeout=5.0)
                except (OSError, subprocess.SubprocessError):
                    pass
        for pipe in (process.stdout, process.stderr):
            if pipe:
                try:
                    pipe.close()
                except OSError:
                    pass
        for thread in self._threads:
            thread.join(timeout=1.0)
        return process.returncode if process.returncode is not None else 0

    def _terminate(self) -> None:
        if self.process is None:
            return
        if self.watchdog is not None:
            self.watchdog.end()
        self._shutdown(self.process, 10.0)
        if self.cleanup and self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)

    def stop(self, timeout: float | None = None) -> int:
        with self._lock:
            if self._stopped:
                return self.process.returncode if self.process is not None and self.process.returncode is not None else 0
            self._stopped = True
            process = self.process
            if process is None:
                if self.cleanup and self.root is not None:
                    shutil.rmtree(self.root, ignore_errors=True)
                return 0
            if self.watchdog is not None:
                self.watchdog.end()
            code = self._shutdown(process, timeout if timeout is not None else 10.0)
            if self.cleanup and self.root is not None:
                shutil.rmtree(self.root, ignore_errors=True)
            return code

    close = stop

    def __enter__(self) -> "NativeRuntime":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


class Session:
    """One OpenCode session; exactly one authorized generation at a time."""

    def __init__(self, runtime: NativeRuntime, session_id: str):
        self.runtime = runtime
        self.session_id = session_id
        self.generation = 0
        self.cursor = 0
        self.closed = False
        self._pending: collections.deque[dict] = collections.deque()
        self._lock = threading.RLock()

    def _guard(self) -> None:
        if self.closed:
            raise SessionRefused(f"session {self.session_id} is closed")
        self.runtime._guard_alive()

    def _read_output(self, timeout: float) -> dict:
        output = self.runtime.rpc(
            "read",
            {"sessionID": self.session_id, "cursor": self.cursor},
            timeout=max(2.0, timeout),
        )
        events = output.get("events") or []
        if not isinstance(events, list):
            raise EventGap("companion read returned non-list events")
        for event in events:
            if not isinstance(event, dict):
                raise EventGap("companion read returned a non-object event")
            seq = event.get("seq")
            if not isinstance(seq, int) or seq <= 0:
                raise EventGap("companion event has no positive sequence")
            if seq != self.cursor + 1:
                raise EventGap(f"event cursor gap: expected {self.cursor + 1}, received {seq}")
            self.cursor = seq
            self._pending.append(event)
        return output

    def _take_pending(self) -> list[dict]:
        events = list(self._pending)
        self._pending.clear()
        return events

    def read(self, timeout: float = 5.0) -> dict:
        self._guard()
        if self._pending:
            return {"events": self._take_pending(), "state": self.state()}
        output = self._read_output(timeout)
        output = dict(output)
        output["events"] = self._take_pending()
        return output

    def events_until(self, predicate: Callable[[dict], bool], timeout: float = 60.0) -> list[dict]:
        self._guard()
        deadline = time.monotonic() + timeout
        collected: list[dict] = []
        while time.monotonic() < deadline:
            remaining = max(0.2, deadline - time.monotonic())
            if self._pending:
                events = self._take_pending()
                state: dict = {}
            else:
                output = self._read_output(min(30.0, remaining))
                events = self._take_pending()
                state = output.get("state") or {}
            for index, event in enumerate(events):
                collected.append(event)
                if predicate(event):
                    # A read may return several events; keep the tail for the
                    # next boundary instead of advancing the cursor past it.
                    self._pending.extend(events[index + 1 :])
                    return collected
            if not events and state.get("closed"):
                break
        raise CompletionRefused(f"no matching native event within {timeout:.1f}s (session {self.session_id})")

    def advance(
        self,
        *,
        snapshot: list[dict],
        system_parts: list[str],
        user_text: str,
        timeout: float = 60.0,
        hermes_tools: list[dict] | None = None,
        native_tools: Iterable[str] | None = None,
    ) -> dict:
        self._guard()
        generation = self.generation + 1
        payload = {
            "sessionID": self.session_id,
            "generation": generation,
            "snapshot": snapshot,
            "systemParts": list(system_parts),
            "userText": user_text,
        }
        if hermes_tools is not None:
            payload["hermesTools"] = hermes_tools
        if native_tools is not None:
            payload["nativeTools"] = sorted(native_tools)
        output = self.runtime.rpc("advance", payload, timeout=timeout)
        self.generation = generation
        # Anything emitted before the authorization belongs to an older
        # generation (late interrupt/step events). Start the read window after
        # the boundary event so it cannot be mistaken for this generation.
        seq = output.get("seq") if isinstance(output, dict) else None
        if isinstance(seq, int) and not isinstance(seq, bool):
            self.cursor = max(self.cursor, seq)
        self._pending.clear()
        return output

    def state(self, timeout: float = 30.0) -> dict:
        self._guard()
        return self.runtime.rpc("state", {"sessionID": self.session_id}, timeout=timeout)

    def wait_pending(self, call_ids: list[str], timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        last: dict = {}
        while time.monotonic() < deadline:
            last = self.state()
            if sorted(last.get("pending", [])) == sorted(call_ids):
                return last
            time.sleep(0.02)
        raise SessionRefused(f"native executors never registered: {last.get('pending')!r}")

    def decide(self, call_id: str, allow: bool, reason: str | None = None, timeout: float = 30.0) -> dict:
        self._guard()
        payload: dict[str, Any] = {"sessionID": self.session_id, "callID": call_id, "allow": bool(allow)}
        if reason is not None:
            payload["reason"] = str(reason)
        return self.runtime.rpc("decide", payload, timeout=timeout)

    def settle(self, results: list[dict], timeout: float = 30.0) -> dict:
        self._guard()
        if not isinstance(results, list) or not results:
            raise SessionRefused("settle requires a non-empty result list")
        try:
            return self.runtime.rpc("results", {"sessionID": self.session_id, "results": results}, timeout=timeout)
        except RpcRefused as error:
            # Companion protocol 3 accepts `settle` as a one-revision alias.
            if error.code not in {"rpc.unavailable", "rpc.internal", "http.404"}:
                raise
            return self.runtime.rpc("settle", {"sessionID": self.session_id, "results": results}, timeout=timeout)

    def finish(self, timeout: float = 20.0) -> dict:
        """Settle the previous generation, keeping the native session identity.

        Parked native tool executions are rejected, an in-flight prompt is
        interrupted, and the event cursor advances past that generation's tail.
        The next ``advance`` installs a fresh canonical snapshot.
        """
        self._guard()
        output = self.runtime.rpc("finish", {"sessionID": self.session_id}, timeout=timeout)
        if not isinstance(output, dict):
            raise SessionRefused("companion finish returned a non-object")
        seq = output.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            self.cursor = max(self.cursor, seq)
        self._pending.clear()
        return output

    def close(self) -> dict:
        with self._lock:
            if self.closed:
                return {"closed": True}
            self.closed = True
            if not self.runtime.alive:
                self._pending.clear()
                return {"closed": True, "skipped": "runtime not running"}
            try:
                return self.runtime.rpc("close", {"sessionID": self.session_id}, timeout=15.0)
            except (RpcRefused, RuntimeClosed):
                return {"closed": True, "skipped": "runtime unavailable"}
            finally:
                self._pending.clear()

    def _completion(self, events: Iterable[dict]) -> dict | None:
        for event in events:
            if event.get("type") != "completion":
                continue
            generation = event.get("generation")
            if generation is not None and generation != self.generation:
                continue
            return event
        return None

    @staticmethod
    def _refusal(events: Iterable[dict], generation: int) -> dict | None:
        events = list(events)
        # A length-finished response makes OpenCode ask for an automatic
        # continuation. Its `unauthorized` event is evidence the gate held, not
        # a failure of the completed primary step.
        valid_completion = any(
            event.get("type") == "completion"
            and event.get("ok") is True
            and event.get("generation") in {None, generation}
            and event.get("finish") in {"stop", "length", "tool_calls"}
            for event in events
        )
        for event in events:
            kind = event.get("type")
            if kind not in {"step_failed", "unauthorized", "tool_guard", "prompt_error", "response_unverified", "correlation_error"}:
                continue
            if kind == "unauthorized" and valid_completion:
                continue
            event_generation = event.get("generation")
            if kind == "step_failed" and event_generation not in {None, generation}:
                continue
            if kind in {"unauthorized", "tool_guard", "prompt_error", "correlation_error", "response_unverified"}:
                return event
            return event
        return None

    def _validate_calls(self, raw_calls: Any) -> list[ToolCall]:
        if not isinstance(raw_calls, list):
            raise CompletionRefused("native step calls are not a list")
        calls: list[ToolCall] = []
        seen: set[str] = set()
        for raw in raw_calls:
            if not isinstance(raw, dict):
                raise CompletionRefused("native tool call is not an object")
            call_id = str(raw.get("callID") or "")
            name = str(raw.get("name") or "")
            if not call_id or not name:
                raise CompletionRefused("native tool call has no id or name")
            if call_id in seen:
                raise CompletionRefused(f"duplicate native tool call id {call_id!r}")
            arguments = raw.get("input")
            if not isinstance(arguments, dict):
                raise CompletionRefused(f"native tool call {call_id!r} input is not an object")
            seen.add(call_id)
            calls.append(ToolCall(call_id=call_id, name=name, arguments=arguments, native=raw.get("native")))
        return calls

    @staticmethod
    def _content(
        events: Iterable[dict], assistant_id: str, calls: list[ToolCall]
    ) -> tuple[str, str | None, dict]:
        """Native text/reasoning/opaque metadata for one assistant message.

        Text and reasoning come from OpenCode's own ending events; the raw
        observer never supplies content. Metadata is copied verbatim with the
        exact part position, so sparse metadata replays onto the same part.
        """
        texts: list[str] = []
        reasonings: list[str] = []
        parts: list[dict] = []
        for event in events:
            if event.get("assistantMessageID") != assistant_id:
                continue
            kind = event.get("type")
            if kind == "reasoning":
                reasonings.append(str(event.get("text") or ""))
                parts.append({"type": "reasoning", "index": len(parts), "native": event.get("native")})
            elif kind == "text":
                texts.append(str(event.get("text") or ""))
                parts.append({"type": "text", "index": len(parts), "native": event.get("native")})
        for call in calls:
            parts.append({"type": "tool-call", "index": len(parts), "id": call.call_id, "native": call.native})
        return "".join(texts), ("".join(reasonings) or None), {"parts": parts}

    def generate(
        self,
        *,
        snapshot: list[dict],
        system_parts: list[str],
        user_text: str,
        timeout: float = 120.0,
        hermes_tools: list[dict] | None = None,
        native_tools: Iterable[str] | None = None,
    ) -> StepResult:
        """Issue one generation permit and return only a validated completion."""
        self._guard()
        # Late official plugins can add tools after `open`; reconcile before
        # the context hook so every advertised tool has a wrapper.
        self.runtime.reconcile()
        deadline = time.monotonic() + timeout
        self.advance(
            snapshot=snapshot,
            system_parts=system_parts,
            user_text=user_text,
            timeout=min(60.0, timeout),
            hermes_tools=hermes_tools,
            native_tools=native_tools,
        )
        generation = self.generation

        def boundary(event: dict) -> bool:
            kind = event.get("type")
            if kind == "unauthorized":
                # Ignore the automatic continuation after a validated length
                # stop; it was refused before vendor traffic.
                valid = any(
                    previous.get("type") == "completion"
                    and previous.get("ok") is True
                    and previous.get("generation") in {None, generation}
                    for previous in seen_events
                )
                return not valid
            if kind in {"tool_guard", "prompt_error", "response_unverified", "correlation_error"}:
                return True
            event_generation = event.get("generation")
            if event_generation is not None and event_generation != generation:
                return False
            if kind == "step_failed":
                return True
            return kind == "step_streamed" and event.get("primary", True) is not False

        seen_events: list[dict] = []

        def watched(event: dict) -> bool:
            seen_events.append(event)
            return boundary(event)

        boundary_events = self.events_until(watched, timeout=max(0.1, min(120.0, deadline - time.monotonic())))
        refusal = self._refusal(boundary_events, generation)
        if refusal is not None:
            reason = refusal.get("reason") or refusal.get("message") or refusal.get("error") or refusal
            raise CompletionRefused(f"native response refused: {reason}", status_code=_upstream_status(refusal))
        streamed = next(
            (
                event
                for event in reversed(boundary_events)
                if event.get("type") == "step_streamed" and event.get("primary", True) is not False
            ),
            None,
        )
        if streamed is None:
            raise CompletionRefused("native step did not reach a primary streamed boundary")
        if streamed.get("completionValidated") is not True:
            unverified = next(
                (event for event in boundary_events if event.get("type") == "response_unverified"), {}
            )
            reason = unverified.get("reason") or "response completion was not validated"
            raise CompletionRefused(f"native response completion rejected: {reason}",
                                    status_code=_upstream_status(unverified))
        assistant_id = str(streamed.get("assistantMessageID") or "")
        calls = self._validate_calls(streamed.get("calls") or [])
        completion = self._completion(boundary_events)

        if calls:
            if completion is None:
                raise CompletionRefused("native tool calls were exposed without an observed completion")
            self._validate_completion(completion, generation, expected_calls=len(calls), allow_length=False)
            finish = str(completion.get("finish") or "tool_calls")
            text, reasoning, native_parts = self._content(boundary_events, assistant_id, calls)
            return StepResult(
                text=text,
                reasoning=reasoning,
                calls=calls,
                finish="tool_calls" if finish == "stop" else finish,
                usage=completion.get("usage") if isinstance(completion.get("usage"), dict) else None,
                generation=generation,
                assistant_message_id=assistant_id,
                session=self,
                completion=completion,
                native_parts=native_parts,
            )

        final_events = self.events_until(
            lambda event: event.get("generation") in {None, generation}
            and event.get("primary", True) is not False
            and event.get("type") in {"step_ended", "step_failed"},
            timeout=max(0.1, min(120.0, deadline - time.monotonic())),
        )
        failed = next(
            (event for event in final_events if event.get("type") == "step_failed" and event.get("generation") in {None, generation}),
            None,
        )
        if failed is not None:
            raise CompletionRefused(f"native step failed: {failed.get('error') or failed}",
                                    status_code=_upstream_status(failed))
        ended = next(
            (
                event
                for event in reversed(final_events)
                if event.get("type") == "step_ended"
                and event.get("generation") in {None, generation}
                and event.get("primary", True) is not False
            ),
            None,
        )
        if ended is None:
            raise CompletionRefused("native step ended without a terminal step event")
        if completion is None:
            completion = self._completion(final_events)
        if completion is None:
            try:
                completion_events = self.events_until(
                    lambda event: event.get("type") == "completion" and event.get("generation") in {None, generation},
                    timeout=max(0.1, min(5.0, deadline - time.monotonic())),
                )
                completion = self._completion(completion_events)
            except CompletionRefused:
                completion = None
        if completion is None:
            raise CompletionRefused("native response completion was not observable")
        self._validate_completion(completion, generation, expected_calls=0, allow_length=True)
        finish = str(ended.get("finish") or "")
        if finish not in {"stop", "length"}:
            raise CompletionRefused(f"native step ended with unsupported finish reason {finish!r}")
        if completion.get("finish") != finish:
            raise CompletionRefused(
                f"conflicting finish reasons: completion={completion.get('finish')!r} step={finish!r}"
            )
        text, reasoning, native_parts = self._content(boundary_events + final_events, assistant_id, [])
        return StepResult(
            text=text,
            reasoning=reasoning,
            calls=[],
            finish=finish,
            usage=completion.get("usage") if isinstance(completion.get("usage"), dict) else None,
            generation=generation,
            assistant_message_id=assistant_id,
            session=self,
            step_usage=ended.get("usage") if isinstance(ended.get("usage"), dict) else None,
            completion=completion,
            native_parts=native_parts,
        )

    @staticmethod
    def _validate_completion(completion: dict, generation: int, *, expected_calls: int, allow_length: bool) -> None:
        if completion.get("ok") is not True or completion.get("done") is not True:
            reason = completion.get("error") or "completion was not positively validated"
            raise CompletionRefused(f"native response completion rejected: {reason}",
                                    status_code=_upstream_status(completion))
        if completion.get("generation") not in {None, generation}:
            raise CompletionRefused("native completion belongs to a different generation")
        if completion.get("error"):
            # ponytail: the companion never sends ok=true with an error; kept consistent anyway.
            raise CompletionRefused(f"native response error: {completion['error']}",
                                    status_code=_upstream_status(completion))
        finish = completion.get("finish")
        if finish is None:
            raise CompletionRefused("native response has no terminal finish reason")
        if finish not in {"stop", "tool_calls", "length"}:
            raise CompletionRefused(f"native response has unsupported finish reason {finish!r}")
        if not _valid_usage(completion.get("usage")):
            raise CompletionRefused("native response has no valid prompt/completion token usage")
        if not allow_length and finish == "length":
            raise CompletionRefused("length-finished response cannot release tool calls")
        if expected_calls == 0 and finish == "tool_calls":
            raise CompletionRefused("finish_reason tool_calls without any native tool call")
        if expected_calls > 0 and finish not in {"tool_calls", "stop"}:
            raise CompletionRefused(f"unexpected finish reason for tool calls: {finish!r}")
        if completion.get("calls") is not None and completion.get("calls") != expected_calls:
            raise CompletionRefused("completion call count does not match the parsed native calls")
