"""Hermes client for direct OpenCode free-model inference.

The client reproduces OpenCode's free-model request envelope while keeping the
agent loop and every tool execution in Hermes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import signal
import string
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

DIRECT_URL = "https://opencode.ai/zen/v1/chat/completions"
RESPONSES_URL = "https://opencode.ai/zen/v1/responses"
LOGICAL_BASE_URL = DIRECT_URL
DEFAULT_MODEL = "big-pickle"
FALLBACK_MODELS = (
    DEFAULT_MODEL,
    "ling-3.0-flash-fin-free",
    "mimo-v2.5-free",
    "muse-spark-1.2-contributor-free",
    "muse-spark-1.3-contributor-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
)
RESPONSES_MODELS = frozenset(
    {
        "muse-spark-1.2-contributor-free",
        "muse-spark-1.3-contributor-free",
    }
)
COMPAT_TOOL_NAMES = (
    "bash",
    "edit",
    "glob",
    "grep",
    "read",
    "skill",
    "task",
    "todowrite",
    "webfetch",
    "websearch",
    "write",
)
_BASE62 = string.digits + string.ascii_letters
_SESSION_LOCK = threading.Lock()
_SESSION_COUNTER = 0


class OpenCodeError(RuntimeError):
    status_code: int | None = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _run_opencode(
    argv: list[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    from tools.environments.local import (  # type: ignore[import-not-found]
        hermes_subprocess_env,
    )

    process = subprocess.Popen(
        argv,
        env=hermes_subprocess_env(inherit_credentials=False),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _status_error(code: int, body: str, context: str) -> OpenCodeError:
    exc = OpenCodeError(f"{context} failed with HTTP {code}: {body[:500]}")
    exc.status_code = code
    return exc


def _urlopen(request: urllib.request.Request, timeout: float):
    return _OPENER.open(request, timeout=timeout)


def _session_id() -> str:
    global _SESSION_COUNTER
    with _SESSION_LOCK:
        _SESSION_COUNTER = (_SESSION_COUNTER + 1) & 0xFFF
        prefix = ((int(time.time() * 1000) << 12) | _SESSION_COUNTER) & ((1 << 48) - 1)
    return (
        "ses_"
        + prefix.to_bytes(6, "big").hex()
        + "".join(secrets.choice(_BASE62) for _ in range(14))
    )


def _compat_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Compatibility marker; unavailable. Never call.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
        return ""
    return str(tool["function"].get("name") or "").strip()


def _wire_tools(
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], set[str]]:
    actual = {_tool_name(tool) for tool in tools or []} - {""}
    by_name = {_tool_name(tool): tool for tool in tools or [] if _tool_name(tool)}
    wire = [by_name.get(name) or _compat_tool(name) for name in COMPAT_TOOL_NAMES]
    wire.extend(
        tool for tool in tools or [] if _tool_name(tool) not in COMPAT_TOOL_NAMES
    )
    return wire, actual


def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        function = tool.get("function") or {}
        result.append(
            {
                "type": "function",
                "name": str(function.get("name") or ""),
                "description": str(function.get("description") or ""),
                "parameters": function.get("parameters") or {"type": "object"},
                "strict": bool(function.get("strict", False)),
            }
        )
    return result


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("text") is not None
        )
    return "" if content is None else str(content)


def _responses_content(content: Any, *, output: bool = False) -> list[dict[str, Any]]:
    text_type = "output_text" if output else "input_text"
    if not isinstance(content, list):
        text = _message_text(content)
        return [{"type": text_type, "text": text}] if text else []
    result = []
    for part in content:
        if not isinstance(part, dict):
            result.append({"type": text_type, "text": str(part)})
            continue
        kind = str(part.get("type") or "")
        if kind in {"text", "input_text", "output_text"}:
            result.append({"type": text_type, "text": str(part.get("text") or "")})
        elif not output and kind in {"image_url", "input_image"}:
            image = part.get("image_url")
            if isinstance(image, dict):
                image = image.get("url")
            if image:
                result.append({"type": "input_image", "image_url": str(image)})
    return result


def _responses_input(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    instructions = ["You are opencode"]
    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "system":
            if text := _message_text(message.get("content")):
                instructions.append(text)
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or message.get("call_id") or "")
            if call_id:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": _message_text(message.get("content")),
                    }
                )
            continue
        if role == "assistant":
            for detail in message.get("reasoning_details") or []:
                if isinstance(detail, dict) and detail.get("type") == "reasoning":
                    items.append(dict(detail))
            if content := _responses_content(message.get("content"), output=True):
                items.append({"role": "assistant", "content": content})
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                arguments = function.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, separators=(",", ":"))
                call_id = str(call.get("id") or call.get("call_id") or "")
                if call_id:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": str(function.get("name") or ""),
                            "arguments": arguments,
                        }
                    )
            continue
        content = _responses_content(message.get("content"))
        if content:
            items.append(
                {"role": "user" if role == "user" else role, "content": content}
            )
    return items, "\n\n".join(instructions)


def _responses_tool_choice(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    function = value.get("function")
    if value.get("type") == "function" and isinstance(function, dict):
        return {"type": "function", "name": str(function.get("name") or "")}
    return value


def _effective_timeout(value: Any, default: float = 900.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    values = [
        getattr(value, key, None)
        for key in ("read", "write", "connect", "pool", "timeout")
    ]
    return max(
        (float(item) for item in values if isinstance(item, (int, float))),
        default=default,
    )


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _usage(value: Any = None) -> SimpleNamespace:
    raw = value if isinstance(value, dict) else {}
    prompt = int(raw.get("prompt_tokens", raw.get("input", 0)) or 0)
    completion = int(raw.get("completion_tokens", raw.get("output", 0)) or 0)
    total = int(raw.get("total_tokens", prompt + completion) or prompt + completion)
    raw_details = raw.get("prompt_tokens_details")
    details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=int(details.get("cached_tokens", 0) or 0)
        ),
    )


def _tool_calls_from_parts(parts: dict[int, dict[str, Any]]) -> list[SimpleNamespace]:
    result = []
    for index in sorted(parts):
        call = parts[index]
        result.append(
            SimpleNamespace(
                id=call.get("id") or f"call_{index}",
                call_id=call.get("id") or f"call_{index}",
                type=call.get("type") or "function",
                function=SimpleNamespace(
                    name=call.get("name") or "", arguments=call.get("arguments") or "{}"
                ),
                response_item_id=None,
            )
        )
    return result


def _phantom_tool_error(name: str, actual: set[str]) -> OpenCodeError | None:
    if name in COMPAT_TOOL_NAMES and name not in actual:
        return OpenCodeError(
            f"OpenCode attempted compatibility-only tool '{name}'. "
            "It was not supplied by Hermes and was not executed."
        )
    return None


def _merge_sse(
    events: Iterator[dict[str, Any]], *, requested_model: str, actual_tools: set[str]
) -> SimpleNamespace:
    content: list[str] = []
    reasoning: list[str] = []
    reasoning_details: list[Any] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason = "stop"
    usage: Any = None
    model = requested_model
    seen = False
    for event in events:
        seen = True
        if event.get("error"):
            raise OpenCodeError(f"OpenCode inference error: {event['error']}")
        model = str(event.get("model") or model)
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if delta.get("content") is not None:
                content.append(str(delta["content"]))
            for key in ("reasoning_content", "reasoning"):
                if delta.get(key) is not None:
                    reasoning.append(str(delta[key]))
                    break
            details = delta.get("reasoning_details")
            if isinstance(details, list):
                reasoning_details.extend(details)
            elif details is not None:
                reasoning_details.append(details)
            for raw_call in delta.get("tool_calls") or []:
                index = int(raw_call.get("index", 0) or 0)
                item = calls.setdefault(
                    index, {"id": "", "type": "function", "name": "", "arguments": ""}
                )
                if raw_call.get("id"):
                    item["id"] = str(raw_call["id"])
                item["type"] = str(raw_call.get("type") or item["type"])
                function = raw_call.get("function") or {}
                if function.get("name"):
                    item["name"] = str(function["name"])
                item["arguments"] += str(function.get("arguments") or "")
                if error := _phantom_tool_error(item["name"], actual_tools):
                    raise error
    if not seen:
        raise OpenCodeError("OpenCode inference returned no events.")
    if finish_reason == "content_filter":
        raise OpenCodeError("OpenCode inference was blocked by the content filter.")
    tool_calls = _tool_calls_from_parts(calls)
    text = "".join(content) or None
    thought = "".join(reasoning) or None
    return SimpleNamespace(
        id=None,
        object="chat.completion",
        model=model,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="tool_calls" if tool_calls else finish_reason,
                message=SimpleNamespace(
                    role="assistant",
                    content=text,
                    tool_calls=tool_calls,
                    reasoning=thought,
                    reasoning_content=thought,
                    reasoning_details=reasoning_details or None,
                ),
            )
        ],
        usage=_usage(usage),
    )


def _next_or_done(iterator: Iterator[Any]) -> tuple[bool, Any]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


class _LazyValue:
    def __init__(self, factory):
        self._factory = factory
        self._value = None
        self._ready = False
        self._lock = threading.Lock()

    def _resolve(self):
        if not self._ready:
            with self._lock:
                if not self._ready:
                    self._value = self._factory()
                    self._ready = True
        return self._value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __await__(self):
        return asyncio.to_thread(self._resolve).__await__()


class _LazyStream:
    def __init__(self, factory):
        self._factory = factory
        self._iterator = None

    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self._factory())
        return self._iterator

    def close(self) -> None:
        close = getattr(self._iterator, "close", None)
        if callable(close):
            close()

    def __await__(self):
        async def _self():
            return self

        return _self().__await__()

    def __aiter__(self):
        async def _iterate():
            iterator = iter(self)
            while True:
                ok, item = await asyncio.to_thread(_next_or_done, iterator)
                if not ok:
                    return
                yield item

        return _iterate()


class OpenCodeClient:
    """Minimal OpenAI-compatible client used by the Hermes provider profile."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "opencode-public"
        self.base_url = base_url or LOGICAL_BASE_URL
        self.command = command or "opencode"
        self.command_argv = [self.command, *(args or [])]
        if (
            self.command == "opencode"
            and not shutil.which(self.command)
            and shutil.which("npx")
        ):
            # ponytail: official package fallback; a configured binary always wins.
            self.command_argv = ["npx", "--yes", "opencode-ai"]
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_chat_completion)
        )
        self.is_closed = False
        self._version: str | None = None
        self._session = _session_id()

    def close(self) -> None:
        self.is_closed = True

    def _opencode_version(self) -> str:
        if self._version is not None:
            return self._version
        try:
            probe = _run_opencode([*self.command_argv, "--version"], timeout=15)
            match = re.search(r"\d+\.\d+(?:\.\d+)?", probe.stdout)
            if match is None:
                match = re.search(r"\d+\.\d+(?:\.\d+)?", probe.stderr)
            self._version = match.group(0) if match else "1.18.31"
        except (OSError, subprocess.SubprocessError):
            self._version = "1.18.31"
        return self._version

    def list_models(self, *, timeout: float = 15.0) -> list[str]:
        try:
            probe = _run_opencode(
                [*self.command_argv, "models", "opencode", "--pure"],
                timeout=timeout,
            )
            if probe.returncode:
                return list(FALLBACK_MODELS)
            models = []
            for line in probe.stdout.splitlines():
                model = line.strip()
                if model.startswith("opencode/"):
                    model = model.split("/", 1)[1]
                if model and not any(char.isspace() for char in model):
                    models.append(model)
            return sorted(dict.fromkeys(models)) or list(FALLBACK_MODELS)
        except (OSError, subprocess.SubprocessError):
            return list(FALLBACK_MODELS)

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        timeout: Any = None,
        temperature: Any = None,
        max_tokens: Any = None,
        top_p: Any = None,
        stop: Any = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        seconds = _effective_timeout(timeout)
        selected = str(model or DEFAULT_MODEL)

        def factory() -> SimpleNamespace:
            events, actual = self._direct_events(
                selected,
                messages or [],
                tools or [],
                tool_choice=tool_choice,
                timeout=seconds,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                extra_body=extra_body,
                extra=kwargs,
            )
            return _merge_sse(events, requested_model=selected, actual_tools=actual)

        if not stream:
            return _LazyValue(factory)

        def stream_factory() -> Iterator[SimpleNamespace]:
            events, actual = self._direct_events(
                selected,
                messages or [],
                tools or [],
                tool_choice=tool_choice,
                timeout=seconds,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                extra_body=extra_body,
                extra=kwargs,
            )
            call_names: dict[int, str] = {}
            seen = False
            for event in events:
                seen = True
                if event.get("error"):
                    raise OpenCodeError(f"OpenCode inference error: {event['error']}")
                event.setdefault("choices", [])
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason") == "content_filter":
                        raise OpenCodeError(
                            "OpenCode inference was blocked by the content filter."
                        )
                    for raw_call in (choice.get("delta") or {}).get("tool_calls") or []:
                        index = int(raw_call.get("index", 0) or 0)
                        name = str((raw_call.get("function") or {}).get("name") or "")
                        if name:
                            call_names[index] = name
                        if error := _phantom_tool_error(
                            call_names.get(index, ""), actual
                        ):
                            raise error
                yield _namespace(event)
            if not seen:
                raise OpenCodeError("OpenCode inference returned no events.")

        return _LazyStream(stream_factory)

    def _direct_events(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: Any,
        timeout: float,
        temperature: Any,
        max_tokens: Any,
        top_p: Any,
        stop: Any,
        extra_body: dict[str, Any] | None,
        extra: dict[str, Any],
    ) -> tuple[Iterator[dict[str, Any]], set[str]]:
        if self.is_closed:
            raise OpenCodeError("OpenCode client is closed.")
        if "/" in model:
            provider, model = model.split("/", 1)
            if provider != "opencode":
                raise OpenCodeError(
                    "Direct mode only supports models from the local "
                    "OpenCode 'opencode' provider."
                )
        if model in RESPONSES_MODELS:
            return self._responses_events(
                model,
                messages,
                tools,
                tool_choice=tool_choice,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_body=extra_body,
                extra=extra,
            )
        wire_tools, actual = _wire_tools(tools)
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": "You are opencode"}, *messages],
            "tools": wire_tools,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        optional = {
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop,
            "parallel_tool_calls": extra.get("parallel_tool_calls"),
            "response_format": extra.get("response_format"),
        }
        body.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        if isinstance(extra_body, dict):
            body.update(
                {
                    key: value
                    for key, value in extra_body.items()
                    if key
                    not in {"model", "messages", "tools", "stream", "stream_options"}
                }
            )
        data = json.dumps(body, separators=(",", ":")).encode()
        request = urllib.request.Request(
            DIRECT_URL,
            data=data,
            method="POST",
            headers={
                "Authorization": "Bearer public",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": f"opencode/{self._opencode_version()}",
                "x-opencode-session": self._session,
            },
        )

        def events() -> Iterator[dict[str, Any]]:
            try:
                response = _urlopen(request, timeout)
            except urllib.error.HTTPError as exc:
                body_text = exc.read().decode("utf-8", "replace")
                raise _status_error(
                    exc.code, body_text, "OpenCode free-model inference"
                ) from exc
            except urllib.error.URLError as exc:
                raise OpenCodeError(
                    f"OpenCode free-model inference failed: {exc.reason}"
                ) from exc
            with response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    if payload:
                        yield json.loads(payload)

        return events(), actual

    def _responses_events(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: Any,
        timeout: float,
        temperature: Any,
        max_tokens: Any,
        top_p: Any,
        extra_body: dict[str, Any] | None,
        extra: dict[str, Any],
    ) -> tuple[Iterator[dict[str, Any]], set[str]]:
        wire_tools, actual = _wire_tools(tools)
        input_items, instructions = _responses_input(messages)
        body: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "instructions": instructions,
            "tools": _responses_tools(wire_tools),
            "stream": True,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": self._session,
        }
        optional = {
            "tool_choice": _responses_tool_choice(tool_choice),
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
            "parallel_tool_calls": extra.get("parallel_tool_calls"),
        }
        body.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        if isinstance(extra_body, dict):
            body.update(
                {
                    key: value
                    for key, value in extra_body.items()
                    if key
                    not in {
                        "model",
                        "input",
                        "instructions",
                        "tools",
                        "stream",
                        "store",
                        "include",
                        "prompt_cache_key",
                    }
                }
            )
        request = urllib.request.Request(
            RESPONSES_URL,
            data=json.dumps(body, separators=(",", ":")).encode(),
            method="POST",
            headers={
                "Authorization": "Bearer public",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": f"opencode/{self._opencode_version()}",
                "x-opencode-session": self._session,
            },
        )

        def source() -> Iterator[dict[str, Any]]:
            try:
                response = _urlopen(request, timeout)
            except urllib.error.HTTPError as exc:
                body_text = exc.read().decode("utf-8", "replace")
                raise _status_error(
                    exc.code, body_text, "OpenCode free-model inference"
                ) from exc
            except urllib.error.URLError as exc:
                raise OpenCodeError(
                    f"OpenCode free-model inference failed: {exc.reason}"
                ) from exc
            with response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    if payload:
                        yield json.loads(payload)

        def events() -> Iterator[dict[str, Any]]:
            seen = False
            saw_tool_call = False
            call_indexes: dict[int, int] = {}
            selected_model = model
            for event in source():
                seen = True
                kind = str(event.get("type") or "")
                response = event.get("response") or {}
                selected_model = str(response.get("model") or selected_model)
                if kind == "error" or event.get("error"):
                    raise OpenCodeError(f"OpenCode inference error: {event}")
                if kind == "response.output_text.delta":
                    yield {
                        "model": selected_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": str(event.get("delta") or "")},
                                "finish_reason": None,
                            }
                        ],
                    }
                elif kind == "response.reasoning_summary_text.delta":
                    yield {
                        "model": selected_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "reasoning_content": str(event.get("delta") or "")
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                elif kind == "response.output_item.done":
                    item = event.get("item") or {}
                    if item.get("type") == "reasoning":
                        yield {
                            "model": selected_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_details": [item]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    elif item.get("type") == "function_call":
                        saw_tool_call = True
                        output_index = int(event.get("output_index", 0) or 0)
                        index = call_indexes.setdefault(output_index, len(call_indexes))
                        yield {
                            "model": selected_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": index,
                                                "id": str(item.get("call_id") or ""),
                                                "type": "function",
                                                "function": {
                                                    "name": str(item.get("name") or ""),
                                                    "arguments": str(
                                                        item.get("arguments") or "{}"
                                                    ),
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                elif kind in {"response.failed", "response.incomplete"}:
                    raise OpenCodeError(
                        f"OpenCode inference did not complete: {response}"
                    )
                elif kind == "response.completed":
                    if response.get("status") not in {None, "completed"}:
                        raise OpenCodeError(
                            f"OpenCode inference did not complete: {response}"
                        )
                    usage = response.get("usage") or {}
                    yield {
                        "model": selected_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls"
                                if saw_tool_call
                                else "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": usage.get("input_tokens", 0),
                            "completion_tokens": usage.get("output_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                            "prompt_tokens_details": usage.get(
                                "input_tokens_details", {}
                            ),
                        },
                    }
            if not seen:
                raise OpenCodeError("OpenCode inference returned no events.")

        return events(), actual
