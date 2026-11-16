"""Stock-Hermes translation for the OpenCode-native provider.

One native compatibility tool only: OpenCode's ``read`` is mapped to Hermes
``read_file``. Every other capability comes from the genuine Hermes schemas the
current session exposes; those are advertised byte-for-byte with a reversible
wire alias when a Hermes name collides with the retained native descriptor.
Nothing executes here.

Call provenance travels in the OpenAI call id as ``ocn1.<wire>.<tag>.<id>``.
The wire name is the name the model actually emitted; a compressed payload is
carried only when the native arguments cannot be recovered by inverting the
Hermes mapping, so canonical history stays unambiguous across a facade rebuild.

Opaque native metadata (per reasoning/text/tool part and the terminal step) is
preserved through Hermes' canonical history in one ``reasoning_details``
carrier of type ``opencode-free.native_assistant``. It is echoed back on replay
and never interpreted.
"""
from __future__ import annotations

import base64
import json
import urllib.parse
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterable

COMPAT_TOOL = "read_file"
NATIVE_KEEP = frozenset({"read"})
NATIVE_CARRIER = "opencode-free.native_assistant"
READ_LIMIT_MAX = 2000
MAX_WIRE_NAME = 64
CALL_PREFIX = "ocn1"
MAX_CALL_ARGUMENT_BYTES = 1024 * 1024


class NativeTranslationError(ValueError):
    code = "translation.error"


class NativeUnsupported(NativeTranslationError):
    code = "native.unsupported"


class NativeInvalidArguments(NativeTranslationError):
    code = "native.invalid_arguments"


class CompatibilityUnavailable(NativeTranslationError):
    code = "native.compatibility_unavailable"


class SnapshotError(NativeTranslationError):
    code = "snapshot.invalid"


# ---------------------------------------------------------------------------
# native -> Hermes
# ---------------------------------------------------------------------------
def _strict(arguments: dict, allowed: set[str], name: str) -> None:
    extra = sorted(set(arguments) - allowed)
    if extra:
        raise NativeInvalidArguments(f"native tool {name!r} has unsupported options: {', '.join(extra)}")


def _string(arguments: dict, key: str, *, required: bool = True, allow_empty: bool = False) -> str | None:
    value = arguments.get(key)
    if value is None:
        if required:
            raise NativeInvalidArguments(f"native tool requires a string {key!r}")
        return None
    if not isinstance(value, str) or (not allow_empty and not value):
        raise NativeInvalidArguments(f"{key!r} must be a {'string' if allow_empty else 'non-empty string'}")
    if "\x00" in value:
        raise NativeInvalidArguments(f"{key!r} must not contain NUL")
    return value


def _integer(value: Any, field_name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativeInvalidArguments(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise NativeInvalidArguments(f"{field_name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise NativeInvalidArguments(f"{field_name} must be at most {maximum}")
    return value


def _map_read(arguments: dict) -> tuple[str, dict]:
    """Native `read` -> Hermes `read_file`.

    Supported: absolute/relative ``path``, 0-or-1-based ``offset`` and a
    positive ``limit``. Native ``offset < 0`` (tail read) has no Hermes
    equivalent and is refused rather than silently truncated. ``limit`` above
    ``read_file``'s own cap is clamped; full Hermes ``read_file`` stays
    available independently with its own schema and semantics.
    """
    _strict(arguments, {"path", "offset", "limit"}, "read")
    mapped: dict[str, Any] = {"path": _string(arguments, "path")}
    if arguments.get("offset") is not None:
        value = _integer(arguments["offset"], "read offset")
        if value < 0:
            raise NativeUnsupported("native negative read offset (tail read) has no Hermes equivalent")
        # Native 0 and 1 both mean the first line; read_file is 1-based.
        mapped["offset"] = max(1, value)
    if arguments.get("limit") is not None:
        mapped["limit"] = min(_integer(arguments["limit"], "read limit", minimum=1), READ_LIMIT_MAX)
    return COMPAT_TOOL, mapped


def translate_call(
    native_name: str,
    arguments: dict,
    *,
    plan: "WirePlan | None" = None,
    enabled: Iterable[str] | None = None,
) -> tuple[str, dict]:
    """Map one model-emitted native call to an ordinary Hermes call.

    Nothing executes here. Native ``read`` maps to ``read_file``. A registered
    Hermes wire name forwards unchanged. Anything else keeps a name Hermes does
    not own (or a sentinel when it collides), so stock dispatch can never run an
    unrelated tool by accident.
    """
    if not isinstance(native_name, str) or not native_name:
        raise NativeInvalidArguments("native tool name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise NativeInvalidArguments(f"native tool {native_name!r} arguments must be an object")
    try:
        arguments = json.loads(json.dumps(arguments, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise NativeInvalidArguments(f"native tool {native_name!r} arguments are not JSON") from error

    if native_name == "read":
        host_name, mapped = _map_read(arguments)
        if enabled is not None and host_name not in set(enabled):
            # The model asked for the compatibility read while Hermes read_file
            # is disabled. Give stock dispatch an unknown name; never run a
            # different tool in its place.
            return "oc_unsupported_read", arguments
        return host_name, mapped

    # An additive Hermes schema the model called directly keeps its identity
    # and arguments byte-for-byte.
    if plan is not None and native_name in plan.wire_to_host:
        return plan.wire_to_host[native_name], arguments

    # Any other native name (execute, write, edit, shell, subagent, ...) has no
    # mapping. A colliding Hermes name gets a sentinel so it cannot run.
    if enabled is not None and native_name in set(enabled):
        return f"oc_unsupported_{native_name}", arguments
    return native_name, arguments


# ---------------------------------------------------------------------------
# Hermes -> native (lossless inverse)
# ---------------------------------------------------------------------------
def _inverse_read(arguments: dict) -> dict:
    return {key: value for key, value in (
        ("path", arguments.get("path")),
        ("offset", arguments.get("offset")),
        ("limit", arguments.get("limit")),
    ) if value is not None}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _payload(value: dict) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_CALL_ARGUMENT_BYTES:
        raise NativeUnsupported("native tool arguments exceed the call-id provenance bound")
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii").rstrip("=")


def _unpayload(value: str) -> dict:
    padded = value + "=" * (-len(value) % 4)
    compressed = base64.urlsafe_b64decode(padded.encode("ascii"))
    inflater = zlib.decompressobj()
    try:
        decoded = inflater.decompress(compressed, MAX_CALL_ARGUMENT_BYTES + 1)
    except zlib.error as error:
        raise NativeTranslationError("call id payload is not valid compressed data") from error
    if len(decoded) > MAX_CALL_ARGUMENT_BYTES or inflater.unconsumed_tail or not inflater.eof:
        raise NativeTranslationError("call id payload exceeds the provenance bound")
    try:
        result = json.loads(decoded.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise NativeTranslationError("call id payload is not JSON") from error
    if not isinstance(result, dict):
        raise NativeTranslationError("call id payload is not an object")
    return result


def encode_call_id(native_name: str, native_arguments: dict, host_arguments: dict, original_id: str) -> str:
    """Return a deterministic call id that carries wire-name (and lossy args) provenance."""
    inverse = _inverse_read if native_name == "read" else None
    recovered = inverse(host_arguments) if inverse is not None else host_arguments
    tag = "v" if _canonical(recovered) == _canonical(native_arguments) else f"p{_payload(native_arguments)}"
    # Names may contain '.', so only the call id's own delimiter is literal.
    # urllib leaves '.' unescaped, so force it after quoting.
    wire = urllib.parse.quote(native_name, safe="").replace(".", "%2E")
    return f"{CALL_PREFIX}.{wire}.{tag}.{original_id}"


def decode_call_id(call_id: str) -> tuple[str, dict | None, str] | None:
    """Return ``(wire_name, native_arguments_or_None, original_id)`` for our ids."""
    if not isinstance(call_id, str) or not call_id.startswith(CALL_PREFIX + "."):
        return None
    parts = call_id.split(".", 3)
    if len(parts) != 4:
        return None
    _, encoded_wire, tag, original = parts
    if not encoded_wire or not tag or not original:
        return None
    try:
        wire = urllib.parse.unquote(encoded_wire, errors="strict")
    except UnicodeDecodeError as error:
        raise SnapshotError(f"tool call id {call_id!r} has an unreadable wire name") from error
    if not wire:
        return None
    if tag == "v":
        return wire, None, original
    if not tag.startswith("p"):
        return None
    try:
        return wire, _unpayload(tag[1:]), original
    except Exception as error:  # noqa: BLE001 - malformed history is explicit
        raise SnapshotError(f"tool call id {call_id!r} has an unreadable payload") from error


# ---------------------------------------------------------------------------
# Opaque native metadata carrier
# ---------------------------------------------------------------------------
def native_carrier(native_parts: Any) -> dict | None:
    """One ``reasoning_details`` entry preserving native per-part metadata."""
    if not isinstance(native_parts, dict):
        return None
    parts = [
        {key: value for key, value in part.items() if value is not None}
        for part in (native_parts.get("parts") or [])
        if isinstance(part, dict) and part.get("native") is not None
    ]
    if not parts:
        return None
    return {"type": NATIVE_CARRIER, "parts": parts}


def _carrier_index(message: dict) -> dict[tuple[str, int], Any]:
    """``{(part_type, position): native}`` from the message's carrier.

    Position-keyed so text/reasoning without upstream metadata does not shift
    the metadata of later parts.
    """
    index: dict[tuple[str, int], Any] = {}
    details = message.get("reasoning_details")
    if not isinstance(details, list):
        return index
    for detail in details:
        if not isinstance(detail, dict) or detail.get("type") != NATIVE_CARRIER:
            continue
        for part in detail.get("parts") or []:
            if not isinstance(part, dict) or part.get("native") is None:
                continue
            kind, position = part.get("type"), part.get("index")
            if isinstance(kind, str) and isinstance(position, int) and not isinstance(position, bool):
                index[(kind, position)] = part["native"]
    return index


def _attach(part: dict, native: Any) -> None:
    if native is not None:
        part["native"] = native


# ---------------------------------------------------------------------------
# Hermes tool registration
# ---------------------------------------------------------------------------
@dataclass
class WirePlan:
    register: list[dict] = field(default_factory=list)
    host_to_wire: dict[str, str] = field(default_factory=dict)
    wire_to_host: dict[str, str] = field(default_factory=dict)

    def wire_for(self, host_name: str) -> str:
        return self.host_to_wire.get(host_name, host_name)

    def host_for(self, wire_name: str) -> str:
        return self.wire_to_host.get(wire_name, wire_name)


def _function(tool: Any) -> dict | None:
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    if isinstance(function, dict):
        return function
    if isinstance(tool.get("name"), str):
        return tool
    return None


def plan_tools(tools: Iterable[dict] | None, native_names: Iterable[str] = NATIVE_KEEP) -> WirePlan:
    """Turn the enabled Hermes schemas into additive, collision-free wire tools.

    Refuses before inference when ``read_file`` is absent: the native ``read``
    descriptor is only advertised because Hermes can actually run it.
    """
    native = {str(name) for name in native_names if isinstance(name, str) and name}
    plan = WirePlan()
    functions: list[dict] = []
    seen: set[str] = set()
    for tool in tools or []:
        function = _function(tool)
        if function is None:
            continue
        host = function.get("name")
        if not isinstance(host, str) or not host or host in seen:
            continue
        description = function.get("description", "")
        parameters = function.get("parameters") or {"type": "object", "properties": {}}
        if not isinstance(description, str) or not isinstance(parameters, dict):
            raise NativeInvalidArguments(f"Hermes tool {host!r} has an invalid schema")
        seen.add(host)
        functions.append({"name": host, "description": description, "parameters": parameters})

    hosts = {function["name"] for function in functions}
    if COMPAT_TOOL not in hosts:
        raise CompatibilityUnavailable(
            "Hermes read_file is unavailable in this session, so the native read "
            "compatibility adapter cannot be advertised; enable the file toolset"
        )

    taken = set(native)
    for function in functions:
        host = function["name"]
        wire = host
        if wire in taken:
            wire = f"hermes_{host}"
            while wire in taken:
                wire = f"_{wire}"
        if len(wire) > MAX_WIRE_NAME:
            raise NativeInvalidArguments(
                f"Hermes tool {host!r} cannot fit a collision-free wire name under {MAX_WIRE_NAME} chars"
            )
        plan.register.append({
            "name": wire,
            "description": function["description"],
            "parameters": function["parameters"],
            "codemode": False,
        })
        plan.host_to_wire[host] = wire
        plan.wire_to_host[wire] = host
        taken.add(wire)
    return plan


# ---------------------------------------------------------------------------
# canonical history -> native snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Snapshot:
    system_parts: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    user_text: str = ""


def _content_text(value: Any, *, field_name: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for part in value:
            if not isinstance(part, dict):
                raise SnapshotError(f"{field_name} content part is not an object")
            if part.get("type") in {"text", "input_text", "output_text"}:
                text = part.get("text")
                if not isinstance(text, str):
                    raise SnapshotError(f"{field_name} text part has no string text")
                parts.append(text)
            elif part.get("type") in {"image_url", "input_image", "image", "input_audio", "audio", "file"}:
                continue
            else:
                raise SnapshotError(f"{field_name} content part {part.get('type')!r} is unsupported")
        return "".join(parts)
    if isinstance(value, dict) and value.get("_multimodal") is True:
        return _content_text(value.get("content"), field_name=field_name)
    raise SnapshotError(f"{field_name} content must be a string or a list of parts")


def _source(url: str) -> dict:
    if url.startswith("data:"):
        import re

        match = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", url, re.DOTALL)
        if match is None or match.group(2) != ";base64":
            raise SnapshotError("only base64 data URLs are representable")
        return {
            "type": "base64", "data": match.group(3),
            "mediaType": match.group(1) or "application/octet-stream",
        }
    if url.startswith(("http://", "https://")):
        return {"type": "url", "url": url, "mediaType": "application/octet-stream"}
    raise SnapshotError(f"media URL scheme {url.split(':', 1)[0]!r} is not representable")


def _media_part(part: dict) -> dict | None:
    kind = part.get("type")
    if kind in {"image_url", "input_image"}:
        image_value = part.get("image_url", part.get("image"))
        url = image_value.get("url") if isinstance(image_value, dict) else image_value
        if not isinstance(url, str) or not url:
            raise SnapshotError("image part has no URL")
        return {"type": "media", "media": {"source": _source(url)}}
    if kind == "image":
        source = part.get("source")
        if not isinstance(source, dict):
            raise SnapshotError("image part has no source")
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            return {"type": "media", "media": {"source": {
                "type": "base64", "data": source["data"],
                "mediaType": source.get("media_type") or source.get("mime") or "application/octet-stream",
            }}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "media", "media": {"source": _source(source["url"])}}
        raise SnapshotError("image source type is not representable")
    if kind == "file":
        file_value = part.get("file")
        body = file_value if isinstance(file_value, dict) else part
        url = body.get("file_data") or body.get("file_url") or body.get("uri")
        if isinstance(url, str) and url:
            return {"type": "file", "uri": url}
    return None


def _arguments(value: Any, call_id: str) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as error:
            raise SnapshotError(f"tool call {call_id!r} arguments are not valid JSON") from error
    if not isinstance(value, dict):
        raise SnapshotError(f"tool call {call_id!r} arguments must be a JSON object")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise SnapshotError(f"tool call {call_id!r} arguments are not valid JSON") from error


def _tool_content(value: Any) -> str:
    if isinstance(value, dict) and value.get("_multimodal") is True:
        return _content_text(value.get("content"), field_name="tool")
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return json.dumps(value, ensure_ascii=False)


def snapshot_from_openai(messages: Iterable[dict], *, plan: WirePlan | None = None) -> Snapshot:
    """Replay Hermes' canonical history in the native snapshot shape."""
    if not isinstance(messages, (list, tuple)):
        raise SnapshotError("messages must be a list")
    plan = plan or WirePlan()
    system_parts: list[str] = []
    transcript: list[dict] = []
    call_names: dict[str, str] = {}
    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    user_text = ""

    for message in messages:
        if not isinstance(message, dict):
            raise SnapshotError("message must be an object")
        role = message.get("role")
        if role in {"system", "developer"}:
            text = _content_text(message.get("content"), field_name=str(role))
            if text:
                system_parts.append(text)
            continue

        if role == "user":
            content = message.get("content")
            parts: list[dict] = []
            if isinstance(content, str):
                if content:
                    parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        raise SnapshotError("user content part is not an object")
                    kind = part.get("type")
                    if kind in {"text", "input_text", "output_text"}:
                        text = part.get("text")
                        if not isinstance(text, str):
                            raise SnapshotError("user text part has no string text")
                        if text:
                            parts.append({"type": "text", "text": text})
                    else:
                        media = _media_part(part)
                        if media is not None:
                            parts.append(media)
            else:
                raise SnapshotError("user content must be a string or a list of parts")
            if not parts:
                raise SnapshotError("user message has no content")
            user_text = "".join(part["text"] for part in parts if part["type"] == "text")
            transcript.append({"role": "user", "content": parts})
            continue

        if role == "assistant":
            parts: list[dict] = []
            natives = _carrier_index(message)
            # Position in the native part order (reasoning/text/tool-call);
            # media parts do not participate, so metadata cannot shift.
            native_index = 0

            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                if not isinstance(reasoning, str):
                    raise SnapshotError("assistant reasoning must be a string")
                part = {"type": "reasoning", "text": reasoning}
                _attach(part, natives.get(("reasoning", native_index)))
                native_index += 1
                parts.append(part)
            content = message.get("content")
            if isinstance(content, str):
                if content:
                    part = {"type": "text", "text": content}
                    _attach(part, natives.get(("text", native_index)))
                    native_index += 1
                    parts.append(part)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        raise SnapshotError("assistant content part is not an object")
                    if item.get("type") in {"text", "input_text", "output_text"}:
                        text = item.get("text")
                        if not isinstance(text, str):
                            raise SnapshotError("assistant text part has no string text")
                        if text:
                            part = {"type": "text", "text": text}
                            _attach(part, natives.get(("text", native_index)))
                            native_index += 1
                            parts.append(part)
                    else:
                        media = _media_part(item)
                        if media is not None:
                            parts.append(media)
            elif content is not None:
                raise SnapshotError("assistant content must be a string or a list of parts")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise SnapshotError("assistant tool_calls must be a list")
            for call in calls:
                if not isinstance(call, dict):
                    raise SnapshotError("assistant tool call must be an object")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise SnapshotError("assistant tool call has no id")
                if call_id in seen_calls:
                    raise SnapshotError(f"duplicate tool call id {call_id!r}")
                function = call.get("function") or {}
                if not isinstance(function, dict):
                    raise SnapshotError(f"tool call {call_id!r} function must be an object")
                host_name = function.get("name")
                if not isinstance(host_name, str) or not host_name:
                    raise SnapshotError(f"tool call {call_id!r} has no name")
                decoded = decode_call_id(call_id)
                if decoded is not None:
                    wire, native_arguments, _original = decoded
                    arguments = native_arguments
                    if arguments is None:
                        arguments = _arguments(function.get("arguments"), call_id)
                        if wire == "read":
                            arguments = _inverse_read(arguments)
                else:
                    # History from another provider (or an older turn): map the
                    # host name back to the wire name where a plan exists.
                    wire = plan.wire_for(host_name)
                    arguments = _arguments(function.get("arguments"), call_id)
                call_names[call_id] = wire
                seen_calls.add(call_id)
                part = {"type": "tool-call", "id": call_id, "name": wire, "input": arguments}
                _attach(part, natives.get(("tool-call", native_index)))
                native_index += 1
                parts.append(part)
            if not parts:
                raise SnapshotError("assistant message has no content or tool calls")
            transcript.append({"role": "assistant", "content": parts})
            continue

        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise SnapshotError("tool result has no tool_call_id")
            if call_id not in call_names:
                raise SnapshotError(f"tool result references unknown call {call_id!r}")
            if call_id in seen_results:
                raise SnapshotError(f"duplicate tool result for call {call_id!r}")
            seen_results.add(call_id)
            wire = call_names[call_id]
            text = _tool_content(message.get("content"))
            if message.get("is_error"):
                result = {"type": "error", "value": text or "tool failed"}
            else:
                result = {"type": "text", "value": text}
            transcript.append({
                "role": "tool",
                "content": [{"type": "tool-result", "id": call_id, "name": wire, "result": result}],
            })
            continue

        raise SnapshotError(f"unsupported message role {role!r}")

    if not user_text:
        raise SnapshotError("a non-empty user message is required as the native prompt anchor")
    return Snapshot(system_parts=system_parts, messages=transcript, user_text=user_text)
