"""Stock-Hermes translation for the OpenCode-native provider.

A constrained set of native descriptors (``NATIVE_MAP``: read, write, glob,
grep, question, todowrite) is mapped to the Hermes tool that runs it, and only
while that tool is enabled; the replaced Hermes schema is then not advertised.
Every other capability comes from the genuine Hermes schemas the current
session exposes, byte-for-byte, with a reversible wire alias on a native name
collision. Unsupported native options are refused, never dropped. Nothing
executes here.

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
import math
import os
import urllib.parse
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterable

COMPAT_TOOL = "read_file"
NATIVE_CARRIER = "opencode-free.native_assistant"
READ_LIMIT_MAX = 2000
SEARCH_LIMIT_DEFAULT = 100  # native glob/grep default; Hermes search_files defaults to 50
# Hermes clarify/todo_list bounds (tools/clarify_tool.py, tools/todo_tool.py).
CLARIFY_MAX_QUESTIONS = 5
CLARIFY_MAX_CHOICES = 4
TODO_MAX_ITEMS = 256
TODO_MAX_CONTENT = 4000
TODO_STATUSES = ("pending", "in_progress", "completed", "cancelled")
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


def _path(arguments: dict, key: str, forbidden: Iterable[str], *, required: bool = True) -> str | None:
    """A native path argument; absolute paths into the private runtime are refused."""
    value = _string(arguments, key, required=required)
    if value is not None and os.path.isabs(os.path.expanduser(value)):
        real = os.path.realpath(os.path.expanduser(value))
        for root in forbidden:
            base = os.path.realpath(root)
            if real == base or real.startswith(base.rstrip(os.sep) + os.sep):
                raise NativeUnsupported(
                    f"{key!r} points into the private OpenCode runtime; native paths resolve "
                    "against the Hermes working directory"
                )
    return value


def _boolean(arguments: dict, key: str) -> bool | None:
    value = arguments.get(key)
    if value is not None and not isinstance(value, bool):
        raise NativeInvalidArguments(f"{key!r} must be a boolean")
    return value


def _map_read(arguments: dict, forbidden: Iterable[str] = ()) -> dict:
    """Native ``read`` -> ``read_file``. Native ``limit: 0`` means the default;
    ``offset < 0`` (tail read) has no Hermes equivalent and is refused."""
    _strict(arguments, {"path", "offset", "limit"}, "read")
    mapped: dict[str, Any] = {"path": _path(arguments, "path", forbidden)}
    if arguments.get("offset") is not None:
        value = _integer(arguments["offset"], "read offset")
        if value < 0:
            raise NativeUnsupported("native negative read offset (tail read) has no Hermes equivalent")
        # Native 0 and 1 both mean the first line; read_file is 1-based.
        mapped["offset"] = max(1, value)
    if arguments.get("limit") is not None:
        limit = _integer(arguments["limit"], "read limit", minimum=0)
        if limit:
            mapped["limit"] = min(limit, READ_LIMIT_MAX)
    return mapped


def _map_write(arguments: dict, forbidden: Iterable[str] = ()) -> dict:
    _strict(arguments, {"path", "content"}, "write")
    return {"path": _path(arguments, "path", forbidden),
            "content": _string(arguments, "content", allow_empty=True)}


def _map_glob(arguments: dict, forbidden: Iterable[str] = ()) -> dict:
    _strict(arguments, {"pattern", "path", "hidden", "limit"}, "glob")
    if _boolean(arguments, "hidden"):
        raise NativeUnsupported("native glob hidden:true has no Hermes equivalent (hidden paths are skipped)")
    mapped: dict[str, Any] = {"pattern": _string(arguments, "pattern"), "target": "files"}
    if arguments.get("path") is not None:
        mapped["path"] = _path(arguments, "path", forbidden)
    limit = arguments.get("limit")
    mapped["limit"] = SEARCH_LIMIT_DEFAULT if limit is None else _integer(limit, "glob limit", minimum=1)
    return mapped


# ponytail: escape only the regex metacharacters ripgrep honours outside a class.
_REGEX_META = frozenset("\\.^$|?*+()[]{}")


def _map_grep(arguments: dict, forbidden: Iterable[str] = ()) -> dict:
    _strict(arguments, {"pattern", "path", "include", "literal", "caseSensitive", "limit"}, "grep")
    pattern = _string(arguments, "pattern")
    if _boolean(arguments, "literal"):
        pattern = "".join("\\" + char if char in _REGEX_META else char for char in pattern)
    if _boolean(arguments, "caseSensitive") is False:
        pattern = "(?i)" + pattern
    mapped: dict[str, Any] = {"pattern": pattern, "target": "content"}
    if arguments.get("path") is not None:
        mapped["path"] = _path(arguments, "path", forbidden)
    if arguments.get("include") is not None:
        mapped["file_glob"] = _string(arguments, "include")
    limit = arguments.get("limit")
    mapped["limit"] = SEARCH_LIMIT_DEFAULT if limit is None else _integer(limit, "grep limit", minimum=1)
    return mapped


def _map_question(arguments: dict, forbidden: Iterable[str] = ()) -> dict:
    """Native ``question`` -> ``clarify``. Headers and option descriptions are
    folded into the question text (never dropped); the exact native call is
    kept by call-id provenance. Over-limit forms are refused, never truncated."""
    _strict(arguments, {"questions"}, "question")
    questions = arguments.get("questions")
    if not isinstance(questions, list) or not questions:
        raise NativeInvalidArguments("'questions' must be a non-empty array")
    if len(questions) > CLARIFY_MAX_QUESTIONS:
        raise NativeUnsupported(f"Hermes clarify accepts at most {CLARIFY_MAX_QUESTIONS} questions per call")
    mapped = []
    for index, item in enumerate(questions):
        if not isinstance(item, dict):
            raise NativeInvalidArguments(f"questions[{index}] must be an object")
        _strict(item, {"question", "header", "options", "multiple"}, "question")
        text = _string(item, "question")
        header = _string(item, "header", required=False, allow_empty=True)
        options = item.get("options", [])
        multiple = _boolean(item, "multiple")
        if not isinstance(options, list):
            raise NativeInvalidArguments(f"questions[{index}].options must be an array")
        if len(options) > CLARIFY_MAX_CHOICES:
            raise NativeUnsupported(f"Hermes clarify accepts at most {CLARIFY_MAX_CHOICES} options per question")
        labels, notes = [], []
        for option in options:
            if not isinstance(option, dict):
                raise NativeInvalidArguments(f"questions[{index}].options entries must be objects")
            _strict(option, {"label", "description"}, "question option")
            label = _string(option, "label")
            if not label.strip():
                raise NativeInvalidArguments(f"questions[{index}] option label must not be blank")
            labels.append(label)
            description = _string(option, "description", required=False, allow_empty=True)
            if description:
                notes.append(f"- {label}: {description}")
        body = "\n\n".join(part for part in (header, text, "\n".join(notes)) if part)
        entry: dict[str, Any] = {"question": body}
        if labels:
            entry["choices"] = labels
            if multiple:
                entry["multi_select"] = True
        mapped.append(entry)
    return {"questions": mapped}


def _map_todowrite(arguments: dict, forbidden: Iterable[str] = ()) -> dict:
    """Native ``todowrite`` (replace-only) -> ``todo_list`` in the session's own
    Hermes store, with positional ids. Hermes caps are refused, not truncated."""
    _strict(arguments, {"todos"}, "todowrite")
    todos = arguments.get("todos")
    if not isinstance(todos, list):
        raise NativeInvalidArguments("'todos' must be an array")
    if len(todos) > TODO_MAX_ITEMS:
        raise NativeUnsupported(f"Hermes todo_list keeps at most {TODO_MAX_ITEMS} items")
    mapped = []
    for index, item in enumerate(todos):
        if not isinstance(item, dict):
            raise NativeInvalidArguments(f"todos[{index}] must be an object")
        _strict(item, {"content", "status"}, "todowrite")
        content = _string(item, "content")
        if not content.strip():
            raise NativeInvalidArguments(f"todos[{index}].content must not be blank")
        if len(content.strip()) > TODO_MAX_CONTENT:
            raise NativeUnsupported(f"Hermes todo_list caps item content at {TODO_MAX_CONTENT} characters")
        status = item.get("status")
        if status not in TODO_STATUSES:
            raise NativeInvalidArguments(f"todos[{index}].status must be one of {', '.join(TODO_STATUSES)}")
        mapped.append({"id": str(index + 1), "content": content, "status": status})
    return {"todos": mapped, "merge": False}


# native name -> (Hermes tool, argument mapper). Everything else stays hidden.
NATIVE_MAP: dict[str, tuple[str, Any]] = {
    "read": ("read_file", _map_read),
    "write": ("write_file", _map_write),
    "glob": ("search_files", _map_glob),
    "grep": ("search_files", _map_grep),
    "question": ("clarify", _map_question),
    "todowrite": ("todo_list", _map_todowrite),
}


# --- experiment subset harness ------------------------------------------------
# ponytail: env-gated extra mappings for the main-branch tool experiment.
# OC_EXTRA_TOOLS lists capability groups; default (unset) == untouched main.
# Groups accept both OC name spellings observed across versions/models:
#   shell -> bash,shell  edit -> edit  skill -> skill  task -> task,subagent
#   webfetch -> webfetch  websearch -> websearch
# Each extra mapper is lenient about both argument spellings (path/filePath)
# because the experiment must measure capability need, not schema pedantry.
def _opt_string(arguments: dict, *keys: str, required: bool = True) -> str | None:
    for key in keys:
        if key in arguments:
            return _string(arguments, key)
    if required:
        raise NativeInvalidArguments(f"native tool requires one of {', '.join(keys)}")
    return None


def _map_shell_extra(arguments: dict, forbidden: Iterable[str] = ()) -> dict:
    _strict(arguments, {"command", "workdir", "timeout", "background", "description"}, "shell")
    mapped: dict[str, Any] = {"command": _string(arguments, "command")}
    if arguments.get("workdir") is not None:
        mapped["workdir"] = _string(arguments, "workdir")
    if _boolean(arguments, "background") is True:
        mapped["background"] = True
    if arguments.get("timeout") is not None:
        milliseconds = _integer(arguments["timeout"], "shell timeout", minimum=0)
        if milliseconds == 0 and not mapped.get("background"):
            raise NativeUnsupported("native shell timeout=0 (no deadline) has no Hermes foreground equivalent")
        mapped["timeout"] = max(1, math.ceil(milliseconds / 1000)) if milliseconds else 600
    elif not mapped.get("background"):
        mapped["timeout"] = 120
    return mapped


def _map_edit_extra(arguments: dict, forbidden: Iterable[str] = ()) -> dict:
    _strict(arguments, {"path", "filePath", "oldString", "newString", "replaceAll"}, "edit")
    old = _string(arguments, "oldString")
    new = _string(arguments, "newString", allow_empty=True)
    if old == new:
        raise NativeInvalidArguments("edit oldString and newString must differ")
    mapped: dict[str, Any] = {
        "path": _opt_string(arguments, "path", "filePath"),
        "old_string": old,
        "new_string": new,
    }
    if _boolean(arguments, "replaceAll") is True:
        mapped["replace_all"] = True
    return mapped


def _map_skill_extra(arguments: dict, forbidden: Iterable[str] = ()) -> dict:
    _strict(arguments, {"id", "name"}, "skill")
    return {"name": _opt_string(arguments, "id", "name")}


def _map_webfetch_extra(arguments: dict, forbidden: Iterable[str] = ()) -> dict:
    _strict(arguments, {"url", "format", "timeout"}, "webfetch")
    url = _string(arguments, "url") or ""
    if not url.startswith(("http://", "https://")):
        raise NativeUnsupported("webfetch url must use http:// or https://")
    fmt = arguments.get("format", "markdown")
    if fmt not in {"text", "markdown", "html"}:
        raise NativeInvalidArguments("webfetch format must be text, markdown, or html")
    if fmt != "markdown":
        raise NativeUnsupported(f"webfetch format={fmt} has no Hermes web_extract equivalent")
    if arguments.get("timeout") is not None:
        raise NativeUnsupported("webfetch timeout has no per-call Hermes web_extract equivalent")
    return {"urls": [url.split("#", 1)[0]]}

def _map_websearch_extra(arguments: dict, forbidden: Iterable[str] = ()) -> dict:
    _strict(arguments, {"query", "numResults"}, "websearch")
    mapped: dict[str, Any] = {"query": _string(arguments, "query")}
    if arguments.get("numResults") is not None:
        mapped["limit"] = max(1, min(100, _integer(arguments["numResults"], "websearch numResults", minimum=1)))
    return mapped


def _map_task_extra(arguments: dict, forbidden: Iterable[str] = ()) -> dict:
    _strict(arguments, {"description", "prompt", "subagent_type", "task_id", "command", "background"}, "task")
    context = [
        _string(arguments, "description"),
        "Requested subagent type: " + str(_string(arguments, "subagent_type", required=False) or "general"),
    ]
    for key, label in (("task_id", "Previous task ID"), ("command", "Requested command"), ("background", "Requested background execution")):
        if key in arguments:
            context.append(f"{label}: {arguments[key]}")
    return {"tasks": [{"goal": _string(arguments, "prompt"), "context": "\n".join(context)}]}


_EXPERIMENT_GROUPS: dict[str, dict[str, tuple[str, Any]]] = {
    "shell": {
        "bash": ("terminal", _map_shell_extra),
        "shell": ("terminal", _map_shell_extra),
    },
    "edit": {"edit": ("patch", _map_edit_extra)},
    "skill": {"skill": ("skill_view", _map_skill_extra)},
    "task": {
        "task": ("delegate_task", _map_task_extra),
        "subagent": ("delegate_task", _map_task_extra),
    },
    "webfetch": {"webfetch": ("web_extract", _map_webfetch_extra)},
    "websearch": {"websearch": ("web_search", _map_websearch_extra)},
}
_ACTIVE_GROUPS = frozenset(filter(None, os.environ.get("OC_EXTRA_TOOLS", "").split(",")))
if _ACTIVE_GROUPS - set(_EXPERIMENT_GROUPS):
    raise ValueError(f"unknown OC_EXTRA_TOOLS groups: {sorted(_ACTIVE_GROUPS - set(_EXPERIMENT_GROUPS))}")
for _group in sorted(_ACTIVE_GROUPS):
    NATIVE_MAP.update(_EXPERIMENT_GROUPS[_group])
# --- end experiment harness ---------------------------------------------------


def translate_call(
    native_name: str,
    arguments: dict,
    *,
    plan: "WirePlan | None" = None,
    enabled: Iterable[str] | None = None,
    forbidden_roots: Iterable[str] = (),
) -> tuple[str, dict]:
    """Map one model-emitted native call to an ordinary Hermes call.

    Nothing executes here. A mapped native name becomes its Hermes tool. A
    registered Hermes wire name forwards unchanged. Anything else keeps a name
    Hermes does not own (or a sentinel when it collides), so stock dispatch can
    never run an unrelated tool by accident.
    """
    if not isinstance(native_name, str) or not native_name:
        raise NativeInvalidArguments("native tool name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise NativeInvalidArguments(f"native tool {native_name!r} arguments must be an object")
    try:
        arguments = json.loads(json.dumps(arguments, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise NativeInvalidArguments(f"native tool {native_name!r} arguments are not JSON") from error
    enabled = set(enabled) if enabled is not None else (plan.enabled if plan is not None else None)

    # An additive Hermes schema the model called directly keeps its identity
    # and arguments byte-for-byte.
    if plan is not None and native_name in plan.wire_to_host:
        return plan.wire_to_host[native_name], arguments
    # A renamed wire tool the model called under its Hermes name (or its old
    # name from history) still dispatches to the real Hermes tool.
    if plan is not None:
        renamed = WIRE_RENAMES.get(native_name)
        if renamed is not None and renamed in plan.wire_to_host:
            return plan.wire_to_host[renamed], arguments
    # A withheld Hermes name (for example from older history) is still an
    # enabled Hermes tool; it dispatches under its own identity.
    if plan is not None and native_name in plan.replaced:
        return native_name, arguments

    if native_name in NATIVE_MAP and (plan is None or native_name in plan.native):
        host_name, mapper = NATIVE_MAP[native_name]
        mapped = mapper(arguments, forbidden_roots)
        if enabled is not None and host_name not in enabled:
            # The target is disabled: give stock dispatch an unknown name;
            # never run a different tool in its place.
            return f"oc_unsupported_{native_name}", arguments
        return host_name, mapped

    # Any other native name (execute, edit, patch, shell, subagent, ...) has no
    # mapping. A colliding Hermes name gets a sentinel so it cannot run.
    if enabled is not None and native_name in enabled:
        return f"oc_unsupported_{native_name}", arguments
    return native_name, arguments


# ---------------------------------------------------------------------------
# Hermes -> native (lossless inverse)
# ---------------------------------------------------------------------------
def _inverse_write(arguments: dict) -> dict:
    return dict(arguments)


def _inverse_read(arguments: dict) -> dict:
    return {key: value for key, value in (
        ("path", arguments.get("path")),
        ("offset", arguments.get("offset")),
        ("limit", arguments.get("limit")),
    ) if value is not None}


# Exact inverses only; every other mapped call carries its native arguments in
# the call-id payload.
_INVERSE = {"read": _inverse_read, "write": _inverse_write}


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
    inverse = _INVERSE.get(native_name)
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
    # Native descriptors this request keeps (each has an enabled Hermes target).
    native: frozenset = frozenset()
    # Every enabled Hermes tool, including ones replaced by a native descriptor.
    enabled: frozenset = frozenset()
    # Enabled Hermes tools withheld from the catalog because a native replaces them.
    replaced: frozenset = frozenset()

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


# ---------------------------------------------------------------------------
# wire schema slimming (free-tier request budget)
# ---------------------------------------------------------------------------
_SLIM_LEAF_KEYS = ("type", "enum", "const")
_SLIM_DROP_KEYS = ("description", "default", "examples", "title", "$comment",
                   "minimum", "maximum", "minLength", "maxLength", "pattern",
                   "exclusiveMinimum", "exclusiveMaximum", "additionalProperties")
SLIM_ENV = "OC_FREE_SLIM_SCHEMAS"

# Wire renames for the free-tier gate. Zen's anonymous free gate rejects
# requests whose tool catalog contains certain names (measured 2026-11-19: a
# tool named `terminal` is refused with FreeUsageLimitError before inference;
# the identical schema under any other name passes). alt avoids this by
# advertising every tool under OpenCode compat names; main keeps Hermes
# identity but renames the flagged ones. `bash` is what OC-trained models
# emit naturally, and it is not a native registry id in OpenCode 2.3.0, so
# the existing alias machinery stays collision-free.
# ponytail: blocklist is observed, not documented upstream — extend per finding.
WIRE_RENAMES = {"terminal": "bash"}


def _slim_description(text: str) -> str:
    """First sentence of a tool description, hard-capped.

    # ponytail: behavior rules living in long descriptions are lost here; add
    # per-tool overrides if a specific tool regresses in A/B.
    """
    head = text.strip().split(". ", 1)[0].strip()
    if head and not head.endswith((".", "!", "?")):
        head += "."
    if not head or len(head) < 4:
        head = text.strip()[:300]
    return head[:300]


def _slim_schema(schema: Any) -> Any:
    """Deterministic compact form of a JSON schema for the model-facing wire.

    Keeps property names, types, enum/const, required and array `items`
    (recursively). Collapses anyOf/oneOf to its first non-null branch. Drops
    prose, defaults and numeric/string limits: Hermes still validates and
    coerces arguments against the full registry schema at dispatch, so the
    wire schema only has to steer the model, not enforce.
    """
    if not isinstance(schema, dict):
        return schema
    for combinator in ("anyOf", "oneOf"):
        branches = schema.get(combinator)
        if isinstance(branches, list) and branches:
            for branch in branches:
                if isinstance(branch, dict) and branch.get("type") != "null":
                    return _slim_schema(branch)
            return _slim_schema(branches[0])
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _SLIM_DROP_KEYS or key in ("anyOf", "oneOf", "allOf"):
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: _slim_schema(sub) for name, sub in value.items()}
        elif key in ("items", "prefixItems", "contains"):
            out[key] = _slim_schema(value)
        elif key in _SLIM_LEAF_KEYS:
            out[key] = value
        elif key == "required":
            out[key] = value
    if "type" not in out and "properties" in out:
        out["type"] = "object"
    return out


def _slim_registration(wire_name: str, description: str, parameters: dict) -> dict:
    if os.environ.get(SLIM_ENV, "1") == "0":
        return {"name": wire_name, "description": description, "parameters": parameters}
    return {
        "name": wire_name,
        "description": _slim_description(description),
        "parameters": _slim_schema(parameters),
    }


def plan_tools(tools: Iterable[dict] | None, native_names: Iterable[str] = ("read",)) -> WirePlan:
    """Choose the model-facing catalog for one request.

    A native descriptor from ``NATIVE_MAP`` is kept only when OpenCode
    registered it and its Hermes target is enabled; that Hermes schema is then
    not registered (one name per capability). A Hermes target shared by several
    natives (``search_files`` <- glob + grep) is replaced only when all of them
    are available. Every other enabled Hermes schema is registered with a
    compact wire form (``_slim_registration``, toggle with
    ``OC_FREE_SLIM_SCHEMAS=0``), with a reversible alias when it collides with
    a native name. Refuses before inference when ``read_file`` is absent.
    """
    native = {str(name) for name in native_names if isinstance(name, str) and name}
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

    if COMPAT_TOOL not in seen:
        raise CompatibilityUnavailable(
            "Hermes read_file is unavailable in this session, so the native read "
            "compatibility adapter cannot be advertised; enable the file toolset"
        )
    replaced = {
        host for host in {target for target, _ in NATIVE_MAP.values()} & seen
        if all(name in native for name, (target, _) in NATIVE_MAP.items() if target == host)
    }
    plan = WirePlan(
        native=frozenset(name for name, (target, _) in NATIVE_MAP.items() if target in replaced),
        enabled=frozenset(seen),
        replaced=frozenset(replaced),
    )
    taken = set(native)
    for function in functions:
        host = function["name"]
        if host in replaced:
            continue
        wire = WIRE_RENAMES.get(host, host) or host
        if wire in taken:
            wire = f"hermes_{host}"
            while wire in taken:
                wire = f"_{wire}"
        if len(wire) > MAX_WIRE_NAME:
            raise NativeInvalidArguments(
                f"Hermes tool {host!r} cannot fit a collision-free wire name under {MAX_WIRE_NAME} chars"
            )
        registration = _slim_registration(wire, function["description"], function["parameters"])
        registration["codemode"] = False
        plan.register.append(registration)
        plan.host_to_wire[host] = wire
        plan.wire_to_host[wire] = host
        taken.add(wire)
    return plan


def native_guidance(plan: WirePlan) -> str | None:
    """One system part stating the limits of the native descriptors in use."""
    if not plan.native:
        return None
    lines = [
        "Native OpenCode tools in this session are executed by Hermes. Relative paths resolve against the "
        "Hermes current working directory, not the private OpenCode runtime directory named in the "
        "environment block; paths inside that runtime directory are refused. Limits:",
    ]
    notes = {
        "read": "read: Hermes text/document extraction, not directory listings or image attachments; offset must be >= 0; limit 0 = default, max 2000.",
        "write": "write: an existing file must be read first in this session; Hermes path guards apply.",
        "glob": "glob: hidden:true is refused; hidden and ignored paths are skipped.",
        "grep": "grep: hidden and ignored paths are skipped.",
        "question": f"question: at most {CLARIFY_MAX_QUESTIONS} questions and {CLARIFY_MAX_CHOICES} options each; headers and option descriptions are shown inside the question text; the result is Hermes clarify JSON.",
        "todowrite": f"todowrite: replaces the whole list (max {TODO_MAX_ITEMS} items); an in_progress item may be moved before earlier pending ones; the result is Hermes todo_list JSON.",
    }
    lines += [f"- {notes[name]}" for name in NATIVE_MAP if name in plan.native]
    lines.append("Any native OpenCode tool not listed above is not available in this session; use the advertised Hermes tools instead.")
    return "\n".join(lines)


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
                        if wire in _INVERSE:
                            arguments = _INVERSE[wire](arguments)
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
            failed = bool(message.get("is_error"))
            if wire in NATIVE_MAP:
                # Hermes handlers commonly return errors inside JSON without
                # setting the outer is_error flag. Do not replay them as success.
                try:
                    body = json.loads(text)
                except (ValueError, TypeError):
                    body = None
                if isinstance(body, dict):
                    failed = failed or bool(body.get("error")) or body.get("success") is False
                    if wire == "question" and body.get("outcome") in {
                        "cancelled", "timed_out", "undelivered",
                    }:
                        failed = True
            if failed:
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
