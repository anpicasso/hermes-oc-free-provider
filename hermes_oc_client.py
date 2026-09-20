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
COMPAT_TOOL_TARGETS = {
    "bash": "terminal",
    "edit": "patch",
    "glob": "search_files",
    "grep": "search_files",
    "read": "read_file",
    "skill": "skill_view",
    "task": "delegate_task",
    "todowrite": "todo_list",
    "webfetch": "web_extract",
    "websearch": "web_search",
    "write": "write_file",
}
COMPAT_TOOL_DESCRIPTIONS = {
    "bash": "Execute a shell command in the local workspace.",
    "edit": "Replace exact text in a local file.",
    "glob": "Find local files by glob pattern.",
    "grep": "Search local file contents with a regular expression.",
    "read": "Read a local file.",
    "skill": "Load a Hermes skill by name.",
    "task": "Delegate a task to a Hermes subagent.",
    "todowrite": "Replace the current Hermes task list.",
    "webfetch": "Extract content from a web page.",
    "websearch": "Search the web.",
    "write": "Write a local file.",
}
COMPAT_TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "bash": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "minimum": 1},
            "workdir": {"type": "string"},
        },
        "required": ["command"],
    },
    "edit": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "oldString": {"type": "string"},
            "newString": {"type": "string"},
            "replaceAll": {"type": "boolean"},
        },
        "required": ["filePath", "oldString", "newString"],
    },
    "glob": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["pattern"],
    },
    "grep": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "include": {"type": "string"},
        },
        "required": ["pattern"],
    },
    "read": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["filePath"],
    },
    "skill": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
    "task": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "prompt": {"type": "string"},
            "subagent_type": {"type": "string"},
            "task_id": {"type": "string"},
            "command": {"type": "string"},
            "background": {"type": "boolean"},
        },
        "required": ["description", "prompt", "subagent_type"],
    },
    "todowrite": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": [
                                "pending",
                                "in_progress",
                                "completed",
                                "cancelled",
                            ],
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                    },
                    "required": ["content", "status", "priority"],
                },
            }
        },
        "required": ["todos"],
    },
    "webfetch": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "format": {"type": "string", "enum": ["text", "markdown", "html"]},
            "timeout": {"type": "integer", "minimum": 1},
        },
        "required": ["url"],
    },
    "websearch": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "numResults": {"type": "integer", "minimum": 1, "maximum": 100},
            "livecrawl": {"type": "string"},
            "type": {"type": "string"},
            "contextMaxCharacters": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
    },
    "write": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["filePath", "content"],
    },
}
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


def _mapped_tool(name: str, target: dict[str, Any]) -> dict[str, Any]:
    function = target.get("function") or {}
    target_name = str(function.get("name") or COMPAT_TOOL_TARGETS[name])
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                f"{COMPAT_TOOL_DESCRIPTIONS[name]} "
                f"OpenCode-compatible alias for Hermes {target_name}."
            ),
            "parameters": COMPAT_TOOL_PARAMETERS[name],
        },
    }


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
        return ""
    return str(tool["function"].get("name") or "").strip()


def _wire_tools(
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_name = {_tool_name(tool): tool for tool in tools or [] if _tool_name(tool)}
    mapped: dict[str, str] = {}
    wire = []
    for name in COMPAT_TOOL_NAMES:
        target_name = COMPAT_TOOL_TARGETS[name]
        if target := by_name.get(target_name):
            wire.append(_mapped_tool(name, target))
            mapped[name] = target_name
        else:
            wire.append(_compat_tool(name))
    mapped_targets = set(mapped.values())
    wire.extend(
        tool
        for tool in tools or []
        if _tool_name(tool) not in COMPAT_TOOL_NAMES
        and _tool_name(tool) not in mapped_targets
    )
    return wire, mapped


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
    mapped_tools: dict[str, str],
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
                name, arguments = _opencode_tool(
                    str(function.get("name") or ""), arguments, mapped_tools
                )
                call_id = str(call.get("id") or call.get("call_id") or "")
                if call_id:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
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


def _responses_tool_choice(value: Any, mapped_tools: dict[str, str]) -> Any:
    value = _wire_tool_choice(value, mapped_tools)
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


def _required(arguments: dict[str, Any], name: str, key: str) -> Any:
    if key not in arguments:
        raise OpenCodeError(
            f"OpenCode tool '{name}' omitted required argument '{key}'."
        )
    return arguments[key]


def _mapped_arguments(name: str, encoded: str) -> str:
    try:
        arguments = json.loads(encoded or "{}")
    except json.JSONDecodeError as exc:
        raise OpenCodeError(
            f"OpenCode tool '{name}' returned invalid JSON arguments."
        ) from exc
    if not isinstance(arguments, dict):
        raise OpenCodeError(f"OpenCode tool '{name}' arguments must be a JSON object.")

    if name == "bash":
        mapped = {"command": _required(arguments, name, "command")}
        if "workdir" in arguments:
            mapped["workdir"] = arguments["workdir"]
        if "timeout" in arguments:
            milliseconds = int(arguments["timeout"])
            mapped["timeout"] = max(1, (milliseconds + 999) // 1000)
    elif name == "edit":
        mapped = {
            "mode": "replace",
            "path": _required(arguments, name, "filePath"),
            "old_string": _required(arguments, name, "oldString"),
            "new_string": _required(arguments, name, "newString"),
        }
        if "replaceAll" in arguments:
            mapped["replace_all"] = arguments["replaceAll"]
    elif name == "glob":
        mapped = {
            "target": "files",
            "pattern": _required(arguments, name, "pattern"),
        }
        if "path" in arguments:
            mapped["path"] = arguments["path"]
    elif name == "grep":
        mapped = {
            "target": "content",
            "pattern": _required(arguments, name, "pattern"),
        }
        if "path" in arguments:
            mapped["path"] = arguments["path"]
        if "include" in arguments:
            mapped["file_glob"] = arguments["include"]
    elif name == "read":
        mapped = {"path": _required(arguments, name, "filePath")}
        for key in ("offset", "limit"):
            if key in arguments:
                mapped[key] = arguments[key]
    elif name == "skill":
        mapped = {"name": _required(arguments, name, "name")}
    elif name == "task":
        context = [
            str(_required(arguments, name, "description")),
            f"Requested OpenCode subagent type: {_required(arguments, name, 'subagent_type')}",
        ]
        for key, label in (
            ("task_id", "Previous task ID"),
            ("command", "Requested command"),
            ("background", "Requested background execution"),
        ):
            if key in arguments:
                context.append(f"{label}: {arguments[key]}")
        mapped = {
            "tasks": [
                {
                    "goal": _required(arguments, name, "prompt"),
                    "context": "\n".join(context),
                }
            ]
        }
    elif name == "todowrite":
        todos = _required(arguments, name, "todos")
        if not isinstance(todos, list):
            raise OpenCodeError(
                "OpenCode tool 'todowrite' argument 'todos' must be a list."
            )
        mapped = {
            "todos": [
                {
                    "id": f"oc-{index + 1}",
                    "content": _required(todo, name, "content"),
                    "status": _required(todo, name, "status"),
                }
                for index, todo in enumerate(todos)
                if isinstance(todo, dict)
            ],
            "merge": False,
        }
        if len(mapped["todos"]) != len(todos):
            raise OpenCodeError("OpenCode tool 'todowrite' contains a non-object item.")
    elif name == "webfetch":
        mapped = {"urls": [_required(arguments, name, "url")]}
    elif name == "websearch":
        mapped = {"query": _required(arguments, name, "query")}
        if "numResults" in arguments:
            mapped["limit"] = max(1, min(100, int(arguments["numResults"])))
    elif name == "write":
        mapped = {
            "path": _required(arguments, name, "filePath"),
            "content": _required(arguments, name, "content"),
        }
    else:
        mapped = arguments
    return json.dumps(mapped, separators=(",", ":"))


def _opencode_arguments(name: str, encoded: str) -> str:
    try:
        arguments = json.loads(encoded or "{}")
    except json.JSONDecodeError as exc:
        raise OpenCodeError(
            f"Hermes tool '{name}' has invalid JSON arguments in conversation history."
        ) from exc
    if not isinstance(arguments, dict):
        raise OpenCodeError(
            f"Hermes tool '{name}' arguments in conversation history must be an object."
        )

    if name == "bash":
        mapped = {"command": _required(arguments, name, "command")}
        if "workdir" in arguments:
            mapped["workdir"] = arguments["workdir"]
        if "timeout" in arguments:
            mapped["timeout"] = max(1, int(arguments["timeout"]) * 1000)
    elif name == "edit":
        if arguments.get("mode", "replace") != "replace":
            raise OpenCodeError(
                "A non-replace Hermes patch cannot be replayed as edit."
            )
        mapped = {
            "filePath": _required(arguments, name, "path"),
            "oldString": _required(arguments, name, "old_string"),
            "newString": _required(arguments, name, "new_string"),
        }
        if "replace_all" in arguments:
            mapped["replaceAll"] = arguments["replace_all"]
    elif name in {"glob", "grep"}:
        mapped = {"pattern": _required(arguments, name, "pattern")}
        if "path" in arguments:
            mapped["path"] = arguments["path"]
        if name == "grep" and "file_glob" in arguments:
            mapped["include"] = arguments["file_glob"]
    elif name == "read":
        mapped = {"filePath": _required(arguments, name, "path")}
        for key in ("offset", "limit"):
            if key in arguments:
                mapped[key] = arguments[key]
    elif name == "skill":
        mapped = {"name": _required(arguments, name, "name")}
    elif name == "task":
        tasks = _required(arguments, name, "tasks")
        if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
            raise OpenCodeError("Hermes delegate history cannot be replayed as task.")
        task = tasks[0]
        context = str(task.get("context") or "")
        lines = context.splitlines()
        mapped = {
            "description": lines[0] if lines else "Delegated Hermes task",
            "prompt": _required(task, name, "goal"),
            "subagent_type": "general",
        }
        prefixes = {
            "Requested OpenCode subagent type: ": "subagent_type",
            "Previous task ID: ": "task_id",
            "Requested command: ": "command",
            "Requested background execution: ": "background",
        }
        for line in lines[1:]:
            for prefix, key in prefixes.items():
                if line.startswith(prefix):
                    value: Any = line.removeprefix(prefix)
                    if key == "background":
                        value = str(value).lower() == "true"
                    mapped[key] = value
                    break
    elif name == "todowrite":
        todos = _required(arguments, name, "todos")
        if not isinstance(todos, list) or any(
            not isinstance(todo, dict) for todo in todos
        ):
            raise OpenCodeError("Hermes todo history cannot be replayed as todowrite.")
        mapped = {
            "todos": [
                {
                    "content": _required(todo, name, "content"),
                    "status": _required(todo, name, "status"),
                    "priority": "medium",
                }
                for todo in todos
            ]
        }
    elif name == "webfetch":
        urls = _required(arguments, name, "urls")
        if not isinstance(urls, list) or not urls:
            raise OpenCodeError(
                "Hermes web extract history cannot be replayed as webfetch."
            )
        mapped = {"url": urls[0]}
    elif name == "websearch":
        mapped = {"query": _required(arguments, name, "query")}
        if "limit" in arguments:
            mapped["numResults"] = arguments["limit"]
    elif name == "write":
        mapped = {
            "filePath": _required(arguments, name, "path"),
            "content": _required(arguments, name, "content"),
        }
    else:
        mapped = arguments
    return json.dumps(mapped, separators=(",", ":"))


def _opencode_alias(name: str, arguments: str, mapped_tools: dict[str, str]) -> str:
    aliases = [alias for alias in COMPAT_TOOL_NAMES if mapped_tools.get(alias) == name]
    if not aliases:
        return name
    if name == "search_files":
        try:
            target = json.loads(arguments or "{}").get("target")
        except (AttributeError, json.JSONDecodeError):
            target = None
        return "glob" if target == "files" else "grep"
    return aliases[0]


def _opencode_tool(
    name: str, arguments: str, mapped_tools: dict[str, str]
) -> tuple[str, str]:
    alias = _opencode_alias(name, arguments, mapped_tools)
    if alias == name:
        return name, arguments or "{}"
    return alias, _opencode_arguments(alias, arguments)


def _wire_messages(
    messages: list[dict[str, Any]], mapped_tools: dict[str, str]
) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        item = dict(message)
        if item.get("role") == "assistant" and item.get("tool_calls"):
            calls = []
            for call in item["tool_calls"]:
                if not isinstance(call, dict):
                    calls.append(call)
                    continue
                wired_call = dict(call)
                function = dict(call.get("function") or {})
                arguments = function.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, separators=(",", ":"))
                function["name"], function["arguments"] = _opencode_tool(
                    str(function.get("name") or ""), arguments, mapped_tools
                )
                wired_call["function"] = function
                calls.append(wired_call)
            item["tool_calls"] = calls
        result.append(item)
    return result


def _wire_tool_choice(value: Any, mapped_tools: dict[str, str]) -> Any:
    if not isinstance(value, dict):
        return value
    function = value.get("function")
    if value.get("type") != "function" or not isinstance(function, dict):
        return value
    result = dict(value)
    wired_function = dict(function)
    wired_function["name"] = _opencode_alias(
        str(wired_function.get("name") or ""), "{}", mapped_tools
    )
    result["function"] = wired_function
    return result


def _translate_tool(
    name: str, arguments: str, mapped_tools: dict[str, str]
) -> tuple[str, str]:
    if error := _phantom_tool_error(name, mapped_tools):
        raise error
    target = mapped_tools.get(name, name)
    if target != name:
        arguments = _mapped_arguments(name, arguments)
    return target, arguments or "{}"


def _tool_calls_from_parts(
    parts: dict[int, dict[str, Any]], mapped_tools: dict[str, str]
) -> list[SimpleNamespace]:
    result = []
    for index in sorted(parts):
        call = parts[index]
        name, arguments = _translate_tool(
            str(call.get("name") or ""),
            str(call.get("arguments") or "{}"),
            mapped_tools,
        )
        result.append(
            SimpleNamespace(
                id=call.get("id") or f"call_{index}",
                call_id=call.get("id") or f"call_{index}",
                type=call.get("type") or "function",
                function=SimpleNamespace(name=name, arguments=arguments),
                response_item_id=None,
            )
        )
    return result


def _tool_call_dicts(
    parts: dict[int, dict[str, Any]], mapped_tools: dict[str, str]
) -> list[dict[str, Any]]:
    result = []
    for index in sorted(parts):
        call = parts[index]
        name, arguments = _translate_tool(
            str(call.get("name") or ""),
            str(call.get("arguments") or "{}"),
            mapped_tools,
        )
        result.append(
            {
                "index": index,
                "id": call.get("id") or f"call_{index}",
                "type": call.get("type") or "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return result


def _phantom_tool_error(
    name: str, mapped_tools: dict[str, str]
) -> OpenCodeError | None:
    if name in COMPAT_TOOL_NAMES and name not in mapped_tools:
        return OpenCodeError(
            f"OpenCode attempted compatibility-only tool '{name}'. "
            "No matching Hermes tool was supplied, so it was not executed."
        )
    return None


def _merge_sse(
    events: Iterator[dict[str, Any]],
    *,
    requested_model: str,
    mapped_tools: dict[str, str],
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
                if error := _phantom_tool_error(item["name"], mapped_tools):
                    raise error
    if not seen:
        raise OpenCodeError("OpenCode inference returned no events.")
    if finish_reason == "content_filter":
        raise OpenCodeError("OpenCode inference was blocked by the content filter.")
    tool_calls = _tool_calls_from_parts(calls, mapped_tools)
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


def _stream_sse(
    events: Iterator[dict[str, Any]],
    *,
    requested_model: str,
    mapped_tools: dict[str, str],
) -> Iterator[dict[str, Any]]:
    calls: dict[int, dict[int, dict[str, Any]]] = {}
    model = requested_model
    seen = False
    for event in events:
        seen = True
        if event.get("error"):
            raise OpenCodeError(f"OpenCode inference error: {event['error']}")
        model = str(event.get("model") or model)
        event.setdefault("choices", [])
        for choice in event.get("choices") or []:
            if choice.get("finish_reason") == "content_filter":
                raise OpenCodeError(
                    "OpenCode inference was blocked by the content filter."
                )
            choice_index = int(choice.get("index", 0) or 0)
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
                choice["delta"] = delta
            raw_calls = delta.pop("tool_calls", [])
            choice_calls = calls.get(choice_index, {})
            for raw_call in raw_calls:
                choice_calls = calls.setdefault(choice_index, {})
                index = int(raw_call.get("index", 0) or 0)
                item = choice_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "name": "", "arguments": ""},
                )
                if raw_call.get("id"):
                    item["id"] = str(raw_call["id"])
                item["type"] = str(raw_call.get("type") or item["type"])
                function = raw_call.get("function") or {}
                if function.get("name"):
                    item["name"] = str(function["name"])
                item["arguments"] += str(function.get("arguments") or "")
                if error := _phantom_tool_error(item["name"], mapped_tools):
                    raise error
            if choice.get("finish_reason") and choice_calls:
                delta["tool_calls"] = _tool_call_dicts(choice_calls, mapped_tools)
                calls.pop(choice_index)
        yield event
    if not seen:
        raise OpenCodeError("OpenCode inference returned no events.")
    if calls:
        yield {
            "model": model,
            "choices": [
                {
                    "index": choice_index,
                    "delta": {
                        "tool_calls": _tool_call_dicts(choice_calls, mapped_tools)
                    },
                    "finish_reason": "tool_calls",
                }
                for choice_index, choice_calls in sorted(calls.items())
            ],
        }


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
            events, mapped_tools = self._direct_events(
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
            return _merge_sse(
                events, requested_model=selected, mapped_tools=mapped_tools
            )

        if not stream:
            return _LazyValue(factory)

        def stream_factory() -> Iterator[SimpleNamespace]:
            events, mapped_tools = self._direct_events(
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
            for event in _stream_sse(
                events, requested_model=selected, mapped_tools=mapped_tools
            ):
                yield _namespace(event)

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
    ) -> tuple[Iterator[dict[str, Any]], dict[str, str]]:
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
        wire_tools, mapped_tools = _wire_tools(tools)
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are opencode"},
                *_wire_messages(messages, mapped_tools),
            ],
            "tools": wire_tools,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        optional = {
            "tool_choice": _wire_tool_choice(tool_choice, mapped_tools),
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

        return events(), mapped_tools

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
    ) -> tuple[Iterator[dict[str, Any]], dict[str, str]]:
        wire_tools, mapped_tools = _wire_tools(tools)
        input_items, instructions = _responses_input(messages, mapped_tools)
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
            "tool_choice": _responses_tool_choice(tool_choice, mapped_tools),
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

        return events(), mapped_tools
