# pyright: reportArgumentType=false
# ruff: noqa: E402 -- the standalone plugin directory must be added before import.
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hermes_oc_client as client  # type: ignore[import-not-found]


# Local synthetic transport: FakeResponse and mocked _urlopen keep every test
# in this file offline; nothing here performs live inference.
class FakeResponse:
    def __init__(self, lines=(), body=b"{}"):
        self._lines = [
            line.encode() if isinstance(line, str) else line for line in lines
        ]
        self._body = body

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def sse(*events):
    return FakeResponse(
        [*(f"data: {json.dumps(event)}\n" for event in events), "data: [DONE]\n"]
    )


def tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Hermes {name}",
            "parameters": {"type": "object"},
        },
    }


class ClientTests(unittest.TestCase):
    def test_session_id_matches_opencode_shape(self):
        self.assertRegex(client._session_id(), r"^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$")

    def test_session_id_is_a_descending_sortable_id(self):
        # The gateway rejects ids whose 12 hex chars are an ascending timestamp:
        # OpenCode inverts (~) them, so the prefix starts high.
        session = client._session_id()
        self.assertGreaterEqual(int(session[4:6], 16), 0xC0)
        self.assertIn(session, client._env_block(session))

    def test_direct_request_preserves_hermes_tools_and_returns_native_call(self):
        captured = {}

        def fake_open(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return sse(
                {
                    "model": "m",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "hermes_probe",
                                            "arguments": '{"value":"PING"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
                {
                    "model": "m",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6,
                    },
                },
            )

        oc = client.OpenCodeClient()
        tool = {
            "type": "function",
            "function": {
                "name": "hermes_probe",
                "description": "probe",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            },
        }
        with mock.patch.object(client, "_open_stream", side_effect=fake_open):
            result = oc.chat.completions.create(
                model="m",
                messages=[{"role": "user", "content": "probe"}],
                tools=[tool],
                timeout=12,
            )
            call = result.choices[0].message.tool_calls[0]

        body = json.loads(captured["request"].data)
        names = [entry["function"]["name"] for entry in body["tools"]]
        self.assertEqual(
            names[: len(client.COMPAT_TOOL_NAMES)], list(client.COMPAT_TOOL_NAMES)
        )
        self.assertIn("hermes_probe", names)
        env = body["messages"][0]
        self.assertEqual(env["role"], "system")
        self.assertIn("<env>", env["content"])
        self.assertIn("Current conversation session ID: ses_", env["content"])
        self.assertIn("Working directory: ", env["content"])
        self.assertEqual(call.function.name, "hermes_probe")
        self.assertEqual(json.loads(call.function.arguments), {"value": "PING"})
        self.assertEqual(result.usage.total_tokens, 6)
        self.assertEqual(captured["timeout"], 12)
        self.assertTrue(
            captured["request"].headers["X-opencode-session"].startswith("ses_")
        )

    def test_repeated_tool_identity_and_reasoning_details_are_preserved(self):
        oc = client.OpenCodeClient()
        response = sse(
            {
                "model": "m",
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"type": "reasoning.text", "text": "r"}
                            ],
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "hermes_probe",
                                        "arguments": '{"value":',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
            {
                "model": "m",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "hermes_probe",
                                        "arguments": '"PING"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
        tool = {
            "type": "function",
            "function": {"name": "hermes_probe", "parameters": {"type": "object"}},
        }
        with mock.patch.object(client, "_open_stream", return_value=response):
            result = oc.chat.completions.create(model="m", messages=[], tools=[tool])
            call = result.choices[0].message.tool_calls[0]
        self.assertEqual(call.id, "call_1")
        self.assertEqual(call.function.name, "hermes_probe")
        self.assertEqual(json.loads(call.function.arguments), {"value": "PING"})
        self.assertEqual(
            result.choices[0].message.reasoning_details,
            [{"type": "reasoning.text", "text": "r"}],
        )

    def test_all_opencode_tools_map_to_available_hermes_tools(self):
        targets = list(dict.fromkeys(client.COMPAT_TOOL_TARGETS.values()))
        wire, mapped = client._wire_tools(
            [*(tool(name) for name in targets), tool("hermes_probe")]
        )
        self.assertEqual(mapped, client.COMPAT_TOOL_TARGETS)
        wire_names = [entry["function"]["name"] for entry in wire]
        self.assertEqual(
            wire_names[: len(client.COMPAT_TOOL_NAMES)],
            list(client.COMPAT_TOOL_NAMES),
        )
        self.assertTrue(set(targets).isdisjoint(wire_names))
        self.assertIn("hermes_probe", wire_names)
        self.assertTrue(
            all(
                entry["function"]["parameters"]
                == client.COMPAT_TOOL_PARAMETERS[entry["function"]["name"]]
                for entry in wire[:11]
            )
        )

        cases = {
            "bash": (
                {"command": "pwd", "timeout": 1501, "workdir": "/repo"},
                {"command": "pwd", "timeout": 2, "workdir": "/repo"},
            ),
            "edit": (
                {
                    "filePath": "a.py",
                    "oldString": "old",
                    "newString": "new",
                    "replaceAll": True,
                },
                {
                    "mode": "replace",
                    "path": "a.py",
                    "old_string": "old",
                    "new_string": "new",
                    "replace_all": True,
                },
            ),
            "glob": (
                {"pattern": "*.py", "path": "src"},
                {"target": "files", "pattern": "*.py", "path": "src"},
            ),
            "grep": (
                {"pattern": "TODO", "path": "src", "include": "*.py"},
                {
                    "target": "content",
                    "pattern": "TODO",
                    "path": "src",
                    "file_glob": "*.py",
                },
            ),
            "read": (
                {"filePath": "a.py", "offset": 2, "limit": 4},
                {"path": "a.py", "offset": 2, "limit": 4},
            ),
            "skill": ({"name": "pdf"}, {"name": "pdf"}),
            "task": (
                {
                    "description": "Inspect",
                    "prompt": "Inspect the parser",
                    "subagent_type": "explore",
                    "background": True,
                },
                {
                    "tasks": [
                        {
                            "goal": "Inspect the parser",
                            "context": "Inspect\nRequested OpenCode subagent type: explore\nRequested background execution: True",
                        }
                    ]
                },
            ),
            "todowrite": (
                {
                    "todos": [
                        {"content": "Ship", "status": "in_progress", "priority": "high"}
                    ]
                },
                {
                    "todos": [
                        {"id": "oc-1", "content": "Ship", "status": "in_progress"}
                    ],
                    "merge": False,
                },
            ),
            "webfetch": (
                {"url": "https://example.com", "format": "html", "timeout": 5},
                {"urls": ["https://example.com"]},
            ),
            "websearch": (
                {"query": "Hermes", "numResults": 8, "livecrawl": "always"},
                {"query": "Hermes", "limit": 8},
            ),
            "write": (
                {"filePath": "a.txt", "content": "hello"},
                {"path": "a.txt", "content": "hello"},
            ),
        }
        for alias, (arguments, expected) in cases.items():
            with self.subTest(alias=alias):
                name, encoded = client._translate_tool(
                    alias, json.dumps(arguments), mapped
                )
                self.assertEqual(name, client.COMPAT_TOOL_TARGETS[alias])
                self.assertEqual(json.loads(encoded), expected)
                replay_name, replay_arguments = client._opencode_tool(
                    name, encoded, mapped
                )
                self.assertEqual(replay_name, alias)
                self.assertTrue(
                    set(client.COMPAT_TOOL_PARAMETERS[alias]["required"])
                    <= set(json.loads(replay_arguments))
                )

        wire, mapped = client._wire_tools([tool("read")])
        self.assertNotIn("read", mapped)
        self.assertIn("unavailable", wire[4]["function"]["description"])
        with self.assertRaisesRegex(client.OpenCodeError, "compatibility-only tool"):
            client._translate_tool("read", '{"filePath":"a.py"}', mapped)

    def test_every_recorded_delegate_shape_replays_as_task(self):
        _, mapped = client._wire_tools([tool("delegate_task")])

        def replay(arguments):
            name, encoded = client._opencode_tool(
                "delegate_task", json.dumps(arguments), mapped
            )
            self.assertEqual(name, "task")
            return json.loads(encoded)

        required = set(client.COMPAT_TOOL_PARAMETERS["task"]["required"])

        for arguments in (
            {"action": "list"},
            {"action": "stop", "subagent_id": "sa-0-f834b4a7"},
            {
                "action": "steer",
                "subagent_id": "sa-1",
                "message": "skip the scratch database",
            },
            {"goal": "Ship the parser", "context": "Parser lives in src/parse.py"},
            {"tasks": json.dumps([{"goal": "Ship it", "context": "src/parse.py"}])},
            {"tasks": None, "goal": "Legacy wins over a null batch"},
            {},
        ):
            with self.subTest(arguments=arguments):
                mapped_arguments = replay(arguments)
                self.assertLessEqual(required, set(mapped_arguments))

        stopped = replay({"action": "stop", "subagent_id": "sa-0-f834b4a7"})
        self.assertEqual(stopped["task_id"], "sa-0-f834b4a7")
        self.assertIn("Requested action: stop", stopped["description"])

        steered = replay(
            {"action": "steer", "subagent_id": "sa-1", "message": "skip the scratch db"}
        )
        self.assertEqual(steered["task_id"], "sa-1")

        legacy = replay({"goal": "Ship the parser", "context": "Parser lives in src"})
        self.assertEqual(legacy["prompt"], "Ship the parser")
        self.assertEqual(legacy["description"], "Parser lives in src")

        unencoded = replay({"tasks": json.dumps([{"goal": "Ship it", "context": "c"}])})
        self.assertEqual(unencoded["prompt"], "Ship it")

    def test_delegate_batch_replay_keeps_every_goal(self):
        _, mapped = client._wire_tools([tool("delegate_task")])

        def prompt(arguments):
            _, encoded = client._opencode_tool(
                "delegate_task", json.dumps(arguments), mapped
            )
            return json.loads(encoded)["prompt"]

        self.assertEqual(prompt({"tasks": [{"goal": "Only", "context": "c"}]}), "Only")

        batch = prompt(
            {
                "tasks": [
                    {"goal": "Fix the blocker", "context": "src/a.py"},
                    {"goal": "Scope by country", "context": "src/b.py"},
                    {"goal": "Write the changelog"},
                ]
            }
        )
        self.assertIn("Fix the blocker", batch)
        self.assertIn("Scope by country", batch)
        self.assertIn("Write the changelog", batch)
        self.assertIn("src/b.py", batch)
        self.assertTrue(batch.startswith("Fix the blocker"))
        self.assertLess(batch.index("Scope by country"), batch.index("Write the changelog"))

    def test_native_tool_history_and_choice_are_rewritten_to_aliases(self):
        captured = {}

        def fake_open(request, _timeout):
            captured["body"] = json.loads(request.data)
            return sse(
                {
                    "model": "m",
                    "choices": [
                        {"delta": {"content": "DONE"}, "finish_reason": "stop"}
                    ],
                }
            )

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command":"pwd","timeout":2}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "/repo",
            },
        ]
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_open_stream", side_effect=fake_open):
            result = oc.chat.completions.create(
                model="m",
                messages=messages,
                tools=[tool("terminal"), tool("hermes_probe")],
                tool_choice={"type": "function", "function": {"name": "terminal"}},
            )
            self.assertEqual(result.choices[0].message.content, "DONE")

        body = captured["body"]
        names = [entry["function"]["name"] for entry in body["tools"]]
        self.assertIn("bash", names)
        self.assertIn("hermes_probe", names)
        self.assertNotIn("terminal", names)
        self.assertEqual(body["tool_choice"]["function"]["name"], "bash")
        replay = body["messages"][1]["tool_calls"][0]["function"]
        self.assertEqual(replay["name"], "bash")
        self.assertEqual(
            json.loads(replay["arguments"]), {"command": "pwd", "timeout": 2000}
        )
        self.assertEqual(body["messages"][2], messages[1])

        mapped = {"bash": "terminal"}
        response_input, _ = client._responses_input(messages, mapped, "ses_test")
        self.assertIn(
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "bash",
                "arguments": '{"command":"pwd","timeout":2000}',
            },
            response_input,
        )

    def test_streamed_bash_call_is_buffered_and_mapped_to_terminal(self):
        response = sse(
            {
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": '{"command":"printf',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ' hi","timeout":1500}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_open_stream", return_value=response):
            chunks = list(
                oc.chat.completions.create(
                    model="m", messages=[], tools=[tool("terminal")], stream=True
                )
            )
        calls = [
            call
            for chunk in chunks
            for choice in chunk.choices
            for call in getattr(choice.delta, "tool_calls", [])
        ]
        self.assertEqual(calls[0].function.name, "terminal")
        self.assertEqual(
            json.loads(calls[0].function.arguments),
            {"command": "printf hi", "timeout": 2},
        )

    def test_stream_error_is_explicit_and_session_is_stable(self):
        requests = []

        def fake_open(request, _timeout):
            requests.append(request)
            if len(requests) == 1:
                return sse({"error": {"message": "boom"}})
            return sse(
                {
                    "model": "m",
                    "choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}],
                }
            )

        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_open_stream", side_effect=fake_open):
            stream = oc.chat.completions.create(model="m", messages=[], stream=True)
            with self.assertRaisesRegex(client.OpenCodeError, "inference error"):
                list(stream)
            self.assertEqual(
                oc.chat.completions.create(model="m", messages=[])
                .choices[0]
                .message.content,
                "OK",
            )
        self.assertEqual(
            requests[0].headers["X-opencode-session"],
            requests[1].headers["X-opencode-session"],
        )

    def test_tool_result_is_forwarded_unchanged(self):
        captured = {}

        def fake_open(request, _timeout):
            captured["body"] = json.loads(request.data)
            return sse(
                {
                    "model": "m",
                    "choices": [
                        {"delta": {"content": "TOOL_OK"}, "finish_reason": "stop"}
                    ],
                }
            )

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "hermes_probe",
                            "arguments": '{"value":"PING"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "LOCAL_HERMES_RESULT:PONG",
            },
        ]
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_open_stream", side_effect=fake_open):
            result = oc.chat.completions.create(model="m", messages=messages, tools=[])
            self.assertEqual(result.choices[0].message.content, "TOOL_OK")
        self.assertEqual(captured["body"]["messages"][2], messages[1])

    def test_responses_model_adapts_native_tool_continuation(self):
        captured = []
        reasoning = {
            "id": "rs_1",
            "type": "reasoning",
            "status": "completed",
            "encrypted_content": "opaque",
            "summary": [],
        }
        first_response = sse(
            {
                "type": "response.created",
                "response": {
                    "model": "free-responses-probe",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": reasoning,
            },
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "completed",
                    "name": "hermes_probe",
                    "call_id": "call_1",
                    "arguments": '{"value":"PING"}',
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "model": "free-responses-probe",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "total_tokens": 13,
                    },
                },
            },
        )
        second_response = sse(
            {
                "type": "response.output_text.delta",
                "delta": "TOOL_OK",
                "response": {"model": "free-responses-probe"},
            },
            {
                "type": "response.completed",
                "response": {
                    "model": "free-responses-probe",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 14,
                        "output_tokens": 2,
                        "total_tokens": 16,
                    },
                },
            },
        )

        def fake_open(request, _timeout):
            captured.append((request.full_url, json.loads(request.data)))
            return first_response if len(captured) == 1 else second_response

        tool = {
            "type": "function",
            "function": {
                "name": "hermes_probe",
                "description": "probe",
                "parameters": {"type": "object"},
            },
        }
        oc = client.OpenCodeClient()
        with (
            mock.patch.object(client, "RESPONSES_MODELS", {"free-responses-probe"}),
            mock.patch.object(client, "_open_stream", side_effect=fake_open),
        ):
            first = oc.chat.completions.create(
                model="free-responses-probe",
                messages=[{"role": "user", "content": "probe"}],
                tools=[tool],
            )
            call = first.choices[0].message.tool_calls[0]
            messages = [
                {"role": "user", "content": "probe"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_details": first.choices[0].message.reasoning_details,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": "LOCAL_HERMES_RESULT:PONG",
                },
            ]
            second = oc.chat.completions.create(
                model="free-responses-probe",
                messages=messages,
                tools=[tool],
            )
            self.assertEqual(second.choices[0].message.content, "TOOL_OK")

        self.assertEqual(captured[0][0], client.RESPONSES_URL)
        first_body = captured[0][1]
        self.assertFalse(first_body["store"])
        self.assertEqual(first_body["include"], ["reasoning.encrypted_content"])
        self.assertEqual(first_body["tools"][-1]["name"], "hermes_probe")
        replay = captured[1][1]["input"]
        self.assertIn(reasoning, replay)
        self.assertIn(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "LOCAL_HERMES_RESULT:PONG",
            },
            replay,
        )

    def test_compatibility_only_tool_call_fails_closed(self):
        oc = client.OpenCodeClient()
        response = sse(
            {
                "model": "m",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "x",
                                    "type": "function",
                                    "function": {"name": "bash", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )
        with mock.patch.object(client, "_open_stream", return_value=response):
            result = oc.chat.completions.create(model="m", messages=[], tools=[])
            with self.assertRaisesRegex(
                client.OpenCodeError, "compatibility-only tool 'bash'"
            ):
                _ = result.choices

    def test_http_error_exposes_status_code(self):
        body = (
            b'{"type":"error","error":{"type":"FreeUsageLimitError",'
            b'"message":"Free usage limit reached."}}'
        )
        error = urllib.error.HTTPError(  # type: ignore[arg-type]
            client.DIRECT_URL, 429, "rate", {}, io.BytesIO(body)
        )
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_open_stream", side_effect=error):
            result = oc.chat.completions.create(model="m", messages=[], tools=[])
            with self.assertRaises(client.OpenCodeError) as caught:
                _ = result.choices
        self.assertEqual(caught.exception.status_code, 429)
        self.assertIn("FreeUsageLimitError", str(caught.exception))

    def test_empty_and_content_filtered_responses_fail_explicitly(self):
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_open_stream", return_value=sse()):
            with self.assertRaisesRegex(client.OpenCodeError, "no events"):
                _ = oc.chat.completions.create(model="m", messages=[]).choices
        filtered = sse(
            {
                "model": "m",
                "choices": [{"delta": {}, "finish_reason": "content_filter"}],
            }
        )
        with mock.patch.object(client, "_open_stream", return_value=filtered):
            with self.assertRaisesRegex(client.OpenCodeError, "content filter"):
                _ = oc.chat.completions.create(model="m", messages=[]).choices

    def test_sync_and_async_client_contract(self):
        def fake_open(_request, _timeout):
            return sse(
                {
                    "model": "m",
                    "choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}],
                }
            )

        oc = client.OpenCodeClient()

        async def run():
            completion = await oc.chat.completions.create(model="m", messages=[])
            self.assertEqual(completion.choices[0].message.content, "OK")
            stream = await oc.chat.completions.create(
                model="m", messages=[], stream=True
            )
            chunks = [chunk async for chunk in stream]
            self.assertEqual(chunks[0].choices[0].delta.content, "OK")

        with mock.patch.object(client, "_open_stream", side_effect=fake_open):
            asyncio.run(run())

    def test_redirects_are_disabled(self):
        self.assertIsNone(  # type: ignore[arg-type]
            client._NoRedirect().redirect_request(
                None, None, 302, "", {}, "https://elsewhere"
            )
        )

    def test_closed_client_fails_before_network(self):
        oc = client.OpenCodeClient()
        oc.close()
        with mock.patch.object(client, "_open_stream") as open_request:
            result = oc.chat.completions.create(model="m", messages=[])
            with self.assertRaisesRegex(client.OpenCodeError, "client is closed"):
                _ = result.choices
        open_request.assert_not_called()

    def test_unmappable_patch_history_replays_natively(self):
        arguments = json.dumps(
            {
                "mode": "patch",
                "patch": "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n*** End Patch",
            }
        )
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_v4a",
                        "type": "function",
                        "function": {"name": "patch", "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_v4a", "content": "Success."},
        ]
        wire = client._wire_messages(messages, {"edit": "patch"})
        self.assertEqual(wire[0]["tool_calls"][0]["function"]["name"], "patch")
        self.assertEqual(wire[0]["tool_calls"][0]["function"]["arguments"], arguments)
        self.assertEqual(wire[1], messages[1])

    def test_catalog_intersects_live_zero_cost_tool_models(self):
        zen = {"data": [{"id": model} for model in ("free-a", "free-b", "paid")]}
        catalog = {
            "opencode": {
                "models": {
                    "free-a": {"tool_call": True, "cost": {"input": 0, "output": 0}},
                    "free-b": {"tool_call": True, "cost": {"input": 0, "output": 0}},
                    "paid": {"tool_call": True, "cost": {"input": 1, "output": 0}},
                    "retired": {
                        "tool_call": True,
                        "status": "deprecated",
                        "cost": {"input": 0, "output": 0},
                    },
                    "no-tools": {"tool_call": False, "cost": {"input": 0, "output": 0}},
                }
            }
        }
        responses = [
            FakeResponse(body=json.dumps(zen).encode()),
            FakeResponse(body=json.dumps(catalog).encode()),
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "models.json"
            with (
                mock.patch.object(client, "_MODEL_SNAPSHOT", None),
                mock.patch.object(client, "_MODEL_SNAPSHOT_AT", 0.0),
                mock.patch.object(client, "_METADATA_SNAPSHOT", None),
                mock.patch.object(client, "_METADATA_SNAPSHOT_AT", 0.0),
                mock.patch.object(client, "_model_cache_path", return_value=cache),
                mock.patch.object(client, "_urlopen", side_effect=responses) as opened,
            ):
                oc = client.OpenCodeClient()
                self.assertEqual(oc.list_models(), ["free-a", "free-b"])
                self.assertEqual(oc.list_models(), ["free-a", "free-b"])
                self.assertEqual(json.loads(cache.read_text()), ["free-a", "free-b"])
        self.assertEqual(
            [call.args[0].full_url for call in opened.call_args_list],
            [client.ZEN_MODELS_URL, client.MODELS_DEV_URL],
        )
        self.assertTrue(
            all(
                call.args[0].headers["User-agent"] == client.OPENCODE_USER_AGENT
                for call in opened.call_args_list
            )
        )

    def test_catalog_failure_uses_last_verified_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "models.json"
            cache.write_text('["last-free"]')
            with (
                mock.patch.object(client, "_MODEL_SNAPSHOT", None),
                mock.patch.object(client, "_model_cache_path", return_value=cache),
                mock.patch.object(
                    client, "_urlopen", side_effect=urllib.error.URLError("offline")
                ),
            ):
                self.assertEqual(client.OpenCodeClient().list_models(), ["last-free"])

    def test_catalog_uses_live_free_tier_when_metadata_lags(self):
        zen = {
            "data": [
                {"id": "glm-5.3-flash-free"},
                {"id": "deepseek-v4.1-flash-free"},
                {"id": "jev-1.13-free"},
                {"id": "big-pickle"},
                {"id": "test"},
                {"id": "paid-model"},
            ]
        }
        catalog = {
            "opencode": {
                "models": {
                    "big-pickle": {"tool_call": True, "cost": {"input": 0, "output": 0}},
                    "paid-model": {"tool_call": True, "cost": {"input": 1, "output": 0}},
                }
            }
        }
        self.assertEqual(
            client._free_live_models(zen, catalog),
            ["big-pickle", "deepseek-v4.1-flash-free", "glm-5.3-flash-free", "jev-1.13-free"],
        )

    def test_catalog_excludes_deprecated_and_non_tool_free_models(self):
        zen = {
            "data": [
                {"id": "muse-spark-1.3-contributor-free"},
                {"id": "no-tools-free"},
                {"id": "ok-free"},
            ]
        }
        catalog = {
            "opencode": {
                "models": {
                    "muse-spark-1.3-contributor-free": {
                        "tool_call": True,
                        "status": "deprecated",
                        "cost": {"input": 0, "output": 0},
                    },
                    "no-tools-free": {"tool_call": False, "cost": {"input": 0, "output": 0}},
                }
            }
        }
        self.assertEqual(client._free_live_models(zen, catalog), ["ok-free"])

    def test_catalog_is_re_read_once_the_snapshot_goes_stale(self):
        """A long-lived gateway must not keep serving its start-up catalog."""
        responses = [
            FakeResponse(body=json.dumps({"data": [{"id": "first-free"}]}).encode()),
            FakeResponse(body=json.dumps({}).encode()),
            FakeResponse(body=json.dumps({"data": [{"id": "second-free"}]}).encode()),
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "models.json"
            with (
                mock.patch.object(client, "_MODEL_SNAPSHOT", None),
                mock.patch.object(client, "_MODEL_SNAPSHOT_AT", 0.0),
                mock.patch.object(client, "_METADATA_SNAPSHOT", None),
                mock.patch.object(client, "_METADATA_SNAPSHOT_AT", 0.0),
                mock.patch.object(client, "_model_cache_path", return_value=cache),
                mock.patch.object(client, "_urlopen", side_effect=responses) as opened,
            ):
                oc = client.OpenCodeClient()
                self.assertEqual(oc.list_models(), ["first-free"])
                self.assertEqual(oc.list_models(), ["first-free"])
                self.assertEqual(len(opened.call_args_list), 2)
                with mock.patch.object(client, "MODEL_SNAPSHOT_TTL", 0.0):
                    self.assertEqual(oc.list_models(), ["second-free"])
                # The stale snapshot is replaced, not appended to.
                self.assertEqual(
                    json.loads(cache.read_text()), ["second-free"]
                )
        # Only the relay's catalog is re-read; models.dev is refreshed rarely.
        self.assertEqual(
            [call.args[0].full_url for call in opened.call_args_list],
            [client.ZEN_MODELS_URL, client.MODELS_DEV_URL, client.ZEN_MODELS_URL],
        )

    def test_expired_disk_cache_is_not_served(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "models.json"
            cache.write_text('["stale-free"]')
            expired = time.time() - client.MODEL_CACHE_MAX_AGE - 60
            os.utime(cache, (expired, expired))
            with (
                mock.patch.object(client, "_MODEL_SNAPSHOT", None),
                mock.patch.object(client, "_MODEL_SNAPSHOT_AT", 0.0),
                mock.patch.object(client, "_METADATA_SNAPSHOT", None),
                mock.patch.object(client, "_METADATA_SNAPSHOT_AT", 0.0),
                mock.patch.object(client, "_model_cache_path", return_value=cache),
                mock.patch.object(
                    client, "_urlopen", side_effect=urllib.error.URLError("offline")
                ),
            ):
                self.assertEqual(
                    client.OpenCodeClient().list_models(),
                    list(client.FALLBACK_MODELS),
                )

    def test_profile_floor_follows_the_discovered_catalog(self):
        class Profile:
            fallback_models = ("retired-free",)
            model_capabilities = {"retired-free": {"supports_tools": True}}

        profile = Profile()
        discovered = client.sync_profile_catalog(profile, ["live-free", "live-free"])
        self.assertEqual(profile.fallback_models, ("live-free",))
        self.assertEqual(
            profile.model_capabilities["live-free"], {"supports_tools": True}
        )
        self.assertEqual(discovered, ("live-free",))

    def test_profile_floor_keeps_the_known_floor_when_nothing_was_discovered(self):
        class Profile:
            fallback_models = ("retired-free",)
            model_capabilities = {}

        profile = Profile()
        self.assertEqual(client.sync_profile_catalog(profile, []), ())
        self.assertEqual(profile.fallback_models, ("retired-free",))


if __name__ == "__main__":
    unittest.main()
