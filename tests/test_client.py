# pyright: reportArgumentType=false
# ruff: noqa: E402 -- the standalone plugin directory must be added before import.
from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hermes_oc_client as client  # type: ignore[import-not-found]


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
        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
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
        self.assertEqual(
            body["messages"][0], {"role": "system", "content": "You are opencode"}
        )
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
        with mock.patch.object(client, "_urlopen", return_value=response):
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
        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
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
        response_input, _ = client._responses_input(messages, mapped)
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
        with mock.patch.object(client, "_urlopen", return_value=response):
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
        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
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
        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
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
                    "model": "muse-spark-1.3-contributor-free",
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
                    "model": "muse-spark-1.3-contributor-free",
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
                "response": {"model": "muse-spark-1.3-contributor-free"},
            },
            {
                "type": "response.completed",
                "response": {
                    "model": "muse-spark-1.3-contributor-free",
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
        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
            first = oc.chat.completions.create(
                model="muse-spark-1.3-contributor-free",
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
                model="muse-spark-1.3-contributor-free",
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
        with mock.patch.object(client, "_urlopen", return_value=response):
            result = oc.chat.completions.create(model="m", messages=[], tools=[])
            with self.assertRaisesRegex(
                client.OpenCodeError, "compatibility-only tool 'bash'"
            ):
                _ = result.choices

    def test_http_error_exposes_status_code(self):
        error = urllib.error.HTTPError(  # type: ignore[arg-type]
            client.DIRECT_URL, 429, "rate", {}, io.BytesIO(b"busy")
        )
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_urlopen", side_effect=error):
            result = oc.chat.completions.create(model="m", messages=[], tools=[])
            with self.assertRaises(client.OpenCodeError) as caught:
                _ = result.choices
        self.assertEqual(caught.exception.status_code, 429)

    def test_empty_and_content_filtered_responses_fail_explicitly(self):
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_urlopen", return_value=sse()):
            with self.assertRaisesRegex(client.OpenCodeError, "no events"):
                _ = oc.chat.completions.create(model="m", messages=[]).choices
        filtered = sse(
            {
                "model": "m",
                "choices": [{"delta": {}, "finish_reason": "content_filter"}],
            }
        )
        with mock.patch.object(client, "_urlopen", return_value=filtered):
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

        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
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
        with mock.patch.object(client, "_urlopen") as open_request:
            result = oc.chat.completions.create(model="m", messages=[])
            with self.assertRaisesRegex(client.OpenCodeError, "client is closed"):
                _ = result.choices
        open_request.assert_not_called()

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


if __name__ == "__main__":
    unittest.main()
