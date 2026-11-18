"""Real stock-Hermes acceptance for the OpenCode-native provider.

Every test uses the registered ``opencode-free`` profile, the real
``OpenCodeClient``, the installed official OpenCode binary, a temporary copy of
the fixture companion and a deterministic loopback model. No core patch, no
provider client injection, no hosted inference.

The model-facing catalog is native `read` plus every genuine Hermes tool the
session exposes. The model emits native `read` (translated to read_file) or
ordinary Hermes calls; Hermes executes them in its own loop and the next
request replays the canonical history with the native wire names and original
arguments. Native tools outside the catalog are refused with no side effect.
"""
from __future__ import annotations

import atexit
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for entry in (str(REPO), str(REPO / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

_SCRATCH = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
# Isolate Hermes state before any repository/core import can read it. Force a
# temp home even when the shell exports an active profile: otherwise discovery
# loads the installed copy and shadows this checkout.
os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="agent-loop-home-", dir=str(_SCRATCH))
# Per-test roots live under this home, so late background writers (session or
# log dirs recreated after tearDown) are removed with it at interpreter exit.
_MODULE_HOME = Path(os.environ["HERMES_HOME"])
atexit.register(shutil.rmtree, _MODULE_HOME, ignore_errors=True)

from run_agent import AIAgent  # noqa: E402
from runtime.fixture import Fixture  # noqa: E402
from tools.registry import registry  # noqa: E402

DEFERRED_TOOL = "hermes_agentloop_deferred"
WANTED = ["file", "terminal", "clarify"]
DISABLED = ["delegation", "web", "browser", "code_execution", "image_gen", "tts", "vision"]
EXPOSED_NATIVE = {"read"}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
        )
    return ""


def _deferred_handler(args, **_kwargs):
    target = Path(args["target"])
    target.write_text("deferred-ok", encoding="utf-8")
    return json.dumps({"deferred_written": target.stat().st_size})


class AgentFixture(Fixture):
    """Adds native read/write and direct Hermes call modes."""

    def sse(self, body, mode):
        model = body.get("model", "unknown")
        messages = body.get("messages", [])
        if any(message.get("role") == "tool" for message in messages):
            text = "fixture final: " + "; ".join(
                _text(m.get("content")) for m in messages if m.get("role") == "tool"
            )
            yield from self._text_events(model, text, usage=self._usage())
            return
        if mode == "native_write":
            yield from self._tool_events(model, "call_write_1", "write",
                                         {"path": self.marker, "content": "fixture-native"})
            return
        if mode == "native_read":
            yield from self._tool_events(model, "call_read_1", "read", {"path": self.marker})
            return
        if mode == "native_read_offset":
            yield from self._tool_events(model, "call_read_2", "read",
                                         {"path": self.marker, "offset": 2, "limit": 5})
            return
        if mode == "direct_read_file":
            yield from self._tool_events(model, "call_direct_read_1", "read_file",
                                         {"path": self.marker})
            return
        if mode == "direct_write":
            yield from self._tool_events(model, "call_direct_1", "write_file",
                                         {"path": self.marker, "content": "direct-hermes"})
            return
        if mode == "deferred_call":
            yield from self._tool_events(model, "call_deferred_1", "tool_call",
                                         {"calls": [{"name": DEFERRED_TOOL,
                                                     "arguments": {"target": self.marker}}]})
            return
        if mode == "execute":
            yield from self._execute_events(
                model,
                f"const r = await tools.write_file({{path: {json.dumps(self.marker)}, content: 'js'}}); return r",
            )
            return
        yield from super().sse(body, mode)

    def _tool_events(self, model, call_id, name, arguments):
        yield self._chunk(model, [{"index": 0, "delta": {"role": "assistant", "content": "calling"},
                                   "finish_reason": None}])
        yield self._chunk(model, [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }]}, "finish_reason": None}])
        yield self._chunk(model, [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])
        yield self._chunk(model, [], self._usage())
        yield b"data: [DONE]\n\n"


def _plugin():
    spec = importlib.util.spec_from_file_location("hermes_opencode_native", REPO / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assistant_calls(fixture):
    calls = []
    for record in fixture.records:
        for message in record["body"].get("messages", []):
            for call in message.get("tool_calls") or []:
                name = (call.get("function") or {}).get("name")
                if isinstance(name, str):
                    calls.append(name)
    return calls


def _outgoing_tools(fixture, index=0):
    return {
        tool["function"]["name"]: tool["function"]
        for tool in fixture.records[index]["body"].get("tools", [])
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }


def _tool_messages(fixture):
    return [
        message
        for record in fixture.records
        for message in record["body"].get("messages", [])
        if message.get("role") == "tool"
    ]


class AgentLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.runtime import helpers

        binary = helpers.installed_binary()
        if binary is None:
            raise unittest.SkipTest("installed official OpenCode binary required")
        # Registration captures the resolvable command, then runtime resolution
        # consumes the same profile setting (no env override).
        import oc_binary

        oc_binary.save_path(binary)
        _plugin()

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="agent-loop-", dir=str(_MODULE_HOME)))
        self.home = self.root / "home"
        self.home.mkdir()
        self.companion = self.root / "companion-root" / "tests" / "fixtures" / "agent_loop_companion"
        self.companion.parent.mkdir(parents=True)
        shutil.copytree(REPO / "tests" / "fixtures" / "agent_loop_companion", self.companion)
        (self.root / "companion-root" / "companion").symlink_to(REPO / "companion")
        self.fixture = None
        self.agent = None
        self._old_home = os.environ.get("HERMES_HOME")
        self._old_companion = os.environ.get("HERMES_OPENCODE_COMPANION")
        os.environ["HERMES_HOME"] = str(self.home)
        os.environ["HERMES_INTERACTIVE"] = "1"
        import oc_binary
        from tests.runtime import helpers

        oc_binary.save_path(helpers.installed_binary())

    def tearDown(self):
        if self.agent is not None:
            try:
                self.agent.client.close()
            except Exception:  # noqa: BLE001 - test cleanup
                pass
        if self.fixture is not None:
            self.fixture.stop()
        if self._old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self._old_home
        if self._old_companion is None:
            os.environ.pop("HERMES_OPENCODE_COMPANION", None)
        else:
            os.environ["HERMES_OPENCODE_COMPANION"] = self._old_companion
        shutil.rmtree(self.root, ignore_errors=True)

    def _agent(self, mode, *, marker="", toolsets=WANTED):
        self.fixture = AgentFixture([mode], marker=marker).start()
        (self.companion / "fixture-url.txt").write_text(self.fixture.base_url, encoding="utf-8")
        os.environ["HERMES_OPENCODE_COMPANION"] = str(self.companion)
        disabled = [name for name in WANTED + DISABLED if name not in toolsets]
        self.agent = AIAgent(
            base_url="opencode-native://local", api_key="", provider="opencode-free",
            model="nonce", max_iterations=5, enabled_toolsets=list(toolsets),
            disabled_toolsets=disabled, skip_memory=True, skip_background_review=True,
            skip_context_files=True, ephemeral_system_prompt="stock agent acceptance",
            quiet_mode=True, save_trajectories=False,
            clarify_callback=lambda qs: {
                "answers": {q["qid"]: (q["choices"][0] if q.get("choices") else "typed") for q in qs},
                "outcome": "submitted",
            },
        )
        import inspect
        self.assertEqual(Path(inspect.getfile(type(self.agent.client))).resolve(),
                         REPO / "hermes_oc_client.py", "wrong provider client loaded")
        return self.agent

    def _run(self, mode, *, marker="", toolsets=WANTED):
        agent = self._agent(mode, marker=marker, toolsets=toolsets)
        return agent.run_conversation(user_message=f"run {mode}")

    def test_native_read_really_reads_and_replays_native_name(self):
        marker = self.root / "read.txt"
        marker.write_text("hello reader\nsecond line", encoding="utf-8")
        result = self._run("native_read", marker=str(marker))
        self.assertIn("hello reader", str(result["final_response"]))
        self.assertEqual(_assistant_calls(self.fixture), ["read"])
        self.assertEqual(self.fixture.request_count, 2)
        tools = _outgoing_tools(self.fixture, 0)
        self.assertEqual({name for name in tools if name in EXPOSED_NATIVE}, {"read"})
        for forbidden in ("write", "edit", "shell", "execute", "subagent", "grep", "glob", "patch"):
            self.assertNotIn(forbidden, tools, f"native {forbidden} was advertised")
        self.assertIn("read_file", tools, "the genuine Hermes read_file schema is missing")

    def test_native_read_offset_and_limit_are_translated(self):
        marker = self.root / "offset.txt"
        marker.write_text("one\ntwo\nthree\nfour", encoding="utf-8")
        result = self._run("native_read_offset", marker=str(marker))
        self.assertIn("two", str(result["final_response"]))
        self.assertEqual(_assistant_calls(self.fixture), ["read"])

    def test_direct_hermes_read_file_is_additive_and_keeps_its_wire_name(self):
        marker = self.root / "direct-read.txt"
        marker.write_text("direct payload", encoding="utf-8")
        result = self._run("direct_read_file", marker=str(marker))
        self.assertIn("direct payload", str(result["final_response"]))
        self.assertEqual(_assistant_calls(self.fixture), ["read_file"])

    def test_direct_hermes_tool_executes_and_replays_unchanged(self):
        marker = self.root / "direct.txt"
        result = self._run("direct_write", marker=str(marker))
        self.assertEqual(marker.read_text(), "direct-hermes")
        self.assertIn('"bytes_written"', str(result["final_response"]))
        self.assertEqual(_assistant_calls(self.fixture), ["write_file"])

    def test_deferred_tool_call_is_ordinary_stock_dispatch(self):
        registry.register(
            name=DEFERRED_TOOL, toolset="file",
            schema={"name": DEFERRED_TOOL, "description": "deferred probe",
                    "parameters": {"type": "object",
                                   "properties": {"target": {"type": "string"}},
                                   "required": ["target"]}},
            handler=_deferred_handler, override=True,
        )
        try:
            marker = self.root / "deferred.txt"
            result = self._run("deferred_call", marker=str(marker))
            self.assertEqual(marker.read_text(), "deferred-ok")
            self.assertIn('"deferred_written"', str(result["final_response"]))
            self.assertEqual(_assistant_calls(self.fixture), ["tool_call"])
        finally:
            registry.deregister(DEFERRED_TOOL)

    def test_new_tool_enters_on_the_next_request(self):
        marker = self.root / "late.txt"
        marker.write_text("x", encoding="utf-8")
        self._run("text_stop", marker=str(marker))
        self.assertNotIn(DEFERRED_TOOL, _outgoing_tools(self.fixture, 0))
        # Provider boundary: the next request advertises the newly available
        # schema; a pure addition is allowed while a generation is live.
        late = {"type": "function", "function": {
            "name": DEFERRED_TOOL, "description": "late tool",
            "parameters": {"type": "object",
                           "properties": {"target": {"type": "string"}},
                           "required": ["target"]},
        }}
        completion = self.agent.client.complete(
            messages=[{"role": "user", "content": "second turn"}],
            tools=list(self.agent.tools) + [late],
            model="nonce",
        )
        self.assertIn("fixture text", completion.text)
        second = _outgoing_tools(self.fixture, 1)
        self.assertIn(DEFERRED_TOOL, second, "a newly available schema did not enter the next request")
        self.assertIn("read_file", second)
        self.assertEqual(json.dumps(second[DEFERRED_TOOL]["parameters"], sort_keys=True),
                         json.dumps(late["function"]["parameters"], sort_keys=True))

    def test_interrupt_reaches_an_in_flight_request(self):
        # Host boundary: Hermes runs the request on a watched worker and polls for
        # interrupts/stale kills only while that worker is alive. The provider work
        # must happen inside create(), not later when the main thread reads the result.
        import threading
        import time

        agent = self._agent("text_stop")
        self.fixture.delay = 8.0
        outcome = {}

        def run():
            started = time.monotonic()
            try:
                outcome["result"] = agent.run_conversation(user_message="slow request")
            except BaseException as error:  # noqa: BLE001 - recorded for the assertion
                outcome["error"] = error
            outcome["elapsed"] = time.monotonic() - started

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        deadline = time.monotonic() + 30
        while self.fixture.request_count == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.fixture.request_count, 1, "the fixture never received the request")
        interrupted_at = time.monotonic()
        agent.interrupt("stop")
        worker.join(timeout=30)
        self.assertFalse(worker.is_alive())
        waited = time.monotonic() - interrupted_at
        # The fixture holds the response for 8s; a watched request stops in ~1s.
        self.assertLess(waited, 4.0, f"interrupt waited for the full response: {outcome}")
        self.assertTrue(outcome["result"]["interrupted"], outcome)
        self.assertNotIn("fixture text", str(outcome["result"].get("final_response")))

    def test_upstream_429_surfaces_as_a_classified_rate_limit_in_the_agent_loop(self):
        # Stock AIAgent with no fallback: the structured 429 must reach the host
        # classifier as `rate_limit` (fallback-eligible), not `unknown`.
        from unittest import mock
        from agent import error_classifier, turn_api_error

        seen = []
        real = error_classifier.classify_api_error

        def spy(error, *args, **kwargs):
            verdict = real(error, *args, **kwargs)
            seen.append((getattr(error, "status_code", None), verdict.reason))
            return verdict

        agent = self._agent("http_429")
        # Observe the turn's own classifier binding; the verdict is unchanged.
        with mock.patch.object(turn_api_error, "classify_api_error", spy):
            result = agent.run_conversation(user_message="quota")
        self.assertTrue(seen, "the host classifier never saw the provider error")
        self.assertEqual(seen[0], (429, error_classifier.FailoverReason.rate_limit), seen)
        self.assertNotIn("fixture text", str(result.get("final_response")))

    def test_second_turn_replays_native_read_with_original_arguments(self):
        marker = self.root / "second.txt"
        marker.write_text("hello reader", encoding="utf-8")
        self._run("native_read", marker=str(marker))
        self.agent.run_conversation(user_message="second user turn")
        assistant = [
            call
            for record in self.fixture.records
            for message in record["body"].get("messages", [])
            if message.get("role") == "assistant"
            for call in message.get("tool_calls") or []
        ]
        names = [(call.get("function") or {}).get("name") for call in assistant]
        self.assertIn("read", names)
        self.assertNotIn("read_file", names, "the native wire name was lost on replay")
        read_arguments = [
            json.loads((call.get("function") or {}).get("arguments") or "{}")
            for call in assistant
            if (call.get("function") or {}).get("name") == "read"
        ]
        self.assertTrue(any(args.get("path") == str(marker) for args in read_arguments), read_arguments)

    def test_unsolicited_native_write_is_refused_without_effect(self):
        marker = self.root / "native-write.txt"
        result = self._run("native_write", marker=str(marker))
        self.assertFalse(marker.exists(), "an unadvertised native tool executed")
        self.assertEqual(self.fixture.request_count, 1)
        self.assertIn("not exposed", str(result["final_response"]))
        self.assertEqual(_assistant_calls(self.fixture), [])

    def test_native_execute_is_refused_without_effect(self):
        marker = self.root / "execute.txt"
        result = self._run("execute", marker=str(marker))
        self.assertFalse(marker.exists())
        self.assertEqual(self.fixture.request_count, 1)
        self.assertIn("native JavaScript execute is unsupported", str(result["final_response"]))
        self.assertEqual(_assistant_calls(self.fixture), [])

    def test_catalog_carries_every_session_tool_schema_unchanged(self):
        marker = self.root / "catalog.txt"
        marker.write_text("x", encoding="utf-8")
        self._run("text_stop", marker=str(marker))
        tools = _outgoing_tools(self.fixture, 0)
        expected = set(self.agent.valid_tool_names)
        self.assertTrue(expected, "the agent exposed no tool names")
        missing = expected - set(tools)
        # A Hermes name that collides with a native registry id is advertised
        # under its reversible wire alias, never dropped.
        for name in sorted(missing):
            self.assertIn(f"hermes_{name}", tools, f"session tool {name} was dropped instead of aliased")
        for name, function in tools.items():
            if name in EXPOSED_NATIVE:
                continue
            host = name[len("hermes_"):] if name.startswith("hermes_") and name not in expected else name
            schema = registry.get_schema(host) or {}
            canonical = schema.get("function") or schema
            if not canonical:
                continue  # host meta-tools (tool_search/tool_call) live outside the registry
            self.assertEqual(json.dumps(function["parameters"], sort_keys=True),
                             json.dumps(canonical.get("parameters"), sort_keys=True),
                             f"schema {name} was rewritten")
            self.assertEqual(function["description"], canonical.get("description", ""),
                             f"description {name} was rewritten")
        for forbidden in ("write", "edit", "shell", "execute", "subagent", "grep", "glob", "task"):
            self.assertNotIn(forbidden, tools, f"native {forbidden} was advertised")
        self.assertEqual(_tool_messages(self.fixture), [])


if __name__ == "__main__":
    unittest.main()
