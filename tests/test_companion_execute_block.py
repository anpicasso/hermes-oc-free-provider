"""Catalog-policy regressions for the official binary + production companion.

The model-facing catalog keeps only native `read`; every other native family is
removed by identity. A model that emits one anyway is refused before any
executor runs. Uses the real installed OpenCode binary with the deterministic
loopback fixture model. No hosted traffic, install, profile change or
production Python import.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
# Isolate Hermes state before any repository/core import can read it.
os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="hermes-oc-policy-home-", dir=str(SCRATCH))
atexit.register(shutil.rmtree, os.environ["HERMES_HOME"], ignore_errors=True)
sys.path.insert(0, str(REPO))

from tests.runtime import helpers  # noqa: E402
from tests.runtime.fixture import Fixture  # noqa: E402
from oc_runtime import RpcRefused  # noqa: E402

USER = [{"role": "user", "content": [{"type": "text", "text": "GO"}]}]
UNSUPPORTED = "native JavaScript execute is unsupported"
NOT_EXPOSED = "not exposed by this provider"
USE_PYTHON = "execute_code"
EXPOSED_NATIVE = {"read"}
REMOVED_NATIVE = {
    "edit", "glob", "grep", "question", "shell", "skill", "subagent",
    "webfetch", "websearch", "write", "execute", "patch",
}
EXECUTE_CODE_TOOL = {
    "name": "execute_code",
    "description": "Run a Python program in the Hermes sandbox.",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source."}},
        "required": ["code"],
        "additionalProperties": False,
    },
    "codemode": False,
}


def body_tools(record: dict) -> dict[str, dict]:
    return {
        tool["function"]["name"]: tool["function"].get("parameters")
        for tool in record["body"].get("tools", [])
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }


def collect(session, timeout: float = 1.0) -> list[dict]:
    events: list[dict] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        output = session.read(timeout=0.4)
        batch = output.get("events") or []
        events.extend(batch)
        if not batch:
            break
    return events


@unittest.skipUnless(
    helpers.BINARY.is_file() and os.access(helpers.BINARY, os.X_OK),
    "installed official OpenCode binary required",
)
class CatalogPolicyCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixtures: list[Fixture] = []
        self.runtimes = []
        self.roots: list[Path] = []

    def tearDown(self) -> None:
        for runtime in reversed(self.runtimes):
            runtime.stop()
        for fixture in reversed(self.fixtures):
            fixture.stop()
        for root in self.roots:
            shutil.rmtree(root, ignore_errors=True)

    def fixture(self, modes: list[str], marker: Path) -> Fixture:
        fixture = Fixture(modes, marker=str(marker))
        fixture.start()
        self.fixtures.append(fixture)
        return fixture

    def runtime(self, fixture: Fixture):
        root = helpers.root_for("catalog-policy")
        self.roots.append(root)
        runtime = helpers.FixtureRuntime(
            fixture.base_url,
            binary=helpers.BINARY,
            companion=helpers.COMPANION,
            root=root,
            startup_timeout=60,
            rpc_timeout=120,
            cleanup=False,
        )
        runtime.start()
        runtime.hello()
        self.runtimes.append(runtime)
        return runtime

    def test_only_native_read_is_advertised_and_hermes_schema_is_exact(self) -> None:
        marker = SCRATCH / "oc-policy-marker.txt"
        marker.unlink(missing_ok=True)
        fixture = self.fixture(["text_stop"], marker)
        runtime = self.runtime(fixture)
        session = runtime.new_session("opencode", "nonce", [helpers.READ_FILE_TOOL, EXECUTE_CODE_TOOL])
        try:
            session.generate(snapshot=USER, system_parts=[], user_text="hi", timeout=90)
            tools = body_tools(fixture.records[0])
            self.assertEqual(set(tools) & (EXPOSED_NATIVE | REMOVED_NATIVE), {"read"}, sorted(tools))
            self.assertIn("read_file", tools)
            self.assertEqual(tools["read_file"], helpers.READ_FILE_TOOL["parameters"])
            self.assertEqual(tools[USE_PYTHON], EXECUTE_CODE_TOOL["parameters"])
            for removed in REMOVED_NATIVE:
                self.assertNotIn(removed, tools, f"native {removed} was advertised")
        finally:
            session.close()

    def test_unsolicited_native_write_is_refused_without_effect(self) -> None:
        marker = SCRATCH / "oc-policy-write-marker.txt"
        marker.unlink(missing_ok=True)
        fixture = self.fixture(["tool_call"], marker)
        runtime = self.runtime(fixture)
        session = runtime.new_session("opencode", "nonce", [helpers.READ_FILE_TOOL])
        try:
            result = session.generate(snapshot=USER, system_parts=[], user_text="write", timeout=90)
            self.assertEqual([call.name for call in result.calls], ["write"])
            events = session.events_until(
                lambda event: event.get("type") in {"native_refused", "step_failed"},
                timeout=20,
            ) + collect(session)
            refusal = next(event for event in events if event.get("type") == "native_refused")
            self.assertEqual(refusal["name"], "write")
            self.assertIn(NOT_EXPOSED, refusal["reason"])
            self.assertFalse(marker.exists(), "an unadvertised native tool executed")
            self.assertEqual(fixture.request_count, 1)
            state = session.state()
            self.assertEqual(state["pending"], [])
        finally:
            session.close()

    def test_native_execute_is_refused_without_side_effect_or_extra_inference(self) -> None:
        marker = SCRATCH / "oc-policy-execute-marker.txt"
        marker.unlink(missing_ok=True)
        fixture = self.fixture(["execute"], marker)
        runtime = self.runtime(fixture)
        session = runtime.new_session("opencode", "nonce", [helpers.READ_FILE_TOOL, EXECUTE_CODE_TOOL])
        try:
            result = session.generate(snapshot=USER, system_parts=[], user_text="EXECUTE", timeout=90)
            self.assertEqual([call.name for call in result.calls], ["execute"])
            call_id = result.calls[0].call_id
            events = session.events_until(
                lambda event: event.get("type") in {"execute_refused", "step_failed"},
                timeout=20,
            ) + collect(session)
            refused = next(event for event in events if event.get("type") == "execute_refused")
            self.assertEqual(refused["callID"], call_id)
            self.assertIn(UNSUPPORTED, refused["reason"])
            self.assertIn(USE_PYTHON, refused["reason"])
            # Even an explicit allow cannot authorize the parked program.
            with self.assertRaises(RpcRefused) as refused_decision:
                session.decide(call_id, True)
            self.assertIn(UNSUPPORTED, str(refused_decision.exception))
            time.sleep(0.3)
            self.assertFalse(marker.exists(), "decide released a JavaScript program")
            self.assertEqual(fixture.request_count, 1)
        finally:
            session.close()

    def test_native_read_parks_before_any_execution(self) -> None:
        marker = SCRATCH / "oc-policy-read-marker.txt"
        marker.write_text("native read payload", encoding="utf-8")
        fixture = self.fixture(["native_read"], marker)
        runtime = self.runtime(fixture)
        session = runtime.new_session("opencode", "nonce", [helpers.READ_FILE_TOOL])
        try:
            result = session.generate(snapshot=USER, system_parts=[], user_text="read", timeout=90)
            self.assertEqual([call.name for call in result.calls], ["read"])
            call_id = result.calls[0].call_id
            state = session.wait_pending([call_id], timeout=10)
            self.assertIn(call_id, state["pending"])
            self.assertEqual(fixture.request_count, 1)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
