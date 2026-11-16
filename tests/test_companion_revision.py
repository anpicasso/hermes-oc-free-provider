"""Companion revision and SSE framing regressions against the official binary.

Uses the deterministic loopback fixture only; no hosted inference, install or
profile change. The fixture companion imports the production companion, so the
real parser, registry revision and live-generation guard run in the pinned
OpenCode binary.

Run from the repository root with the pinned binary:

    HERMES_OPENCODE_BINARY=/path/to/opencode \
    TMPDIR=/path/to/scratch \
    python -m unittest tests.test_companion_revision -v
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Isolate Hermes state before any repository import reads it.
_SCRATCH = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="hermes-oc-revision-home-", dir=str(_SCRATCH))
atexit.register(shutil.rmtree, os.environ["HERMES_HOME"], ignore_errors=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oc_runtime import CompletionRefused, SessionRefused  # noqa: E402
from tests.runtime import helpers  # noqa: E402
from tests.runtime.fixture import Fixture  # noqa: E402

USER_SNAPSHOT = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]

WRITE_FILE = {
    "name": "write_file",
    "description": "Write content to a file.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
    "codemode": True,
}

AUX_TOOL = {
    "name": "aux_tool",
    "description": "Auxiliary tool.",
    "parameters": {"type": "object", "properties": {}},
    "codemode": False,
}


def revised_write_file(description: str) -> dict:
    tool = dict(WRITE_FILE)
    tool["description"] = description
    tool["parameters"] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "mode": {"type": "string"},
        },
        "required": ["path", "content"],
    }
    return tool


class BurstFixture(Fixture):
    """Same SSE body under different server write sizes; optional oversize line."""

    EVENTS = 8_000

    def sse(self, body: dict, mode: str):
        if mode not in {"burst", "oversized"}:
            yield from super().sse(body, mode)
            return
        model = body.get("model", "unknown")
        if mode == "oversized":
            yield self._chunk(model, [{
                "index": 0,
                "delta": {"role": "assistant", "content": "z" * (300 * 1024)},
                "finish_reason": None,
            }])
            yield self._chunk(model, [{"index": 0, "delta": {}, "finish_reason": "stop"}], self._usage())
            yield b"data: [DONE]\n\n"
            return
        for index in range(self.EVENTS):
            delta = {"role": "assistant", "content": "x"} if index == 0 else {"content": "x"}
            yield self._chunk(model, [{"index": 0, "delta": delta, "finish_reason": None}])
        yield self._chunk(model, [{"index": 0, "delta": {}, "finish_reason": "stop"}], self._usage())
        yield b"data: [DONE]\n\n"


@unittest.skipUnless(
    helpers.BINARY.is_file() and os.access(helpers.BINARY, os.X_OK),
    "pinned OpenCode binary required (set HERMES_OPENCODE_BINARY)",
)
class CompanionRevisionCase(unittest.TestCase):
    def setUp(self) -> None:
        self._runtimes = []
        self._fixtures = []
        self._roots: list[Path] = []

    def tearDown(self) -> None:
        for runtime in reversed(self._runtimes):
            runtime.stop()
        for fixture in reversed(self._fixtures):
            fixture.stop()
        for root in self._roots:
            shutil.rmtree(root, ignore_errors=True)

    def fixture(self, modes: list[str]) -> Fixture:
        fixture = BurstFixture(modes).start()
        self._fixtures.append(fixture)
        return fixture

    def runtime(self, fixture: Fixture):
        root = helpers.root_for("revision")
        self._roots.append(root)
        kwargs: dict[str, object] = {
            "binary": helpers.BINARY,
            "companion": helpers.COMPANION,
            "root": root,
            "startup_timeout": 60,
            "rpc_timeout": 120,
            "cleanup": False,
        }
        runtime = helpers.FixtureRuntime(fixture.base_url, **kwargs)
        runtime.start()
        runtime.hello()
        self._runtimes.append(runtime)
        return runtime

    @staticmethod
    def definition(runtime, name: str) -> str:
        hello = runtime.hello(refresh=True)
        return next(tool["definition"] for tool in hello["tools"] if tool["id"] == name)

    @staticmethod
    def tool_names(runtime) -> set[str]:
        return {tool["id"] for tool in runtime.hello(refresh=True)["tools"]}

    # -- dynamic schema revisions -----------------------------------------
    def test_revision_between_finished_generations_is_accepted(self) -> None:
        runtime = self.runtime(self.fixture(["text_stop"]))
        first = runtime.new_session("opencode", "nonce", [WRITE_FILE])
        before = self.definition(runtime, "write_file")
        first.close()

        second = runtime.new_session("opencode", "nonce", [revised_write_file("Write content v2.")])
        after = self.definition(runtime, "write_file")
        self.assertNotEqual(before, after, "the registered definition must change")
        result = second.generate(
            snapshot=USER_SNAPSHOT, system_parts=[], user_text="go", timeout=90,
        )
        self.assertEqual(result.finish, "stop")
        second.close()

    def test_open_while_live_cannot_mutate_registry(self) -> None:
        runtime = self.runtime(self.fixture(["text_stop"]))
        first = runtime.new_session("opencode", "nonce", [WRITE_FILE])
        first.advance(snapshot=USER_SNAPSHOT, system_parts=[], user_text="go", timeout=60)
        before = self.definition(runtime, "write_file")
        with self.assertRaises(SessionRefused) as caught:
            runtime.new_session("opencode", "nonce", [revised_write_file("Write content v2.")])
        self.assertIn("live session limit 1", str(caught.exception))
        self.assertEqual(self.definition(runtime, "write_file"), before)
        first.close()

        second = runtime.new_session("opencode", "nonce", [revised_write_file("Write content v2.")])
        second.close()

    def test_stale_tool_removed_between_generations(self) -> None:
        runtime = self.runtime(self.fixture(["text_stop"]))
        first = runtime.new_session("opencode", "nonce", [WRITE_FILE, AUX_TOOL])
        self.assertIn("aux_tool", self.tool_names(runtime))
        first.close()

        second = runtime.new_session("opencode", "nonce", [WRITE_FILE])
        names = self.tool_names(runtime)
        self.assertNotIn("aux_tool", names)
        self.assertIn("write_file", names)
        result = second.generate(
            snapshot=USER_SNAPSHOT, system_parts=[], user_text="go", timeout=90,
        )
        self.assertEqual(result.finish, "stop")
        second.close()

    def test_stale_tool_cannot_be_removed_while_live(self) -> None:
        runtime = self.runtime(self.fixture(["text_stop"]))
        first = runtime.new_session("opencode", "nonce", [WRITE_FILE, AUX_TOOL])
        first.advance(snapshot=USER_SNAPSHOT, system_parts=[], user_text="go", timeout=60)
        with self.assertRaises(SessionRefused) as caught:
            runtime.new_session("opencode", "nonce", [WRITE_FILE])
        self.assertIn("live session limit 1", str(caught.exception))
        self.assertIn("aux_tool", self.tool_names(runtime))
        first.close()

        second = runtime.new_session("opencode", "nonce", [WRITE_FILE])
        self.assertNotIn("aux_tool", self.tool_names(runtime))
        second.close()

    # -- SSE framing --------------------------------------------------------
    def test_large_burst_is_chunking_independent(self) -> None:
        fixture = self.fixture(["burst"])
        runtime = self.runtime(fixture)
        expected = "x" * BurstFixture.EVENTS
        for sizes in ([1024 * 1024], [16 * 1024]):
            fixture.chunk_sizes = sizes
            session = runtime.new_session("opencode", "nonce")
            result = session.generate(
                snapshot=USER_SNAPSHOT, system_parts=[], user_text="go", timeout=120,
            )
            self.assertEqual(result.text, expected, f"chunk sizes {sizes[:2]}")
            self.assertEqual(result.finish, "stop")
            session.close()
        self.assertEqual(fixture.request_count, 2)

    def test_oversized_single_event_still_refused(self) -> None:
        fixture = self.fixture(["oversized"])
        fixture.chunk_sizes = [8 * 1024]
        runtime = self.runtime(fixture)
        session = runtime.new_session("opencode", "nonce")
        with self.assertRaises(CompletionRefused):
            session.generate(
                snapshot=USER_SNAPSHOT, system_parts=[], user_text="go", timeout=90,
            )
        session.close()


if __name__ == "__main__":
    unittest.main()
