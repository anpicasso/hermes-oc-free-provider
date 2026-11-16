"""Packaged companion contract tests against the official OpenCode binary.

The fixture is a deterministic local protocol instrument, not a model. It
proves the public boundary only: no hosted inference, no install, no profile or
deployment change.

The production companion must be exercised without any fixture option. The
deterministic provider is registered only by `tests/fixtures/companion/`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tests.runtime.helpers import installed_binary  # noqa: E402

BINARY = installed_binary() or Path("opencode")
PRODUCTION_COMPANION = REPO / "companion"
TEST_COMPANION = REPO / "tests" / "fixtures" / "companion"
SCRATCH = Path(tempfile.gettempdir())
SUPPORTED_PACKAGE = "@opencode/ai/providers/openai-compatible"
NATIVE_TOOLS = {
    "read", "shell", "edit", "glob", "grep", "question", "skill", "subagent",
    "webfetch", "websearch", "write",
}
WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": "Write content to a file.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
    "codemode": True,
}
DIRECT_WRITE_TOOL = {**WRITE_FILE_TOOL, "codemode": False}
READ_FILE_TOOL = {
    "name": "read_file",
    "description": "Read a file.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    "codemode": False,
}


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
        )
    return ""


class Fixture:
    """Deterministic OpenAI Chat Completions SSE fixture over loopback."""

    def __init__(self) -> None:
        self.records: list[dict] = []
        self.plan: list[dict] = []
        self.lock = threading.Lock()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                with fixture.lock:
                    fixture.records.append({"path": self.path, "body": body})
                if not self.path.endswith("/chat/completions"):
                    self.send_error(404)
                    return
                payload = b"".join(fixture.sse(body))
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "close")
                self.end_headers()
                offset = position = 0
                sizes = [1, 2, 3, 5, 8, 13, 21, 34, 55]
                while offset < len(payload):
                    size = sizes[position % len(sizes)]
                    self.wfile.write(payload[offset : offset + size])
                    self.wfile.flush()
                    offset += size
                    position += 1
                self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def request_count(self) -> int:
        return len(self.records)

    def start(self) -> "Fixture":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    # -- protocol shapes -----------------------------------------------------
    @staticmethod
    def _chunk(model: str, choices: list, usage: dict | None = None) -> bytes:
        payload = {
            "id": "chatcmpl-fixture",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

    @staticmethod
    def _usage() -> dict:
        return {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}

    def _text(self, model: str, text: str, finish: str = "stop", usage: dict | None = None,
              reasoning: str | None = None):
        if reasoning is not None:
            yield self._chunk(model, [{"index": 0, "delta": {"role": "assistant", "reasoning_content": reasoning}, "finish_reason": None}])
        yield self._chunk(model, [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}])
        yield self._chunk(model, [{"index": 0, "delta": {}, "finish_reason": finish}])
        if usage is not None:
            yield self._chunk(model, [], usage)
        yield b"data: [DONE]\n\n"

    def _tool_call(self, model: str, calls: list[dict], done: bool = True):
        for position, call in enumerate(calls):
            yield self._chunk(
                model,
                [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": position,
                                    "id": call["id"],
                                    "type": "function",
                                    "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            )
        yield self._chunk(model, [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])
        yield self._chunk(model, [], self._usage())
        if done:
            yield b"data: [DONE]\n\n"

    def sse(self, body: dict):
        model = body.get("model", "unknown")
        if any(message.get("role") == "tool" for message in body.get("messages", [])):
            results = [message for message in body["messages"] if message.get("role") == "tool"]
            yield from self._text(model, "fixture final: " + "; ".join(_content_text(m.get("content")) for m in results),
                                  usage=self._usage())
            return
        with self.lock:
            item = self.plan.pop(0) if self.plan else {"mode": "text_stop"}
        mode = item["mode"]
        if mode == "text_stop":
            yield from self._text(model, "fixture text", usage=self._usage())
            return
        if mode == "text_reasoning":
            yield from self._text(model, "fixture text", usage=self._usage(), reasoning="think-more")
            return
        if mode == "text_zero_usage":
            yield from self._text(model, "zero usage", usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            return
        if mode == "text_no_usage":
            yield from self._text(model, "no usage")
            return
        if mode == "text_bad_usage":
            yield from self._text(model, "bad usage", usage={"prompt_tokens": 3})
            return
        if mode == "text_conflicting_finish":
            yield self._chunk(model, [{"index": 0, "delta": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}])
            yield self._chunk(model, [{"index": 0, "delta": {}, "finish_reason": "length"}])
            yield self._chunk(model, [], self._usage())
            yield b"data: [DONE]\n\n"
            return
        if mode == "text_unsupported_finish":
            yield from self._text(model, "filtered", finish="content_filter", usage=self._usage())
            return
        if mode == "tool_call_no_usage":
            arguments = json.dumps(item["arguments"])
            yield self._chunk(model, [{"index": 0, "delta": {
                "role": "assistant",
                "tool_calls": [{"index": 0, "id": "call_write_1", "type": "function",
                                "function": {"name": "write", "arguments": arguments}}],
            }, "finish_reason": None}])
            yield self._chunk(model, [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])
            yield b"data: [DONE]\n\n"
            return
        if mode == "tool_call":
            yield from self._tool_call(model, [{"id": "call_write_1", "name": "write", "arguments": item["arguments"]}])
            return
        if mode == "tool_call_no_done":
            yield from self._tool_call(model, [{"id": "call_write_1", "name": "write", "arguments": item["arguments"]}], done=False)
            return
        if mode == "tool_call_error":
            for chunk in self._tool_call(model, [{"id": "call_write_1", "name": "write", "arguments": item["arguments"]}], done=False):
                yield chunk
            yield b'data: {"error": {"message": "upstream exploded", "code": 500}}\n\n'
            yield b"data: [DONE]\n\n"
            return
        if mode == "tool_call_invalid":
            yield from self._tool_call(model, [{"id": "", "name": "write", "arguments": {}}])
            return
        if mode == "execute":
            yield from self._tool_call(model, [{"id": "call_exec_1", "name": "execute", "arguments": {"code": item["code"]}}])
            return
        raise AssertionError(f"unknown fixture mode: {mode}")


class RpcFailure(RuntimeError):
    def __init__(self, method: str, status: int, body: str):
        super().__init__(f"{method} HTTP {status}: {body}")
        self.status = status
        self.body = body


class Runtime:
    """One private official OpenCode server with a chosen companion directory."""

    def __init__(self, companion: Path, *, fixture_url: str | None = None):
        self.companion = companion
        self.fixture_url = fixture_url
        self.root = SCRATCH / f"oc-companion-test-{uuid.uuid4().hex[:10]}"
        self.home = self.root / "home"
        self.project = self.root / "project"
        self.password = secrets.token_urlsafe(24)
        self.process: subprocess.Popen | None = None
        self.url: str | None = None
        self.stderr: list[str] = []

    def start(self, timeout: float = 60.0) -> "Runtime":
        for directory in (
            self.home, self.home / "config", self.home / "data", self.home / "cache",
            self.home / "state", self.project, self.root / "tmp",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        plugin: dict = {"package": str(self.companion)}
        if self.fixture_url is not None:
            plugin["options"] = {"fixtureURL": self.fixture_url}
        config = {
            "$schema": "https://opencode.ai/config.json",
            "plugins": [plugin],
            "compaction": {"auto": False},
            "snapshots": False,
        }
        (self.project / "opencode.json").write_text(json.dumps(config, indent=2))
        (self.project / "AGENTS.md").write_text("fixture project; no repository\n")
        subprocess.run(["git", "init", "-q", str(self.project)], check=False)
        env = {
            key: os.environ[key]
            for key in ("PATH", "LANG", "SSL_CERT_FILE", "SSL_CERT_DIR")
            if key in os.environ
        }
        env.update(
            HOME=str(self.home),
            XDG_CONFIG_HOME=str(self.home / "config"),
            XDG_DATA_HOME=str(self.home / "data"),
            XDG_CACHE_HOME=str(self.home / "cache"),
            XDG_STATE_HOME=str(self.home / "state"),
            TMPDIR=str(self.root / "tmp"),
            OPENCODE_PASSWORD=self.password,
            OPENCODE_PRINT_LOGS="1",
            NO_COLOR="1",
        )
        self.process = subprocess.Popen(
            [str(BINARY), "serve", "--stdio", "--port", "0", "--log-level", "warning"],
            cwd=str(self.project),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        ready: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._drain_stdout, args=(ready,), daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        line = None
        try:
            line = ready.get(timeout=timeout)
        except queue.Empty:
            self.stop()
            raise RuntimeError(f"OpenCode startup timed out: {self.stderr[-2000:]}")
        if not line:
            self.stop()
            raise RuntimeError(f"OpenCode exited before readiness: {self.stderr[-2000:]}")
        self.url = json.loads(line)["url"]
        return self

    def _drain_stdout(self, ready: queue.Queue):
        assert self.process and self.process.stdout
        first = True
        for line in self.process.stdout:
            if first:
                first = False
                ready.put(line.strip())
        ready.put(None)

    def _drain_stderr(self):
        assert self.process and self.process.stderr
        for line in self.process.stderr:
            self.stderr.append(line.rstrip())

    def rpc(self, method: str, payload: dict | None = None, timeout: float = 60.0) -> dict:
        assert self.url
        query = urllib.parse.urlencode({"location[directory]": str(self.project)})
        request = urllib.request.Request(
            f"{self.url}/api/rpc/hermes-oc-native/{method}?{query}",
            data=json.dumps({"input": payload or {}}).encode(),
            headers={
                "content-type": "application/json",
                "authorization": "Basic " + base64.b64encode(f"opencode:{self.password}".encode()).decode(),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            error.close()
            raise RpcFailure(method, error.code, body) from None

    def hello(self, timeout: float = 60.0) -> dict:
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self.rpc("hello", {}, timeout=30)["output"]
            except RpcFailure as error:
                last = error
                if "rpc.unavailable" not in error.body and "rpc.internal" not in error.body:
                    raise
                time.sleep(0.2)
        raise RuntimeError(f"companion RPC never became available: {last}")

    @property
    def alive(self) -> bool:
        return bool(self.process and self.process.poll() is None)

    def stop(self, timeout: float = 15.0) -> int:
        if not self.process:
            return 0
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        for pipe in (self.process.stdout, self.process.stderr):
            if pipe:
                pipe.close()
        return self.process.returncode or 0


class Session:
    """Event reader for one companion session."""

    def __init__(self, runtime: Runtime, session_id: str):
        self.runtime = runtime
        self.session_id = session_id
        self.cursor = 0
        self.events: list[dict] = []

    def read(self, timeout: float = 30.0) -> dict:
        output = self.runtime.rpc(
            "read", {"sessionID": self.session_id, "cursor": self.cursor}, timeout=timeout
        )["output"]
        for event in output.get("events", []):
            if event["seq"] != self.cursor + 1:
                raise AssertionError(f"event gap: expected {self.cursor + 1}, got {event['seq']}")
            self.cursor = event["seq"]
            self.events.append(event)
        return output

    def until(self, predicate, timeout: float = 30.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                output = self.read(timeout=1.0)
            except (TimeoutError, OSError):
                continue
            if any(predicate(event) for event in output.get("events", [])):
                return self.events
            if not output.get("events") and output.get("state", {}).get("closed"):
                break
        raise AssertionError(f"no matching event within {timeout}s; got {[e['type'] for e in self.events]}")

    def wait_for(self, predicate, timeout: float = 30.0) -> list[dict]:
        """Like `until`, but also matches an event already read."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(predicate(event) for event in self.events):
                return self.events
            try:
                output = self.read(timeout=1.0)
            except (TimeoutError, OSError):
                continue
            if not output.get("events") and output.get("state", {}).get("closed"):
                break
        raise AssertionError(f"no matching event within {timeout}s; got {[e['type'] for e in self.events]}")

    def types(self) -> list[str]:
        return [event["type"] for event in self.events]

    def state(self) -> dict:
        return self.runtime.rpc("state", {"sessionID": self.session_id})["output"]

    def close(self) -> dict:
        return self.runtime.rpc("close", {"sessionID": self.session_id})["output"]


class CompanionCase(unittest.TestCase):
    def setUp(self) -> None:
        self.runtimes: list[Runtime] = []
        self.fixtures: list[Fixture] = []

    def tearDown(self) -> None:
        for runtime in reversed(self.runtimes):
            if runtime.alive:
                runtime.stop()
        for fixture in reversed(self.fixtures):
            fixture.stop()

    def fixture(self) -> Fixture:
        fixture = Fixture().start()
        self.fixtures.append(fixture)
        return fixture

    def runtime(self, companion: Path, fixture: Fixture | None = None) -> Runtime:
        runtime = Runtime(companion, fixture_url=fixture.base_url if fixture else None)
        # Register before start so a failed handshake still gets torn down.
        self.runtimes.append(runtime)
        runtime.start()
        runtime.hello()
        return runtime

    def open(self, runtime: Runtime, tools: list[dict] | None = None) -> Session:
        payload: dict = {"providerID": "opencode", "modelID": "nonce"}
        if tools:
            payload["hermesTools"] = tools
        output = runtime.rpc("open", payload)["output"]
        return Session(runtime, output["sessionID"])

    def advance(self, session: Session, generation: int = 1, text: str = "go") -> None:
        session.runtime.rpc(
            "advance",
            {
                "sessionID": session.session_id,
                "generation": generation,
                "snapshot": [{"role": "user", "content": [{"type": "text", "text": text}]}],
                "systemParts": ["HERMES_COMPANION_TEST_SYSTEM"],
                "userText": text,
            },
        )

    @staticmethod
    def body_tools(record: dict) -> dict[str, dict]:
        return {
            tool["function"]["name"]: tool["function"].get("parameters")
            for tool in record["body"].get("tools", [])
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }


class PackagingTests(CompanionCase):
    def test_idle_read_returns_after_one_bounded_wait(self) -> None:
        fixture = self.fixture()
        runtime = self.runtime(TEST_COMPANION, fixture)
        session = self.open(runtime)
        session.read()  # drain the opened event
        started = time.monotonic()
        result = session.read(timeout=3.0)
        self.assertEqual(result["events"], [])
        self.assertLess(time.monotonic() - started, 2.5)
        session.close()

    def test_production_companion_has_no_fixture_surface(self) -> None:
        source = (PRODUCTION_COMPANION / "index.ts").read_text()
        self.assertNotIn("ctx.options.fixtureURL", source)
        self.assertNotIn("parseProbe", source)
        self.assertNotIn("FIXTURE_PROVIDER", source)
        self.assertNotIn("Hermes deterministic protocol fixture", source)

    def test_handshake_and_live_catalog(self) -> None:
        runtime = self.runtime(PRODUCTION_COMPANION)
        hello = runtime.hello()
        self.assertEqual(hello["protocol"], 4)
        self.assertEqual(hello["companion"], "hermes-oc-native")
        self.assertEqual(hello["binaryVersion"], "2.3.0")
        self.assertEqual(Path(hello["location"]["directory"]).resolve(), runtime.project.resolve())
        tools = {tool["id"]: tool for tool in hello["tools"]}
        self.assertLessEqual(NATIVE_TOOLS, set(tools), f"native tools missing: {NATIVE_TOOLS - set(tools)}")
        self.assertTrue(all(tools[name]["forwarded"] is True for name in NATIVE_TOOLS))
        self.assertTrue(all(tools[name]["hermes"] is False for name in NATIVE_TOOLS))
        self.assertNotIn("hermes-fixture", json.dumps(hello))
        # Output schemas participate in the structural signature.
        null_output = hashlib.sha256(
            json.dumps(
                {"name": "", "description": "", "input": None, "output": None},
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.assertTrue(
            any(tools[name].get("outputDefinition") not in {None, null_output} for name in NATIVE_TOOLS),
            "native output definitions are not represented in the signature",
        )
        second_hello = runtime.hello()
        self.assertEqual(
            {tool["id"]: tool["outputDefinition"] for tool in second_hello["tools"]},
            {tool["id"]: tool["outputDefinition"] for tool in hello["tools"]},
            "output definition signature is not stable",
        )

        catalog = runtime.rpc("catalog")["output"]
        providers = {row["id"]: row for row in catalog["providers"]}
        models = {row["id"]: row for row in catalog["models"]}
        self.assertIn("opencode", providers)
        self.assertEqual(set(providers["opencode"]), {"id", "name", "package"})
        self.assertNotIn("hermes-fixture", providers)
        self.assertNotIn("nonce", models)
        big = models["big-pickle"]
        self.assertEqual(big["providerID"], "opencode")
        self.assertEqual(big["package"], SUPPORTED_PACKAGE)
        self.assertEqual(big["providerPackage"], SUPPORTED_PACKAGE)
        self.assertTrue(big["enabled"])
        self.assertTrue(big["capabilities"]["tools"])
        self.assertGreater(big["limit"]["context"], 0)
        self.assertIsInstance(big["cost"], list)
        self.assertTrue(big["cost"])
        for tier in big["cost"]:
            self.assertEqual(set(tier["cache"]), {"read", "write"})
            self.assertIn("input", tier)
            self.assertIn("output", tier)
        serialized = json.dumps(catalog)
        for secret in ("settings", "headers", "body", "apiKey", "baseURL", "authorization"):
            self.assertNotIn(f'"{secret}"', serialized)

    def test_test_companion_registers_fixture_only_when_selected(self) -> None:
        fixture = self.fixture()
        runtime = self.runtime(TEST_COMPANION, fixture)
        catalog = runtime.rpc("catalog")["output"]
        models = {row["id"]: row for row in catalog["models"]}
        providers = {row["id"]: row for row in catalog["providers"]}
        self.assertIn("opencode", providers)
        self.assertIn("nonce", models)
        nonce = models["nonce"]
        self.assertEqual(nonce["providerID"], "opencode")
        self.assertEqual(nonce["providerPackage"], SUPPORTED_PACKAGE)
        self.assertEqual(nonce["cost"], [{"input": 0, "output": 0, "cache": {"read": 0, "write": 0}}])
        self.assertEqual(nonce["limit"], {"context": 32000, "output": 4096})


    def test_closed_sessions_are_evicted_and_live_sessions_are_bounded(self) -> None:
        fixture = self.fixture()
        runtime = self.runtime(TEST_COMPANION, fixture)
        ids = []
        for _ in range(16):
            session = self.open(runtime)
            ids.append(session.session_id)
            session.close()
        hello = runtime.hello()
        self.assertEqual(hello["sessions"], 0, "closed sessions were retained")
        self.assertEqual(hello["closedSessions"], 16)
        closed = Session(runtime, ids[-1])
        self.assertTrue(closed.state()["closed"])
        self.assertEqual(closed.close()["skipped"], "already closed")
        with self.assertRaises(RpcFailure) as refused:
            self.advance(closed)
        self.assertIn("session closed", refused.exception.body)

        live = [self.open(runtime) for _ in range(16)]
        with self.assertRaises(RpcFailure) as capped:
            self.open(runtime)
        self.assertIn("live session limit", capped.exception.body)
        for session in live:
            session.close()
        self.assertEqual(runtime.hello()["sessions"], 0)


class RegistryTests(CompanionCase):
    def test_only_native_read_is_advertised_and_hermes_schemas_are_exact(self):
        fixture = self.fixture()
        runtime = self.runtime(TEST_COMPANION, fixture)
        fixture.plan.append({"mode": "text_stop"})
        first = self.open(runtime, [READ_FILE_TOOL])
        self.advance(first)
        first.until(lambda event: event["type"] == "step_ended")
        baseline = self.body_tools(fixture.records[0])
        self.assertEqual(set(baseline) & NATIVE_TOOLS, {"read"}, baseline)
        for removed in NATIVE_TOOLS - {"read"}:
            self.assertNotIn(removed, baseline, f"native {removed} was advertised")
        native_read = baseline["read"]
        before = {tool["id"]: tool["definition"] for tool in runtime.hello()["tools"]}
        first.close()

        fixture.plan.append({"mode": "text_stop"})
        second = self.open(runtime, [READ_FILE_TOOL, DIRECT_WRITE_TOOL])
        self.advance(second)
        second.until(lambda event: event["type"] == "step_ended")
        after_body = self.body_tools(fixture.records[1])
        self.assertEqual(after_body["read"], native_read, "native read definition changed")
        self.assertEqual(after_body["write_file"], WRITE_FILE_TOOL["parameters"])
        for removed in NATIVE_TOOLS - {"read"}:
            self.assertNotIn(removed, after_body, f"native {removed} was advertised")
        after = {tool["id"]: tool["definition"] for tool in runtime.hello()["tools"]}
        for name in NATIVE_TOOLS:
            self.assertEqual(after[name], before[name], f"native signature changed: {name}")
        second.close()

        # A later generation without the tool must not still see it.
        fixture.plan.append({"mode": "text_stop"})
        third = self.open(runtime, [READ_FILE_TOOL])
        self.advance(third)
        third.until(lambda event: event["type"] == "step_ended")
        self.assertNotIn("write_file", self.body_tools(fixture.records[2]))
        third.close()

    def test_native_name_collision_is_refused_before_traffic(self) -> None:
        fixture = self.fixture()
        runtime = self.runtime(TEST_COMPANION, fixture)
        conflicting = {
            "name": "write",
            "description": "must never replace the native write",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
        with self.assertRaises(RpcFailure) as caught:
            runtime.rpc("open", {"providerID": "opencode", "modelID": "nonce", "hermesTools": [conflicting]})
        self.assertEqual(caught.exception.status, 400)
        self.assertIn("collide", caught.exception.body)
        self.assertEqual(fixture.request_count, 0)

    def test_changed_hermes_definition_is_refused(self) -> None:
        fixture = self.fixture()
        runtime = self.runtime(TEST_COMPANION, fixture)
        first = self.open(runtime, [WRITE_FILE_TOOL])
        changed = json.loads(json.dumps(WRITE_FILE_TOOL))
        changed["parameters"]["properties"]["mode"] = {"type": "string"}
        with self.assertRaises(RpcFailure) as caught:
            runtime.rpc("open", {"providerID": "opencode", "modelID": "nonce", "hermesTools": [changed]})
        self.assertIn("definition changed", caught.exception.body)
        self.assertEqual(fixture.request_count, 0)
        first.close()


class CompletionGateTests(CompanionCase):
    def test_positive_call_is_released_only_after_validated_completion(self) -> None:
        fixture = self.fixture()
        runtime = self.runtime(TEST_COMPANION, fixture)
        marker = runtime.root / "native-marker.txt"
        marker.unlink(missing_ok=True)
        fixture.plan.append({"mode": "tool_call", "arguments": {"path": str(marker), "content": "native"}})
        session = self.open(runtime)
        self.advance(session)
        events = session.until(lambda event: event["type"] == "step_streamed")
        completion = next(event for event in events if event["type"] == "completion")
        self.assertTrue(completion["ok"], completion)
        self.assertTrue(completion["done"])
        self.assertEqual(completion["finish"], "tool_calls")
        self.assertEqual(completion["calls"], 1)
        self.assertEqual(completion["usage"], {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18})
        streamed = next(event for event in events if event["type"] == "step_streamed")
        self.assertTrue(streamed["completionValidated"])
        self.assertEqual(streamed["calls"][0]["name"], "write")
        self.assertFalse(marker.exists(), "the native executor ran before Hermes dispatch")
        self.assertEqual(fixture.request_count, 1)
        with self.assertRaises(RpcFailure) as same_session:
            runtime.rpc(
                "advance",
                {
                    "sessionID": session.session_id,
                    "generation": 2,
                    "snapshot": [{"role": "user", "content": [{"type": "text", "text": "again"}]}],
                    "systemParts": [],
                    "userText": "again",
                },
            )
        self.assertIn("fresh session", same_session.exception.body)
        session.close()

    def test_empty_snapshot_is_refused_before_prompt(self) -> None:
        fixture = self.fixture()
        runtime = self.runtime(TEST_COMPANION, fixture)
        fixture.plan.append({"mode": "text_stop"})
        session = self.open(runtime)
        with self.assertRaises(RpcFailure) as empty:
            runtime.rpc(
                "advance",
                {
                    "sessionID": session.session_id,
                    "generation": 1,
                    "snapshot": [],
                    "systemParts": [],
                    "userText": "x",
                },
            )
        self.assertIn("snapshot must not be empty", empty.exception.body)
        self.assertEqual(fixture.request_count, 0)
        # The rejected advance did not consume the generation permit.
        self.advance(session)
        session.until(lambda event: event["type"] == "step_ended")
        self.assertEqual(fixture.request_count, 1)
        session.close()

    def test_completion_contract_requires_done_finish_and_usage(self) -> None:
        for mode, needle in (
            ("text_no_usage", "valid prompt/completion token usage"),
            ("text_bad_usage", "valid prompt/completion token usage"),
            ("tool_call_no_usage", "valid prompt/completion token usage"),
            ("text_conflicting_finish", "conflicting finish_reason"),
            ("text_unsupported_finish", "unsupported finish_reason"),
        ):
            with self.subTest(mode=mode):
                fixture = self.fixture()
                runtime = self.runtime(TEST_COMPANION, fixture)
                marker = runtime.root / f"{mode}.txt"
                marker.unlink(missing_ok=True)
                fixture.plan.append({"mode": mode, "arguments": {"path": str(marker), "content": "bad"}})
                session = self.open(runtime)
                self.advance(session)
                session.wait_for(lambda event: event["type"] == "completion")
                session.wait_for(lambda event: event["type"] in {"step_streamed", "step_failed"})
                completion = next(event for event in session.events if event["type"] == "completion")
                self.assertFalse(completion["ok"], completion)
                self.assertIn(needle, completion["error"])
                streamed = next((event for event in session.events if event["type"] == "step_streamed"), None)
                if streamed is not None:
                    self.assertFalse(streamed["completionValidated"])
                    self.assertEqual(streamed["calls"], [])
                self.assertFalse(marker.exists())
                self.assertEqual(fixture.request_count, 1)
                session.close()
                runtime.stop()

    def test_zero_usage_and_native_metadata_are_accepted(self) -> None:
        for mode, expected_reasoning in (("text_zero_usage", None), ("text_reasoning", "think-more")):
            with self.subTest(mode=mode):
                fixture = self.fixture()
                runtime = self.runtime(TEST_COMPANION, fixture)
                fixture.plan.append({"mode": mode})
                session = self.open(runtime)
                self.advance(session)
                events = session.until(lambda event: event["type"] == "step_ended")
                completion = next(event for event in events if event["type"] == "completion")
                self.assertTrue(completion["ok"], completion)
                if mode == "text_zero_usage":
                    self.assertEqual(completion["usage"],
                                     {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                texts = [event for event in events if event["type"] == "text"]
                self.assertTrue(texts, "native text.ended was not forwarded")
                # Opaque metadata is optional; absent upstream metadata is
                # never invented.
                self.assertIn(texts[-1].get("native"), (None, {}, {"textField": "content"}))
                reasonings = [event for event in events if event["type"] == "reasoning"]
                if expected_reasoning is None:
                    self.assertEqual(reasonings, [])
                else:
                    self.assertEqual(reasonings[-1]["text"], expected_reasoning)
                    self.assertEqual(reasonings[-1]["native"], {"reasoningField": "reasoning_content"})
                session.close()
                runtime.stop()

    def test_truncated_error_and_invalid_responses_release_nothing(self) -> None:
        for mode, needle in (
            ("tool_call_no_done", "without [DONE]"),
            ("tool_call_error", "upstream exploded"),
            ("tool_call_invalid", "missing id or name"),
        ):
            with self.subTest(mode=mode):
                fixture = self.fixture()
                runtime = self.runtime(TEST_COMPANION, fixture)
                marker = runtime.root / f"{mode}.txt"
                marker.unlink(missing_ok=True)
                fixture.plan.append({"mode": mode, "arguments": {"path": str(marker), "content": "bad"}})
                session = self.open(runtime)
                self.advance(session)
                session.wait_for(lambda event: event["type"] == "completion")
                session.wait_for(lambda event: event["type"] in {"step_streamed", "step_failed"})
                events = session.events
                completion = next(event for event in events if event["type"] == "completion")
                self.assertFalse(completion["ok"], completion)
                self.assertIn(needle, completion["error"])
                streamed = next((event for event in events if event["type"] == "step_streamed"), None)
                if streamed is not None:
                    self.assertFalse(streamed["completionValidated"])
                    self.assertEqual(streamed["calls"], [])
                else:
                    self.assertIn("step_failed", [event["type"] for event in events])
                self.assertFalse(marker.exists())
                self.assertEqual(fixture.request_count, 1)
                session.close()
                runtime.stop()



if __name__ == "__main__":
    unittest.main(verbosity=2)
