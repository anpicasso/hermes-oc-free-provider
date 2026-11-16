"""Native process, completion gate and stock client tests.

Runs the official pinned binary with the deterministic local fixture. No hosted
inference, install, commit or active-profile change. The client is provider-only:
it returns translated tool calls to Hermes and never runs a program itself.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import oc_catalog  # noqa: E402
from hermes_oc_client import (  # noqa: E402
    ControlRefused,
    ModelRefused,
    OpenCodeClient,
    RuntimeOwner,
    RuntimePool,
)
from oc_runtime import (  # noqa: E402
    BinaryRefused,
    CompletionRefused,
    HandshakeRefused,
    NativeRuntime,
    SnapshotRefused,
    canonical_messages,
)
from tests.runtime import helpers  # noqa: E402
from tests.runtime.fixture import Fixture  # noqa: E402

SUPPORTED_PACKAGE = "@opencode/ai/providers/openai-compatible"
FREE_COST = [{"input": 0, "output": 0, "cache": {"read": 0, "write": 0}}]


class RuntimeCase(unittest.TestCase):
    def setUp(self) -> None:
        self._runtimes: list[NativeRuntime] = []
        self._fixtures: list[Fixture] = []
        self._clients: list[OpenCodeClient] = []
        self._roots: list[Path] = []

    def tearDown(self) -> None:
        for client in reversed(self._clients):
            try:
                client.close()
            except Exception:  # noqa: BLE001 - test cleanup
                pass
        for runtime in reversed(self._runtimes):
            try:
                runtime.stop()
            except Exception:  # noqa: BLE001 - test cleanup
                pass
        for fixture in reversed(self._fixtures):
            try:
                fixture.stop()
            except Exception:  # noqa: BLE001 - test cleanup
                pass
        for root in self._roots:
            shutil.rmtree(root, ignore_errors=True)

    def fixture(self, modes: list[str], *, marker: str | None = None) -> Fixture:
        fixture = Fixture(modes, marker=marker).start()
        self._fixtures.append(fixture)
        return fixture

    def runtime(self, fixture: Fixture, **kwargs) -> NativeRuntime:
        root = helpers.root_for(self._testMethodName)
        self._roots.append(root)
        runtime = helpers.FixtureRuntime(
            fixture.base_url,
            binary=helpers.BINARY,
            companion=helpers.COMPANION,
            root=root,
            cleanup=False,
            **kwargs,
        )
        runtime.start()
        self._runtimes.append(runtime)
        return runtime

    def client(self, runtime: NativeRuntime, **kwargs) -> OpenCodeClient:
        client = OpenCodeClient(runtime=runtime, model="nonce", **kwargs)
        self._clients.append(client)
        return client

    @staticmethod
    def marker(runtime: NativeRuntime) -> Path:
        path = Path(runtime.root) / "marker.txt"
        path.unlink(missing_ok=True)
        return path

    @staticmethod
    def native_body(fixture: Fixture, index: int) -> dict:
        return fixture.records[index]["body"]


class RuntimeTests(RuntimeCase):
    def test_private_runtime_handshake_catalog_and_ownership(self) -> None:
        fixture = self.fixture(["text_stop"])
        runtime = self.runtime(fixture)
        info = runtime.hello()
        self.assertEqual(info["protocol"], 5)
        self.assertEqual(info["companion"], "hermes-oc-native")
        self.assertEqual(info["binaryVersion"], "2.3.0")
        self.assertEqual(runtime.verify_ownership(), runtime.process.pid)
        from urllib.parse import urlparse

        self.assertIn(urlparse(runtime.url).hostname, {"127.0.0.1", "::1", "localhost"})
        catalog = runtime.catalog()
        self.assertTrue(catalog)
        self.assertIn("nonce", {row["id"] for row in catalog})
        self.assertTrue(all("settings" not in row for row in catalog))
        self.assertIn("nonce", runtime.list_models())
        pid = runtime.process.pid
        runtime.stop()
        self.assertFalse(runtime.alive)
        helpers.assert_process_gone(pid)

    def test_validated_host_hint_is_used_without_path_or_config(self) -> None:
        fixture = self.fixture(["text_stop"])
        root = helpers.root_for("binary-hint")
        self._roots.append(root)
        old_path = os.environ.get("PATH")
        os.environ["PATH"] = str(root)  # no opencode on PATH
        try:
            runtime = helpers.FixtureRuntime(
                fixture.base_url,
                binary_hints=(str(helpers.BINARY),),
                companion=helpers.COMPANION,
                root=root,
                cleanup=False,
            )
            runtime.start()
            self._runtimes.append(runtime)
            self.assertEqual(runtime.binary, helpers.BINARY)
            self.assertEqual(runtime.hello()["protocol"], 5)
        finally:
            if old_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = old_path

    def test_old_or_missing_binary_refuses_before_spawn(self) -> None:
        fake = helpers.root_for("old-binary")
        self._roots.append(fake)
        old = fake / "opencode"
        old.write_text("#!/bin/sh\necho 'opencode v2.2.9'\n", encoding="utf-8")
        old.chmod(0o755)
        with self.assertRaises(BinaryRefused) as refused:
            NativeRuntime(binary=old, companion=helpers.COMPANION, root=fake).start()
        self.assertIn("2.3.0", str(refused.exception))
        with self.assertRaises(BinaryRefused):
            NativeRuntime(binary="/bin/true", companion=helpers.COMPANION, root=fake).start()
        missing = helpers.root_for("missing-companion")
        self._roots.append(missing)
        with self.assertRaises(BinaryRefused):
            NativeRuntime(binary=helpers.BINARY, companion=missing / "nope.ts", root=missing).start()

    def test_handshake_rejects_wrong_protocol_or_version(self) -> None:
        good = {
            "protocol": 5,
            "companion": "hermes-oc-native",
            "binaryVersion": "2.3.0",
            "tools": [{"id": "read", "forwarded": True}],
        }
        NativeRuntime.validate_handshake(good)
        newer = dict(good)
        newer["binaryVersion"] = "2.4.1"
        NativeRuntime.validate_handshake(newer)
        for key, value in (("protocol", 3), ("companion", "other"), ("binaryVersion", "2.2.0")):
            with self.subTest(key=key):
                bad = dict(good)
                bad[key] = value
                with self.assertRaises(HandshakeRefused):
                    NativeRuntime.validate_handshake(bad)
        bad = dict(good)
        bad["tools"] = [{"id": "read", "forwarded": False}]
        with self.assertRaises(HandshakeRefused):
            NativeRuntime.validate_handshake(bad)

    def test_production_companion_handshake_has_no_fixture_provider(self) -> None:
        root = helpers.root_for("production-companion")
        self._roots.append(root)
        runtime = NativeRuntime(
            binary=helpers.BINARY,
            companion=helpers.PRODUCTION_COMPANION,
            root=root,
            cleanup=False,
        ).start()
        self._runtimes.append(runtime)
        self.assertEqual(runtime.hello()["protocol"], 5)
        self.assertNotIn("nonce", runtime.list_models())
        self.assertTrue(all(row.get("package") for row in runtime.catalog() if row.get("providerID") == "opencode"))

    def test_explicit_root_is_parent_only_and_preexisting_files_survive(self) -> None:
        parent = helpers.root_for("caller-root")
        self._roots.append(parent)
        agents = parent / "AGENTS.md"
        config = parent / "opencode.json"
        agents.write_text("CALLER AGENTS\n")
        config.write_text('{"caller": true}\n')
        runtime = NativeRuntime(
            binary=helpers.BINARY,
            companion=helpers.COMPANION,
            root=parent,
            workspace=parent,
            cleanup=True,
        )
        runtime._prepare()
        self.assertNotEqual(runtime.root, parent)
        self.assertEqual(runtime.root.parent, parent)
        self.assertEqual(agents.read_text(), "CALLER AGENTS\n")
        self.assertEqual(config.read_text(), '{"caller": true}\n')
        owned = runtime.root
        runtime.stop()
        self.assertFalse(owned.exists())
        self.assertTrue(parent.exists())

    def test_version_probe_uses_the_remaining_startup_budget(self) -> None:
        seen: dict[str, float] = {}
        runtime = NativeRuntime(binary=helpers.BINARY, companion=helpers.COMPANION, cleanup=True)

        def probe(path, timeout=10.0):
            seen["timeout"] = timeout
            return "2.3.0"

        runtime._version_of_binary = probe  # type: ignore[method-assign]
        try:
            runtime._prepare(time.monotonic() + 30.0)
        finally:
            runtime.stop()
        self.assertGreaterEqual(seen["timeout"], 25.0)

    def test_startup_timeout_reaps_the_owned_group(self) -> None:
        fixture = self.fixture(["text_stop"])
        root = helpers.root_for("startup-timeout")
        self._roots.append(root)
        runtime = NativeRuntime(
            binary=helpers.BINARY,
            companion=helpers.COMPANION,
            root=root,
            startup_timeout=0.01,
            cleanup=False,
        )
        with self.assertRaises(Exception) as caught:
            runtime.start()
        self.assertIn("startup", str(caught.exception).lower())
        self.assertFalse(runtime.alive)

    def test_canonical_snapshot_preserves_instructions_and_rejects_orphans(self) -> None:
        snapshot = canonical_messages(
            [
                {"role": "system", "content": [{"type": "text", "text": "SYS"}]},
                {"role": "developer", "content": "DEV"},
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "calling",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "write", "arguments": json.dumps({"path": "x", "content": "y"})}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            ]
        )
        self.assertEqual(snapshot.system_parts, ["SYS", "DEV"])
        self.assertEqual(snapshot.user_text, "hello")
        self.assertEqual([m["role"] for m in snapshot.messages], ["user", "assistant", "tool"])
        self.assertEqual(snapshot.messages[1]["content"][1]["id"], "call_1")
        self.assertEqual(snapshot.messages[2]["content"][0]["name"], "write")
        with self.assertRaises(SnapshotRefused):
            canonical_messages(
                [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]
            )
        with self.assertRaises(SnapshotRefused):
            canonical_messages([{"role": "tool", "tool_call_id": "orphan", "content": "x"}, {"role": "user", "content": "hi"}])
        with self.assertRaises(SnapshotRefused):
            canonical_messages(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "dup", "type": "function", "function": {"name": "write", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "dup", "content": "1"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "dup", "type": "function", "function": {"name": "write", "arguments": "{}"}}]},
                ]
            )


class CompletionTests(RuntimeCase):
    def test_text_completion_preserves_native_and_hermes_system_parts(self) -> None:
        fixture = self.fixture(["text_stop"])
        runtime = self.runtime(fixture)
        client = self.client(runtime)
        completion = client.complete(
            messages=[
                {"role": "system", "content": "HERMES_SYSTEM_MARKER_ONE"},
                {"role": "developer", "content": "HERMES_DEVELOPER_MARKER_TWO"},
                {"role": "user", "content": "hello"},
            ],
            tools=[helpers.READ_FILE_TOOL],
        )
        self.assertEqual(completion.finish_reason, "stop")
        self.assertEqual(completion.text, "fixture text")
        self.assertEqual(completion.choices[0].message.content, "fixture text")
        self.assertEqual(completion.usage.total_tokens, 18)
        self.assertEqual(fixture.request_count, 1)
        system = self.native_body(fixture, 0)["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("You are an AI agent running in OpenCode", system["content"])
        self.assertIn("HERMES_SYSTEM_MARKER_ONE", system["content"])
        self.assertIn("HERMES_DEVELOPER_MARKER_TWO", system["content"])

    def test_usage_absent_is_refused_before_any_completion(self) -> None:
        fixture = self.fixture(["text_no_usage"])
        client = self.client(self.runtime(fixture))
        with self.assertRaises(CompletionRefused) as refused:
            client.complete(messages=[{"role": "user", "content": "hi"}], tools=[helpers.READ_FILE_TOOL])
        self.assertIn("valid prompt/completion token usage", str(refused.exception))
        self.assertEqual(fixture.request_count, 1)

    def test_zero_usage_is_valid_not_missing(self) -> None:
        fixture = self.fixture(["text_zero_usage"])
        client = self.client(self.runtime(fixture))
        completion = client.complete(messages=[{"role": "user", "content": "hi"}], tools=[helpers.READ_FILE_TOOL])
        self.assertEqual(completion.usage.prompt_tokens, 0)
        self.assertEqual(completion.usage.completion_tokens, 0)
        self.assertEqual(completion.usage.total_tokens, 0)

    def test_native_read_is_returned_then_fresh_snapshot_continues(self) -> None:
        fixture = self.fixture(["native_read"])
        runtime = self.runtime(fixture)
        marker = self.marker(runtime)
        marker.write_text("native read payload", encoding="utf-8")
        fixture.marker = str(marker)
        client = self.client(runtime)
        first = client.complete(
            messages=[{"role": "user", "content": "read the marker"}],
            tools=[helpers.READ_FILE_TOOL],
        )
        self.assertEqual(first.finish_reason, "tool_calls")
        self.assertEqual(len(first.tool_calls), 1)
        call = first.tool_calls[0]
        self.assertTrue(call["id"].startswith("ocn1.read.v."))
        self.assertEqual(call["function"]["name"], "read_file")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": str(marker)})
        self.assertEqual(marker.read_text(), "native read payload")
        self.assertEqual(fixture.request_count, 1)

        messages = [
            {"role": "user", "content": "read the marker"},
            {"role": "assistant", "content": "reading", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "name": "read_file", "content": "native read payload"},
        ]
        second = client.complete(messages=messages, tools=[helpers.READ_FILE_TOOL])
        self.assertEqual(second.finish_reason, "stop")
        self.assertIn("fixture final", second.text)
        self.assertEqual(fixture.request_count, 2)
        body = self.native_body(fixture, 1)
        assistant = next(message for message in body["messages"] if message["role"] == "assistant")
        self.assertEqual(assistant["tool_calls"][0]["id"], call["id"])
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "read")
        tool = next(message for message in body["messages"] if message["role"] == "tool")
        self.assertEqual(tool["tool_call_id"], call["id"])

    def test_finish_reason_with_argument_fragment_is_assembled(self) -> None:
        fixture = self.fixture(["tool_call_finish_with_arguments"])
        runtime = self.runtime(fixture)
        marker = self.marker(runtime)
        marker.write_text("payload", encoding="utf-8")
        fixture.marker = str(marker)
        client = self.client(runtime)
        completion = client.complete(
            messages=[{"role": "user", "content": "read"}], tools=[helpers.READ_FILE_TOOL]
        )
        self.assertEqual(completion.finish_reason, "tool_calls")
        self.assertEqual(len(completion.tool_calls), 1)
        self.assertEqual(completion.tool_calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(completion.tool_calls[0]["function"]["arguments"]),
                         {"path": str(marker)})
        self.assertEqual(fixture.request_count, 1)

    def test_truncated_error_and_invalid_completions_are_refused(self) -> None:
        for mode in ("tool_call_no_done", "tool_call_error", "tool_call_invalid"):
            with self.subTest(mode=mode):
                fixture = self.fixture([mode])
                runtime = self.runtime(fixture)
                marker = self.marker(runtime)
                fixture.marker = str(marker)
                client = self.client(runtime)
                with self.assertRaises(CompletionRefused):
                    client.complete(messages=[{"role": "user", "content": "write"}], tools=[helpers.READ_FILE_TOOL])
                self.assertFalse(marker.exists())
                self.assertEqual(fixture.request_count, 1)

    def test_length_finish_is_terminal_without_unsolicited_request(self) -> None:
        fixture = self.fixture(["text_length"])
        client = self.client(self.runtime(fixture))
        completion = client.complete(messages=[{"role": "user", "content": "long"}], tools=[helpers.READ_FILE_TOOL])
        self.assertEqual(completion.finish_reason, "length")
        self.assertEqual(completion.text, "truncated text")
        time.sleep(0.3)
        self.assertEqual(fixture.request_count, 1)

    def test_relay_trace_header_is_local_metadata_only(self) -> None:
        fixture = self.fixture(["text_stop"])
        client = self.client(self.runtime(fixture))
        trace = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}], tools=[helpers.READ_FILE_TOOL],
            extra_headers={"Traceparent": trace})
        self.assertTrue(callable(getattr(type(completion), "model_dump", None)))
        payload = json.loads(json.dumps(completion.model_dump(mode="json", warnings=False)))
        self.assertEqual(payload["choices"][0]["message"]["content"], completion.text)
        assert completion.usage is not None
        self.assertEqual(payload["usage"]["prompt_tokens"], completion.usage.prompt_tokens)
        self.assertEqual(fixture.request_count, 1)
        self.assertNotIn(trace, json.dumps(fixture.records))
        for headers in (None, {}):
            OpenCodeClient._validate_request(extra_headers=headers)
        for headers in ({"Authorization": "secret"}, {"traceparent": trace, "X-Custom": "x"},
                        {"traceparent": 123}, [], "traceparent", {1: "x"}):
            with self.subTest(headers=headers), self.assertRaises(ControlRefused):
                client.complete(messages=[], extra_headers=headers)
        self.assertEqual(fixture.request_count, 1)

    def test_unsupported_controls_fail_before_any_request(self) -> None:
        fixture = self.fixture(["text_stop"])
        runtime = self.runtime(fixture)
        client = self.client(runtime)
        controls = [
            {"stream": True}, {"temperature": 0.5}, {"top_p": 0.5}, {"max_tokens": 10},
            {"max_completion_tokens": 10}, {"stop": ["x"]},
            {"response_format": {"type": "json_object"}}, {"reasoning_effort": "high"},
            {"tool_choice": "required"}, {"tool_choice": "none"}, {"unknown_control": 1},
        ]
        for control in controls:
            with self.subTest(control=control), self.assertRaises(ControlRefused):
                client.complete(messages=[{"role": "user", "content": "hi"}], tools=[helpers.READ_FILE_TOOL], **control)
        self.assertEqual(fixture.request_count, 0)
        compatible = OpenCodeClient(api_key="ignored-placeholder", base_url="https://ignored.invalid/v1", runtime=runtime)
        self.assertEqual(compatible.api_key, "")
        self.assertEqual(compatible.base_url, "opencode-native://local")


class ClientPolicyTests(RuntimeCase):
    def test_runtime_owner_survives_facade_reconstruction(self) -> None:
        class FakeRuntime:
            pid = 1234
            alive = True
            started = 0
            stopped = 0

            def start(self):
                self.started += 1
                self.alive = True
                return self

            def stop(self):
                self.stopped += 1
                self.alive = False

        owner = RuntimeOwner()
        fake = FakeRuntime()
        owner._runtime = fake
        first = OpenCodeClient(runtime_owner=owner)
        second = OpenCodeClient(runtime_owner=owner)
        self.assertIs(first._ensure_runtime(), fake)
        self.assertIs(second._ensure_runtime(), fake)
        first.close()
        self.assertFalse(fake.stopped, "facade close must not kill the shared owner")
        second.close()
        self.assertFalse(fake.stopped)
        owner.close()
        self.assertTrue(fake.stopped)

    def test_idle_owner_is_reaped_and_keys_are_unambiguous(self) -> None:
        class FakeRuntime:
            def stop(self):
                pass

        key = "profile-test:session-prune"
        owner = RuntimePool.acquire(key)
        owner._runtime = FakeRuntime()
        owner.last_used = 0.0
        RuntimePool.prune(now=RuntimePool.TTL_SECONDS + 1)
        self.assertNotIn(key, RuntimePool._owners)
        RuntimePool.close_all()
        self.assertNotEqual(RuntimePool.key_for("/profiles/a:b", "c"),
                            RuntimePool.key_for("/profiles/a", "b:c"))

    def test_parent_catalog_selection_and_discovery_surfaces(self) -> None:
        fixture = self.fixture(["text_stop"])
        runtime = self.runtime(fixture)
        rows = [
            {
                "id": "new-free", "modelID": "new-free", "providerID": "opencode",
                "name": "New Free", "type": "chat", "package": SUPPORTED_PACKAGE,
                "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
                "cost": FREE_COST, "limit": {"context": 1000, "output": 100},
                "status": "active", "enabled": True,
            },
            {
                "id": "paid", "modelID": "paid", "providerID": "opencode",
                "name": "Paid", "type": "chat", "package": SUPPORTED_PACKAGE,
                "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
                "cost": [{"input": 1, "output": 1, "cache": {"read": 0, "write": 0}}],
                "status": "active", "enabled": True,
            },
        ]
        runtime.catalog = lambda timeout=None: rows  # type: ignore[method-assign]
        client = self.client(runtime)
        self.assertEqual(client.list_models(), ["new-free"])
        details = oc_catalog.describe_models(client.list_model_details())
        self.assertEqual(len(details), 2)
        self.assertTrue(details[0]["free_eligible"])
        self.assertFalse(details[1]["free_eligible"])
        self.assertEqual(client.select_model("new-free")["id"], "new-free")
        with self.assertRaises(ModelRefused):
            client.select_model("paid")
        with self.assertRaises(ModelRefused):
            client.select_model("missing")

    def test_async_completion_does_not_block_the_event_loop(self) -> None:
        fixture = self.fixture(["text_stop"])
        client = self.client(self.runtime(fixture))

        def slow(**_kwargs):
            time.sleep(0.3)
            return "done"

        client.complete = slow  # type: ignore[method-assign]

        async def main():
            ticks = 0

            async def ticker():
                nonlocal ticks
                for _ in range(6):
                    await asyncio.sleep(0.05)
                    ticks += 1

            task = asyncio.create_task(ticker())
            result = await client.acomplete(messages=[{"role": "user", "content": "hi"}], tools=[helpers.READ_FILE_TOOL])
            lazy = await client.chat.completions.create(messages=[{"role": "user", "content": "hi"}], tools=[helpers.READ_FILE_TOOL])
            await task
            return ticks, result, lazy

        ticks, result, lazy = asyncio.run(main())
        self.assertEqual(result, "done")
        self.assertEqual(lazy, "done")
        self.assertGreaterEqual(ticks, 5)


if __name__ == "__main__":
    unittest.main()
