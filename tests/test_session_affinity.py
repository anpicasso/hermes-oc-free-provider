"""Session identity and retained cache affinity through the stock Hermes seam.

Runs the official OpenCode binary with the deterministic loopback fixture. No
hosted inference, no active-profile change: every test uses an isolated
``HERMES_HOME`` set before the first core import.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

_SCRATCH = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
HOME = Path(tempfile.mkdtemp(prefix="oc-affinity-home-", dir=str(_SCRATCH)))
# Must precede every repository/core import so discovery cannot read or alter
# the active profile.
os.environ["HERMES_HOME"] = str(HOME)

REPO = Path(__file__).resolve().parents[1]
for entry in (str(REPO), str(REPO / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import oc_affinity  # noqa: E402
import oc_binary  # noqa: E402
from agent.auxiliary_client import _project_provider_profile  # noqa: E402
from agent.chat_completion_helpers import build_api_kwargs  # noqa: E402
from hermes_oc_client import (  # noqa: E402
    PROVIDER_NAME,
    ControlRefused,
    OpenCodeClient,
    RuntimePool,
)
from run_agent import AIAgent  # noqa: E402
from tests.runtime import helpers  # noqa: E402
from tests.runtime.fixture import Fixture  # noqa: E402

PLUGIN_DATA = HOME / "plugin-data" / "oc-free-provider"


def _plugin():
    spec = importlib.util.spec_from_file_location("hermes_opencode_native", REPO / "__init__.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[index]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _context(session_id: str, cache_scope_id: str | None = None) -> dict:
    return {
        "version": 2,
        "session_id": session_id,
        "cache_scope_id": session_id if cache_scope_id is None else cache_scope_id,
        "profile_home": str(HOME),
    }


def _rows() -> list[tuple]:
    db = PLUGIN_DATA / "data.db"
    if not db.exists():
        return []
    conn = sqlite3.connect(db)
    try:
        try:
            return conn.execute(
                "SELECT scope_key, version, native_session_id, provider_id, model_id, last_used "
                "FROM native_sessions"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    finally:
        conn.close()


class SeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        binary = helpers.installed_binary()
        if binary is None:
            raise unittest.SkipTest("installed official OpenCode binary required")
        oc_binary.save_path(binary)
        _plugin()

    def setUp(self) -> None:
        os.environ["HERMES_HOME"] = str(HOME)

    def test_profile_precedence_and_absent_scope(self) -> None:
        profile = _plugin().opencode_free
        self.assertEqual(profile.build_api_kwargs_extras(), ({}, {}))
        _, top = profile.build_api_kwargs_extras(session_id="session-one")
        self.assertEqual(
            top["hermes_session_context"],
            {
                "version": 2,
                "session_id": "session-one",
                "cache_scope_id": "",
                "profile_home": str(HOME),
            },
        )
        _, top = profile.build_api_kwargs_extras(session_id="rotated", cache_scope_id="lineage-root")
        self.assertEqual(top["hermes_session_context"]["session_id"], "rotated")
        self.assertEqual(top["hermes_session_context"]["cache_scope_id"], "lineage-root")
        self.assertEqual(top["hermes_session_context"]["profile_home"], str(HOME))
        self.assertEqual(profile.build_api_kwargs_extras(session_id=""), ({}, {}))

    def test_real_aiagent_builder_publishes_marker_and_aux_never_sees_it(self) -> None:
        agent = AIAgent(
            base_url="opencode-native://local", api_key="", provider="opencode-free",
            model="nonce", max_iterations=2, enabled_toolsets=["file"], disabled_toolsets=["web"],
            skip_memory=True, skip_background_review=True, skip_context_files=True,
            ephemeral_system_prompt="affinity seam", quiet_mode=True, save_trajectories=False,
        )
        try:
            agent.session_id = "seam-session-1"
            kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}], [])
            marker = kwargs.get("hermes_session_context")
            self.assertIsInstance(marker, dict)
            self.assertEqual(marker["version"], 2)
            self.assertEqual(marker["session_id"], "seam-session-1")
            self.assertEqual(marker["cache_scope_id"], "seam-session-1")
            self.assertEqual(marker["profile_home"], str(HOME))
            client = agent.client
            assert client is not None
            scope = client._session_scope(marker)
            self.assertEqual(scope["profile_home"], str(HOME))
            self.assertNotIn("hermes_session_context", kwargs.get("extra_body") or {})
            self.assertNotIn("hermes_session_context", (kwargs.get("messages") or [{}])[0])
        finally:
            agent.client.close()
        projection = _project_provider_profile(
            "opencode-free", "opencode-free", "nonce", "opencode-native://local", None, frozenset()
        )
        self.assertNotIn("hermes_session_context", projection.top_level)

    def test_profile_switch_is_refused_at_lazy_evaluation_before_runtime_or_db(self) -> None:
        profile = _plugin().opencode_free
        _, top = profile.build_api_kwargs_extras(session_id="pinned-session")
        marker = top["hermes_session_context"]
        client = OpenCodeClient(model="nonce")
        try:
            with mock.patch.object(
                client, "_ensure_runtime", side_effect=AssertionError("runtime was touched")
            ), mock.patch(
                "oc_affinity.native_root", side_effect=AssertionError("affinity db was touched")
            ):
                lazy = client.create_lazy(
                    messages=[{"role": "user", "content": "hi"}],
                    model="nonce",
                    hermes_session_context=marker,
                )
                from hermes_constants import reset_hermes_home_override, set_hermes_home_override

                token = set_hermes_home_override(str(HOME / "other-profile"))
                try:
                    with self.assertRaises(ControlRefused) as caught:
                        lazy.text
                finally:
                    reset_hermes_home_override(token)
            self.assertIn("different Hermes profile", str(caught.exception))
        finally:
            client.close()

    def test_scoped_marker_without_pinned_profile_is_refused(self) -> None:
        with self.assertRaises(ControlRefused) as caught:
            OpenCodeClient._session_scope(
                {"version": 2, "session_id": "unpinned", "cache_scope_id": "unpinned"}
            )
        self.assertIn("profile_home", str(caught.exception))

    def test_delegate_siblings_separate_while_compression_and_gateway_stay(self) -> None:
        agent = AIAgent(
            base_url="opencode-native://local", api_key="", provider="opencode-free",
            model="nonce", max_iterations=2, enabled_toolsets=["file"], disabled_toolsets=["web"],
            skip_memory=True, skip_background_review=True, skip_context_files=True,
            ephemeral_system_prompt="lane seam", quiet_mode=True, save_trajectories=False,
        )
        client = agent.client

        def scope_for(session_id: str, cache_scope_id: str) -> dict | None:
            agent.session_id = session_id
            with mock.patch(
                "agent.prompt_cache_scope.resolve_prompt_cache_scope_safe",
                return_value=cache_scope_id,
            ):
                kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}], [])
            return client._session_scope(kwargs["hermes_session_context"])

        def lane(marker: dict | None) -> str:
            self.assertIsNotNone(marker)
            return marker["lane"]

        try:
            child_a = lane(scope_for("child_a", "sub_same"))
            child_b = lane(scope_for("child_b", "sub_same"))
            self.assertNotEqual(child_a, child_b, "delegate siblings merged into one lane")
            self.assertEqual(
                lane(scope_for("comp_child", "lineage-root")),
                lane(scope_for("lineage-root", "lineage-root")),
                "compression rotation lost logical affinity",
            )
            self.assertEqual(
                lane(scope_for("resp_1", "gwk_same")),
                lane(scope_for("resp_2", "gwk_same")),
                "gateway per-response ids split one conversation",
            )
            self.assertEqual(
                lane(scope_for("parent", "parent")),
                lane(scope_for("parent", "parent")),
                "same-identity /btw or review fork must share one serialized lane",
            )
            self.assertIsNone(scope_for("", "sub_without_session"))
        finally:
            agent.client.close()

    def test_keys_separate_profile_scope_and_model(self) -> None:
        base = oc_affinity.scope_key(str(HOME), "scope", PROVIDER_NAME, "model-a")
        self.assertEqual(base, oc_affinity.scope_key(str(HOME), "scope", PROVIDER_NAME, "model-a"))
        self.assertNotEqual(base, oc_affinity.scope_key(str(HOME), "scope", PROVIDER_NAME, "model-b"))
        self.assertNotEqual(base, oc_affinity.scope_key(str(HOME), "other", PROVIDER_NAME, "model-a"))
        self.assertNotEqual(base, oc_affinity.scope_key("/other/profile", "scope", PROVIDER_NAME, "model-a"))
        self.assertEqual(len(base), 64)


class RetainedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        binary = helpers.installed_binary()
        if binary is None:
            raise unittest.SkipTest("installed official OpenCode binary required")
        oc_binary.save_path(binary)
        _plugin()

    def setUp(self) -> None:
        os.environ["HERMES_HOME"] = str(HOME)
        RuntimePool.close_all()
        self.root = Path(tempfile.mkdtemp(prefix="oc-affinity-case-", dir=str(_SCRATCH)))
        self.companion = self.root / "companion-root" / "tests" / "fixtures" / "agent_loop_companion"
        self.companion.parent.mkdir(parents=True)
        shutil.copytree(REPO / "tests" / "fixtures" / "agent_loop_companion", self.companion)
        (self.root / "companion-root" / "companion").symlink_to(REPO / "companion")
        self.fixture = None
        self.clients: list[OpenCodeClient] = []
        self.scope = f"{self._testMethodName}-{uuid.uuid4().hex}"

    def tearDown(self) -> None:
        for client in reversed(self.clients):
            try:
                client.close()
            except Exception:  # noqa: BLE001 - test cleanup
                pass
        RuntimePool.close_all()
        if self.fixture is not None:
            self.fixture.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def _fixture(self, modes: list[str]) -> Fixture:
        fixture = Fixture(modes).start()
        self.fixture = fixture
        (self.companion / "fixture-url.txt").write_text(fixture.base_url, encoding="utf-8")
        return fixture

    def _client(self) -> OpenCodeClient:
        client = OpenCodeClient(
            binary=str(helpers.BINARY), companion=str(self.companion), model="nonce", timeout=60.0
        )
        self.clients.append(client)
        return client

    def _row(self) -> tuple | None:
        key = oc_affinity.scope_key(str(HOME), self.scope, PROVIDER_NAME, "nonce")
        for row in _rows():
            if row[0] == key:
                return row
        return None

    def _native_root(self) -> Path:
        row = self._row()
        self.assertIsNotNone(row)
        return PLUGIN_DATA / "native" / row[0]

    def _headers(self) -> list[str | None]:
        return [record["headers"].get("x-opencode-session") for record in self.fixture.records]

    def test_two_turns_retain_one_native_session_and_replace_the_snapshot(self) -> None:
        self._fixture(["text_stop"])
        client = self._client()
        first = client.complete(
            messages=[
                {"role": "system", "content": "SYSTEM_MARKER"},
                {"role": "user", "content": "first turn"},
            ],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        self.assertEqual(first.finish_reason, "stop")
        self.assertFalse(first.affinity_reset)
        second = client.complete(
            messages=[
                {"role": "system", "content": "SYSTEM_MARKER"},
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": first.text},
                {"role": "user", "content": "second turn"},
            ],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        self.assertEqual(second.finish_reason, "stop")
        self.assertEqual(self.fixture.request_count, 2)
        session_id = self._row()[2]
        self.assertEqual(self._headers(), [session_id, session_id])
        first_body = self.fixture.records[0]["body"]
        second_body = self.fixture.records[1]["body"]
        self.assertEqual(first_body["messages"][0]["content"], second_body["messages"][0]["content"],
                         "the stable native system prefix changed between turns")
        first_tools = [tool["function"]["name"] for tool in first_body.get("tools", [])]
        second_tools = [tool["function"]["name"] for tool in second_body.get("tools", [])]
        self.assertEqual(first_tools, second_tools, "the tool prefix changed between unchanged turns")
        project = str(self._native_root() / "project")
        self.assertIn(project, first_body["messages"][0]["content"])
        self.assertEqual([m["role"] for m in second_body["messages"]],
                         ["system", "user", "assistant", "user"])
        self.assertEqual([m.get("content") for m in second_body["messages"] if m["role"] == "user"],
                         ["first turn", "second turn"])

    def test_tool_continuation_retains_identity_and_discards_parked_results(self) -> None:
        self._fixture(["native_read"])
        marker = self.root / "marker.txt"
        marker.write_text("native payload", encoding="utf-8")
        self.fixture.marker = str(marker)
        client = self._client()
        first = client.complete(
            messages=[{"role": "user", "content": "read the marker"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        self.assertEqual(first.finish_reason, "tool_calls")
        self.assertEqual(len(first.tool_calls), 1)
        call = first.tool_calls[0]
        second = client.complete(
            messages=[
                {"role": "user", "content": "read the marker"},
                {"role": "assistant", "content": first.text or "calling", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "name": "read_file",
                 "content": "native payload"},
            ],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        self.assertEqual(second.finish_reason, "stop")
        self.assertEqual(self.fixture.request_count, 2)
        session_id = self._row()[2]
        self.assertEqual(self._headers(), [session_id, session_id])
        second_body = self.fixture.records[1]["body"]
        self.assertEqual([m["role"] for m in second_body["messages"]],
                         ["system", "user", "assistant", "tool"])
        assistant = next(m for m in second_body["messages"] if m["role"] == "assistant")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "read")
        self.assertEqual(assistant["tool_calls"][0]["id"], call["id"])

    def test_facade_rebuild_and_runtime_eviction_reopen_the_same_session(self) -> None:
        self._fixture(["text_stop"])
        client = self._client()
        client.complete(
            messages=[{"role": "user", "content": "one"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        session_id = self._row()[2]
        client.close()
        rebuilt = self._client()
        rebuilt.complete(
            messages=[{"role": "user", "content": "two"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        self.assertEqual(self._row()[2], session_id)
        RuntimePool.close_all()  # process eviction keeps the durable native root
        self.assertTrue(self._native_root().is_dir())
        rebuilt.complete(
            messages=[{"role": "user", "content": "three"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        self.assertEqual(self._row()[2], session_id)
        self.assertEqual(self._headers(), [session_id, session_id, session_id])

    def test_missing_native_state_allocates_a_new_official_id_and_reports_reset(self) -> None:
        self._fixture(["text_stop"])
        client = self._client()
        client.complete(
            messages=[{"role": "user", "content": "one"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        old_id = self._row()[2]
        RuntimePool.close_all()
        shutil.rmtree(self._native_root())
        second = client.complete(
            messages=[{"role": "user", "content": "two"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        new_id = self._row()[2]
        self.assertTrue(second.affinity_reset)
        self.assertNotEqual(new_id, old_id)
        self.assertEqual(self._headers(), [old_id, new_id])

    def test_runtime_crash_reopens_the_same_session_from_durable_data(self) -> None:
        self._fixture(["text_stop"])
        client = self._client()
        client.complete(
            messages=[{"role": "user", "content": "one"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        session_id = self._row()[2]
        key = self._row()[0]
        owner = RuntimePool._owners[key]
        process = owner.runtime.process
        self.assertIsNotNone(process)
        process.kill()
        process.wait(timeout=30)
        second = client.complete(
            messages=[{"role": "user", "content": "after crash"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        self.assertEqual(self._row()[2], session_id)
        self.assertFalse(second.affinity_reset)
        self.assertEqual(self._headers(), [session_id, session_id])

    def test_unrecoverable_finish_allocates_a_fresh_native_session(self) -> None:
        self._fixture(["text_stop"])
        client = self._client()
        client.complete(
            messages=[{"role": "user", "content": "one"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        old_id = self._row()[2]
        owner = RuntimePool._owners[self._row()[0]]
        with mock.patch.object(owner.session, "finish", return_value={"settled": False}):
            second = client.complete(
                messages=[{"role": "user", "content": "two"}],
                tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
            )
        new_id = self._row()[2]
        self.assertTrue(second.affinity_reset)
        self.assertNotEqual(new_id, old_id)
        self.assertEqual(self._headers(), [old_id, new_id])

    def test_corrupt_mapping_is_reported_before_inference(self) -> None:
        self._fixture(["text_stop"])
        client = self._client()
        key = oc_affinity.scope_key(str(HOME), self.scope, PROVIDER_NAME, "nonce")
        self.assertIsNone(oc_affinity.load(str(HOME), key, "opencode", "nonce"))
        conn = sqlite3.connect(PLUGIN_DATA / "data.db")
        try:
            with conn:
                conn.execute(
                    "INSERT INTO native_sessions VALUES (?, 1, ?, 'opencode', 'nonce', ?)",
                    (key, "not-a-native-session", time.time()),
                )
        finally:
            conn.close()
        with self.assertRaises(oc_affinity.AffinityRefused):
            client.complete(
                messages=[{"role": "user", "content": "must not run"}],
                tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
            )
        self.assertEqual(self.fixture.request_count, 0)

    def test_cross_process_lock_refuses_before_starting_a_runtime(self) -> None:
        self._fixture(["text_stop"])
        client = self._client()
        key = oc_affinity.scope_key(str(HOME), self.scope, PROVIDER_NAME, "nonce")
        with oc_affinity.try_lock(str(HOME), key):
            with self.assertRaises(ControlRefused) as refused:
                client.complete(
                    messages=[{"role": "user", "content": "busy"}],
                    tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
                )
        self.assertIn("owned by another process", str(refused.exception))
        self.assertEqual(self.fixture.request_count, 0)

    def test_lifetime_lock_blocks_other_owner_and_prune_while_idle(self) -> None:
        self._fixture(["text_stop"])
        client = self._client()
        client.complete(
            messages=[{"role": "user", "content": "one"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        key = self._row()[0]
        with self.assertRaises(oc_affinity.AffinityBusy):
            with oc_affinity.try_lock(str(HOME), key):
                pass
        others = {row[0] for row in _rows()} - {key}
        removed = oc_affinity.prune(str(HOME), active_keys=others, max_age=0)
        self.assertNotIn(key, removed)
        self.assertTrue(self._native_root().is_dir())
        RuntimePool.close_all()
        with oc_affinity.try_lock(str(HOME), key):
            pass

    def test_prune_rechecks_recent_row_under_lock(self) -> None:
        key = oc_affinity.scope_key(str(HOME), "prune-recheck", PROVIDER_NAME, "nonce")
        native = oc_affinity.native_root(str(HOME), key)
        oc_affinity.save(str(HOME), key, "ses_" + "c" * 26, "opencode", "nonce")
        conn = sqlite3.connect(PLUGIN_DATA / "data.db")
        try:
            with conn:
                conn.execute("UPDATE native_sessions SET last_used = 0 WHERE scope_key = ?", (key,))
        finally:
            conn.close()
        others = {row[0] for row in _rows()} - {key}
        real_lock = oc_affinity.try_lock

        @contextlib.contextmanager
        def touch_then_lock(profile_home: str, candidate: str):
            touch = sqlite3.connect(PLUGIN_DATA / "data.db")
            try:
                with touch:
                    touch.execute(
                        "UPDATE native_sessions SET last_used = ? WHERE scope_key = ?",
                        (time.time(), candidate),
                    )
            finally:
                touch.close()
            with real_lock(profile_home, candidate):
                yield

        with mock.patch("oc_affinity.try_lock", touch_then_lock):
            removed = oc_affinity.prune(str(HOME), active_keys=others, max_age=1.0)
        self.assertNotIn(key, removed)
        self.assertTrue(native.exists())
        self.assertIn(key, {row[0] for row in _rows()})

    def test_same_scope_concurrent_call_is_a_bounded_busy_error(self) -> None:
        self._fixture(["text_stop"])
        self.fixture.delay = 1.2
        client = self._client()
        outcome: dict[str, BaseException | None] = {}

        def first():
            try:
                client.complete(
                    messages=[{"role": "user", "content": "first"}],
                    tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
                )
            except BaseException as error:  # noqa: BLE001 - surfaced in the assertion
                outcome["first"] = error

        thread = threading.Thread(target=first)
        thread.start()
        deadline = time.monotonic() + 30.0
        while self.fixture.request_count < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        with self.assertRaises(ControlRefused) as refused:
            client.complete(
                messages=[{"role": "user", "content": "second"}],
                tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
            )
        self.assertIn("already running", str(refused.exception))
        thread.join(timeout=60.0)
        self.assertIsNone(outcome.get("first"))

    def test_cancel_poisons_only_the_generation_and_keeps_affinity(self) -> None:
        self._fixture(["text_stop", "text_stop"])
        client = self._client()
        first = client.complete(
            messages=[{"role": "user", "content": "one"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        session_id = self._row()[2]
        self.fixture.delay = 2.0
        outcome: dict[str, BaseException | None] = {}

        def slow():
            try:
                client.complete(
                    messages=[{"role": "user", "content": "two"}],
                    tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
                )
            except BaseException as error:  # noqa: BLE001 - asserted below
                outcome["error"] = error

        thread = threading.Thread(target=slow)
        thread.start()
        deadline = time.monotonic() + 30.0
        while self.fixture.request_count < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        client.cancel()
        thread.join(timeout=30.0)
        self.assertIsNotNone(outcome.get("error"), "cancel did not abort the in-flight generation")
        self.fixture.delay = 0.0
        recovered = client.complete(
            messages=[{"role": "user", "content": "recovered"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        self.assertEqual(recovered.finish_reason, "stop")
        self.assertEqual(self._row()[2], session_id)
        self.assertEqual(self._headers()[0], session_id)
        self.assertEqual(self._headers()[-1], session_id)
        self.assertFalse(recovered.affinity_reset)

    def test_unsafe_finish_cleanup_failure_keeps_one_lease(self) -> None:
        self._fixture(["text_stop"])
        client = self._client()
        client.complete(
            messages=[{"role": "user", "content": "one"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        old_id = self._row()[2]
        owner = RuntimePool._owners[self._row()[0]]
        with mock.patch.object(owner.session, "finish", return_value={"settled": False}), \
             mock.patch.object(owner.session, "close", side_effect=RuntimeError("close failed")):
            second = client.complete(
                messages=[{"role": "user", "content": "two"}],
                tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
            )
        self.assertTrue(second.affinity_reset)
        self.assertNotEqual(self._row()[2], old_id)
        self.assertEqual(owner._refs, 0)
        self.assertIsNotNone(owner.ownership_lock)
        third = client.complete(
            messages=[{"role": "user", "content": "three"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        self.assertEqual(third.finish_reason, "stop")
        self.assertEqual(owner._refs, 0)

    def test_absent_scope_stays_ephemeral_and_creates_no_mapping(self) -> None:
        self._fixture(["text_stop"])
        client = self._client()
        before = _rows()
        client.complete(messages=[{"role": "user", "content": "one"}],
                        tools=[helpers.READ_FILE_TOOL])
        client.complete(messages=[{"role": "user", "content": "two"}],
                        tools=[helpers.READ_FILE_TOOL])
        self.assertEqual(self.fixture.request_count, 2)
        first, second = self._headers()
        self.assertTrue(first and second)
        self.assertNotEqual(first, second)
        self.assertEqual(_rows(), before)

    def test_late_hermes_tool_enters_the_retained_generation(self) -> None:
        self._fixture(["text_stop", "text_stop"])
        client = self._client()
        late = {
            "name": "late_affinity_tool",
            "description": "Late additive tool.",
            "parameters": {"type": "object", "properties": {"target": {"type": "string"}}},
        }
        first = client.complete(
            messages=[{"role": "user", "content": "one"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        client.complete(
            messages=[
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": first.text},
                {"role": "user", "content": "two"},
            ],
            tools=[helpers.READ_FILE_TOOL, late], hermes_session_context=_context(self.scope),
        )
        tools = {
            tool["function"]["name"]
            for tool in self.fixture.records[1]["body"].get("tools", [])
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }
        self.assertIn("late_affinity_tool", tools)
        # read_file is replaced by its native `read` descriptor (no duplicate).
        self.assertIn("read", tools)
        self.assertNotIn("read_file", tools)
        self.assertEqual(self._headers(), [self._row()[2], self._row()[2]])

    def test_completion_has_no_invented_wire_id_and_mapping_matches_official_shape(self) -> None:
        self._fixture(["text_stop"])
        client = self._client()
        completion = client.complete(
            messages=[{"role": "user", "content": "shape"}],
            tools=[helpers.READ_FILE_TOOL], hermes_session_context=_context(self.scope),
        )
        session_id = self._row()[2]
        self.assertRegex(session_id, r"^ses_[0-9A-Za-z]{20,40}$")
        self.assertEqual(self._headers(), [session_id])
        self.assertEqual(completion.model, "nonce")
        self.assertNotEqual(session_id, self.scope)
        self.assertNotEqual(session_id, completion.model)


class RetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["HERMES_HOME"] = str(HOME)

    def test_inactive_scope_is_removed_under_lock_and_loses_affinity(self) -> None:
        key = oc_affinity.scope_key(str(HOME), "retention-side", PROVIDER_NAME, "nonce")
        native = oc_affinity.native_root(str(HOME), key)
        (native / "project").mkdir()
        oc_affinity.save(str(HOME), key, "ses_" + "a" * 26, "opencode", "nonce")
        others = {row[0] for row in _rows()} - {key}
        conn = sqlite3.connect(PLUGIN_DATA / "data.db")
        try:
            with conn:
                conn.execute("UPDATE native_sessions SET last_used = 0 WHERE scope_key = ?", (key,))
        finally:
            conn.close()
        removed = oc_affinity.prune(str(HOME), active_keys=others, max_age=0)
        self.assertIn(key, removed)
        self.assertFalse(native.exists())
        self.assertNotIn(key, {row[0] for row in _rows()})

    def test_active_scope_is_never_pruned(self) -> None:
        key = oc_affinity.scope_key(str(HOME), "retention-active", PROVIDER_NAME, "nonce")
        native = oc_affinity.native_root(str(HOME), key)
        oc_affinity.save(str(HOME), key, "ses_" + "b" * 26, "opencode", "nonce")
        conn = sqlite3.connect(PLUGIN_DATA / "data.db")
        try:
            with conn:
                conn.execute("UPDATE native_sessions SET last_used = 0 WHERE scope_key = ?", (key,))
        finally:
            conn.close()
        removed = oc_affinity.prune(str(HOME), active_keys={key}, max_age=0)
        self.assertNotIn(key, removed)
        self.assertTrue(native.exists())


def tearDownModule() -> None:
    RuntimePool.close_all()
    shutil.rmtree(HOME, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
