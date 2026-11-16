"""Idle-owner lifecycle and dead-runtime replacement; no execute machinery."""
from __future__ import annotations

import os
import shutil
import sys
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
for entry in (str(REPO), str(REPO / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from hermes_oc_client import OpenCodeClient, RuntimeOwner, RuntimePool  # noqa: E402
from oc_runtime import RuntimeClosed  # noqa: E402
from tests.runtime import helpers  # noqa: E402
from tests.runtime.fixture import Fixture  # noqa: E402


class FakeRuntime:
    _started = False

    def __init__(self):
        self.pid = 1234
        self.alive = True
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1
        self.alive = True
        return self

    def stop(self):
        self.stopped += 1
        self.alive = False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class RuntimeRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixtures: list[Fixture] = []
        self.clients: list[OpenCodeClient] = []
        self.owners: list[RuntimeOwner] = []
        self.roots: list[Path] = []
        self._ttl = RuntimePool.TTL_SECONDS
        self._interval = RuntimePool.REAP_INTERVAL
        RuntimePool.close_all()

    def tearDown(self):
        for client in reversed(self.clients):
            try:
                client.close()
            except Exception:  # noqa: BLE001 - test cleanup
                pass
        for owner in reversed(self.owners):
            try:
                owner.close()
            except Exception:  # noqa: BLE001 - test cleanup
                pass
        RuntimePool.close_all()
        RuntimePool.TTL_SECONDS = self._ttl
        RuntimePool.REAP_INTERVAL = self._interval
        for fixture in reversed(self.fixtures):
            try:
                fixture.stop()
            except Exception:  # noqa: BLE001 - test cleanup
                pass
        for root in self.roots:
            shutil.rmtree(root, ignore_errors=True)

    def _root(self, name: str) -> Path:
        root = helpers.root_for(name)
        self.roots.append(root)
        return root

    def test_owner_replaces_a_dead_runtime(self):
        owner = RuntimeOwner(factory=FakeRuntime)
        self.owners.append(owner)
        first = owner.acquire()
        first.alive = False
        second = owner.acquire()
        self.assertIsNot(second, first)
        self.assertEqual(first.stopped, 1)
        self.assertTrue(second.alive)

    def test_isolated_client_replaces_its_own_dead_runtime_but_not_an_injected_one(self):
        old = FakeRuntime()
        old._started = True
        old.alive = False
        replacement = FakeRuntime()
        client = OpenCodeClient()
        client._runtime = old  # type: ignore[assignment]
        with mock.patch("hermes_oc_client.NativeRuntime", return_value=replacement):
            self.assertIs(client._ensure_runtime(), replacement)
        self.assertEqual(old.stopped, 1)
        self.assertTrue(replacement.alive)

        injected = FakeRuntime()
        injected._started = True
        injected.alive = False
        explicit = OpenCodeClient(runtime=injected)  # type: ignore[arg-type]
        with self.assertRaises(RuntimeClosed):
            explicit._ensure_runtime()
        self.assertEqual(injected.stopped, 0)

    def test_idle_pool_owner_reaps_its_process_without_a_followup_request(self):
        fixture = Fixture(["text_stop"]).start()
        self.fixtures.append(fixture)
        runtime = helpers.FixtureRuntime(
            fixture.base_url,
            binary=helpers.BINARY,
            companion=helpers.COMPANION,
            root=self._root("idle-reap"),
            cleanup=False,
        )
        runtime.start()
        pid = runtime.process.pid
        key = RuntimePool.key_for(f"/profiles/{uuid.uuid4().hex}:scope", "session")
        owner = RuntimePool.acquire(key)
        owner._runtime = runtime
        owner.last_used = time.monotonic()
        RuntimePool.TTL_SECONDS = 0.2
        RuntimePool.REAP_INTERVAL = 0.05

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if key not in RuntimePool._owners and not _alive(pid):
                break
            time.sleep(0.05)

        self.assertNotIn(key, RuntimePool._owners)
        self.assertFalse(runtime.alive)
        helpers.assert_process_gone(pid)

    def test_owner_keys_distinguish_profile_and_session_boundaries(self):
        self.assertNotEqual(
            RuntimePool.key_for("/profiles/a:b", "c"),
            RuntimePool.key_for("/profiles/a", "b:c"),
        )
        self.assertEqual(
            RuntimePool.key_for("/profiles/a:b", "c"),
            RuntimePool.key_for("/profiles/a:b", "c"),
        )


if __name__ == "__main__":
    unittest.main()
