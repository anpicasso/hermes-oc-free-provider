"""Security checks against the pinned runtime, without model inference."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from oc_runtime import NativeRuntime
from tests.runtime.helpers import BINARY, PRODUCTION_COMPANION, assert_process_gone


class RuntimeIsolationTests(unittest.TestCase):
    def test_credentials_project_plugins_and_unauthenticated_rpc_are_isolated(self):
        with tempfile.TemporaryDirectory(prefix="oc-isolation-") as directory:
            parent = Path(directory)
            hostile = parent / "hostile.ts"
            hostile.write_text('throw new Error("UNAUTHORIZED_PARENT_PLUGIN_LOADED");\n')
            config = {"plugins": [{"package": str(hostile)}]}
            (parent / "opencode.json").write_text(json.dumps(config))
            original = (parent / "opencode.json").read_bytes()
            runtime = NativeRuntime(binary=BINARY, companion=PRODUCTION_COMPANION,
                                    workspace=parent, startup_timeout=90)
            self.addCleanup(runtime.stop)
            sentinels = {
                "OPENAI_API_KEY": "isolation-openai-not-a-real-key",
                "OPENCODE_API_KEY": "isolation-opencode-not-a-real-key",
                "GITHUB_TOKEN": "isolation-github-not-a-real-key",
                "OPENCODE_CONFIG": str(parent / "opencode.json"),
                "OPENCODE_CONFIG_CONTENT": json.dumps(config),
                "BUN_OPTIONS": "--preload=" + str(hostile),
            }
            with patch.dict(os.environ, sentinels):
                runtime.start()
            self.assertEqual(runtime.hello()["companion"], "hermes-oc-native")
            assert runtime.process is not None
            pid = runtime.process.pid
            child_env = dict(entry.split(b"=", 1) for entry in
                             Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if entry)
            for key, value in sentinels.items():
                self.assertNotIn(key.encode(), child_env)
                self.assertNotIn(value.encode(), b"\0".join(child_env.values()))
            self.assertEqual(child_env[b"HOME"].decode(), str(runtime.home))
            self.assertNotEqual(runtime.workspace, parent)
            self.assertEqual((parent / "opencode.json").read_bytes(), original)
            self.assertNotIn("UNAUTHORIZED_PARENT_PLUGIN_LOADED", runtime.stderr_tail())

            # Disabling proxy lookup keeps this probe on the actual loopback socket.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for authorization in (None, "Basic aW52YWxpZDppbnZhbGlk"):
                headers = {"Content-Type": "application/json"}
                if authorization:
                    headers["Authorization"] = authorization
                request = urllib.request.Request(runtime._endpoint("hello"),
                                                 data=b'{"input":{}}', headers=headers)
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    opener.open(request, timeout=10)
                try:
                    self.assertIn(caught.exception.code, (401, 403))
                finally:
                    caught.exception.close()
            self.assertEqual(runtime.hello(refresh=True)["protocol"], 5)
            runtime.stop()
            assert_process_gone(pid)
            self.assertTrue(hostile.exists())
            self.assertEqual((parent / "opencode.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
