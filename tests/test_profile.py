"""Exercise real Hermes catalog consumers without network or active-profile state."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ProfileTests(unittest.TestCase):
    def test_real_catalog_consumers_and_profile_isolation(self):
        script = r'''
import importlib.util, os, sys, time
from pathlib import Path
from unittest.mock import patch
import oc_binary
oc_binary.save_path(os.environ["TEST_BINARY"])
import providers
providers.list_providers()
spec = importlib.util.spec_from_file_location("native_profile_test", "__init__.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
p = module.opencode_free
from hermes_oc_client import LOGICAL_BASE_URL as CLIENT_BASE_URL
assert p.base_url == CLIENT_BASE_URL, (p.base_url, CLIENT_BASE_URL)
assert providers.get_provider_profile("oc-free") is p
from hermes_cli.models import provider_model_ids, merge_profile_catalog
class Client:
    rows = []
    closed = 0
    def list_model_details(self, timeout=0): return self.rows
    def close(self): self.closed += 1
row = {"id":"dynamic-free-model", "providerID":"opencode", "enabled":True,
       "providerPackage":"@opencode/ai/providers/openai-compatible",
       "cost":[{"input":0,"output":0,"cache":{"read":0,"write":0}}]}
c = Client()
c.rows = [row]
with patch.object(p, "create_client", return_value=c):
    assert p.fetch_models() == ["dynamic-free-model"]
    assert provider_model_ids("oc-free", force_refresh=True) == ["dynamic-free-model"]
    assert merge_profile_catalog("opencode-free", p, p.fetch_models()) == ["dynamic-free-model"]
    from hermes_cli.model_setup_flows import _plugin_flow_live_rows
    assert _plugin_flow_live_rows(p, "", "")[0] == ["dynamic-free-model"]
    assert p.probe_models(base_url=p.base_url)[0] == ["dynamic-free-model"]
    # Auxiliary requests send temperature/max_tokens; control parity is
    # deferred, so this provider advertises no aux model instead of silently
    # ignoring controls.
    assert p.resolve_aux_model() == ""
    assert p.default_aux_model == ""
    from hermes_cli import models_pricing as pricing
    probed = []
    def no_probe(url, headers, timeout, opener=None):
        probed.append(url)
        return None
    with patch.object(pricing, "_get_json", no_probe):
        from agent.auxiliary_client import _get_aux_model_for_provider
        assert _get_aux_model_for_provider("opencode-free", prefer_fast=True) == ""
    assert not any(str(url).startswith(("http://", "https://")) for url in probed), probed
    assert not p.fallback_models
    c.rows = []
    assert provider_model_ids("opencode-free", force_refresh=True) == []
    assert p.fetch_models() == []
    assert p.resolve_aux_model() == ""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(os.environ["SECOND_HOME"])
    try:
        assert p.resolve_aux_model() == ""
        c.rows = [row]
        assert p.fetch_models() == ["dynamic-free-model"]
    finally:
        reset_hermes_home_override(token)
    assert p.fetch_models() == []
    p.invalidate_catalog_cache()
    with patch.object(c, "list_model_details", side_effect=RuntimeError("discovery unavailable")):
        assert p.fetch_models() is None
        assert p.resolve_aux_model() == ""
    p._native_catalog_cache = {p._catalog_key(): (time.monotonic(), [
        {"id": "bad-caps", "free_eligible": True, "supported": True, "capabilities": None},
        {"id": "vision-free", "free_eligible": True, "supported": True,
         "capabilities": {"input": ["text", "image"]}},
    ])}
    assert p.resolve_aux_model(vision=True) == ""
    p._native_catalog_cache = {p._catalog_key(): (time.monotonic(), [
        {"id": "bad-string", "free_eligible": True, "supported": True, "capabilities": {"input": "image"}},
    ])}
    assert p.resolve_aux_model(vision=True) == ""
assert c.closed >= 4
p.invalidate_catalog_cache()
# Actual auxiliary seam: the shared router returns this provider's client, and
# a normal auxiliary control request is refused loudly at create time instead
# of being silently dropped. No runtime is started because validation precedes
# transport.
from agent.auxiliary_client import resolve_provider_client
aux_client, aux_model = resolve_provider_client("opencode-free", model="dynamic-free-model")
assert aux_client is not None, "aux router did not return the provider client"
import inspect as _inspect
assert Path(_inspect.getfile(type(aux_client))).name == "hermes_oc_client.py"
try:
    aux_client.chat.completions.create(
        model="dynamic-free-model", messages=[{"role": "user", "content": "x"}],
        temperature=0.3, max_tokens=16,
    )
except Exception as error:  # noqa: BLE001 - the exact refusal type is asserted
    assert type(error).__name__ == "ControlRefused", error
else:
    raise AssertionError("auxiliary controls were silently accepted")
aux_client.close()
with patch.object(p, "create_client", return_value=c), patch.object(c, "close", side_effect=RuntimeError("cleanup failed")):
    with patch.object(c, "list_model_details", side_effect=RuntimeError("discovery unavailable")):
        assert p.fetch_models() is None
print("PROFILE_CONSUMERS_OK")
'''
        with tempfile.TemporaryDirectory(prefix="oc-profile-", dir=os.environ.get("TMPDIR")) as root:
            from tests.runtime import helpers

            binary = helpers.installed_binary()
            if binary is None:
                self.skipTest("installed official OpenCode binary required")
            env = dict(os.environ, HERMES_HOME=root, SECOND_HOME=str(Path(root) / "second"),
                       TEST_BINARY=str(binary))
            run = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                                 env=env, text=True, capture_output=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("PROFILE_CONSUMERS_OK", run.stdout)


if __name__ == "__main__":
    unittest.main()
