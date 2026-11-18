# ruff: noqa: E402 -- the plugin must be imported the way Hermes imports it.
"""Profile hooks through Hermes' own consumers. Needs Hermes core on sys.path
and a temporary HERMES_HOME exported before the interpreter starts."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_oc_free_profile_under_test",
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
assert _SPEC and _SPEC.loader
plugin = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = plugin
_SPEC.loader.exec_module(plugin)
client = sys.modules[f"{_SPEC.name}.hermes_oc_client"]

from agent.model_metadata import get_model_context_length  # type: ignore[import-not-found]
from agent.transports.chat_completions import ChatCompletionsTransport  # type: ignore[import-not-found]
from providers import get_provider_profile  # type: ignore[import-not-found]

# Shape of models.dev's `opencode` rows (values copied from the live catalog).
ROWS = {
            "glm-5.3-flash-free": {
                "limit": {"context": 200000, "output": 32000},
                "reasoning_options": [{"type": "effort", "values": ["low", "high", "max"]}],
            },
            "inkling-2-small-free": {
                "limit": {"context": 1048576, "output": 131072},
                "reasoning_options": [
                    {"type": "effort", "values": ["none", "minimal", "low", "medium", "high", "max"]}
                ],
            },
            "longcat-2.5-free": {
                "limit": {"context": 1000000, "output": 262144},
                "reasoning_options": [{"type": "toggle"}],
            },
            "big-pickle": {"limit": {"context": 200000}, "reasoning_options": []},
            "broken-free": {"limit": {"context": "lots"}, "reasoning_options": "effort"},
            "list-limit-free": {"limit": [1], "reasoning_options": [{"type": "effort", "values": "high"}]},
            "junk-values-free": {"reasoning_options": [{"type": "effort", "values": [None, 3, "high"]}]},
            "high-only-free": {"reasoning_options": [{"type": "effort", "values": ["high", "max"]}]},
            "off-high-free": {"reasoning_options": [{"type": "effort", "values": ["none", "high"]}]},
            "resp-free": {"reasoning_options": [{"type": "effort", "values": ["low", "high"]}]},
}


class ProfileTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(client, "_MODEL_ROWS", {str(client.get_hermes_home()): ROWS})
        patcher.start()
        self.addCleanup(patcher.stop)
        profile = get_provider_profile("opencode-free")
        assert profile is not None
        self.profile = profile

    def test_checkout_profile_is_the_registered_one(self):
        self.assertIs(self.profile, plugin.opencode_free)
        self.assertTrue(str(client.__file__).startswith(str(ROOT)))

    def test_context_length_comes_from_the_catalog_row(self):
        def host(model):
            return get_model_context_length(
                model, base_url=client.LOGICAL_BASE_URL, provider="opencode-free"
            )

        self.assertEqual(host("inkling-2-small-free"), 1048576)
        self.assertEqual(host("glm-5.3-flash-free"), 200000)
        self.assertIsNone(self.profile.get_model_context_length("broken-free"))
        self.assertIsNone(self.profile.get_model_context_length("list-limit-free"))
        self.assertIsNone(self.profile.get_model_context_length(None))  # type: ignore[arg-type]
        self.assertEqual(host("opencode/glm-5.3-flash-free"), 200000)
        self.assertIsNone(self.profile.get_model_context_length("unknown-free"))

    def _wire(self, model, reasoning_config):
        kwargs = ChatCompletionsTransport().build_kwargs(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            provider_profile=self.profile,
            reasoning_config=reasoning_config,
            base_url=client.LOGICAL_BASE_URL,
        )
        self.assertNotIn("reasoning", kwargs.get("extra_body") or {})
        return kwargs.get("reasoning_effort")

    def test_reasoning_effort_is_clamped_to_declared_levels(self):
        on = {"enabled": True, "effort": "medium"}
        self.assertEqual(self._wire("glm-5.3-flash-free", on), "low")
        self.assertEqual(self._wire("glm-5.3-flash-free", {"enabled": True, "effort": "xhigh"}), "high")
        self.assertIsNone(self._wire("glm-5.3-flash-free", {"enabled": True, "effort": "minimal"}))
        self.assertEqual(self._wire("glm-5.3-flash-free", {"enabled": True, "effort": "ultra"}), "max")
        self.assertEqual(self._wire("inkling-2-small-free", on), "medium")

    def test_effort_below_the_declared_floor_is_omitted_not_escalated(self):
        for effort in ("minimal", "low", "medium"):
            self.assertIsNone(self._wire("high-only-free", {"enabled": True, "effort": effort}))
            # "none" is declared but is thinking-off, never a stand-in for an enabled ask.
            self.assertIsNone(self._wire("off-high-free", {"enabled": True, "effort": effort}))
        self.assertEqual(self._wire("off-high-free", {"enabled": True, "effort": "none"}), "none")
        self.assertEqual(self._wire("high-only-free", {"enabled": True, "effort": "xhigh"}), "high")

    def test_malformed_rows_and_responses_models_fail_soft(self):
        on = {"enabled": True, "effort": "high"}
        self.assertIsNone(self._wire("broken-free", on))
        self.assertIsNone(self._wire("list-limit-free", on))
        self.assertEqual(self._wire("junk-values-free", on), "high")
        self.assertEqual(self._wire("opencode/resp-free", on), "high")
        with mock.patch.object(client, "RESPONSES_MODELS", frozenset({"resp-free"})):
            self.assertIsNone(self._wire("opencode/resp-free", on))
            self.assertIsNone(self._wire("resp-free", on))

    def test_catalog_rows_follow_the_active_home_in_one_module(self):
        from hermes_constants import (  # type: ignore[import-not-found]
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        client._MODEL_ROWS.clear()
        seen = {}
        for name, context in (("a", 111111), ("b", 222222), ("a", 111111)):
            home = Path(tempfile.gettempdir()) / f"oc-rows-{os.getpid()}-{name}"
            path = home / "cache" / "oc-free-provider" / "model-rows.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"m-free": {"limit": {"context": context}}}))
            token = set_hermes_home_override(home)
            try:
                seen.setdefault(name, []).append(self.profile.get_model_context_length("m-free"))
            finally:
                reset_hermes_home_override(token)
        self.assertEqual(seen, {"a": [111111, 111111], "b": [222222]})

    def test_thinking_off_is_sent_only_where_declared(self):
        off = {"enabled": False}
        self.assertEqual(self._wire("inkling-2-small-free", off), "none")
        self.assertIsNone(self._wire("glm-5.3-flash-free", off))

    def test_unknown_toggle_only_and_unset_send_nothing(self):
        on = {"enabled": True, "effort": "high"}
        self.assertIsNone(self._wire("unknown-free", on))
        self.assertIsNone(self._wire("longcat-2.5-free", on))
        self.assertIsNone(self._wire("big-pickle", on))
        self.assertIsNone(self._wire("glm-5.3-flash-free", None))


if __name__ == "__main__":
    unittest.main()
