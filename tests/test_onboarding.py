"""Binary onboarding and request-control contracts; no hosted traffic.

Covers the approved resolution order (process PATH, then the active profile's
``plugins.entries.oc-free-provider.settings.binary_path``), the version gate,
the interactive save path, and eager refusal of unsupported request controls.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="oc-onboarding-home-", dir=str(SCRATCH))
atexit.register(shutil.rmtree, os.environ["HERMES_HOME"], ignore_errors=True)
sys.path.insert(0, str(REPO))

import oc_binary  # noqa: E402
from hermes_oc_client import ControlRefused, OpenCodeClient  # noqa: E402
from tests.runtime.helpers import installed_binary  # noqa: E402


def _script(directory: Path, name: str, version: str) -> Path:
    path = directory / name
    path.write_text(f"#!/bin/sh\necho 'opencode v{version}'\n", encoding="utf-8")
    path.chmod(0o755)
    return path


class BinaryCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="oc-binary-", dir=str(SCRATCH)))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.old_path = os.environ.get("PATH")
        config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("plugins:\n  entries: {}\n", encoding="utf-8")

    def tearDown(self) -> None:
        if self.old_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = self.old_path

    def test_path_first_then_profile_setting(self) -> None:
        known = installed_binary()
        if known is None:
            self.skipTest("installed official OpenCode binary required")
        # No opencode on PATH: the saved absolute path wins.
        os.environ["PATH"] = str(self.dir)
        oc_binary.save_path(known)
        self.assertEqual(oc_binary.resolve(), str(known))
        # A usable PATH entry comes first.
        path_bin = _script(self.dir, "opencode", "2.3.0")
        os.environ["PATH"] = str(self.dir)
        self.assertEqual(oc_binary.resolve(), str(path_bin.resolve()))

    def test_old_path_candidate_falls_back_to_valid_setting(self) -> None:
        known = installed_binary()
        if known is None:
            self.skipTest("installed official OpenCode binary required")
        _script(self.dir, "opencode", "2.2.9")
        os.environ["PATH"] = str(self.dir)
        oc_binary.save_path(known)
        self.assertEqual(oc_binary.resolve(), str(known))

    def test_stale_saved_path_and_old_hint_never_beat_a_good_path(self) -> None:
        stale = _script(self.dir, "stale-opencode", "2.0.10")
        config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "plugins:\n  entries:\n    oc-free-provider:\n      settings:\n"
            f"        binary_path: {stale}\n",
            encoding="utf-8",
        )
        good = _script(self.dir, "opencode", "2.3.0")
        os.environ["PATH"] = str(self.dir)
        self.assertEqual(oc_binary.resolve(hints=(str(stale),)), str(good.resolve()))
        # Config outranks a hint even when PATH has nothing usable.
        os.environ["PATH"] = str(self.dir / "empty")
        (self.dir / "empty").mkdir(exist_ok=True)
        configured = _script(self.dir, "configured-opencode", "2.3.0")
        config_path.write_text(
            "plugins:\n  entries:\n    oc-free-provider:\n      settings:\n"
            f"        binary_path: {configured}\n",
            encoding="utf-8",
        )
        self.assertEqual(oc_binary.resolve(hints=(str(stale),)), str(configured.resolve()))

    def test_newer_version_is_accepted_and_old_refused(self) -> None:
        newer = _script(self.dir, "newer", "2.4.1")
        self.assertEqual(oc_binary.validate(newer)[1], "2.4.1")
        old = _script(self.dir, "old", "2.2.9")
        with self.assertRaises(oc_binary.BinaryUnavailable) as refused:
            oc_binary.validate(old)
        self.assertIn("2.3.0", str(refused.exception))

    def test_no_environment_configuration_path(self) -> None:
        os.environ["PATH"] = str(self.dir)
        saved = os.environ.get("HERMES_OPENCODE_BINARY")
        os.environ["HERMES_OPENCODE_BINARY"] = "/does/not/exist"
        os.environ["OPENCODE_BIN_PATH"] = "/also/missing"
        try:
            with self.assertRaises(oc_binary.BinaryUnavailable):
                oc_binary.resolve()
        finally:
            if saved is None:
                os.environ.pop("HERMES_OPENCODE_BINARY", None)
            else:
                os.environ["HERMES_OPENCODE_BINARY"] = saved
            os.environ.pop("OPENCODE_BIN_PATH", None)

    def test_invalid_interactive_input_leaves_configuration_unchanged(self) -> None:
        known = installed_binary()
        if known is None:
            self.skipTest("installed official OpenCode binary required")
        oc_binary.save_path(known)
        messages: list[str] = []
        self.assertIsNone(oc_binary.ask_for_path(ask=lambda _prompt: "/nope/opencode",
                                                 say=messages.append, max_attempts=1))
        self.assertEqual(oc_binary.configured_path(), str(known))
        self.assertTrue(any("does not exist" in message for message in messages), messages)
        self.assertIsNone(oc_binary.ask_for_path(ask=lambda _prompt: "", say=messages.append))
        self.assertEqual(oc_binary.configured_path(), str(known))

    def test_save_preserves_unrelated_plugin_configuration(self) -> None:
        known = installed_binary()
        if known is None:
            self.skipTest("installed official OpenCode binary required")
        config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "plugins:\n  entries:\n    other-plugin:\n      settings:\n        keep: me\n",
            encoding="utf-8",
        )
        oc_binary.save_path(known)
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["plugins"]["entries"]["other-plugin"]["settings"]["keep"], "me")
        self.assertEqual(
            config["plugins"]["entries"]["oc-free-provider"]["settings"]["binary_path"], str(known)
        )


class SameProcessOnboardingCase(unittest.TestCase):
    def test_saved_path_refreshes_the_registered_launch_command(self) -> None:
        known = installed_binary()
        if known is None:
            self.skipTest("installed official OpenCode binary required")
        config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("plugins:\n  entries: {}\n", encoding="utf-8")
        empty = Path(tempfile.mkdtemp(prefix="oc-empty-path-", dir=str(SCRATCH)))
        self.addCleanup(shutil.rmtree, empty, True)
        old_path = os.environ.get("PATH")
        os.environ["PATH"] = str(empty)
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("oc_plugin_onboarding", REPO / "__init__.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            # Registration could not resolve anything yet.
            self.assertEqual(module.opencode_free.process_command, "opencode")
            oc_binary.save_path(known)
            self.assertTrue(module._auth_handler("add", None))
            self.assertEqual(module.opencode_free.process_command, str(known))
            # External config update after registration is visible immediately.
            scripts = Path(tempfile.mkdtemp(prefix="oc-switch-scripts-", dir=str(SCRATCH)))
            self.addCleanup(shutil.rmtree, scripts, True)
            second = _script(scripts, "second-opencode", "2.3.0")
            (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
                "plugins:\n  entries:\n    oc-free-provider:\n      settings:\n"
                f"        binary_path: {second}\n",
                encoding="utf-8",
            )
            self.assertEqual(module.opencode_free.process_command, str(second.resolve()))
            # Profile switch: the active home's setting wins in the same process.
            other_home = Path(tempfile.mkdtemp(prefix="oc-other-home-", dir=str(SCRATCH)))
            self.addCleanup(shutil.rmtree, other_home, True)
            third = _script(scripts, "third-opencode", "2.3.0")
            (other_home / "config.yaml").write_text(
                "plugins:\n  entries:\n    oc-free-provider:\n      settings:\n"
                f"        binary_path: {third}\n",
                encoding="utf-8",
            )
            from hermes_constants import reset_hermes_home_override, set_hermes_home_override

            token = set_hermes_home_override(str(other_home))
            try:
                self.assertEqual(module.opencode_free.process_command, str(third.resolve()))
            finally:
                reset_hermes_home_override(token)
        finally:
            if old_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = old_path


class RequestControlCase(unittest.TestCase):
    def test_stream_and_generation_controls_fail_at_create_time(self) -> None:
        client = OpenCodeClient(runtime=object())
        for kwargs in (
            {"stream": True},
            {"temperature": 0.2},
            {"max_tokens": 64},
            {"top_p": 0.9},
            {"stop": ["x"]},
            {"reasoning_effort": "high"},
            {"unknown_control": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ControlRefused):
                client.chat.completions.create(messages=[], **kwargs)

    def test_iterating_a_completion_is_a_clear_refusal(self) -> None:
        client = OpenCodeClient(runtime=object())
        lazy = client.create_lazy(messages=[])
        with self.assertRaises(ControlRefused) as refused:
            iter(lazy)
        self.assertIn("streaming is not advertised", str(refused.exception))


if __name__ == "__main__":
    unittest.main()
