"""Two profile copies loaded by real Hermes provider discovery stay module-isolated.

Each temporary home gets its own installed copy under `plugins/oc-free-provider/`.
Discovery runs in a fresh process with HERMES_HOME set before startup, a context-local
home override selects the second home, and no host function is patched. Every
sibling module reachable from each registered profile must come from that profile's
own copy, and no bare top-level sibling may appear in `sys.modules`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIBLINGS = ("oc_binary", "oc_catalog", "oc_translate", "oc_runtime", "oc_watchdog",
            "oc_affinity", "hermes_oc_client")

SCRIPT = r'''
import os, sys
from pathlib import Path
import providers
from hermes_constants import set_hermes_home_override, reset_hermes_home_override

SIBLINGS = %r
seen = {}
for label, home in (("a", os.environ["HOME_A"]), ("b", os.environ["HOME_B"])):
    token = set_hermes_home_override(home)
    try:
        profile = providers.get_provider_profile("opencode-free")
        assert profile is not None, f"{label}: provider not discovered"
        package = sys.modules[type(profile).__module__]
        # Core imports from a frozen per-home snapshot of the installed copy; the
        # marker proves which copy it is, the root proves which files load.
        root = Path(package.__file__).resolve().parent
        assert package.COPY_MARKER == label, (label, root)
        assert "workspace/oc-free-implementation" not in str(root), ("checkout loaded", root)
        client = profile.create_client()  # constructor only: no runtime, no traffic
        client_file = Path(sys.modules[type(client).__module__].__file__).resolve()
        assert client_file.parent == root, (label, client_file)
        for name in SIBLINGS:
            module = sys.modules.get(f"{package.__name__}.{name}")
            assert module is not None, f"{label}: {name} not loaded under its package"
            assert Path(module.__file__).resolve().parent == root, (label, name, module.__file__)
            assert module.COPY_MARKER == label, (label, name, module.COPY_MARKER)
        client.close()
        seen[label] = (package.__name__, str(root))
    finally:
        reset_hermes_home_override(token)
assert seen["a"][0] != seen["b"][0] and seen["a"][1] != seen["b"][1], seen
leaked = sorted(name for name in SIBLINGS if name in sys.modules)
assert not leaked, f"bare sibling modules shared across profiles: {leaked}"
print("TWO_COPIES_ISOLATED", seen)
''' % (SIBLINGS,)


def _install_copy(home: Path, label: str) -> None:
    target = home / "plugins" / "oc-free-provider"
    shutil.copytree(REPO, target, ignore=shutil.ignore_patterns(
        ".git", "tests", "__pycache__", "*.pyc", "docs", ".runtime"))
    for name in ("__init__", *SIBLINGS):
        with open(target / f"{name}.py", "a", encoding="utf-8") as handle:
            handle.write(f"\nCOPY_MARKER = {label!r}\n")


class ProfileCopyIsolationTests(unittest.TestCase):
    def test_real_discovery_keeps_two_profile_copies_apart(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oc-two-copies-", dir=os.environ.get("TMPDIR")) as root:
            home_a, home_b = Path(root) / "a", Path(root) / "b"
            for home, label in ((home_a, "a"), (home_b, "b")):
                home.mkdir()
                _install_copy(home, label)
            # The checkout must not be importable: only the installed copies may load.
            core = os.environ.get("PYTHONPATH", "")
            env = dict(os.environ, HERMES_HOME=str(home_a), HOME_A=str(home_a), HOME_B=str(home_b),
                       HERMES_DISABLE_LAZY_INSTALLS="1", PYTHONPATH=core)
            run = subprocess.run([sys.executable, "-c", SCRIPT], cwd=root, env=env,
                                 text=True, capture_output=True, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("TWO_COPIES_ISOLATED", run.stdout)


if __name__ == "__main__":
    unittest.main()
