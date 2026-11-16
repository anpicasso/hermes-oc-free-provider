"""Test-only helpers for the installed official binary and deterministic fixture."""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_ROOT = Path(tempfile.gettempdir())

# Packaging owns the bundled companion. Fixture tests load the test-only
# wrapper; production never registers the fixture provider.
PRODUCTION_COMPANION = REPO_ROOT / "companion"
FIXTURE_COMPANION = REPO_ROOT / "tests" / "fixtures" / "companion"
COMPANION = Path(
    os.environ.get(
        "HERMES_OPENCODE_COMPANION",
        str(FIXTURE_COMPANION if FIXTURE_COMPANION.exists() else PRODUCTION_COMPANION),
    )
)


def installed_binary() -> Path | None:
    """First usable official binary on this machine, without product settings.

    The product resolves PATH then the profile setting; tests accept a test-only
    explicit path and otherwise use PATH so no machine-specific location is
    baked into the public suite.
    """
    from oc_binary import BinaryUnavailable, validate

    candidates = []
    override = os.environ.get("HERMES_OPENCODE_BINARY")
    if override:
        candidates.append(Path(override))
    found = shutil.which("opencode")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if not str(candidate) or not candidate.is_file():
            continue
        try:
            absolute, _version = validate(candidate)
        except (BinaryUnavailable, OSError):
            continue
        return Path(absolute)
    return None


BINARY = installed_binary() or Path("opencode")

from oc_runtime import NativeRuntime  # noqa: E402

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

READ_FILE_TOOL = {
    "name": "read_file",
    "description": "Read a file from the local filesystem.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["path"],
    },
    "codemode": False,
}


def root_for(name: str) -> Path:
    path = SCRATCH_ROOT / f"oc-native-runtime-{name}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class FixtureRuntime(NativeRuntime):
    """Test-only runtime config: loopback fixture provider, no production hooks."""

    def __init__(self, fixture_url: str, **kwargs):
        super().__init__(**kwargs)
        self.plugin_options = {"fixtureURL": fixture_url}


def assert_process_gone(pid: int) -> None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise AssertionError(f"process {pid} is still alive")
