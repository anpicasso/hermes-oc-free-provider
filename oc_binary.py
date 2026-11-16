"""Resolve the owner-managed official OpenCode executable and its onboarding.

The provider never downloads, installs or upgrades OpenCode. A usable binary
(the process ``PATH`` first, then ``plugins.entries.oc-free-provider.settings.binary_path``
in the active profile's ``config.yaml``) is validated with a bounded version
probe; only a validated absolute path is ever stored. There is no env-var
configuration path, and no shell is involved: launch is an argv list.

Version >= 2.3.0 is necessary, not proof that every later release keeps the
companion hooks. The runtime handshake still verifies the loaded companion.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PLUGIN_ID = "oc-free-provider"
SETTING_KEY = "binary_path"
MIN_VERSION = (2, 3, 0)
PROBE_TIMEOUT = 15.0
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


class BinaryUnavailable(RuntimeError):
    """No usable official binary; the message is the user-facing next step."""

    code = "binary.unavailable"


def parse_version(text: str | None) -> tuple[int, int, int] | None:
    """Semantic version from CLI output; unrelated semantic versions are ignored."""
    for line in reversed((text or "").strip().splitlines()):
        match = _VERSION_RE.search(line)
        if match is not None:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


_PLATFORM_ENV = (
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE", "SSL_CERT_DIR",
    # Windows loaders need these to start the executable at all.
    "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "SYSTEMDRIVE",
)


def _probe_env() -> dict[str, str]:
    # The probe must not inherit unrelated provider or GitHub credentials, but
    # platform loader/path variables are required for the child to start.
    return {key: os.environ[key] for key in _PLATFORM_ENV if key in os.environ}


def _probe(path: Path, timeout: float) -> subprocess.CompletedProcess:
    """Bounded probe with owned-group teardown; argv only, no shell."""
    kwargs: dict = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if flags:
            kwargs["creationflags"] = flags
    process = subprocess.Popen(
        [str(path), "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=_probe_env(), shell=False, **kwargs,
    )
    try:
        stdout, stderr = process.communicate(timeout=max(1.0, float(timeout)))
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, 9)
            else:
                process.kill()
        except (ProcessLookupError, OSError):
            pass
        process.wait(timeout=5.0)
        raise
    return subprocess.CompletedProcess(process.args, process.returncode or 0, stdout, stderr)


def probe_version(path: str | Path, *, timeout: float = PROBE_TIMEOUT) -> tuple[str, tuple[int, int, int]]:
    """Bounded ``<binary> --version`` probe; argv only, no shell."""
    binary = Path(path).expanduser()
    run = _probe(binary, timeout)
    if run.returncode != 0:
        raise BinaryUnavailable(
            f"OpenCode version probe failed for {binary} (exit {run.returncode}): {(run.stderr or '').strip()[-300:]}"
        )
    version = parse_version(run.stdout) or parse_version(run.stderr)
    if version is None:
        raise BinaryUnavailable(f"OpenCode version probe returned no version for {binary}")
    text = ".".join(str(part) for part in version)
    if version < MIN_VERSION:
        raise BinaryUnavailable(
            f"OpenCode {text} at {binary} is too old; version "
            f"{'.'.join(str(p) for p in MIN_VERSION)} or newer is required"
        )
    return text, version


def validate(path: str | Path) -> tuple[str, str]:
    """Return ``(absolute_path, version)`` or raise with an actionable message."""
    binary = Path(str(path)).expanduser()
    if not binary.is_absolute():
        raise BinaryUnavailable(f"OpenCode binary path must be absolute: {path}")
    binary = binary.resolve()
    if not binary.is_file():
        raise BinaryUnavailable(f"OpenCode binary does not exist: {binary}")
    if not os.access(binary, os.X_OK):
        raise BinaryUnavailable(f"OpenCode binary is not executable: {binary}")
    version, _parsed = probe_version(binary)
    return str(binary), version


def configured_path() -> str | None:
    """Raw ``plugins.entries.oc-free-provider.settings.binary_path`` value.

    A model-provider plugin has no ``PluginContext``, so there is no
    ``ctx.get_config`` to call here. ``load_config_readonly`` plus the private
    nested-value reader is the host's profile-scoped read path; nothing is
    fabricated, cached across profiles, or written outside the shared writer
    below.
    """
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.plugins_state import _nested_plugin_value

        value = _nested_plugin_value(
            load_config_readonly() or {}, ("plugins", "entries", PLUGIN_ID, "settings", SETTING_KEY), None
        )
    except Exception:  # noqa: BLE001 - standalone provider import / unreadable profile
        return None
    return value if isinstance(value, str) and value.strip() else None


def save_path(path: str | Path) -> str:
    """Persist a validated absolute path through the shared plugin-settings writer.

    ``save_plugin_setting`` is the same writer ``ctx.set_config`` uses for a
    standalone plugin. A model-provider plugin has no PluginContext, so the
    native writer is called directly; no context object is fabricated.
    """
    absolute, _version = validate(path)
    from hermes_cli.plugins_state import _plugin_relative_segments, save_plugin_setting

    save_plugin_setting(PLUGIN_ID, _plugin_relative_segments(SETTING_KEY), absolute)
    return absolute


def _from_config() -> str | None:
    raw = configured_path()
    if raw is None:
        return None
    try:
        absolute, _version = validate(raw)
    except (BinaryUnavailable, OSError, subprocess.SubprocessError):
        return None
    return absolute


def resolve(explicit: str | Path | None = None, *, hints: Iterable[str] = ()) -> str:
    """Usable absolute binary: explicit argument, PATH, profile setting, then hints.

    Precedence is fixed: a usable PATH entry, then the active profile setting,
    then ``hints`` (execution hints supplied by the host, for example the
    command core resolved for its external-process eligibility gate). A hint is
    validated like any other candidate and never outranks PATH or config, so a
    stale hint cannot win over a working installation.
    """
    if explicit is not None:
        absolute, _version = validate(explicit)
        return absolute
    # Precedence is fixed: usable PATH, then the active profile setting, then a
    # validated host/current-core fallback hint. A hint never outranks either.
    candidates = [shutil.which("opencode"), configured_path(), *hints]
    for raw in candidates:
        if not raw:
            continue
        try:
            absolute, _version = validate(raw)
        except (BinaryUnavailable, OSError, subprocess.SubprocessError):
            continue
        return absolute
    configured = configured_path()
    detail = (
        f"; configured {SETTING_KEY}={configured!r} is not usable"
        if configured
        else ""
    )
    raise BinaryUnavailable(
        "OpenCode is not configured. Install `opencode` >= "
        f"{'.'.join(str(p) for p in MIN_VERSION)} on the Hermes process PATH, or run "
        "`hermes auth add opencode-free` / `hermes model` to save an absolute "
        f"plugins.entries.{PLUGIN_ID}.settings.{SETTING_KEY}{detail}"
    )


def location_hint() -> str:
    """Cheap cache-key component: configured path plus the PATH candidate.

    No version probe, so profile discovery and auxiliary hot paths stay free of
    subprocesses; the catalog cache TTL handles a replaced binary.
    """
    return f"{configured_path() or ''}|{shutil.which('opencode') or ''}"


def ask_for_path(*, ask=input, say=print, max_attempts: int = 3) -> str | None:
    """Interactive onboarding: validate before saving; None on cancel/invalid input.

    Configuration is untouched unless a path validates, so cancelling leaves the
    profile exactly as it was.
    """
    say(f"OpenCode >= {'.'.join(str(p) for p in MIN_VERSION)} is required and is never installed by Hermes.")
    say("Enter the absolute path to the official `opencode` executable.")
    for _attempt in range(max_attempts):
        try:
            answer = ask("OpenCode binary path (blank cancels): ")
        except (EOFError, KeyboardInterrupt):
            say("Cancelled; configuration unchanged.")
            return None
        raw = (answer or "").strip()
        if not raw:
            say("Cancelled; configuration unchanged.")
            return None
        try:
            absolute, version = validate(raw)
        except (BinaryUnavailable, OSError, subprocess.SubprocessError) as error:
            say(f"  ✗ {error}")
            continue
        try:
            saved = save_path(absolute)
        except Exception as error:  # noqa: BLE001 - permission/managed-install refusal
            say(f"  ✗ could not save {SETTING_KEY}: {error}")
            return None
        say(f"  ✓ OpenCode {version} at {saved}")
        return saved
    say("No usable path provided; configuration unchanged.")
    return None


def ensure(*, ask=input, say=print, interactive: bool | None = None) -> str | None:
    """Resolve for onboarding: reuse a usable binary, else prompt when interactive."""
    try:
        return resolve()
    except BinaryUnavailable:
        pass
    if interactive is None:
        interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
    if not interactive:
        say(
            "No usable OpenCode binary. Run `hermes auth add opencode-free` or "
            "`hermes model` in a terminal to save one."
        )
        return None
    return ask_for_path(ask=ask, say=say)
