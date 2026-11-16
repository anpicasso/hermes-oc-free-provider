"""Profile-scoped native-session affinity: mapping DB, ownership locks, durable roots.

One row per logical scope/model maps to the exact native session ID the official
OpenCode binary allocated. The mapping lives in the documented plugin storage
(``plugins.plugin_storage.plugin_db``), never in Hermes' session database. The
native ``project/``/``data/`` workspace lives in a durable per-scope directory so
an idle runtime can be evicted without losing the cached prefix.

Nothing here is sent over the wire and no ID is ever synthesised: the official
binary allocates it, this module only records the exact string it returned.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterator

PLUGIN_NAME = "oc-free-provider"
SCHEMA_VERSION = 1
RETENTION_SECONDS = 14 * 24 * 60 * 60.0

# Bounds, not ownership proof: the native GET corroborates id/location before use.
_NATIVE_ID_RE = re.compile(r"^ses_[0-9A-Za-z_-]{8,120}$")
_SCOPE_KEY_RE = re.compile(r"^[0-9a-f]{64}$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS native_sessions (
    scope_key TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    native_session_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    last_used REAL NOT NULL
)
"""


class AffinityRefused(RuntimeError):
    """Corrupt, cross-profile or otherwise unsafe affinity state."""


class AffinityBusy(AffinityRefused):
    """Another process owns the scope right now."""


def _home_matches(profile_home: str) -> None:
    """Refuse to write profile A's mapping under profile B's active home."""
    try:
        from hermes_constants import get_hermes_home
    except Exception:  # noqa: BLE001 - standalone provider import
        return
    current = str(get_hermes_home())
    if os.path.realpath(current) != os.path.realpath(str(profile_home)):
        raise AffinityRefused(
            "active Hermes profile changed during the request; refusing a cross-profile affinity write"
        )


def _data_root(profile_home: str) -> Path:
    _home_matches(profile_home)
    from plugins.plugin_storage import plugin_data_dir

    return Path(plugin_data_dir(PLUGIN_NAME))


def scope_key(profile_home: str, scope: str, provider_id: str, model_id: str) -> str:
    """Versioned, unambiguous local key. Never used as a wire/session identifier."""
    if not profile_home or not scope:
        raise AffinityRefused("profile home and logical scope are required")
    payload = json.dumps(
        ["oc-native-affinity", 1, str(profile_home), str(scope), str(provider_id), str(model_id)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def native_root(profile_home: str, key: str) -> Path:
    if not _SCOPE_KEY_RE.fullmatch(str(key)):
        raise AffinityRefused(f"invalid scope key {key!r}")
    root = _data_root(profile_home) / "native" / key
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    return root


def _open(profile_home: str) -> sqlite3.Connection:
    _home_matches(profile_home)
    from plugins.plugin_storage import plugin_db

    return plugin_db(PLUGIN_NAME)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_SCHEMA)
    conn.commit()


def _valid_id(native_session_id: object) -> bool:
    return isinstance(native_session_id, str) and bool(_NATIVE_ID_RE.fullmatch(native_session_id))


def load(profile_home: str, key: str, provider_id: str, model_id: str) -> dict | None:
    """Return the stored mapping or None. Corruption is reported, never repaired."""
    conn = _open(profile_home)
    try:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT scope_key, version, native_session_id, provider_id, model_id, last_used "
            "FROM native_sessions WHERE scope_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        scope_key_value, version, native_session_id, row_provider, row_model, last_used = row
        if (
            scope_key_value != key
            or version != SCHEMA_VERSION
            or not _valid_id(native_session_id)
            or row_provider != provider_id
            or row_model != model_id
            or not isinstance(last_used, (int, float))
        ):
            raise AffinityRefused(
                f"native affinity mapping for scope {key[:12]}… is corrupt; refusing to use or rewrite it"
            )
        with conn:
            conn.execute(
                "UPDATE native_sessions SET last_used = ? WHERE scope_key = ?", (time.time(), key)
            )
        return {
            "scope_key": scope_key_value,
            "version": version,
            "native_session_id": native_session_id,
            "provider_id": row_provider,
            "model_id": row_model,
            "last_used": float(last_used),
        }
    except sqlite3.DatabaseError as error:
        raise AffinityRefused(f"native affinity mapping is unreadable: {error}") from error
    finally:
        conn.close()


def save(profile_home: str, key: str, native_session_id: str, provider_id: str, model_id: str) -> None:
    if not _valid_id(native_session_id):
        raise AffinityRefused(f"refusing to persist a non-native session id {native_session_id!r}")
    conn = _open(profile_home)
    try:
        _ensure_schema(conn)
        with conn:
            conn.execute(
                "INSERT INTO native_sessions "
                "(scope_key, version, native_session_id, provider_id, model_id, last_used) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(scope_key) DO UPDATE SET "
                "version = excluded.version, native_session_id = excluded.native_session_id, "
                "provider_id = excluded.provider_id, model_id = excluded.model_id, "
                "last_used = excluded.last_used",
                (key, SCHEMA_VERSION, native_session_id, provider_id, model_id, time.time()),
            )
    except sqlite3.DatabaseError as error:
        raise AffinityRefused(f"native affinity mapping could not be written: {error}") from error
    finally:
        conn.close()


def delete(profile_home: str, key: str) -> None:
    conn = _open(profile_home)
    try:
        _ensure_schema(conn)
        with conn:
            conn.execute("DELETE FROM native_sessions WHERE scope_key = ?", (key,))
    except sqlite3.DatabaseError as error:
        raise AffinityRefused(f"native affinity mapping could not be updated: {error}") from error
    finally:
        conn.close()


# -- cross-process ownership -------------------------------------------------
class ScopeLock:
    """A held advisory scope lock. Never unlink the lock file (inode races)."""

    def __init__(self, handle):
        self._handle = handle
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover - exercised on Windows only
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        finally:
            self._handle.close()


def acquire_lock(profile_home: str, key: str) -> ScopeLock:
    """Non-blocking ownership; the caller holds it for the runtime lifetime."""
    locks = _data_root(profile_home) / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    path = locks / f"{key}.lock"
    handle = open(path, "a+b")
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:  # pragma: no cover - exercised on Windows only
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        handle.close()
        raise AffinityBusy("native scope is owned by another process") from error
    return ScopeLock(handle)


@contextlib.contextmanager
def try_lock(profile_home: str, key: str) -> Iterator[None]:
    """Short-lived lock for prune/reset; a live owner holds the same lock."""
    lock = acquire_lock(profile_home, key)
    try:
        yield
    finally:
        lock.release()


def _last_used(profile_home: str, key: str) -> float | None:
    conn = _open(profile_home)
    try:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT last_used FROM native_sessions WHERE scope_key = ?", (key,)
        ).fetchone()
    except sqlite3.DatabaseError as error:
        raise AffinityRefused(f"native affinity mapping is unreadable: {error}") from error
    finally:
        conn.close()
    if row is None:
        return None
    if not isinstance(row[0], (int, float)):
        raise AffinityRefused(f"native affinity mapping for scope {key[:12]}… is corrupt")
    return float(row[0])


# -- bounded retention -------------------------------------------------------
_LAST_PRUNE: dict[str, float] = {}
_PRUNE_LOCK = threading.Lock()


def prune_once(profile_home: str, *, active_keys: set[str], max_age: float | None = None) -> list[str]:
    """Age-based disk retention, at most once per interval per profile.

    Files are removed before the mapping row, so an interrupted prune leaves a
    missing native session (reported as an affinity reset), not a stale row
    pointing at deleted state. Locks make active scopes untouchable.
    """
    now = time.time()
    age = RETENTION_SECONDS if max_age is None else max(0.0, float(max_age))
    with _PRUNE_LOCK:
        last = _LAST_PRUNE.get(profile_home, 0.0)
        if max_age is None and now - last < 3600.0:
            return []
        _LAST_PRUNE[profile_home] = now
    return prune(profile_home, active_keys=active_keys, max_age=age)


def prune(profile_home: str, *, active_keys: set[str], max_age: float) -> list[str]:
    root = _data_root(profile_home) / "native"
    if not root.is_dir():
        return []
    now = time.time()
    removed: list[str] = []

    conn = _open(profile_home)
    try:
        _ensure_schema(conn)
        rows = conn.execute("SELECT scope_key, last_used FROM native_sessions").fetchall()
    except sqlite3.DatabaseError as error:
        raise AffinityRefused(f"native affinity mapping is unreadable: {error}") from error
    finally:
        conn.close()

    known = {row[0]: float(row[1]) for row in rows if isinstance(row[1], (int, float))}
    candidates: dict[str, float] = {}
    for key, last_used in known.items():
        if _SCOPE_KEY_RE.fullmatch(str(key)) and now - last_used >= max_age:
            candidates[key] = last_used
    for child in root.iterdir() if root.is_dir() else []:
        if not child.is_dir() or not _SCOPE_KEY_RE.fullmatch(child.name):
            continue
        if child.name in known or child.name in active_keys:
            continue
        try:
            candidates.setdefault(child.name, child.stat().st_mtime)
        except OSError:
            candidates.setdefault(child.name, now)
    for key, touched in candidates.items():
        if key in active_keys or now - touched < max_age:
            continue
        try:
            with try_lock(profile_home, key):
                if key in active_keys:
                    continue
                target = root / key
                # Re-read eligibility under the ownership lock: a concurrent
                # request may have touched the row after candidate selection.
                current = _last_used(profile_home, key)
                if current is not None:
                    if time.time() - current < max_age:
                        continue
                elif target.exists():
                    try:
                        if time.time() - target.stat().st_mtime < max_age:
                            continue
                    except OSError:
                        continue
                if target.exists():
                    shutil.rmtree(target)
                delete(profile_home, key)
                removed.append(key)
        except AffinityBusy:
            continue
        except OSError:
            continue
    return removed


def reset_scope(profile_home: str, key: str) -> bool:
    """Explicit reset: remove owned native data and its mapping under the lock."""
    if not _SCOPE_KEY_RE.fullmatch(str(key)):
        raise AffinityRefused(f"invalid scope key {key!r}")
    with try_lock(profile_home, key):
        target = _data_root(profile_home) / "native" / key
        existed = target.exists()
        if existed:
            shutil.rmtree(target)
        delete(profile_home, key)
        return existed
