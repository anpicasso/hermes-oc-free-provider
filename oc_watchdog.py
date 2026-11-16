"""Parent-side hard bound for one isolated per-conversation OpenCode runtime.

The official v2.3 binary has no execute timeout and its interpreter cannot be
interrupted out of a non-suspending program. The owner of the subprocess is the
host, so the bound lives here: one process group per conversation, an absolute
deadline per approved program, and a kill/reap that first proves the target is
still the child we started.

Nothing in this module executes tools or model requests. It only decides when a
conversation is dead, and refuses every later call or result for it.
"""
from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path


class WatchdogRefused(RuntimeError):
    """Ownership check failed; refusing to signal a process we do not own."""


class ConversationFailed(RuntimeError):
    """The conversation is dead: no more execution, dispatch, or results."""


def group_members(pgid: int) -> list[int]:
    """Live pids in one process group, read from /proc when it exists."""
    root = Path("/proc")
    if not root.is_dir():
        return []
    members: list[int] = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat.rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid:
                members.append(int(entry.name))
        except (OSError, IndexError, ValueError):
            continue
    return sorted(members)


_CPU_SAMPLES: dict[int, tuple[int, float]] = {}


def cpu_percent(pid: int) -> float:
    """CPU% of one pid between consecutive samples, -1.0 when it is gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        ticks = int(fields[11]) + int(fields[12])
        at = time.monotonic()
        previous = _CPU_SAMPLES.get(pid)
        _CPU_SAMPLES[pid] = (ticks, at)
        if previous is None:
            return 0.0
        ticks_delta = ticks - previous[0]
        seconds = max(1e-6, at - previous[1])
        hz = getattr(os, "sysconf", None)
        rate = 100
        if callable(hz):
            try:
                rate = int(hz("SC_CLK_TCK"))  # type: ignore[arg-type]
            except (ValueError, OSError):
                rate = 100
        return round(ticks_delta / max(1, rate) / seconds * 100.0, 1)
    except (OSError, IndexError, ValueError, KeyError):
        return -1.0


class Watchdog:
    """One conversation's hard bound.

    `begin()` arms an absolute deadline for an approved `execute` program; when
    it expires the whole process group is SIGKILLed and reaped. The deadline is
    never extended by nested tool traffic, so a program that keeps calling tools
    cannot outlive its budget.
    """

    def __init__(self, process, *, binary: str | Path, execute_deadline: float = 2.0, poll: float = 0.05):
        self.process = process
        self.binary = Path(binary).resolve()
        self.execute_deadline = execute_deadline
        self.poll = poll
        self.failure: dict | None = None
        # Visible before the kill/reap starts: readers must never see a
        # conversation as alive while its teardown is in progress.
        self.failing = False
        self.killed = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._label: str | None = None

    @property
    def active(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive() and not self._stop.is_set())

    # -- ownership ---------------------------------------------------------
    def verify_ownership(self) -> int:
        """Prove the target is still the child this host started."""
        pid = self.process.pid
        if self.process.poll() is not None:
            raise WatchdogRefused(f"pid {pid} already exited")
        if os.name != "posix":
            # Popen owns the live child handle; no portable group/exe identity.
            return pid
        if os.getpgid(pid) != pid:
            raise WatchdogRefused(f"pid {pid} is not its own process-group leader")
        stat_path = Path(f"/proc/{pid}/stat")
        if not stat_path.is_file():
            return pid
        try:
            stat = stat_path.read_text()
            fields = stat.rsplit(")", 1)[1].split()
            ppid, pgrp = int(fields[1]), int(fields[2])
            exe = os.readlink(f"/proc/{pid}/exe")
        except OSError as error:
            raise WatchdogRefused(f"pid {pid} vanished: {error}") from None
        if ppid != os.getpid():
            raise WatchdogRefused(f"pid {pid} is not our child (ppid={ppid})")
        if pgrp != pid:
            raise WatchdogRefused(f"pid {pid} group {pgrp} is not private")
        if Path(exe).resolve() != self.binary:
            raise WatchdogRefused(f"pid {pid} runs {exe}, expected {self.binary}")
        return pid

    # -- lifecycle ---------------------------------------------------------
    def begin(self, label: str) -> None:
        if self.failure is not None:
            raise ConversationFailed(self.failure["reason"])
        if self.failing:
            raise ConversationFailed("conversation teardown is already in progress")
        # One tracked program per conversation. A second approval while a
        # program is still running would reset the shared stop event and leave
        # two timers racing; refuse it instead.
        if self._thread is not None and self._thread.is_alive() and not self._stop.is_set():
            raise ConversationFailed(f"execute already active: {self._label}")
        self._label = label
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, name=f"watchdog:{label}", daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        deadline = time.monotonic() + self.execute_deadline
        while not self._stop.wait(self.poll):
            if time.monotonic() < deadline:
                continue
            self.fail(f"execute deadline exceeded ({self.execute_deadline:.1f}s): {self._label}")
            return

    def end(self) -> None:
        """The bounded unit of work finished on its own."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def fail(self, reason: str) -> dict:
        """Hard-stop the conversation; explicit, idempotent, recorded."""
        with self._lock:
            if self.failure is not None:
                return self.failure
            self.failing = True
            record: dict = {"reason": reason, "label": self._label, "at": time.time()}
            try:
                pid = self.verify_ownership()
                record["pid"] = pid
                record["cpuBeforeKill"] = cpu_percent(pid)
                if os.name == "posix":
                    os.killpg(pid, signal.SIGKILL)
                else:
                    self.process.kill()
                try:
                    self.process.wait(timeout=5.0)
                except Exception:  # noqa: BLE001 - wait failure is reported, not hidden
                    record["reap"] = "wait timeout"
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and group_members(pid):
                    time.sleep(0.02)
                record["groupMembersAfterKill"] = group_members(pid)
                record["cpuAfterKill"] = cpu_percent(pid)
                record["returncode"] = self.process.returncode
                self.killed = True
            except Exception as error:  # noqa: BLE001 - recorded as the explicit failure
                record["killError"] = f"{type(error).__name__}: {error}"
            self.failure = record
            self._stop.set()
            return record
