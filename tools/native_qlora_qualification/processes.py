# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Track owned descendants across elastic rank sessions; signal stable Linux pidfds.

Polling reads only known processes and their task children, not the node's full
process table. A detached rank's orphaned children remain owned only if observed
while their ancestry was live. A receipt cannot grant ownership of a new PID.
"""

from __future__ import annotations

import errno
import os
import signal
import threading
from pathlib import Path
from typing import Any


def process_identity(pid: int) -> dict[str, Any] | None:
    try:
        root = Path("/proc") / str(pid)
        tail = (root / "stat").read_text().rsplit(")", 1)[1].split()
        uid = int(
            next(line.split()[1] for line in (root / "status").read_text().splitlines() if line.startswith("Uid:"))
        )
        return {
            "pid": pid,
            "state": tail[0],
            "ppid": int(tail[1]),
            "pgid": int(tail[2]),
            "session": int(tail[3]),
            "start_ticks": int(tail[19]),
            "uid": uid,
            "cgroup": (root / "cgroup").read_text(),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left[key] == right[key] for key in ("pid", "start_ticks", "uid", "cgroup"))


def children(pid: int) -> set[int]:
    found = set()
    try:
        for task in (Path("/proc") / str(pid) / "task").iterdir():
            try:
                found.update(int(value) for value in (task / "children").read_text().split())
            except (FileNotFoundError, ProcessLookupError):
                pass
    except (FileNotFoundError, ProcessLookupError):
        pass
    return found


class OwnedProcessTree:
    """Start immediately after Popen; call cleanup in finally, even after root exit."""

    def __init__(self, root_pid: int, *, interval: float = 0.5):
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("qualification process cleanup requires Linux pidfd support")
        root = process_identity(root_pid)
        if root is None or root["ppid"] != os.getpid() or root["uid"] != os.getuid():
            raise ValueError("process tracker root must be this process's freshly launched child")
        self.root = root
        self.known = {root_pid: root}
        self.interval = interval
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._error = None
        self.refresh()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def refresh(self) -> None:
        with self._lock:
            queue = list(self.known)
            visited = set()
            while queue:
                pid = queue.pop()
                if pid in visited:
                    continue
                visited.add(pid)
                actual = process_identity(pid)
                if actual is None or not same_process(actual, self.known[pid]):
                    continue
                self.known[pid] = actual
                for child_pid in children(pid):
                    child = process_identity(child_pid)
                    if child is None:
                        continue
                    if (
                        child["ppid"] != pid
                        or child["uid"] != self.root["uid"]
                        or child["cgroup"] != self.root["cgroup"]
                    ):
                        raise RuntimeError("descendant ownership changed during qualification tracking")
                    previous = self.known.get(child_pid)
                    if previous is not None and not same_process(previous, child):
                        raise RuntimeError("owned descendant PID was reused during qualification")
                    self.known[child_pid] = child
                    queue.append(child_pid)

    def _watch(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.refresh()
            except BaseException as error:
                self._error = error
                return

    def validate_interruption(self, snapshot: dict[str, Any]) -> None:
        """Validate the rank receipt against independently observed live ancestry."""
        self.refresh()
        with self._lock:
            rank = snapshot["rank"]
            if rank["pid"] == self.root["pid"] or rank["pgid"] != rank["pid"] or rank["session"] != rank["pid"]:
                raise ValueError("expected elastic rank in its own recorded session")
            for record in [rank, *snapshot["descendants"]]:
                previous = self.known.get(record["pid"])
                if previous is None or not same_process(previous, record):
                    raise ValueError("interruption receipt contains a process whose ownership was not observed")

    def _signal(self, record: dict[str, Any], signum: int) -> bool:
        actual = process_identity(record["pid"])
        if actual is None or actual["state"] == "Z":
            return False
        try:
            fd = os.pidfd_open(record["pid"])
        except OSError as error:
            # A dead session/group leader can retain its PID through live group
            # members: pidfd_open returns EINVAL, not ESRCH, for that non-task PID.
            if error.errno in (errno.ESRCH, errno.EINVAL) and process_identity(record["pid"]) is None:
                return False
            raise
        try:
            actual = process_identity(record["pid"])
            if actual is None or actual["state"] == "Z":
                return False
            if not same_process(record, actual):
                raise RuntimeError("refusing to signal a reused or moved process identity")
            try:
                signal.pidfd_send_signal(fd, signum)
                return True
            except ProcessLookupError:
                return False
        finally:
            os.close(fd)

    def cleanup(self) -> dict[str, int]:
        self._stop.set()
        self._thread.join(timeout=max(5, 4 * self.interval))
        if self._thread.is_alive():
            raise RuntimeError("process ownership watcher failed to stop")
        self.refresh()
        # Stop observed processes before the final ancestry walk, preventing them
        # from spawning children between that walk and termination.
        stopped = set()
        for _ in range(3):
            fresh = set(self.known) - stopped
            for pid in fresh:
                self._signal(self.known[pid], signal.SIGSTOP)
            stopped.update(fresh)
            self.refresh()
            if not (set(self.known) - stopped):
                break
        killed = sum(self._signal(record, signal.SIGKILL) for record in reversed(list(self.known.values())))
        if self._error is not None:
            raise RuntimeError("process ownership tracking failed") from self._error
        return {"observed_processes": len(self.known), "signaled_processes": killed}
