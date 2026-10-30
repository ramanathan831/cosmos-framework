# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real Linux process cleanup across elastic-like detached rank sessions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from tools.native_qlora_qualification.processes import OwnedProcessTree, process_identity


def wait_for(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise TimeoutError("test process condition did not arrive")


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux pidfds required")
@pytest.mark.parametrize("rank_exits", [True, False], ids=["interrupted-rank", "timeout-live-rank"])
def test_detached_rank_exit_does_not_leave_prefetch_worker_or_kill_unrelated_process(tmp_path, rank_exits):
    ready, release = tmp_path / "ready.json", tmp_path / "release"
    rank_script = """
import json,os,subprocess,sys,time
from pathlib import Path
worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
Path(sys.argv[1]).write_text(json.dumps({'rank':os.getpid(),'worker':worker.pid}))
while not Path(sys.argv[2]).exists():time.sleep(.01)
os._exit(86)
"""
    launcher_script = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',sys.argv[1],*sys.argv[2:]],start_new_session=True); p.wait()"
    launcher = subprocess.Popen(
        [sys.executable, "-c", launcher_script, rank_script, str(ready), str(release)], start_new_session=True
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    tracker = OwnedProcessTree(launcher.pid, interval=0.01)
    try:
        wait_for(ready.exists)
        record = json.loads(ready.read_text())
        wait_for(lambda: record["worker"] in tracker.known)
        snapshot = {"rank": process_identity(record["rank"]), "descendants": [process_identity(record["worker"])]}
        assert snapshot["rank"]["pgid"] != launcher.pid
        if rank_exits:
            release.touch()
            launcher.wait(timeout=8)
        tracker.validate_interruption(snapshot)
        forged = {"rank": snapshot["rank"], "descendants": [process_identity(unrelated.pid)]}
        with pytest.raises(ValueError, match="ownership was not observed"):
            tracker.validate_interruption(forged)
        result = tracker.cleanup()
        launcher.wait(timeout=8)
        assert result["observed_processes"] == 3
        wait_for(lambda: (identity := process_identity(record["worker"])) is None or identity["state"] == "Z")
        assert unrelated.poll() is None
    finally:
        try:
            tracker.cleanup()
        finally:
            if launcher.poll() is None:
                launcher.kill()
                launcher.wait()
            unrelated.kill()
            unrelated.wait()


def test_pid_birth_change_refuses_signal(monkeypatch):
    from tools.native_qlora_qualification import processes

    tracker = object.__new__(OwnedProcessTree)
    expected = {"pid": 100, "start_ticks": 10, "uid": 0, "cgroup": "fixture"}
    monkeypatch.setattr(processes.os, "pidfd_open", lambda _: 123)
    monkeypatch.setattr(processes.os, "close", lambda _: None)
    monkeypatch.setattr(processes, "process_identity", lambda _: {**expected, "start_ticks": 11, "state": "S"})
    monkeypatch.setattr(processes.signal, "pidfd_send_signal", lambda *a: pytest.fail("signaled reused PID"))
    with pytest.raises(RuntimeError, match="reused"):
        tracker._signal(expected, processes.signal.SIGKILL)
