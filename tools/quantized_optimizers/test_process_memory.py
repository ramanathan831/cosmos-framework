"""A real subprocess spawned by a background thread belongs in host accounting."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.quantized_optimizers.run_deployment import process_tree_memory


def test_thread_spawned_worker_is_measured():
    script = """
import subprocess, sys, threading
def worker():
    child = subprocess.Popen([sys.executable, '-c',
        'import time; data=bytearray(16*1024*1024); print(\"ready\",flush=True); time.sleep(60)'],
        stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        print(child.pid, flush=True)
        sys.stdin.readline()
    finally:
        child.terminate()
        child.wait(timeout=10)
thread = threading.Thread(target=worker)
thread.start()
thread.join()
"""
    root = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        worker_pid = int(root.stdout.readline().strip())
        # This proves the old main-thread-only traversal would miss this worker.
        assert worker_pid not in map(int, Path(f"/proc/{root.pid}/task/{root.pid}/children").read_text().split())
        report = process_tree_memory(root.pid)
        assert set(report["process_tree_pids"]) == {root.pid, worker_pid}
        assert report["process_tree_measured_pids"] == report["process_tree_pids"]
        assert report["process_tree_pss"] >= 16 * 1024 * 1024
        json.dumps(report, allow_nan=False)
    finally:
        root.communicate("done\n", timeout=20)
    assert root.returncode == 0


@pytest.mark.parametrize("fail", [False, True])
def test_native_workflow_sampler_covers_preflight_and_closes_on_failure(tmp_path, monkeypatch, fail):
    import pynvml

    from tools.quantized_optimizers import gr00t_native, run_deployment

    samples, lifecycle = [], []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("RESULTS", str(tmp_path))
    monkeypatch.setenv("RESOURCE_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["gr00t_native"])
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: lifecycle.append("start"))
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: lifecycle.append("stop"))
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", lambda _: 0)

    def sample(_handle, process, _output, stop):
        assert process.pid == os.getpid()
        while not stop.is_set():
            samples.append(process_tree_memory(process.pid))
            stop.wait(0.01)

    monkeypatch.setattr(run_deployment, "monitor", sample)

    def preflight(_args, _root, _output):
        assert lifecycle == ["start"]
        child = subprocess.Popen(
            [sys.executable, "-c", 'import sys; data=bytearray(16*1024*1024); print("ready",flush=True); input()'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert child.stdout.readline().strip() == "ready"
            deadline = time.monotonic() + 10
            while not any(child.pid in row["process_tree_measured_pids"] for row in samples):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            if fail:
                raise ValueError("preflight rejected")
        finally:
            child.communicate("done\n", timeout=10)

    monkeypatch.setattr(gr00t_native, "run", preflight)
    if fail:
        with pytest.raises(ValueError, match="preflight rejected"):
            gr00t_native.main()
    else:
        gr00t_native.main()
    assert lifecycle == ["start", "stop"]
    count = len(samples)
    time.sleep(0.03)
    assert len(samples) == count  # sampler joined, including on preflight failure
    scope = json.loads((tmp_path / "memory-scope.json").read_text())
    assert scope["root_pid"] == os.getpid()


def test_launcher_phase_retains_subprocess_failure(tmp_path):
    from tools.quantized_optimizers.gr00t_native import checked_phase

    with pytest.raises(subprocess.CalledProcessError):
        checked_phase([sys.executable, "-c", "raise SystemExit(3)"], tmp_path, "preflight")
    phase = json.loads((tmp_path / "launcher-phases.jsonl").read_text())
    assert phase["phase"] == "preflight" and phase["passed"] is False and phase["seconds"] > 0
