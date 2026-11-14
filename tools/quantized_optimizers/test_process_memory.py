"""A real subprocess spawned by a background thread belongs in host accounting."""

import json
import subprocess
import sys
from pathlib import Path

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
