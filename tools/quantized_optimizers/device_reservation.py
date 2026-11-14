"""Explicit co-resident CUDA memory for native paging stress, never a training backend.

The buffers are ordinary CUDA allocations in separate owned processes. They do no
work after initialization, and never change model inputs or optimizer state. A
pressure result must name this synthetic reservation separately from model memory.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def worker_command(device, size, ready):
    return [
        sys.executable,
        "-m",
        "tools.quantized_optimizers.device_reservation",
        "--device",
        str(device),
        "--bytes",
        str(size),
        "--ready",
        str(ready),
    ]


class ResidentDeviceMemory:
    def __init__(self, gib, output, *, timeout=180):
        if type(gib) is not int or gib < 0:
            raise ValueError("co-resident GiB must be a nonnegative integer")
        self.size, self.output, self.timeout = gib * 2**30, Path(output), timeout
        self.processes, self.logs, self.receipts = [], [], []

    def check(self):
        if any(process.poll() is not None for process in self.processes):
            raise RuntimeError("co-resident allocation worker exited; pressure run is invalid")

    def __enter__(self):
        if not self.size:
            return self
        started = time.monotonic()
        try:
            for device, _ in enumerate(os.environ["CUDA_VISIBLE_DEVICES"].split(",")):
                ready = self.output / f"reservation-gpu{device}.json"
                if ready.exists():
                    raise FileExistsError(ready)
                log = (self.output / f"reservation-gpu{device}.log").open("x")
                self.logs.append(log)
                process = subprocess.Popen(worker_command(device, self.size, ready), stdout=log, stderr=log)
                self.processes.append(process)
                self.receipts.append((device, ready, process))
            while not all(ready.exists() for _, ready, _ in self.receipts):
                self.check()
                if time.monotonic() - started > self.timeout:
                    raise TimeoutError("co-resident CUDA reservation did not become ready")
                time.sleep(0.1)
            for device, ready, process in self.receipts:
                receipt = json.loads(ready.read_text())
                if (receipt["device"], receipt["bytes"], receipt["pid"]) != (device, self.size, process.pid):
                    raise RuntimeError("co-resident CUDA reservation identity differs")
            self.check()
            (self.output / "reservation-scope.json").write_text(
                json.dumps(
                    {
                        "bytes_per_gpu": self.size,
                        "pids": [process.pid for process in self.processes],
                        "startup_seconds": time.monotonic() - started,
                        "scope": "synthetic co-resident ordinary CUDA buffers; separate from native model memory",
                    },
                    indent=2,
                )
            )
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in self.logs:
            log.close()

    def __exit__(self, *exc):
        try:
            if exc[0] is None:
                self.check()
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    if args.bytes <= 0 or args.device < 0 or args.ready.exists():
        parser.error("positive size, valid device and fresh receipt required")
    import torch

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    torch.cuda.set_device(args.device)
    buffer = torch.empty(args.bytes, dtype=torch.uint8, device="cuda").zero_()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    temporary = args.ready.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "device": args.device,
                "bytes": buffer.numel() * buffer.element_size(),
                "ordinary_cuda_allocated_bytes": torch.cuda.memory_allocated(),
                "ordinary_cuda_reserved_bytes": torch.cuda.memory_reserved(),
                "device_free_bytes_after_reservation": free,
                "device_total_bytes": total,
            },
            indent=2,
        )
    )
    os.replace(temporary, args.ready)
    stop.wait()
    del buffer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
