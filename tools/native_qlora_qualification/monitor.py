# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Sample owned CUDA processes and host process-tree memory; peaks are lower bounds."""

import json
import os
import signal
import sys
import time
from pathlib import Path

import pynvml

root_pid = int(sys.argv[1])
out = Path(sys.argv[2])
stage_file = Path(sys.argv[3])
pynvml.nvmlInit()
device = os.environ["COSMOS_QLORA_GPU_UUID"]
handle = pynvml.nvmlDeviceGetHandleByUUID(device)
active = True


def stop(*unused):
    global active
    active = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
with out.open("x") as stream:
    while active:
        parents = {}
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                try:
                    fields = (path / "stat").read_text().split(") ", 1)[1].split()
                    parents[int(path.name)] = int(fields[1])
                except (OSError, ValueError, IndexError):
                    pass
        selected = {root_pid}
        for _ in range(16):
            children = {pid for pid, parent in parents.items() if parent in selected}
            if children <= selected:
                break
            selected |= children
        memory = {}
        for pid in selected - {os.getpid()}:
            try:
                values = {}
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    name, _, value = line.partition(":")
                    if name in ("VmRSS", "VmHWM"):
                        values[name] = int(value.split()[0]) * 1024
                try:
                    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                        name, _, value = line.partition(":")
                        if name in ("Pss", "Rss"):
                            values[name + "_rollup"] = int(value.split()[0]) * 1024
                except (OSError, ValueError):
                    values["rollup_unavailable"] = True
                memory[str(pid)] = values
            except (OSError, ValueError):
                pass
        gpu = pynvml.nvmlDeviceGetMemoryInfo(handle)
        record = {
            "unix_seconds": time.time(),
            "sampling_seconds": 1,
            "stage": stage_file.read_text().strip() if stage_file.exists() else "startup",
            "device_uuid": device,
            "device_memory_used_bytes": gpu.used,
            "device_memory_total_bytes": gpu.total,
            "process_host_memory": memory,
            "selected_host_rss_sum_bytes": sum(v.get("VmRSS", 0) for v in memory.values()),
        }
        record["device_compute_processes"] = [
            {"pid": p.pid, "used_gpu_memory_bytes": p.usedGpuMemory, "belongs_to_run": p.pid in selected}
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        ]
        record["selected_host_pss_sum_bytes"] = sum(v.get("Pss_rollup", 0) for v in memory.values())
        record["host_pss_complete"] = all("Pss_rollup" in v for v in memory.values())
        telemetry = {}
        for name, call in {
            "utilization": lambda: (lambda value: {"gpu_percent": value.gpu, "memory_percent": value.memory})(
                pynvml.nvmlDeviceGetUtilizationRates(handle)
            ),
            "sm_clock_mhz": lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            "memory_clock_mhz": lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
            "power_milliwatts": lambda: pynvml.nvmlDeviceGetPowerUsage(handle),
            "temperature_celsius": lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        }.items():
            try:
                telemetry[name] = call()
            except pynvml.NVMLError as error:
                telemetry[name] = {"unavailable": type(error).__name__}
        record["telemetry"] = telemetry
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        time.sleep(1)
