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

try:
    from .monitor_host import HostSampler, finish_sample
except ImportError:
    from monitor_host import HostSampler, finish_sample

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
host_sampler = HostSampler(root_pid, stage_file)
with out.open("x") as stream:
    while active:
        host_record = host_sampler.collect()
        selected = {
            int(pid) for attempt in host_record["host_collection"]["attempts"] for pid in attempt["selected_after"]
        }
        gpu = pynvml.nvmlDeviceGetMemoryInfo(handle)
        record = {
            "unix_seconds": time.time(),
            "sampling_seconds": 1,
            "device_uuid": device,
            "device_memory_used_bytes": gpu.used,
            "device_memory_total_bytes": gpu.total,
            **host_record,
        }
        record["device_compute_processes"] = [
            {"pid": p.pid, "used_gpu_memory_bytes": p.usedGpuMemory, "belongs_to_run": p.pid in selected}
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        ]
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
        finish_sample(record, stage_file)
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        time.sleep(1)

host_sampler.close()
pynvml.nvmlShutdown()
