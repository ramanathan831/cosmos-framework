"""Process-isolated resident/pressure allocation diagnostic, not task qualification.

A CUDA ballast models competing resident activations. It leaves less free device
memory than two eight-bit moments require; it is not a full-model capacity result.
Each child has a 120-second deadline and retains failures in the JSON evidence.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def child(arm, pressure):
    import importlib.metadata

    import torch

    from cosmos_framework.posttrain.config.build import build_optimizer
    from cosmos_framework.posttrain.config.rl import OptimizerConfig

    torch.cuda.init()
    row = {
        "arm": arm,
        "pressure": pressure,
        "torch": torch.__version__,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "device": torch.cuda.get_device_name(),
        "pid": os.getpid(),
        "samples": [],
    }
    try:
        net = torch.nn.Module()
        net.register_parameter("weight", torch.nn.Parameter(torch.ones(268435456, device="cuda")))
        net.weight.grad = torch.full_like(net.weight, 0.01)
        opt = build_optimizer(net, OptimizerConfig(name=arm))
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        row.update(free_before_ballast=free, total_memory=total)
        ballast = None
        if pressure:
            # Moment arrays require 512 MiB; leave 256 MiB for local allocations.
            ballast = torch.empty(free - 256 * 1024**2, dtype=torch.uint8, device="cuda")
            ballast.fill_(1)
            torch.cuda.synchronize()
        row["ballast_bytes"] = 0 if ballast is None else ballast.numel()
        row["free_after_ballast"] = torch.cuda.mem_get_info()[0]
        for index in range(3):
            started = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            row["samples"].append(
                {
                    "index": index,
                    "step_seconds": time.perf_counter() - started,
                    "torch_allocated": torch.cuda.memory_allocated(),
                    "torch_peak": torch.cuda.max_memory_allocated(),
                    "torch_reserved": torch.cuda.memory_reserved(),
                    "device_used": torch.cuda.device_memory_used(),
                    "host": {
                        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                        for line in Path("/proc/self/status").read_text().splitlines()
                        if line.startswith(("VmRSS:", "VmHWM:"))
                    },
                }
            )
        row["state"] = opt.state_report()
        row["finite_parameter_sample"] = bool(torch.isfinite(net.weight[:4096]).all())
        row["status"] = "completed"
    except Exception as error:
        row.update(status="failed", error_type=type(error).__name__, error=str(error))
    print(json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--child", choices=("adamw_bnb_8bit", "paged_adamw_8bit"))
    p.add_argument("--pressure", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.child:
        child(args.child, args.pressure)
        return
    rows = []
    for pressure in (False, True):
        for arm in ("adamw_bnb_8bit", "paged_adamw_8bit"):
            cmd = [sys.executable, __file__, "--child", arm] + (["--pressure"] if pressure else [])
            started = time.monotonic()
            try:
                run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                row = {"returncode": run.returncode, "stderr": run.stderr, "elapsed": time.monotonic() - started}
                row["result"] = json.loads(run.stdout.splitlines()[-1]) if run.stdout.strip() else None
            except subprocess.TimeoutExpired:
                row = {"arm": arm, "pressure": pressure, "status": "timeout", "elapsed": time.monotonic() - started}
            rows.append(row)
            args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
