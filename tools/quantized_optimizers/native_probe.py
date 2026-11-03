"""Native SFT with bounded shape/state/timing telemetry for DEVELOPMENT only.

The hooks synchronize CUDA and therefore are not uninstrumented speed evidence.
They never record token contents or replace model/optimizer arithmetic.
"""

import json
import os
import time
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

from cosmos_framework.posttrain.entrypoints.sft import main
from cosmos_framework.posttrain.model.hf import HFModel

out = Path(os.environ["RESULTS"]) / (os.environ.get("PROBE_ARM", "native") + ".telemetry.jsonl")
forward = HFModel.forward_policy_batch
counts = {"forward": 0}
timers = {}


def record(row):
    with out.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def observed_forward(self, batch):
    if counts["forward"] < 256:
        row = {
            "kind": "forward",
            "index": counts["forward"],
            "grad_enabled": torch.is_grad_enabled(),
            "shapes": {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)},
        }
        for key in ("image_grid_thw", "video_grid_thw"):
            if key in batch:
                row[key] = batch[key].tolist()
        if "attention_mask" in batch:
            row["sequence_lengths"] = batch["attention_mask"].sum(dim=-1).tolist()
        record(row)
    counts["forward"] += 1
    return forward(self, batch)


def before(opt, args, kwargs):
    timer = timers.setdefault(id(opt), {"depth": 0})
    if timer["depth"] == 0:
        torch.cuda.synchronize()
        timer["start"] = time.perf_counter()
    timer["depth"] += 1


def after(opt, args, kwargs):
    timer = timers[id(opt)]
    timer["depth"] -= 1
    if timer["depth"]:
        return
    del timers[id(opt)]
    torch.cuda.synchronize()
    row = {
        "kind": "optimizer",
        "backend": type(opt).__name__,
        "optimizer_seconds": time.perf_counter() - timer["start"],
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "device_used": torch.cuda.device_memory_used(),
        "host_rss_kib": int(
            next(
                line.split()[1]
                for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith("VmRSS:")
            )
        ),
    }
    if hasattr(opt, "state_report"):
        row["state"] = opt.state_report()
    else:
        row["trainable_elements"] = sum(p.numel() for g in opt.param_groups for p in g["params"])
        row["state_bytes"] = sum(
            v.numel() * v.element_size() for s in opt.state.values() for v in s.values() if isinstance(v, torch.Tensor)
        )
    try:
        import pynvml

        pynvml.nvmlInit()
        row["process_gpu_bytes"] = (
            sum(
                p.usedGpuMemory
                for i in range(pynvml.nvmlDeviceGetCount())
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(i))
                if p.pid == os.getpid()
            )
            or None
        )  # Container PID namespaces may prevent NVML PID matching.
    except Exception as error:
        row["process_gpu_bytes"] = None
        row["nvml_error"] = str(error)
    record(row)


if __name__ == "__main__":
    HFModel.forward_policy_batch = observed_forward
    pre = register_optimizer_step_pre_hook(before)
    post = register_optimizer_step_post_hook(after)
    try:
        main()
    finally:
        pre.remove()
        post.remove()
        HFModel.forward_policy_batch = forward
