"""Bounded real-CUDA optimizer isolation diagnostic; not a task-quality benchmark."""

import argparse
import gc
import json
import resource
import time

import torch

from cosmos_framework.posttrain.config.build import build_optimizer
from cosmos_framework.posttrain.config.rl import OptimizerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(8)
    rows = []
    start = time.monotonic()
    for seed in (17, 29, 43):
        for trial in range(5):
            for name in ("adamw", "adamw_bnb_8bit", "paged_adamw_8bit", "bnb32_diagnostic"):
                # Hard local budget: 15 minutes, counted including failed attempts.
                if time.monotonic() - start > 900:
                    raise RuntimeError("diagnostic wall budget exhausted")
                torch.manual_seed(seed)
                net = torch.nn.Sequential(
                    *[torch.nn.Linear(args.width, args.width, bias=False, device="cuda") for _ in range(args.depth)]
                )
                x = torch.randn(64, args.width, device="cuda")
                y = torch.randn(64, args.width, device="cuda")
                if name == "bnb32_diagnostic":
                    import bitsandbytes

                    opt = bitsandbytes.optim.AdamW32bit(net.parameters(), lr=1e-4, weight_decay=0.01)
                else:
                    opt = build_optimizer(
                        net,
                        OptimizerConfig(name=name, impl="fused" if name == "adamw" else "", lr=1e-4, weight_decay=0.01),
                    )
                torch.cuda.reset_peak_memory_stats()
                step_times, all_times, losses = [], [], []
                for i in range(15):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = (net(x) - y).square().mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    opt.step()
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    if i >= 5:
                        step_times.append(t2 - t1)
                        all_times.append(t2 - t0)
                        losses.append(loss.item())
                report = opt.state_report() if hasattr(opt, "state_report") else None
                rows.append(
                    dict(
                        seed=seed,
                        trial=trial,
                        optimizer=name,
                        optimizer_seconds=step_times,
                        training_seconds=all_times,
                        losses=losses,
                        state=report,
                        peak_allocated=torch.cuda.max_memory_allocated(),
                        peak_reserved=torch.cuda.max_memory_reserved(),
                        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        device_used=torch.cuda.device_memory_used(),
                    )
                )
                del opt, net, x, y, loss
                gc.collect()
                torch.cuda.empty_cache()
                with open(args.output, "w") as f:
                    json.dump(
                        dict(
                            torch=torch.__version__,
                            width=args.width,
                            depth=args.depth,
                            gpu=torch.cuda.get_device_name(),
                            elapsed_seconds=time.monotonic() - start,
                            rows=rows,
                            caveat="Synthetic all-resident development diagnostic; no task quality or capacity claim",
                        ),
                        f,
                        indent=2,
                    )


if __name__ == "__main__":
    main()
