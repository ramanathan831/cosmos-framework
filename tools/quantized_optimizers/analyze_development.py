"""Paired DEVELOPMENT evidence; reused groups cannot qualify a final release.

Read the loss/time curve and allocation telemetry from the preserved native files.
Never substitute incomplete trials or synthetic losses for held-out task results.
"""

import argparse
import csv
import gzip
import io
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


def interval(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return {"mean": float(a.mean()) if len(a) else None, "ci95": None, "n": len(a)}
    delta = t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / len(a) ** 0.5
    return {"mean": float(a.mean()), "ci95": [float(a.mean() - delta), float(a.mean() + delta)], "n": len(a)}


def parse_time(value):
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    read = gzip.open if args.raw.suffix == ".gz" else open
    with read(args.raw, "rt") as stream:
        raw = json.load(stream)
    arms = ("adamw", "adamw_bnb_8bit", "paged_adamw_8bit")
    runs = {}
    for key, content in raw.items():
        if not key.endswith("metrics.jsonl") or "/capacity/" in key:
            continue
        match = re.search(r"seed(17|29|43)/(adamw|adamw_bnb_8bit|paged_adamw_8bit)/", key)
        if not match:
            continue
        seed, arm = int(match[1]), match[2]
        prefix = key[: match.end()]
        rows = [json.loads(line) for line in content.splitlines()]
        complete = rows[-1].get("optim/optimizer_step") == 40 and rows[-1].get("optim/nonfinite_skips_total") == 0
        clock, curve, tokens, sequences = 0.0, [], [], []
        for row in rows:
            clock += row.get("timer/e2e_step_s", 0) + row.get("timer/validation_s", 0)
            if "val/loss_avg" in row:
                curve.append({"step": row["step"], "seconds": clock, "nll": row["val/loss_avg"], "time": row["time"]})
            if "train/valid_token_count" in row:
                tokens.append(row["train/valid_token_count"])
                sequences.append(row["train/valid_sequence_count"])
        target = curve[0]["nll"] * 0.95
        crossed = next((point for point in curve if point["nll"] <= target), None)
        telemetry = [json.loads(line) for line in raw.get(prefix + arm + ".telemetry.jsonl", "").splitlines()]
        updates = [
            r
            for r in telemetry
            if r["kind"] == "optimizer"
            and r.get("trainable_elements", r.get("state", {}).get("trainable_elements", 0)) > 1e9
        ]
        forwards = [r for r in telemetry if r["kind"] == "forward"]
        device_samples = list(
            csv.DictReader(io.StringIO(raw.get(prefix + arm + ".device_samples.csv", "")), skipinitialspace=True)
        )
        gpu_peaks = []
        for sample in device_samples:
            try:
                gpu_peaks.append(float(sample["memory.used [MiB]"].split()[0]))
            except (KeyError, ValueError, AttributeError):
                pass
        start = raw.get(prefix + arm + ".start")
        end = raw.get(prefix + arm + ".end")
        for point in curve:
            point["wall_seconds"] = parse_time(point["time"]) - parse_time(start) if start else None
        training_rows = [r for r in rows if "timer/e2e_step_s" in r]
        seconds = sum(r["timer/e2e_step_s"] for r in training_rows)
        timing_blocks = [
            {
                "first_step": block[0]["step"],
                "last_step": block[-1]["step"],
                "seconds": sum(row["timer/e2e_step_s"] for row in block),
                "sequences": sum(row["train/valid_sequence_count"] for row in block),
            }
            for index in range(5)
            if len(block := training_rows[5 + index * 7 : 12 + index * 7]) == 7
        ]
        run = {
            "seed": seed,
            "arm": arm,
            "complete": complete,
            "curve": curve,
            "target_nll": target,
            "target_seconds": crossed["seconds"] if crossed else None,
            "wall_target_seconds": parse_time(crossed["time"]) - parse_time(start) if crossed and start else None,
            "wall_seconds": parse_time(end) - parse_time(start) if start and end else None,
            "training_seconds": seconds,
            "training_sequences": sum(sequences),
            "supervised_tokens_per_update": tokens,
            "sequences_per_second": sum(sequences) / seconds if seconds else None,
            "timing_blocks_after_five_updates": timing_blocks,
            "timing_block_note": "paired windows within the same run; not independent trial restarts",
            "optimizer_seconds": [u["optimizer_seconds"] for u in updates],
            "stage_seconds": {
                "preprocess_train_and_val": sum(r["seconds"] for r in telemetry if r["kind"] == "preprocess"),
                "collate_train_and_val": sum(r["seconds"] for r in telemetry if r["kind"] == "collate"),
                "forward_train": sum(
                    r["seconds"] for r in telemetry if r["kind"] == "forward_time" and r["grad_enabled"]
                ),
                "forward_val": sum(
                    r["seconds"] for r in telemetry if r["kind"] == "forward_time" and not r["grad_enabled"]
                ),
                "backward": sum(r["seconds"] for r in telemetry if r["kind"] == "backward"),
                "optimizer": sum(u["optimizer_seconds"] for u in updates),
            },
            "collectives": "single rank; no multi-rank gradient communication",
            "checkpoint_writes": "disabled in comparison recipe; recovery validated separately",
            "torch_peak_bytes": max((u["peak_allocated"] for u in updates), default=None),
            "sampled_device_peak_mib": max(gpu_peaks, default=None),
            "sampled_process_gpu_peak_bytes": max(
                (u["process_gpu_bytes"] for u in updates if u.get("process_gpu_bytes") is not None), default=None
            ),
            "sampled_host_rss_peak_kib": max((u["host_rss_kib"] for u in updates), default=None),
            "state": updates[-1].get("state", {"state_bytes": updates[-1].get("state_bytes")}) if updates else None,
            "forward_shapes": forwards,
        }
        if (seed, arm) in runs:
            raise ValueError("ambiguous repeated comparison arm")
        runs[seed, arm] = run
    # A matched-quality comparison uses one absolute target per paired seed.
    # Quantized arms cannot set an easier target from their own initial loss.
    for (seed, arm), run in runs.items():
        baseline = runs.get((seed, "adamw"))
        if baseline is None:
            run["target_nll"] = run["target_seconds"] = run["wall_target_seconds"] = None
            continue
        target = baseline["curve"][0]["nll"] * 0.95
        crossed = next((point for point in run["curve"] if point["nll"] <= target), None)
        run["target_nll"] = target
        run["target_seconds"] = crossed["seconds"] if crossed else None
        start = next((value for key, value in raw.items() if key.endswith(f"seed{seed}/{arm}/{arm}.start")), None)
        run["wall_target_seconds"] = parse_time(crossed["time"]) - parse_time(start) if crossed and start else None
    complete = len(runs) == 9 and all(r["complete"] for r in runs.values())
    pairs = {}
    if complete:
        budget = min(r["curve"][-1]["seconds"] for r in runs.values())
        wall_budget = min(r["wall_seconds"] for r in runs.values())
        for r in runs.values():
            point = [p for p in r["curve"] if p["seconds"] <= budget][-1]
            r["equal_budget_nll"] = point["nll"]
            r["equal_budget_validation_step"] = point["step"]
            r["equal_budget_training_sequences"] = int(point["step"]) * 4
            available = [p for p in r["curve"] if p["wall_seconds"] is not None and p["wall_seconds"] <= wall_budget]
            r["equal_wall_budget_point"] = available[-1] if available else None
        for arm in arms[1:]:
            base = [runs[s, "adamw"] for s in (17, 29, 43)]
            other = [runs[s, arm] for s in (17, 29, 43)]
            ttq = interval(
                [
                    b["target_seconds"] / a["target_seconds"]
                    for b, a in zip(base, other)
                    if a["target_seconds"] and b["target_seconds"]
                ]
            )
            controls = []
            for b, a in zip(base, other):
                token_counts_equal = a["supervised_tokens_per_update"] == b["supervised_tokens_per_update"]

                def shapes(run):
                    return [
                        (f.get("sequence_lengths"), f.get("video_grid_thw"))
                        for f in run["forward_shapes"]
                        if f["grad_enabled"]
                    ]

                shape_order_equal = shapes(a) == shapes(b)
                controls.append(
                    {
                        "seed": a["seed"],
                        "supervised_token_order_equal": token_counts_equal,
                        "training_shape_order_equal": shape_order_equal,
                    }
                )
                if not token_counts_equal or not shape_order_equal:
                    raise ValueError(f"paired training controls differ for seed {a['seed']} {arm}")
            pairs[arm] = {
                "controls": controls,
                "final_nll_ratio": interval(
                    [a["curve"][-1]["nll"] / b["curve"][-1]["nll"] for b, a in zip(base, other)]
                ),
                "throughput_ratio": interval(
                    [a["sequences_per_second"] / b["sequences_per_second"] for b, a in zip(base, other)]
                ),
                "target_time_ratio": ttq,
                "equal_budget_nll_improvement": interval(
                    [b["equal_budget_nll"] - a["equal_budget_nll"] for b, a in zip(base, other)]
                ),
            }
            pair = pairs[arm]
            pair["wall_target_time_ratio"] = interval(
                [
                    b["wall_target_seconds"] / a["wall_target_seconds"]
                    for b, a in zip(base, other)
                    if a["wall_target_seconds"] and b["wall_target_seconds"]
                ]
            )
            pair["equal_wall_budget_nll_improvement"] = interval(
                [
                    b["equal_wall_budget_point"]["nll"] - a["equal_wall_budget_point"]["nll"]
                    for b, a in zip(base, other)
                    if a["equal_wall_budget_point"] and b["equal_wall_budget_point"]
                ]
            )
            pair["state_memory_saving"] = interval(
                [
                    1 - (a["state"]["moment_bytes"] + a["state"]["metadata_bytes"]) / b["state"]["state_bytes"]
                    for b, a in zip(base, other)
                ]
            )
            pair["sampled_process_gpu_memory_saving"] = interval(
                [
                    1 - a["sampled_process_gpu_peak_bytes"] / b["sampled_process_gpu_peak_bytes"]
                    for b, a in zip(base, other)
                    if a["sampled_process_gpu_peak_bytes"] and b["sampled_process_gpu_peak_bytes"]
                ]
            )
            pair["gates"] = {
                "validation_nll": pair["final_nll_ratio"]["ci95"][1] <= 1.01,
                "training_speed": pair["throughput_ratio"]["ci95"][0] > 1.05,
                "time_to_quality_speed": ttq["n"] == 3 and ttq["ci95"][0] > 1.05,
                "equal_budget_quality_gain": pair["equal_budget_nll_improvement"]["ci95"][0] > 0,
                "wall_time_to_quality_speed": pair["wall_target_time_ratio"]["n"] == 3
                and pair["wall_target_time_ratio"]["ci95"][0] > 1.05,
                "equal_wall_budget_quality_gain": pair["equal_wall_budget_nll_improvement"]["n"] == 3
                and pair["equal_wall_budget_nll_improvement"]["ci95"][0] > 0,
                "recording_separation": json.loads(raw.get("quality_data_audit.json", "{}")).get("quality_gate")
                == "PASS_RECORDING_SEPARATION_ONLY",
                "state_memory": pair["state_memory_saving"]["ci95"][0] >= 0.60,
                "original_task_accuracy": None,
                "uninstrumented_speed_confirmation": None,
            }
    else:
        budget = wall_budget = None
    output = {
        "release": "NOT QUALIFIED: instrumented development evidence; mainline task-score and uninstrumented speed gates remain unmet",
        "complete_three_seed_development": complete,
        "equal_budget_seconds": budget,
        "equal_wall_budget_seconds_including_initialization": wall_budget,
        "data_integrity": json.loads(raw.get("quality_data_audit.json", "{}")),
        "paired": pairs,
        "runs": list(runs.values()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "native_development_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    fig, grid = plt.subplots(2, 2, figsize=(13, 9))
    axes = [grid[0, 0], grid[1, 0], grid[1, 1]]
    wall_ax = grid[0, 1]
    for arm in arms:
        group = [r for r in runs.values() if r["arm"] == arm and r["complete"]]
        if not group:
            continue
        label = ("torch fused FP32 baseline" if arm == "adamw" else arm) + f" (n={len(group)})"
        lengths = {len(r["curve"]) for r in group}
        if len(lengths) == 1:
            times = np.mean([[p["seconds"] for p in r["curve"]] for r in group], axis=0)
            losses = np.array([[p["nll"] for p in r["curve"]] for r in group])
            axes[0].plot(times, losses.mean(axis=0), label=label)
            wall_times = np.mean([[p["wall_seconds"] for p in r["curve"]] for r in group], axis=0)
            wall_ax.plot(wall_times, losses.mean(axis=0), label=label)
            if len(group) > 1:
                ci = t.ppf(0.975, len(group) - 1) * losses.std(axis=0, ddof=1) / len(group) ** 0.5
                axes[0].fill_between(times, losses.mean(axis=0) - ci, losses.mean(axis=0) + ci, alpha=0.15)
                wall_ax.fill_between(wall_times, losses.mean(axis=0) - ci, losses.mean(axis=0) + ci, alpha=0.15)
        x = interval([r["torch_peak_bytes"] / 2**30 for r in group])
        y = interval([r["sequences_per_second"] for r in group])
        axes[1].errorbar(
            x["mean"],
            y["mean"],
            xerr=(x["ci95"][1] - x["mean"]) if x["ci95"] else None,
            yerr=(y["ci95"][1] - y["mean"]) if y["ci95"] else None,
            fmt="o",
            capsize=4,
            label=label,
        )
        process_peaks = [
            r["sampled_process_gpu_peak_bytes"] / 2**30
            for r in group
            if r["sampled_process_gpu_peak_bytes"] is not None
        ]
        if len(process_peaks) == len(group):
            physical = interval(process_peaks)
            axes[2].errorbar(
                physical["mean"],
                y["mean"],
                xerr=physical["ci95"][1] - physical["mean"] if physical["ci95"] else None,
                yerr=y["ci95"][1] - y["mean"] if y["ci95"] else None,
                fmt="o",
                capsize=4,
                label=label,
            )
    axes[2].set(
        xlabel="Sampled process GPU peak GiB (NVML)",
        ylabel="Training video sequences/s",
        title="Post-update process samples; not total paged residency",
    )
    axes[0].set(
        xlabel="Training + validation seconds (initialization excluded)",
        ylabel="Development token NLL",
        title="Reused video groups; not final quality qualification",
    )
    axes[1].set(
        xlabel="Peak PyTorch allocated GiB (managed moments excluded)",
        ylabel="Training video sequences/s",
        title="Native SFT development, 95% seed intervals",
    )
    wall_ax.set(
        xlabel="Wall seconds from process launch (initialization included)",
        ylabel="Validation token NLL",
        title="Actual elapsed training budget; 95% seed intervals",
    )
    for ax in [*axes, wall_ax]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "native_development.png", dpi=180)


if __name__ == "__main__":
    main()
