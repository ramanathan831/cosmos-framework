"""Prespecified round-two paired analysis; incomplete cells never count as passes."""

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.quantized_optimizers.analyze_development import interval

ARMS = ("adamw", "adamw_bnb_8bit")
SEEDS = (17, 29, 43)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = json.load(gzip.open(args.raw, "rt"))
    args.output.mkdir(exist_ok=True, parents=True)
    preparations = [json.loads(v) for k, v in raw.items() if k.endswith("preparation_receipt.json")]
    if len(preparations) != 1:
        raise ValueError("expected one verified preparation receipt")
    preparation = preparations[0]["total_seconds_with_independent_exact_replay"]
    runs = {}
    exposures, runtimes = {}, {}
    model_files = None
    for seed in SEEDS:
        runtime_paths = [
            k
            for k in raw
            if k.startswith(f"seed{seed}/runtime-")
            and json.loads(raw[k]).get("mode", "trials") in ("trials", "train_only")
        ]
        if len(runtime_paths) != 1:
            raise ValueError("seed group must have exactly one allocation identity")
        runtimes[seed] = json.loads(raw[runtime_paths[0]])
        restart_parity = json.loads(raw[f"seed{seed}/restart_parity.json"])
        if not restart_parity or not all(restart_parity.values()):
            raise ValueError("restart correctness gate failed")
        exposure = json.loads(raw[f"seed{seed}/exposure.json"])["rows"]
        exposures[seed] = {
            "sequences": len(exposure),
            "sequence_lengths": [x["sequence_length"] for x in exposure],
            "supervised_tokens": [x["supervised_tokens"] for x in exposure],
            "media_shapes": sorted({json.dumps(x["media_shapes"], sort_keys=True) for x in exposure}),
            "video_grids": sorted({json.dumps(x["video_grid_thw"]) for x in exposure}),
        }
        for restart in range(3):
            for arm in ARMS:
                prefix = f"seed{seed}/restart{restart}_{arm}/"
                complete = json.loads(raw[prefix + "complete.json"])
                attempt = json.loads(raw[prefix + "attempt.json"])
                identities = json.loads(raw[prefix + "checkpoint_digest.json"])
                files = json.loads(raw[prefix + "model-file-reads.json"])["files"]
                if model_files is None:
                    model_files = files
                if model_files != files:
                    raise ValueError("pretrained weight bytes differ across comparison arms")
                paths = [k for k in raw if k.startswith(prefix) and k.endswith("metrics.jsonl")]
                if len(paths) != 1:
                    raise ValueError("missing/ambiguous metrics")
                rows = [json.loads(line) for line in raw[paths[0]].splitlines()]
                if rows[-1].get("optim/optimizer_step") != 40 or rows[-1].get("optim/nonfinite_skips_total") != 0:
                    raise ValueError("incomplete/invalid run")
                if attempt["returncode"] or attempt["canary_instrumented"]:
                    raise ValueError("invalid speed run")
                clock, curve = 0.0, []
                for row in rows:
                    clock += row.get("timer/e2e_step_s", 0) + row.get("timer/validation_s", 0)
                    if "val/loss_avg" in row:
                        launch = datetime.fromisoformat(row["time"]).timestamp() - attempt["start_time_ns"] / 1e9
                        curve.append(
                            {
                                "step": row["step"],
                                "nll": row["val/loss_avg"],
                                "native_seconds": clock,
                                "launch_seconds": launch,
                                "cold_seconds": launch + preparation,
                            }
                        )
                samples = [json.loads(line) for line in raw[prefix + "memory.jsonl"].splitlines()]
                if not samples or any("sample_error" in row for row in samples):
                    raise ValueError("missing/failed memory samples")
                seconds = sum(row.get("timer/e2e_step_s", 0) for row in rows)
                sequences = sum(row.get("train/valid_sequence_count", 0) for row in rows)
                runs[seed, restart, arm] = {
                    "seed": seed,
                    "restart": restart,
                    "arm": arm,
                    "training_seconds": seconds,
                    "validation_seconds": sum(row.get("timer/validation_s", 0) for row in rows),
                    "launch_seconds_after_last_validation": complete["elapsed_seconds"] - curve[-1]["launch_seconds"],
                    "model_file_read_seconds": json.loads(raw[prefix + "model-file-reads.json"])["seconds"],
                    "sequences_per_second": sequences / seconds,
                    "launch_completion_seconds": complete["elapsed_seconds"],
                    "supervised_tokens": sum(row.get("train/valid_token_count", 0) for row in rows),
                    "sequences": sequences,
                    "curve": curve,
                    "device_peak_bytes": max(row["device_used"] for row in samples),
                    "process_gpu_peak_bytes": max(
                        (
                            row["matched_process_gpu_bytes"]
                            for row in samples
                            if row["matched_process_gpu_bytes"] is not None
                        ),
                        default=None,
                    ),
                    "host_pss_peak_bytes": max(row["process_tree_pss"] for row in samples),
                    "checkpoint_tensor_state_bytes": identities["state_bytes"],
                    "observed_sm_clock_mhz_range": [
                        min(x["sm_mhz"] for x in samples),
                        max(x["sm_mhz"] for x in samples),
                    ],
                }
                if sum(x["supervised_tokens"] for x in exposure) != runs[seed, restart, arm]["supervised_tokens"]:
                    raise ValueError("actual supervised token count differs from reconstructed native exposure")
    if len({v["source_archive_sha256"] for v in runtimes.values()}) != 1:
        compatibility = json.loads(raw["source_compatibility.json"])
        records = [compatibility["archive_records"][v["source_archive_sha256"]] for v in runtimes.values()]
        if not compatibility["verified_identical"] or len({json.dumps(v, sort_keys=True) for v in records}) != 1:
            raise ValueError("comparison sources changed native training or launch code")
    if len({json.dumps(v["versions"], sort_keys=True) for v in runtimes.values()}) != 1:
        raise ValueError("comparison seeds used different dependencies")
    paired = {}
    for clock in ("native_seconds", "launch_seconds", "cold_seconds"):
        per_seed_speed, per_seed_equal, restart_values = [], [], []
        for seed in SEEDS:
            speeds, equals = [], []
            for restart in range(3):
                base, quant = (runs[seed, restart, arm] for arm in ARMS)
                if base["supervised_tokens"] != quant["supervised_tokens"] or base["sequences"] != quant["sequences"]:
                    raise ValueError("unmatched data exposure")
                target = 0.95 * base["curve"][0]["nll"]
                crossings = [
                    next((p[clock] for p in run["curve"] if p["nll"] <= target), None) for run in (base, quant)
                ]
                if any(value is None for value in crossings):
                    speeds.append(None)
                else:
                    speeds.append(crossings[0] / crossings[1])
                budget = min(run["curve"][-1][clock] for run in (base, quant))
                points = [
                    max((p for p in run["curve"] if p[clock] <= budget), key=lambda p: p[clock])
                    for run in (base, quant)
                ]
                equals.append(points[0]["nll"] - points[1]["nll"])
                restart_values.append(
                    {
                        "seed": seed,
                        "restart": restart,
                        "target": target,
                        "ttq_seconds": crossings,
                        "equal_budget_seconds": budget,
                        "equal_budget_points": points,
                    }
                )
            per_seed_speed.append(float(np.mean(speeds)) if all(v is not None for v in speeds) else None)
            per_seed_equal.append(float(np.mean(equals)))
        speed = interval(per_seed_speed) if all(v is not None for v in per_seed_speed) else {"mean": None, "ci95": None}
        equal = interval(per_seed_equal)
        paired[clock] = {
            "time_to_quality_ratio": speed,
            "equal_budget_nll_improvement": equal,
            "speed_gate": speed["ci95"] is not None and speed["ci95"][0] > 1.05,
            "equal_budget_quality_gain_gate": equal["ci95"][0] > 0,
            "restarts": restart_values,
        }
    throughput = interval(
        [
            float(
                np.mean(
                    [
                        runs[s, r, ARMS[1]]["sequences_per_second"] / runs[s, r, ARMS[0]]["sequences_per_second"]
                        for r in range(3)
                    ]
                )
            )
            for s in SEEDS
        ]
    )
    completion = interval(
        [
            float(
                np.mean(
                    [
                        runs[s, r, ARMS[0]]["launch_completion_seconds"]
                        / runs[s, r, ARMS[1]]["launch_completion_seconds"]
                        for r in range(3)
                    ]
                )
            )
            for s in SEEDS
        ]
    )
    final = {}
    score_rows = {}
    for seed in SEEDS:
        for arm in ARMS:
            prefix = f"seed{seed}/final_{arm}/"
            final[seed, arm] = json.loads(raw[prefix + "summary.json"])
            score_rows[seed, arm] = [json.loads(line) for line in raw[prefix + "scores.jsonl"].splitlines()]
            expected_checkpoint = json.loads(raw[f"seed{seed}/restart0_{arm}/complete.json"])["checkpoint"]
            if (
                not final[seed, arm]["complete"]
                or final[seed, arm]["checkpoint"] != expected_checkpoint
                or final[seed, arm]["seed"] != seed
            ):
                raise ValueError("final score commit is incomplete or belongs to another checkpoint/seed")
    frozen = [json.loads(line) for line in raw["seed17/final_frozen/scores.jsonl"].splitlines()]
    ids = [row["id"] for row in frozen]
    for rows in score_rows.values():
        if [row["id"] for row in rows] != ids or [row["input_sha256"] for row in rows] != [
            row["input_sha256"] for row in frozen
        ]:
            raise ValueError("final scoring input mismatch")
    nll_ratio = interval(
        [final[s, ARMS[1]]["metrics"]["answer_nll"] / final[s, ARMS[0]]["metrics"]["answer_nll"] for s in SEEDS]
    )
    accuracy_drop = interval(
        [
            final[s, ARMS[0]]["metrics"]["restricted_choice_accuracy"]
            - final[s, ARMS[1]]["metrics"]["restricted_choice_accuracy"]
            for s in SEEDS
        ]
    )
    development_nll_ratio = interval(
        [
            float(
                np.mean(
                    [
                        runs[s, r, ARMS[1]]["curve"][-1]["nll"] / runs[s, r, ARMS[0]]["curve"][-1]["nll"]
                        for r in range(3)
                    ]
                )
            )
            for s in SEEDS
        ]
    )
    # Crossed recording/seed bootstrap, fixed before all outcomes.
    rng = np.random.default_rng(2026)
    quant_nll = np.asarray([[row["answer_nll"] for row in score_rows[s, ARMS[1]]] for s in SEEDS])
    base_nll = np.asarray([[row["answer_nll"] for row in score_rows[s, ARMS[0]]] for s in SEEDS])
    quant_acc = np.asarray(
        [[row["restricted_choice_correct"] for row in score_rows[s, ARMS[1]]] for s in SEEDS], dtype=float
    )
    base_acc = np.asarray(
        [[row["restricted_choice_correct"] for row in score_rows[s, ARMS[0]]] for s in SEEDS], dtype=float
    )
    frozen_acc = np.asarray([row["restricted_choice_correct"] for row in frozen], dtype=float)
    boot = {"nll_ratio": [], "accuracy_drop": [], "adaptation": []}
    for _ in range(10000):
        seeds, groups = rng.integers(0, 3, 3), rng.integers(0, len(ids), len(ids))
        index = np.ix_(seeds, groups)
        boot["nll_ratio"].append(float(quant_nll[index].mean() / base_nll[index].mean()))
        boot["accuracy_drop"].append(float((base_acc - quant_acc)[index].mean()))
        boot["adaptation"].append(float(quant_acc[index].mean() - frozen_acc[groups].mean()))
    bootstrap = {key: {"ci95": np.quantile(values, [0.025, 0.975]).tolist()} for key, values in boot.items()}
    saving = (
        1
        - runs[17, 0, ARMS[1]]["checkpoint_tensor_state_bytes"] / runs[17, 0, ARMS[0]]["checkpoint_tensor_state_bytes"]
    )
    memory = {}
    for metric in ("device_peak_bytes", "process_gpu_peak_bytes", "host_pss_peak_bytes"):
        if any(run[metric] is None for run in runs.values()):
            raise ValueError(f"missing memory measurement: {metric}")
        memory[metric] = {
            arm: interval([float(np.mean([runs[s, r, arm][metric] for r in range(3)])) for s in SEEDS]) for arm in ARMS
        }
        memory[metric]["paired_reduction_fraction"] = interval(
            [
                float(np.mean([1 - runs[s, r, ARMS[1]][metric] / runs[s, r, ARMS[0]][metric] for r in range(3)]))
                for s in SEEDS
            ]
        )
    allocation = {name: json.loads(raw[f"canary-v7/{name}/state.json"]) for name in ("prepared_fused", "prepared_8bit")}
    allocation = {
        name: {k: value[k] for k in ("state_report", "peak_allocated", "peak_reserved")}
        for name, value in allocation.items()
    }
    gates = {
        "development_nll": development_nll_ratio["ci95"][1] <= 1.01,
        "task_nll": nll_ratio["ci95"][1] <= 1.01 and bootstrap["nll_ratio"]["ci95"][1] <= 1.01,
        "task_accuracy": accuracy_drop["ci95"][1] <= 0.01 and bootstrap["accuracy_drop"]["ci95"][1] <= 0.01,
        "positive_adaptation": bootstrap["adaptation"]["ci95"][0] > 0,
        "training_throughput": throughput["ci95"][0] > 1.05,
        "launch_time_to_quality": paired["launch_seconds"]["speed_gate"],
        "cold_time_to_quality": paired["cold_seconds"]["speed_gate"],
        "state_saving": saving >= 0.6,
        "equal_wall_budget_quality_gain": paired["launch_seconds"]["equal_budget_quality_gain_gate"],
    }
    unresolved_release_evidence = [
        "No historically untouched evaluation pool was supplied; approved reused groups support this disclosed comparison only.",
        "Equal-time quality uses development validation checkpoints; final task scores exist at matched update40, not at both arms' equal-time stopping points.",
    ]
    report = {
        "complete_runs": len(runs),
        "paired_seeds": 3,
        "restarts_per_seed": 3,
        "preparation_cpu_seconds": preparation,
        "model_files": model_files,
        "throughput_ratio": throughput,
        "launch_completion_ratio": completion,
        "clocks": paired,
        "task_nll_ratio": nll_ratio,
        "development_nll_ratio": development_nll_ratio,
        "task_accuracy_drop": accuracy_drop,
        "crossed_recording_seed_bootstrap": bootstrap,
        "checkpoint_tensor_state_saving": saving,
        "allocation_canary": allocation,
        "memory": memory,
        "exposure": exposures,
        "runtimes": runtimes,
        "gates": gates,
        "numeric_gates_pass": all(gates.values()),
        "unresolved_release_evidence": unresolved_release_evidence,
        "release_qualified": all(gates.values()) and not unresolved_release_evidence,
        "runs": list(runs.values()),
        "final_metrics": {f"seed{s}_{a}": v["metrics"] for (s, a), v in final.items()},
        "limitations": [
            "Historically reused QLoRA evaluation pool; newly selected campaign groups are not an untouched-pool claim.",
            "Device peaks are 250ms NVML sampled lower bounds; host PSS excludes unmapped host allocations.",
            "Quality at equal time uses the last completed validation, without interpolation or extra tuning.",
            "Model score is teacher-forced, not free generation.",
            "Launch time after the last validation includes checkpoint writing and teardown; it is not a synchronized checkpoint-only timer.",
            "Model-file reads are identical controls included in each launch clock; shared filesystem contention is retained.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, label in zip(ARMS, ("Fused FP32 AdamW + recomputation", "8-bit AdamW, no recomputation"), strict=True):
        for step_index in range(9):
            x, y = (
                interval([float(np.mean([runs[s, r, arm]["curve"][step_index][key] for r in range(3)])) for s in SEEDS])
                for key in ("launch_seconds", "nll")
            )
            axes[0].errorbar(
                x["mean"],
                y["mean"],
                xerr=[[x["mean"] - x["ci95"][0]], [x["ci95"][1] - x["mean"]]],
                yerr=[[y["mean"] - y["ci95"][0]], [y["ci95"][1] - y["mean"]]],
                fmt="o",
                color="C0" if arm == ARMS[0] else "C1",
                label=label if step_index == 0 else None,
            )
        x = interval([float(np.mean([runs[s, r, arm]["sequences_per_second"] for r in range(3)])) for s in SEEDS])
        y = interval([float(np.mean([runs[s, r, arm]["device_peak_bytes"] / 2**30 for r in range(3)])) for s in SEEDS])
        axes[1].errorbar(
            x["mean"],
            y["mean"],
            xerr=[[x["mean"] - x["ci95"][0]], [x["ci95"][1] - x["mean"]]],
            yerr=[[y["mean"] - y["ci95"][0]], [y["ci95"][1] - y["mean"]]],
            fmt="o",
            label=label,
        )
    axes[0].set(xlabel="Launch seconds, prepared corpus", ylabel="Development supervised-token NLL")
    axes[1].set(xlabel="Native training sequences / second", ylabel="Sampled whole-device peak GiB")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("16-block native video SFT · matched effective batch4 · 3 seeds, 95% t intervals")
    fig.tight_layout()
    fig.savefig(args.output / "deployment.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"gates": gates, "release_qualified": report["release_qualified"]}, indent=2))


if __name__ == "__main__":
    main()
