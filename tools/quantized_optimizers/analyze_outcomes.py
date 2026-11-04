"""Analyze actual committed checkpoints under the locked outcome protocol.

Retains the historical analyses unchanged. Missing or ineligible budget states
are censored; a later checkpoint or development curve is never substituted.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.quantized_optimizers.analyze_development import interval
from tools.quantized_optimizers.outcome_statistics import paired_quality
from tools.quantized_optimizers.plan_outcomes import ARMS
from tools.quantized_optimizers.score_dense import aggregate


def analyze(raw, protocol, selection, protocol_sha256, cold_preparation_seconds):
    seeds = protocol["paired_seeds"]

    def find(seed, ending):
        keys = [
            k
            for k in raw
            if (k.startswith(f"final-seed{seed}/") or k.startswith(f"final-seed{seed}-")) and k.endswith(ending)
        ]
        if len(keys) != 1:
            raise ValueError(f"missing or ambiguous completed evidence: seed{seed} {ending}: {keys}")
        return keys[0]

    def lines(key):
        return [json.loads(line) for line in raw[key].splitlines()]

    frozen = lines(find(17, "/frozen/score/scores.jsonl"))
    frozen_summary = json.loads(raw[find(17, "/frozen/score/summary.json")])
    if (
        not frozen_summary["complete"]
        or frozen_summary["checkpoint"] is not None
        or frozen_summary["selection_sha256"] != protocol["evaluation"]["manifest_sha256"]
        or frozen_summary["metrics"] != aggregate(frozen)
    ):
        raise ValueError("frozen reference identity or aggregate mismatch")
    ids = [row["id"] for row in selection["rows"]]
    if [(row["id"], row["group"]) for row in frozen] != [(row["id"], row["group"]) for row in selection["rows"]]:
        raise ValueError("frozen score does not match locked final manifest")
    designs = ["matched", "training", "end_to_end", *[f"target_restart{i}" for i in range(3)]]
    runs, scores, identities, models = {}, {}, {}, []
    for seed in seeds:
        exposure = json.loads(raw[find(seed, "/training_exposure/exposure.json")])["rows"]
        for design in designs:
            for arm in ARMS:
                name = f"{design}_{arm}"
                summary_path = find(seed, f"/{name}/score/summary.json")
                prefix = summary_path.removesuffix("score/summary.json")
                root = prefix.removesuffix(name + "/")
                spec = json.loads(raw[root + "executed_spec.json"])
                if spec.get("protocol_sha256") != protocol_sha256 or not spec.get("locked_before_evaluation"):
                    raise ValueError("result does not use the frozen comparison protocol")
                runtime_keys = [k for k in raw if k.startswith(root + "runtime-") and k.endswith(".json")]
                if len(runtime_keys) != 1:
                    raise ValueError("ambiguous allocation runtime")
                runtime = json.loads(raw[runtime_keys[0]])
                identities[root] = runtime
                models.append(json.loads(raw[prefix + "model-file-reads.json"])["files"])
                attempt = json.loads(raw[prefix + "attempt.json"])
                summary = json.loads(raw[summary_path])
                delivered = json.loads(raw[prefix + "delivery.json"])
                if attempt["returncode"] or not summary["complete"]:
                    raise ValueError("failed training/scoring cannot enter a comparison")
                metric_keys = [k for k in raw if k.startswith(prefix + "native/") and k.endswith("metrics.jsonl")]
                if len(metric_keys) != 1:
                    raise ValueError("missing native metrics")
                metrics = lines(metric_keys[0])
                steps = [row for row in metrics if "optim/optimizer_step" in row]
                last = int(steps[-1]["optim/optimizer_step"])
                if [int(row["optim/optimizer_step"]) for row in steps] != list(range(1, last + 1)) or any(
                    row["optim/nonfinite_skips_total"] for row in steps
                ):
                    raise ValueError("missing or skipped native updates")
                tokens = sum(row.get("train/valid_token_count", 0) for row in steps)
                sequences = sum(row.get("train/valid_sequence_count", 0) for row in steps)
                actual_exposure = [row for row in exposure if row["update"] <= last]
                if sequences != len(actual_exposure) or tokens != sum(
                    row["supervised_tokens"] for row in actual_exposure
                ):
                    raise ValueError("native sequence/token totals disagree with actual data order")
                budget = delivered["budget_report"]
                eligible = True
                if design == "matched" and last != 15:
                    raise ValueError("matched-update recipe did not complete15 updates")
                if budget:
                    if budget["committed_optimizer_step"] != last:
                        raise ValueError("scored state is not the recorded actual budget update")
                    eligible = budget["training_checkpoint_eligible"]
                    if design == "end_to_end":
                        eligible = eligible and delivered["end_to_end_budget_eligible"]
                    if design.startswith("target"):
                        eligible = eligible and budget["target_hit"] is not None
                score_rows = lines(prefix + "score/scores.jsonl")
                if [row["id"] for row in score_rows] != ids:
                    raise ValueError("comparison changes the fixed task manifest")
                if (
                    summary["selection_sha256"] != protocol["evaluation"]["manifest_sha256"]
                    or summary["checkpoint"] != delivered["checkpoint"]
                    or summary["metrics"] != aggregate(score_rows)
                ):
                    raise ValueError("scored checkpoint/manifest/aggregate mismatch")
                for key in ("corpus_sha256", "objective_identity", "processor_identity", "metric_contract", "versions"):
                    if summary[key] != frozen_summary[key]:
                        raise ValueError(f"scoring contract differs from frozen reference: {key}")
                for reference, scored in zip(frozen, score_rows, strict=True):
                    for key in (
                        "group",
                        "gold",
                        "candidate_token_ids",
                        "media_sha256",
                        "input_sha256",
                        "sequence_length",
                        "native_supervised_tokens",
                        "answer_token_position",
                        "prediction_logit_position",
                    ):
                        if reference[key] != scored[key]:
                            raise ValueError(f"scoring input/label differs from frozen reference: {key}")
                scores[seed, design, arm] = score_rows
                memory = lines(prefix + "memory.jsonl")
                score_memory = lines(prefix + "score-memory.jsonl")
                if any("sample_error" in row for row in memory + score_memory):
                    raise ValueError("failed external memory sampling")
                complete = json.loads(raw[prefix + "diagnostics/complete.json"])
                state = complete["state_report"]
                state_bytes = state.get("state_bytes", state.get("moment_bytes", 0) + state.get("metadata_bytes", 0))
                curve = lines(prefix + "diagnostics/task_curve.jsonl")
                training_seconds = sum(row.get("timer/e2e_step_s", 0) for row in steps)
                record = {
                    "seed": seed,
                    "design": design,
                    "arm": arm,
                    "prefix": prefix,
                    "optimizer_step": last,
                    "eligible": bool(eligible),
                    "budget": budget,
                    "metrics": summary["metrics"],
                    "training_seconds": training_seconds,
                    "training_validation_seconds": budget["target_hit"]["clock_seconds"]
                    - budget["loop_start_clock_seconds"]
                    if budget and budget["target_hit"]
                    else training_seconds + sum(row.get("timer/validation_s", 0) for row in metrics),
                    "sequences_per_second": sequences / training_seconds,
                    "sequences": sequences,
                    "supervised_tokens": tokens,
                    "sequence_lengths": [row["sequence_length"] for row in actual_exposure],
                    "media_shapes": sorted(
                        {json.dumps(row["media_shapes"], sort_keys=True) for row in actual_exposure}
                    ),
                    "prepared_delivery_seconds": delivered["prepared_launch_to_scored_checkpoint_seconds"],
                    "cold_delivery_seconds": delivered["prepared_launch_to_scored_checkpoint_seconds"]
                    + cold_preparation_seconds,
                    "scoring_seconds": delivered["scoring_seconds"],
                    "state_bytes": state_bytes,
                    "allocator_peak_bytes": complete["peak_allocated"],
                    "substantive_dev_nll": curve[-1]["substantive_validation"]["val/loss_avg"],
                    "task_dev_nll": curve[-1]["metrics"]["answer_nll"],
                    "phases": lines(prefix + "diagnostics/phases.jsonl"),
                    "checkpoint_provenance": summary["checkpoint_provenance"],
                }
                for scope, samples in (("training", memory), ("delivery", memory + score_memory)):
                    for key in ("device_used", "matched_process_gpu_bytes", "process_tree_pss", "process_tree_rss"):
                        values = [row[key] for row in samples if row.get(key) is not None]
                        record[f"{scope}_{key}_peak"] = max(values) if values else None
                runs[seed, design, arm] = record
    if any(model != models[0] for model in models):
        raise ValueError("pretrained model bytes differ across arms")
    if (
        len({json.dumps(r["versions"], sort_keys=True) for r in identities.values()}) != 1
        or len({r["source_archive_sha256"] for r in identities.values()}) != 1
    ):
        raise ValueError("source/dependencies differ across final comparisons")
    quality = {}
    for design in ("matched", "training", "end_to_end", "target_restart0"):
        if not all(runs[s, design, a]["eligible"] for s in seeds for a in ARMS):
            quality[design] = {
                "all_checkpoints_eligible": False,
                "equal_budget_gain_gate": False,
                "censored": [
                    runs[s, design, a]["prefix"] for s in seeds for a in ARMS if not runs[s, design, a]["eligible"]
                ],
                "reason": "No paired budget estimate: at least one arm lacks an eligible actual state. Raw scores remain observations of those saved states only.",
            }
            continue
        result = paired_quality(
            [scores[s, design, ARMS[0]] for s in seeds], [scores[s, design, ARMS[1]] for s in seeds], frozen
        )
        result["all_checkpoints_eligible"] = all(runs[s, design, a]["eligible"] for s in seeds for a in ARMS)
        result["updates"] = {a: [runs[s, design, a]["optimizer_step"] for s in seeds] for a in ARMS}
        result["equal_budget_gain_gate"] = (
            result["all_checkpoints_eligible"]
            and result["guards"]["equal_budget_nll_gain"]
            and result["guards"]["accuracy"]
        )
        quality[design] = result
    timing = {}
    for clock in ("prepared_delivery_seconds", "cold_delivery_seconds", "training_validation_seconds"):
        per_seed, repeats = [], []
        for seed in seeds:
            paired = []
            for restart in range(3):
                base, quant = (runs[seed, f"target_restart{restart}", a] for a in ARMS)
                if base["eligible"] and quant["eligible"]:
                    paired.append(base[clock] / quant[clock])
            per_seed.append(float(np.mean(paired)) if len(paired) == 3 else None)
            repeats.append(paired)
        report = interval(per_seed) if None not in per_seed else {"mean": None, "ci95": None, "n": 0}
        timing[clock] = {
            **report,
            "paired_restarts_by_seed": repeats,
            "speed_gate": report["ci95"] is not None and report["ci95"][0] > 1.05,
        }
    dev = {
        metric: interval([runs[s, "matched", ARMS[1]][metric] / runs[s, "matched", ARMS[0]][metric] for s in seeds])
        for metric in ("substantive_dev_nll", "task_dev_nll")
    }
    saving = 1 - runs[17, "matched", ARMS[1]]["state_bytes"] / runs[17, "matched", ARMS[0]]["state_bytes"]
    throughput = interval(
        [
            float(
                np.mean(
                    [
                        runs[s, f"target_restart{r}", ARMS[1]]["sequences_per_second"]
                        / runs[s, f"target_restart{r}", ARMS[0]]["sequences_per_second"]
                        for r in range(3)
                    ]
                )
            )
            for s in seeds
        ]
    )
    gates = {
        "matched_task_nll": quality["matched"]["guards"]["nll"],
        "matched_task_accuracy": quality["matched"]["guards"]["accuracy"],
        "substantive_development_nll": dev["substantive_dev_nll"]["ci95"][1] <= 1.01,
        "positive_baseline_adaptation": quality["matched"]["guards"]["positive_baseline_adaptation"],
        "positive_quantized_adaptation": quality["matched"]["guards"]["positive_quantized_adaptation"],
        "equal_training_budget_quality_gain": quality["training"]["equal_budget_gain_gate"],
        "equal_product_budget_quality_gain": quality["end_to_end"]["equal_budget_gain_gate"],
        "prepared_end_to_end_time_to_quality": timing["prepared_delivery_seconds"]["speed_gate"],
        "cold_end_to_end_time_to_quality": timing["cold_delivery_seconds"]["speed_gate"],
        "optimizer_state_saving": saving >= 0.6,
        "repeated_native_throughput": throughput["ci95"][0] > 1.05,
    }
    historical = []
    for selection_part in selection["historical_selections"]:
        subset = set(selection_part["ids"])
        historical.append(
            {
                "selection_sha256": selection_part["sha256"],
                "frozen_accuracy": float(
                    np.mean([r["restricted_choice_correct"] for r in frozen if r["id"] in subset])
                ),
                "matched_accuracy": {
                    a: [
                        float(
                            np.mean(
                                [r["restricted_choice_correct"] for r in scores[s, "matched", a] if r["id"] in subset]
                            )
                        )
                        for s in seeds
                    ]
                    for a in ARMS
                },
            }
        )
    return {
        "protocol_sha256": protocol_sha256,
        "complete_training_runs": len(runs),
        "quality": quality,
        "time_to_quality": timing,
        "repeated_native_throughput_ratio": throughput,
        "development": dev,
        "state_saving_fraction": saving,
        "gates": gates,
        "numeric_gates_pass": all(gates.values()),
        "cold_preparation_seconds": cold_preparation_seconds,
        "cold_clock_limitation": "Common measured full preparation cost; one full preparation measurement does not establish its between-run uncertainty.",
        "historical_subsets": historical,
        "runtime_identities": identities,
        "model_files": models[0],
        "runs": list(runs.values()),
        "independent_replication_qualified": False,
        "evaluation_limitation": "Authorized reused pool; recording groups disjoint from training/development but historically exposed. No independent replication claim.",
    }


def plots(report, output):
    runs = report["runs"]
    colors = {ARMS[0]: "#475569", ARMS[1]: "#0d9488"}
    labels = {ARMS[0]: "Fused FP32 AdamW", ARMS[1]: "Eight-bit AdamW, no recomputation"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for arm in ARMS:
        for design, marker in (("matched", "o"), ("training", "s"), ("end_to_end", "^")):
            if not report["quality"][design]["all_checkpoints_eligible"]:
                continue
            selected = [r for r in runs if r["arm"] == arm and r["design"] == design]
            x = interval([r["prepared_delivery_seconds"] for r in selected])
            y = interval([r["metrics"]["answer_nll"] for r in selected])
            ax.errorbar(
                x["mean"],
                y["mean"],
                xerr=x["mean"] - x["ci95"][0],
                yerr=y["mean"] - y["ci95"][0],
                fmt=marker,
                color=colors[arm],
                capsize=3,
                label=f"{labels[arm]} · {design}",
            )
    ax.set(
        xlabel="Seconds from prepared launch to scored committed checkpoint",
        ylabel="Held-out gold-answer NLL (lower is better)",
        title="Actual states · 3 paired seeds · 95% t intervals\n128 questions /32 reused recording groups; censored states remain in the report",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "quality_time.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm in ARMS:
        selected = [r for r in runs if r["arm"] == arm and r["design"] == "matched"]
        x = interval([r["sequences_per_second"] for r in selected])
        for axis, key, title in zip(
            axes,
            ("training_device_used_peak", "delivery_process_tree_pss_peak"),
            ("Whole-device peak", "Host PSS peak through final scoring"),
            strict=True,
        ):
            y = interval([r[key] / 2**30 for r in selected])
            axis.errorbar(
                x["mean"],
                y["mean"],
                xerr=x["mean"] - x["ci95"][0],
                yerr=y["mean"] - y["ci95"][0],
                fmt="o",
                color=colors[arm],
                capsize=3,
                label=labels[arm],
            )
            axis.set(xlabel="Native training sequences/s", ylabel="GiB", title=title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Matched15 updates · effective batch4 · A10080GB · 3 seeds,95% t intervals")
    fig.tight_layout()
    fig.savefig(output / "memory_throughput.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cold-preparation-seconds", type=float, required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    report = analyze(
        json.load(gzip.open(args.raw, "rt")),
        json.loads(protocol_bytes),
        json.loads(args.selection.read_text()),
        hashlib.sha256(protocol_bytes).hexdigest(),
        args.cold_preparation_seconds,
    )
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plots(report, args.output)
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
