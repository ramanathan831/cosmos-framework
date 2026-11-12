"""Apply prospectively locked development gates, keeping the short study separate."""

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from tools.quantized_optimizers.outcome_statistics import paired_quality

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "quantized_optimizers_round8"))
from analyze_replay import components
from analyze_screen import IDENTITY, read_raw, rows


def descriptive_changes(raw, prefix, task_rows):
    """Retain every development row; component views never replace native guards."""
    prior = ROOT.parent / "quantized_optimizers_round8"
    texts = {x["id"]: x for x in json.loads((prior / "development_texts.json").read_text())["rows"]}
    vocabulary = json.loads((prior / "token_strings.json").read_text())["raw_vocabulary_strings"]
    observations = rows(raw, prefix + "diagnostics/late_substantive.jsonl")
    assert [x["step"] for x in observations] == [87, 174]
    parts = []
    for observation in observations:
        assert observation["batch_size"] == 4 and observation["native_tokens"] == 1346
        grouped = defaultdict(lambda: dict(tokens=0, loss_sum=0.0))
        for batch in observation["batches"]:
            for row in batch["rows"]:
                labels = components(row, texts[row["id"]], vocabulary)
                for loss, (question, component) in zip(row["token_losses"], labels, strict=True):
                    cell = grouped[question + " / " + component]
                    cell["tokens"] += 1
                    cell["loss_sum"] += loss
        assert sum(v["tokens"] for v in grouped.values()) == 1346
        assert math.isclose(
            sum(v["loss_sum"] for v in grouped.values()) / 1346, observation["native_loss"], abs_tol=1e-7
        )
        parts.append(dict(step=observation["step"], native_loss=observation["native_loss"], components=dict(grouped)))
    earlier = {x["id"]: x for x in task_rows if x["step"] == 87}
    changes = []
    for row in task_rows:
        if row["step"] != 174:
            continue
        previous = earlier[row["id"]]
        assert all(row[k] == previous[k] for k in IDENTITY if k != "step")
        changes.append(
            dict(
                id=row["id"],
                gold=row["gold"],
                nll=[previous["answer_nll"], row["answer_nll"]],
                correct=[previous["restricted_choice_correct"], row["restricted_choice_correct"]],
                prediction=[previous["restricted_choice_prediction"], row["restricted_choice_prediction"]],
                candidate_logits=[previous["candidate_logits"], row["candidate_logits"]],
            )
        )
    assert len(changes) == 32
    return dict(native_substantive_components=parts, task_changes_87_to_174=changes)


def plot_curves(all_seeds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for arm, color, label in (
        ("fused", "#4477AA", "Fused + recomputation"),
        ("eightbit", "#CC6677", "Eight-bit, no recomputation"),
    ):
        for panel, (key, title) in enumerate(
            (
                ("answer_nll", "Choice gold-answer NLL"),
                ("restricted_choice_accuracy", "Restricted-choice accuracy"),
                ("val/loss_avg", "Native substantive NLL"),
            )
        ):
            traces = []
            for seed in all_seeds:
                curve = seed["arms"][arm]["curve"]
                steps = [x["step"] for x in curve]
                values = [x["substantive_validation" if panel == 2 else "metrics"][key] for x in curve]
                traces.append(values)
                axes[panel].plot(steps, values, color=color, alpha=0.25, linewidth=1)
            axes[panel].plot(steps, np.mean(traces, axis=0), color=color, marker="o", label=label)
            axes[panel].set(xlabel="Completed optimizer updates", title=title)
            axes[panel].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Two complete passes: three development seeds, fixed update-174 endpoint")
    fig.text(
        0.5,
        0.01,
        "Thin lines: individual seeds; bold: mean. Not final held-out quality or product timing evidence.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(ROOT / "full_exposure_curves.png", dpi=160)
    plt.close(fig)


def main():
    all_seeds, curves = [], {}
    paired = {"fused": [], "eightbit": []}
    frozen = None
    metadata = json.loads((ROOT / "development_manifest.json").read_text())["rows"]
    groups = {row["id"]: row["group"] for row in metadata}
    for seed in (17, 29, 43):
        raw = read_raw(ROOT / f"full_seed{seed}_raw.json.gz")
        assert "complete.json" in raw
        arms = {}
        for arm in ("fused", "eightbit"):
            prefix = f"{arm}_seed{seed}/"
            assert json.loads(raw[prefix + "attempt.json"])["returncode"] == 0
            curve = rows(raw, prefix + "diagnostics/task_curve.jsonl")
            assert [x["step"] for x in curve] == [0, 29, 58, 87, 116, 145, 174]
            assert all(x["substantive_validation"]["val/valid_token_count"] == 1346 for x in curve)
            inputs = rows(raw, prefix + "diagnostics/training_inputs.jsonl")
            assert [len(x["ids"]) for x in inputs] == ([4] * 86 + [2]) * 2
            for start in (0, 87):
                ids = [i for x in inputs[start : start + 87] for i in x["ids"]]
                assert len(ids) == len(set(ids)) == 346
                expected = list(range(346))
                random.Random(seed + start // 87).shuffle(expected)
                assert [int(i.rsplit(":", 1)[1]) for i in ids] == expected
            metrics = rows(
                raw, next(k for k in raw if k.startswith(prefix + "native/") and k.endswith("/metrics.jsonl"))
            )
            updates = [x for x in metrics if "train/valid_token_count" in x]
            assert len(updates) == 174
            assert all(
                x["optim/update_outcome"] == "applied" and x["optim/nonfinite_skips_total"] == 0 for x in updates
            )
            assert [x["optim/optimizer_step"] for x in updates] == list(range(1, 175))
            assert [x["train/valid_sequence_count"] for x in updates] == ([4] * 86 + [2]) * 2
            complete = json.loads(raw[prefix + "diagnostics/complete.json"])
            assert complete["complete"] and complete["trainable_elements"] == 3087142912
            state = complete["state_report"]
            if arm == "eightbit":
                assert state["uninitialized_elements"] == 0 and state["paged_moment_bytes"] == 0
                assert state["quantized_elements"] + state["fp32_elements"] == 3087142912
                assert state["backend"]["name"] == "adamw_bnb_8bit" and state["backend"]["version"] == "0.49.2"
                tensor_state_bytes = state["moment_bytes"] + state["metadata_bytes"]
            else:
                tensor_state_bytes = state["state_bytes"]
            mem = rows(raw, prefix + "memory.jsonl")
            phases = rows(raw, prefix + "diagnostics/phases.jsonl")
            arms[arm] = dict(
                curve=curve,
                task_rows=rows(raw, prefix + "diagnostics/task_rows.jsonl"),
                inputs=inputs,
                task_nll=curve[-1]["metrics"]["answer_nll"],
                accuracy=curve[-1]["metrics"]["restricted_choice_accuracy"],
                substantive_nll=curve[-1]["substantive_validation"]["val/loss_avg"],
                complete=complete,
                optimizer_tensor_state_bytes=tensor_state_bytes,
                attempt=json.loads(raw[prefix + "attempt.json"]),
                memory_peaks={
                    k: max(x.get(k) or 0 for x in mem)
                    for k in ("device_used", "process_tree_pss", "matched_process_gpu_bytes")
                },
                phase_seconds={
                    k: sum(x["seconds"] for x in phases if x["phase"] == k)
                    for k in sorted({x["phase"] for x in phases})
                },
            )
            arms[arm]["descriptive_changes"] = descriptive_changes(raw, prefix, arms[arm]["task_rows"])
            curves[f"{arm}{seed}"] = curve
        a, b = arms["fused"], arms["eightbit"]
        assert a["inputs"] == b["inputs"]
        for x, y in zip(a["task_rows"], b["task_rows"], strict=True):
            assert all(x[k] == y[k] for k in IDENTITY)
            if x["step"] == 0:
                assert x == y
        if seed == 17:
            for arm in arms:
                assert json.loads(raw[arm + "_recovery_verified.json"])["complete"]
        for arm in arms:
            records = arms[arm]["task_rows"]
            initial = [dict(row, group=groups[row["id"]]) for row in records if row["step"] == 0]
            if frozen is None:
                frozen = initial
            else:
                assert frozen == initial
            paired[arm].append([dict(row, group=groups[row["id"]]) for row in records if row["step"] == 174])
            arms[arm].pop("inputs")
            arms[arm].pop("task_rows")
        all_seeds.append(
            dict(
                seed=seed,
                arms=arms,
                nll_ratio=b["task_nll"] / a["task_nll"],
                accuracy_drop=a["accuracy"] - b["accuracy"],
                substantive_ratio=b["substantive_nll"] / a["substantive_nll"],
                optimizer_tensor_state_saving=1 - b["optimizer_tensor_state_bytes"] / a["optimizer_tensor_state_bytes"],
            )
        )
    quality = paired_quality(paired["fused"], paired["eightbit"], frozen)
    nll_ci = quality["nll_ratio"]["ci95"]
    acc_ci = quality["accuracy_drop"]["ci95"]
    adaptation = {
        arm: mean(x["arms"][arm]["task_nll"] for x in all_seeds)
        < mean(x["arms"][arm]["curve"][0]["metrics"]["answer_nll"] for x in all_seeds)
        and mean(x["arms"][arm]["accuracy"] for x in all_seeds)
        > mean(x["arms"][arm]["curve"][0]["metrics"]["restricted_choice_accuracy"] for x in all_seeds)
        for arm in ("fused", "eightbit")
    }
    gates = dict(
        nll_upper_le_1_01=quality["guards"]["nll"],
        accuracy_drop_upper_le_01=quality["guards"]["accuracy"],
        each_substantive_ratio_le_1_01=all(x["substantive_ratio"] <= 1.01 for x in all_seeds),
        retained_state_saving_ge_60=all(x["optimizer_tensor_state_saving"] >= 0.60 for x in all_seeds),
        adaptation=all(adaptation.values())
        and quality["guards"]["positive_baseline_adaptation"]
        and quality["guards"]["positive_quantized_adaptation"],
    )
    result = dict(
        complete=True,
        study="Two-pass development, update174 fixed endpoint; not original15-update or final held-out qualification",
        per_seed=all_seeds,
        paired_seed_family_quality=quality,
        nll_ratio_ci95=nll_ci,
        accuracy_degradation_ci95=acc_ci,
        adaptation=adaptation,
        gates=gates,
        promote=all(gates.values()),
        timing_scope="Instrumented development includes native attribution and seed17 recovery digests. No product speed inference; phase timers nest, do not sum them.",
    )
    (ROOT / "full_exposure_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_curves(all_seeds)
    print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=2))


if __name__ == "__main__":
    main()
