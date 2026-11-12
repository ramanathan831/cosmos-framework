"""Apply prospectively locked development gates, keeping the short study separate."""

import json
import sys
from pathlib import Path
from statistics import mean

from tools.quantized_optimizers.outcome_statistics import paired_quality

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "quantized_optimizers_round8"))
from analyze_screen import IDENTITY, read_raw, rows


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
            metrics = rows(
                raw, next(k for k in raw if k.startswith(prefix + "native/") and k.endswith("/metrics.jsonl"))
            )
            updates = [x for x in metrics if "train/valid_token_count" in x]
            assert len(updates) == 174
            assert all(
                x["optim/update_outcome"] == "applied" and x["optim/nonfinite_skips_total"] == 0 for x in updates
            )
            assert [x["optim/optimizer_step"] for x in updates] == list(range(1, 175))
            complete = json.loads(raw[prefix + "diagnostics/complete.json"])
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
    print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=2))


if __name__ == "__main__":
    main()
