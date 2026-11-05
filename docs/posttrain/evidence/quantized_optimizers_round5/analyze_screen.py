"""Apply the unchanged three-seed development guards to one admitted stage.

References only native development diagnostics inside the sealed round-three
archive, never its final scoring files. No thresholds are inferred from results.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).parent
IDENTITY = (
    "id",
    "step",
    "gold",
    "candidate_token_ids",
    "answer_token_position",
    "prediction_logit_position",
    "native_supervised_tokens",
    "sequence_length",
    "inputs",
    "input_sha256",
)


def read_raw(path):
    raw = json.loads(gzip.decompress(path.read_bytes()))
    for name, receipt in json.loads(raw["_collection.json"])["files"].items():
        value = raw[name].encode()
        assert hashlib.sha256(value).hexdigest() == receipt["sha256"]
        assert len(value) == receipt["bytes"]
    return raw


def rows(raw, key):
    return [json.loads(line) for line in raw[key].splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("prefix")
    parser.add_argument("arm", choices=("adamw", "adamw_bnb_8bit"))
    parser.add_argument("--lr-ratio", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = read_raw(args.raw)
    control = read_raw(ROOT.parent / "quantized_optimizers_round3/outcomes_raw.json.gz")
    result = []
    for seed in (17, 29, 43):
        prefix = f"{args.prefix}_seed{seed}/"
        old = f"final-seed{seed}/matched_{args.arm}/"
        attempt = json.loads(raw[prefix + "attempt.json"])
        assert attempt["returncode"] == 0
        a = rows(raw, prefix + "diagnostics/task_rows.jsonl")
        b = rows(control, old + "diagnostics/task_rows.jsonl")
        assert len(a) == len(b) == 128
        for x, y in zip(a, b, strict=True):
            assert all(x[k] == y[k] for k in IDENTITY)
            if x["step"] == 0:
                assert x == y
        curve = rows(raw, prefix + "diagnostics/task_curve.jsonl")
        old_curve = rows(control, old + "diagnostics/task_curve.jsonl")
        assert [x["step"] for x in curve] == [0, 5, 10, 15]
        for x, y in zip(curve, old_curve, strict=True):
            for key in ("val/valid_token_count", "val/valid_sequence_count"):
                assert x["substantive_validation"][key] == y["substantive_validation"][key]
        metrics_key = next(k for k in raw if k.startswith(prefix + "native/") and k.endswith("/metrics.jsonl"))
        old_key = next(k for k in control if k.startswith(old + "native/") and k.endswith("/metrics.jsonl"))
        updates = [x for x in rows(raw, metrics_key) if "train/valid_token_count" in x]
        old_updates = [x for x in rows(control, old_key) if "train/valid_token_count" in x]
        assert len(updates) == len(old_updates) == 15
        for step, (x, y) in enumerate(zip(updates, old_updates, strict=True), 1):
            assert x["optim/optimizer_step"] == step and x["optim/update_outcome"] == "applied"
            assert x["optim/nonfinite_skips_total"] == 0
            for key in ("train/valid_token_count", "train/valid_sequence_count"):
                assert x[key] == y[key]
            assert math.isclose(x["train/learning_rate"], y["train/learning_rate"] * args.lr_ratio, abs_tol=1e-15)
        state = json.loads(raw[prefix + "diagnostics/complete.json"])
        assert state["complete"] and state["trainable_elements"] == 3087142912
        current = [x for x in a if x["step"] == 15]
        previous = [x for x in b if x["step"] == 15]
        delta = [x["answer_nll"] - y["answer_nll"] for x, y in zip(current, previous, strict=True)]
        result.append(
            {
                "seed": seed,
                "task_nll": curve[-1]["metrics"]["answer_nll"],
                "control_task_nll": old_curve[-1]["metrics"]["answer_nll"],
                "accuracy": curve[-1]["metrics"]["restricted_choice_accuracy"],
                "control_accuracy": old_curve[-1]["metrics"]["restricted_choice_accuracy"],
                "substantive_nll_ratio": curve[-1]["substantive_validation"]["val/loss_avg"]
                / old_curve[-1]["substantive_validation"]["val/loss_avg"],
                "curve": curve,
                "control_curve": old_curve,
                "task_nll_delta_median": median(delta),
                "nll_improved_questions": sum(x < 0 for x in delta),
                "mean_delta_excluding_largest_improvement": (sum(delta) - min(delta)) / 31,
                "update15_questions": [{"candidate": x, "control": y} for x, y in zip(current, previous, strict=True)],
                "attempt": attempt,
                "state": state,
                "identity_and_update_checks_pass": True,
            }
        )
    ratio = mean(x["task_nll"] for x in result) / mean(x["control_task_nll"] for x in result)
    gates = {
        "mean_task_nll_improvement_at_least_2_percent": ratio <= 0.98,
        "no_seed_accuracy_loss": all(x["accuracy"] >= x["control_accuracy"] for x in result),
        "each_substantive_nll_ratio_at_most_1_01": all(x["substantive_nll_ratio"] <= 1.01 for x in result),
    }
    output = {
        "complete": True,
        "raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "arm": args.arm,
        "cases_prefix": args.prefix,
        "candidate_mean_task_nll_over_control": ratio,
        "gates": gates,
        "accepted": all(gates.values()),
        "rows": result,
        "scope": "Development only. Same inputs, masks, counts, initialization and15actual applied updates. No final scores read; no free-generation quality claim.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}))


if __name__ == "__main__":
    main()
