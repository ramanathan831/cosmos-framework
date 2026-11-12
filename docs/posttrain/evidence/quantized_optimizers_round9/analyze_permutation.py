"""Describe same-update permutation effects after exact reference replay checks."""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "quantized_optimizers_round8"))
from analyze_screen import IDENTITY, read_raw, rows


def differences(a, b):
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "l2": float(np.linalg.norm(delta)),
        "exact": bool(np.all(delta == 0)),
    }


def main():
    raw = read_raw(ROOT / "permutation_raw.json.gz")
    assert "complete.json" in raw
    parity = json.loads(raw["default_verified.json"])
    assert parity["complete"] and parity["excluded"] == []
    names = ("same_order43", "permuted43")
    series = {}
    for name in names:
        assert json.loads(raw[name + "/attempt.json"])["returncode"] == 0
        series[name] = {
            "inputs": rows(raw, name + "/diagnostics/training_inputs.jsonl"),
            "observations": rows(raw, name + "/diagnostics/training_observations.jsonl"),
            "curve": rows(raw, name + "/diagnostics/task_curve.jsonl"),
            "task": rows(raw, name + "/diagnostics/task_rows.jsonl"),
            "audit": json.loads(raw[name + "/diagnostics/stochastic_audit.json"]),
        }
    a, b = (series[n] for n in names)
    assert len(a["inputs"]) == len(b["inputs"]) == 15
    changed = []
    for step, (x, y) in enumerate(zip(a["inputs"], b["inputs"], strict=True), 1):
        assert sorted(x["ids"]) == sorted(y["ids"]) and len(x["ids"]) == 4
        if x["ids"] != y["ids"]:
            changed.append(step)
    assert changed == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
    assert a["audit"] == b["audit"] and a["audit"]["grad_accum"] == 1
    observations = []
    for x, y in zip(a["observations"], b["observations"], strict=True):
        assert x["step"] == y["step"]
        left = {r["id"]: r for r in x["rows"]}
        right = {r["id"]: r for r in y["rows"]}
        assert left.keys() == right.keys()
        aligned = []
        for identity, row in left.items():
            other = right[identity]
            for key in ("tokens", "labels", "length", "media", "media_shapes", "supervised_tokens", "target_ids"):
                assert row[key] == other[key], (x["step"], identity, key)
            aligned.append(
                {
                    "id": identity,
                    "tokens": row["supervised_tokens"],
                    "token_ce": differences(row["token_losses"], other["token_losses"]),
                    "top4_logits": differences(row["top4_logits"], other["top4_logits"]),
                }
            )
        gradients, updates = [], []
        assert x["parameters"].keys() == y["parameters"].keys()
        full_error_sq = full_reference_sq = 0.0
        full_exact = True
        for name, value in x["parameters"].items():
            other = y["parameters"][name]
            if value is None or other is None:
                assert value is other
                continue
            assert value["elements"] == other["elements"]
            gradients.append(dict(name=name, **differences(value["samples"], other["samples"])))
            updates.append(dict(name=name, **differences(value["update_samples"], other["update_samples"])))
            if x["step"] == 1:
                full_error_sq += other["reference_difference"]["squared_l2"]
                full_reference_sq += value["l2"] ** 2
                full_exact &= other["reference_difference"]["exact"]
        observations.append(
            {
                "step": x["step"],
                "aligned_rows": aligned,
                "rng_before_equal": x["rng_before"] == y["rng_before"],
                "rng_after_equal": x["rng_after"] == y["rng_after"],
                "rng_unchanged_within_training": [v["rng_before"] == v["rng_after"] for v in (x, y)],
                "grad_norms": [v["metrics"].get("train/grad_norm") for v in (x, y)],
                "clip_coefficients": [v["metrics"].get("train/grad_clip_coef") for v in (x, y)],
                "training_losses": [v["metrics"].get("train/loss_avg") for v in (x, y)],
                "full_first_gradient": None
                if x["step"] != 1
                else {
                    "exact": full_exact,
                    "difference_l2": math.sqrt(full_error_sq),
                    "relative_l2": math.sqrt(full_error_sq / full_reference_sq),
                },
                "sampled_gradients": gradients,
                "sampled_updates": updates,
            }
        )
    assert [x["step"] for x in observations] == [1, 13, 14, 15]
    for x, y in zip(a["task"], b["task"], strict=True):
        assert all(x[k] == y[k] for k in IDENTITY)
        if x["step"] == 0:
            assert x == y
    result = {
        "complete": True,
        "reference_nonperturbation": parity,
        "changed_updates": changed,
        "stochastic_audit": a["audit"],
        "observations": observations,
        "curves": {n: series[n]["curve"] for n in names},
        "limits": [
            "Only step1 compares gradients at identical parameter/moment state.",
            "Later differences include inherited trajectory changes; 256 sampled gradient/update entries per parameter.",
            "No production recipe selection from this single permutation.",
            "Task and teacher-forced substantive losses do not establish free-generation quality.",
        ],
    }
    (ROOT / "permutation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "changed_updates": changed,
                "stochastic_audit": a["audit"],
                "first_gradient": observations[0]["full_first_gradient"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
