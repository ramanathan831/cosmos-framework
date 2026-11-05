"""Localize the failed development text-loss guard without changing its reduction."""

import json
import math
from pathlib import Path

from analyze_screen import read_raw, rows

ROOT = Path(__file__).parent


def main():
    raw = read_raw(ROOT / "substantive_diagnosis_raw.json.gz")
    old = read_raw(ROOT.parent / "quantized_optimizers_round3/outcomes_raw.json.gz")
    candidates = {a: read_raw(ROOT / f"weight95_{a}_raw.json.gz") for a in ("fused", "eightbit")}
    result, identities = {}, {}
    for arm, optimizer in (("fused", "adamw"), ("eightbit", "adamw_bnb_8bit")):
        cases = {}
        for mode in ("control", "candidate"):
            name = f"{arm}_{mode}"
            original = old if mode == "control" else candidates[arm]
            prefix = f"final-seed29/matched_{optimizer}/" if mode == "control" else f"{arm}_weight95_seed29/"
            task = [x for x in rows(original, prefix + "diagnostics/task_rows.jsonl") if x["step"] == 15]
            scored = rows(raw, name + "/scores/scores.jsonl")
            assert len(task) == len(scored) == 32
            for x, y in zip(task, scored, strict=True):
                for key in x.keys() - {"step", "strict_first_token_text"}:
                    assert x[key] == y[key], (name, key)
            text = json.loads(raw[name + "/scores/substantive.json"])
            assert len(text) == 16 and sum(x["tokens"] for x in text) == 1346
            assert all(math.isfinite(x["loss"]) and x["tokens"] > 0 for x in text)
            native = rows(original, prefix + "diagnostics/task_curve.jsonl")[-1]["substantive_validation"][
                "val/loss_avg"
            ]
            average = math.fsum(x["loss"] * x["tokens"] for x in text) / 1346
            summary = json.loads(raw[name + "/scores/summary.json"])
            attempt = json.loads(raw[name + "/attempt.json"])
            assert attempt["returncode"] == 0 and summary["complete"]
            cases[mode] = {
                "native_batch4_token_nll": native,
                "fresh_batch1_token_nll": average,
                "fresh_minus_native": average - native,
                "rows": text,
                "checkpoint_read": json.loads(raw[name + "/scores/checkpoint-read.json"]),
                "checkpoint_provenance": summary["checkpoint_provenance"],
                "attempt_seconds": attempt["elapsed_seconds"],
                "exact_task_rows": len(task),
            }
            identities[name] = [(x["id"], x["tokens"]) for x in text]
        delta = []
        for a, b in zip(cases["candidate"]["rows"], cases["control"]["rows"], strict=True):
            assert (a["id"], a["tokens"]) == (b["id"], b["tokens"])
            delta.append(
                {
                    "id": a["id"],
                    "tokens": a["tokens"],
                    "candidate_nll": a["loss"],
                    "control_nll": b["loss"],
                    "contribution_to_token_mean_delta": (a["loss"] - b["loss"]) * a["tokens"] / 1346,
                }
            )
        result[arm] = {
            "cases": cases,
            "native_guard_ratio": cases["candidate"]["native_batch4_token_nll"]
            / cases["control"]["native_batch4_token_nll"],
            "fresh_diagnostic_ratio": cases["candidate"]["fresh_batch1_token_nll"]
            / cases["control"]["fresh_batch1_token_nll"],
            "rows_by_worsening_contribution": sorted(
                delta, key=lambda x: x["contribution_to_token_mean_delta"], reverse=True
            ),
        }
    assert all(value == identities["fused_control"] for value in identities.values())
    assert set(json.loads(raw["complete.json"])["cases"]) == set(identities)
    output = {
        "complete": True,
        "exact_native_task_observations": 128,
        "same_substantive_ids_and_tokens": True,
        "arms": result,
        "scope": "Seed29 development only, four already committed checkpoints. No training or final scoring. Native batch4 token mean remains the acceptance metric. Fresh batch1 per-row teacher-forced answer-text losses localize contributions; any batch-shape numerical differences are retained, not used to replace the guard. This does not measure generated-answer quality or establish calibration across the task.",
    }
    (ROOT / "substantive_diagnosis_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({a: {k: v for k, v in r.items() if k != "cases"} for a, r in result.items()}))


if __name__ == "__main__":
    main()
