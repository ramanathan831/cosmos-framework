"""Localize the rejected order policy using unchanged native batch-four loss."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from analyze_replay import components
from analyze_screen import read_raw, rows

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("interleave", "smoothing"), default="interleave")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate = args.candidate
    seeds = (29, 43) if candidate == "interleave" else (43,)
    raw = read_raw(ROOT / f"{candidate}_failure_raw.json.gz")
    expected_cases = (["original29"] if candidate == "interleave" else []) + [f"{candidate}{s}" for s in seeds]
    assert json.loads(raw["complete.json"])["cases"] == expected_cases
    training = read_raw(ROOT / f"fused_{candidate}_raw.json.gz")
    original = read_raw(ROOT.parent / "quantized_optimizers_round3/outcomes_raw.json.gz")
    replay = read_raw(ROOT / "replay_raw.json.gz")
    vocabulary = json.loads((ROOT / "token_strings.json").read_text())["raw_vocabulary_strings"]
    texts = {r["id"]: r for r in json.loads((ROOT / "development_texts.json").read_text())["rows"]}
    checks = [(f"{candidate}{seed}", seed, training, f"fused_{candidate}_seed{seed}") for seed in seeds]
    if candidate == "interleave":
        checks.insert(0, ("original29", 29, original, "final-seed29/matched_adamw"))
    for case, seed, source, prefix in checks:
        assert json.loads(raw[case + "/attempt.json"])["returncode"] == 0
        current = rows(raw, case + "/scores/scores.jsonl")
        recorded = [r for r in rows(source, prefix + "/diagnostics/task_rows.jsonl") if r["step"] == 15]
        assert len(current) == len(recorded) == 32
        for a, b in zip(current, recorded, strict=True):
            assert all(a[k] == b[k] for k in a.keys() & b.keys())
        native = json.loads(raw[case + "/scores/substantive-batches.json"])
        curve = rows(source, prefix + "/diagnostics/task_curve.jsonl")[-1]
        assert native["native_loss"] == curve["substantive_validation"]["val/loss_avg"]
    output = []
    for seed in seeds:
        a = json.loads(
            raw["original29/scores/substantive-batches.json"]
            if seed == 29
            else replay["original/reload15/substantive-batches.json"]
        )
        b = json.loads(raw[f"{candidate}{seed}/scores/substantive-batches.json"])
        assert a["native_tokens"] == b["native_tokens"] == 1346
        assert a["source_fingerprint"] == b["source_fingerprint"]
        grouped = defaultdict(float)
        token_changes, row_changes = [], []
        for ba, bb in zip(a["batches"], b["batches"], strict=True):
            assert ba["input_sha256"] == bb["input_sha256"] and ba["input_shapes"] == bb["input_shapes"]
            for ra, rb in zip(ba["rows"], bb["rows"], strict=True):
                assert all(
                    ra[k] == rb[k] for k in ("id", "tokens", "target_token_ids", "trajectory_id", "sequence_length")
                )
                parts = components(ra, texts[ra["id"]], vocabulary)
                deltas = [y - x for x, y in zip(ra["token_losses"], rb["token_losses"], strict=True)]
                for i, delta in enumerate(deltas):
                    question, kind = parts[i]
                    grouped[question + " / " + kind] += delta / 1346
                    token_changes.append(
                        {
                            "id": ra["id"],
                            "question": question,
                            "kind": kind,
                            "token": vocabulary[str(ra["target_token_ids"][i])],
                            "delta": delta,
                            "original_loss": ra["token_losses"][i],
                            "candidate_loss": rb["token_losses"][i],
                        }
                    )
                row_changes.append({"id": ra["id"], "native_delta_contribution": math.fsum(deltas) / 1346})
        assert math.isclose(math.fsum(grouped.values()), b["native_loss"] - a["native_loss"], abs_tol=1e-7)
        output.append(
            {
                "seed": seed,
                "original_native_loss": a["native_loss"],
                "candidate_native_loss": b["native_loss"],
                "ratio": b["native_loss"] / a["native_loss"],
                "exact_native_tokens": 1346,
                "question_components": dict(sorted(grouped.items(), key=lambda x: abs(x[1]), reverse=True)),
                "rows": row_changes,
                "largest_token_changes": sorted(token_changes, key=lambda x: abs(x["delta"]), reverse=True)[:24],
            }
        )
    result = {
        "complete": True,
        "exact_task_observations": 32 * len(checks),
        "scope": "Development only, native batch four. Teacher-forced text loss; no free-generation or optimizer-specific claim. Original seed43 attribution reuses its already verified exact replay.",
        "comparisons": output,
    }
    (args.output or ROOT / f"{candidate}_failure_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            [{k: v for k, v in x.items() if k not in ("rows", "largest_token_changes")} for x in output], indent=2
        )
    )


if __name__ == "__main__":
    main()
