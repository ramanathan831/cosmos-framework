"""Temporal attribution after exact replay parity; never final-data selection."""

import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    raw = json.loads(gzip.decompress((ROOT / "replay_raw.json.gz").read_bytes()))
    assert json.loads(raw["replay_verified.json"])["complete"]
    series = {
        arm: {
            row["step"]: row for row in map(json.loads, raw[arm + "/diagnostics/late_substantive.jsonl"].splitlines())
        }
        for arm in ("original", "weight975")
    }
    order = {
        row["step"]: row
        for row in json.loads((ROOT.parent / "quantized_optimizers_round7/training_order_summary.json").read_text())[
            "batches"
        ]
    }
    observations = []
    prior = {}
    for step in range(10, 16):
        a, b = series["original"][step], series["weight975"][step]
        assert a["native_tokens"] == b["native_tokens"] == 1346
        assert a["source_fingerprint"] == b["source_fingerprint"]
        tokens = []
        rows = []
        for ba, bb in zip(a["batches"], b["batches"], strict=True):
            assert ba["input_sha256"] == bb["input_sha256"] and ba["input_shapes"] == bb["input_shapes"]
            for ra, rb in zip(ba["rows"], bb["rows"], strict=True):
                assert all(
                    ra[k] == rb[k] for k in ("id", "tokens", "target_token_ids", "trajectory_id", "sequence_length")
                )
                delta = [y - x for x, y in zip(ra["token_losses"], rb["token_losses"], strict=True)]
                for position, value in enumerate(delta):
                    key = (ra["id"], position)
                    tokens.append(
                        {
                            "id": ra["id"],
                            "position": position,
                            "token_id": ra["target_token_ids"][position],
                            "delta": value,
                            "increment_since_previous": value - prior[key] if key in prior else None,
                            "original_loss": ra["token_losses"][position],
                            "candidate_loss": rb["token_losses"][position],
                        }
                    )
                    prior[key] = value
                rows.append(
                    {
                        "id": ra["id"],
                        "native_mean_delta_contribution": math.fsum(delta) / 1346,
                        "original_nll": ra["token_loss_sum"] / ra["tokens"],
                        "candidate_nll": rb["token_loss_sum"] / rb["tokens"],
                    }
                )
        assert math.isclose(
            math.fsum(t["delta"] for t in tokens) / 1346, b["native_loss"] - a["native_loss"], abs_tol=1e-7
        )
        observation = {
            "step": step,
            "original_native_nll": a["native_loss"],
            "candidate_native_nll": b["native_loss"],
            "ratio": b["native_loss"] / a["native_loss"],
            "delta": b["native_loss"] - a["native_loss"],
            "training_text_rows": order[step]["text_rows"],
            "rows": rows,
            "largest_token_relative_changes": sorted(
                tokens, key=lambda x: abs(x["increment_since_previous"] or 0), reverse=True
            )[:30],
        }
        observations.append(observation)
    output = {
        "complete": True,
        "scope": "One controlled fused seed43 replay; exact native batch-four teacher-forced text loss, not free generation or optimizer-specific evidence.",
        "parity": json.loads(raw["replay_verified.json"]),
        "updates": observations,
        "interpretation_limit": "Temporal contributions localize where the recipe effect changes; they do not isolate a sole causal role for batch composition versus inherited parameters/moments.",
    }
    (ROOT / "replay_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            [{k: v for k, v in x.items() if k not in ("rows", "largest_token_relative_changes")} for x in observations],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
