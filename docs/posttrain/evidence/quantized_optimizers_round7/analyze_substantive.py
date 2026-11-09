"""Attribute the existing seed43 guard using unchanged native physical batches."""

import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def main():
    raw = read(ROOT / "substantive_raw.json.gz")
    old = read(ROOT.parent / "quantized_optimizers_round3/outcomes_raw.json.gz")
    w95 = read(ROOT.parent / "quantized_optimizers_round5/weight95_fused_raw.json.gz")
    w975 = read(ROOT.parent / "quantized_optimizers_round6/weight975_fused_raw.json.gz")
    historical = {
        "original": (old, "final-seed43/matched_adamw/"),
        "weight95": (w95, "fused_weight95_seed43/"),
        "weight975": (w975, "fused_weight975_seed43/"),
    }
    assert set(json.loads(raw["complete.json"])["cases"]) == set(historical)
    result = {}
    for name, (previous, prefix) in historical.items():
        text = json.loads(raw[name + "/scores/substantive-batches.json"])
        summary = json.loads(raw[name + "/scores/summary.json"])
        attempt = json.loads(raw[name + "/attempt.json"])
        assert summary["complete"] and attempt["returncode"] == 0
        curve = [json.loads(line) for line in previous[prefix + "diagnostics/task_curve.jsonl"].splitlines()]
        expected = curve[-1]["substantive_validation"]["val/loss_avg"]
        assert text["native_loss"] == text["expected_native_loss"] == expected
        prior = [json.loads(line) for line in previous[prefix + "diagnostics/task_rows.jsonl"].splitlines()]
        prior = [r for r in prior if r["step"] == 15]
        current = [json.loads(line) for line in raw[name + "/scores/scores.jsonl"].splitlines()]
        assert len(prior) == len(current) == 32
        for a, b in zip(prior, current, strict=True):
            assert all(a[k] == b[k] for k in a.keys() - {"step", "strict_first_token_text"})
        pinned = json.loads(raw[name + "/pinned-checkpoint.json"])
        restored = json.loads(raw[name + "/scores/checkpoint-read.json"])
        assert pinned["sha256"] == restored["sha256"] == summary["expected_checkpoint_sha256"]
        result[name] = dict(
            native_loss=text["native_loss"],
            native_tokens=text["native_tokens"],
            batches=text["batches"],
            source_fingerprint=text["source_fingerprint"],
            metrics=summary["metrics"],
            checkpoint=summary["checkpoint"],
            checkpoint_sha256=pinned["sha256"],
            checkpoint_provenance=summary["checkpoint_provenance"],
            attempt_seconds=attempt["elapsed_seconds"],
        )
    comparisons = {}
    control = result["original"]
    for name in ["weight95", "weight975"]:
        candidate = result[name]
        assert candidate["source_fingerprint"] == control["source_fingerprint"]
        rows = []
        for a, b in zip(candidate["batches"], control["batches"], strict=True):
            assert a["input_sha256"] == b["input_sha256"] and a["input_shapes"] == b["input_shapes"]
            for x, y in zip(a["rows"], b["rows"], strict=True):
                assert all(
                    x[k] == y[k] for k in ["id", "trajectory_id", "tokens", "target_token_ids", "sequence_length"]
                )
                deltas = [u - v for u, v in zip(x["token_losses"], y["token_losses"], strict=True)]
                rows.append(
                    dict(
                        id=x["id"],
                        tokens=x["tokens"],
                        control_nll=y["token_loss_sum"] / y["tokens"],
                        candidate_nll=x["token_loss_sum"] / x["tokens"],
                        contribution_to_native_mean_delta=math.fsum(deltas) / control["native_tokens"],
                        top_worsening_tokens=sorted(
                            [
                                dict(
                                    position=i,
                                    token_id=x["target_token_ids"][i],
                                    delta=d,
                                    control_loss=y["token_losses"][i],
                                    candidate_loss=x["token_losses"][i],
                                )
                                for i, d in enumerate(deltas)
                            ],
                            key=lambda v: v["delta"],
                            reverse=True,
                        )[:3],
                    )
                )
        assert math.isclose(
            sum(r["contribution_to_native_mean_delta"] for r in rows),
            candidate["native_loss"] - control["native_loss"],
            abs_tol=1e-7,
        )
        comparisons[name] = dict(
            native_ratio=candidate["native_loss"] / control["native_loss"],
            worsened_rows=sum(r["contribution_to_native_mean_delta"] > 0 for r in rows),
            improved_rows=sum(r["contribution_to_native_mean_delta"] < 0 for r in rows),
            rows=sorted(rows, key=lambda r: r["contribution_to_native_mean_delta"], reverse=True),
        )
    output = dict(
        complete=True,
        scope="Development fused seed43 only; original/.95/.975 checkpoints. Native batch4 remains authoritative; no new training, final evaluation or free-generation quality claim.",
        exact_task_observations=96,
        exact_native_substantive_aggregates=3,
        identical_native_batch_inputs=True,
        cases={
            name: {
                **{key: value for key, value in case.items() if key != "batches"},
                "batch_identities": [
                    {key: batch[key] for key in ("input_sha256", "input_shapes", "native_loss", "native_tokens")}
                    for batch in case["batches"]
                ],
            }
            for name, case in result.items()
        },
        raw_token_attribution="substantive_raw.json.gz: */scores/substantive-batches.json",
        comparisons=comparisons,
    )
    (ROOT / "substantive_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: dict(ratio=v["native_ratio"], worse=v["worsened_rows"], better=v["improved_rows"], top=v["rows"][:3])
                for k, v in comparisons.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
