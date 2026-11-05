"""Reproduce this campaign's locked checkpoint-overlap development screen."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent


def load(name):
    data = json.loads(gzip.decompress((ROOT / name).read_bytes()))
    for key, receipt in json.loads(data["_collection.json"])["files"].items():
        value = data[key].encode()
        assert len(value) == receipt["bytes"]
        assert hashlib.sha256(value).hexdigest() == receipt["sha256"]
    return data


def main():
    data = load("checkpoint_overlap_raw.json.gz")
    previous = load("runtime_canary_raw.json.gz")
    cases = {}
    for arm in ("fused", "eightbit"):
        for mode in ("serial", "overlap"):
            name = f"{arm}_{mode}"
            summary = json.loads(data[name + "/scores/summary.json"])
            attempt = json.loads(data[name + "/attempt.json"])
            checksum = json.loads(data[name + "/scores/checkpoint-read.json"])
            rows = [json.loads(line) for line in data[name + "/scores/scores.jsonl"].splitlines()]
            control = [json.loads(line) for line in previous[arm + "_serial/scores/scores.jsonl"].splitlines()]
            old = json.loads(previous[arm + "_serial/scores/summary.json"])
            old_hash = json.loads(previous[arm + "_serial/scores/checkpoint-read.json"])
            assert rows == control and len(rows) == 32
            for key in (
                "checkpoint_provenance",
                "checkpoint",
                "metrics",
                "objective_identity",
                "corpus_sha256",
                "selection_sha256",
            ):
                assert summary[key] == old[key], key
            assert checksum["sha256"] == old_hash["sha256"]
            assert attempt["returncode"] == 0
            memory = [json.loads(line) for line in data[name + "/memory.jsonl"].splitlines()]
            cases[name] = {
                "attempt_seconds": attempt["elapsed_seconds"],
                "scoring_scope_seconds": summary["elapsed_seconds"],
                "phase_seconds": summary["phase_seconds"],
                "peak_host_pss_bytes": max(x.get("process_tree_pss") or 0 for x in memory),
                "peak_process_gpu_bytes": max(x.get("matched_process_gpu_bytes") or 0 for x in memory),
                "peak_device_bytes": max(x.get("device_used") or 0 for x in memory),
                "checkpoint_sha256": checksum["sha256"],
                "checkpoint_provenance": summary["checkpoint_provenance"],
                "exact_rows": len(rows),
                "metrics": summary["metrics"],
            }
    ratios = {}
    for name, metric in (
        ("combined_read_restore_ratio", "phase_seconds"),
        ("aggregate_attempt_time_ratio", "attempt_seconds"),
    ):
        totals = {}
        for mode in ("serial", "overlap"):
            values = [cases[f"{arm}_{mode}"][metric] for arm in ("fused", "eightbit")]
            totals[mode] = (
                sum(x["checkpoint_read_and_restore"] for x in values) if metric == "phase_seconds" else sum(values)
            )
        ratios[name] = totals["overlap"] / totals["serial"]
    assert "6 passed" in data["pinned_controls/process.log"]
    assert set(json.loads(data["complete.json"])["cases"]) == {"pinned_controls", *cases}
    result = {
        "complete": True,
        "job": "38039130",
        "allocated_gpu_seconds": 390,
        "source_commit": json.loads((ROOT / "checkpoint_overlap_source.json").read_text())["commit"],
        "raw_sha256": hashlib.sha256((ROOT / "checkpoint_overlap_raw.json.gz").read_bytes()).hexdigest(),
        "cases": cases,
        **ratios,
        "gates": {
            "exact_input_prediction_ce_nll_parity": True,
            "exact_checkpoint_sha_and_provenance": True,
            "phase_reduction_at_least_10_percent": ratios["combined_read_restore_ratio"] <= 0.9,
            "attempt_ratio_at_most_1_02": ratios["aggregate_attempt_time_ratio"] <= 1.02,
        },
        "pinned_helper_tests": 6,
        "scope": "Two optimizer checkpoints, one counterbalanced pair per arm;32development questions per process. No final evaluation, independent seed replication or CI. Phase timers include concurrent work; do not add checkpoint_read and committed_model_restore. Aggregate attempt timing also includes variable startup; the combined phase is the direct mechanism evidence.",
    }
    result["accepted"] = all(result["gates"].values())
    (ROOT / "checkpoint_overlap_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accepted": result["accepted"], **ratios}))


if __name__ == "__main__":
    main()
