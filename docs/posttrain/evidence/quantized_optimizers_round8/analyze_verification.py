"""Locked three-pair verification-phase analysis; never a cold pipeline estimate."""

import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def main():
    raw = json.loads(gzip.decompress((ROOT / "verification_raw.json.gz").read_bytes()))
    assert raw["complete.txt"].strip() == "complete"
    cases = {}
    for arm in ("tensors", "bytes"):
        for i in range(3):
            name = f"{arm}{i}"
            case = json.loads(raw[name + ".json"])
            process = json.loads(raw[name + ".process.json"])
            assert case["complete"] and process["returncode"] == 0
            assert case["logical_rows"] == 751 and case["targets"] == 562
            assert case["manifest_sha256"] == "ea2e64893ab3d7da0f4c8be33001ae99c96dfd2c54fe293449c29255cd1f2d70"
            case["whole_process_seconds"] = process["elapsed_seconds"]
            cases[name] = case
    summaries = {}
    for field in ("verification_seconds", "whole_process_seconds"):
        logs = np.log([cases[f"tensors{i}"][field] / cases[f"bytes{i}"][field] for i in range(3)])
        rng = np.random.default_rng(20261111)
        samples = np.exp(rng.choice(logs, size=(10000, 3), replace=True).mean(axis=1))
        ci = np.quantile(samples, [0.025, 0.975]).tolist()
        summaries[field] = {"geometric_ratio": float(np.exp(logs.mean())), "ci95": ci, "excludes_parity": ci[0] > 1}
    receipt = json.loads(raw["fresh-slice/preparation_receipt.json"])
    assert receipt["reference_verification"] == "bytes"
    assert receipt["exact_raw_prepared_rows"] == receipt["independently_replayed_unique_targets"] == 15
    assert receipt["total_verified_rows"] == 15 and not receipt["cuda_initialized"]
    output = {
        "complete": True,
        "scope": "Post-publication verification on existing exact full corpora,3counterbalanced pairs. No new cold preparation timings or optimizer-quality comparison.",
        "cases": cases,
        "timing": summaries,
        "fresh_slice": receipt,
        "defaults": "tensor reference verification remains default",
        "limitations": [
            "Shared storage caches uncontrolled",
            "One CPU node/session;3timing pairs are not independent model seeds",
            "New full producer timing replications absent; no composed cold claim for the new byte-verification path",
        ],
    }
    (ROOT / "verification_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
