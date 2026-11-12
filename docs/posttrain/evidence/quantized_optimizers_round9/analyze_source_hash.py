"""Locked paired analysis of source identity construction, not cold delivery."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "quantized_optimizers_round8"))
from analyze_screen import read_raw


def main():
    raw = read_raw(ROOT / "source_hash_raw.json.gz")
    assert raw["pairs/complete.txt"].strip() == "complete"
    cases, identities = {}, None
    for pair in range(3):
        for workers in (1, 4):
            name = f"pair{pair}-workers{workers}"
            result = json.loads(raw[f"pairs/{name}.json"])
            process = json.loads(raw[f"pairs/{name}.process.json"])
            assert result["complete"] and process["returncode"] == 0
            assert result["workers"] == workers
            if identities is None:
                identities = result["identities"]
            assert identities == result.pop("identities")
            result["whole_process_seconds"] = process["elapsed_seconds"]
            result["peak_process_tree_memory"] = process["peak_process_tree_memory"]
            cases[name] = result
    timing = {}
    for field in ("identity_seconds", "whole_process_seconds"):
        ratios = [cases[f"pair{i}-workers1"][field] / cases[f"pair{i}-workers4"][field] for i in range(3)]
        logs = np.log(ratios)
        rng = np.random.default_rng(20261112)
        draws = np.exp(rng.choice(logs, (10000, 3), replace=True).mean(1))
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
        timing[field] = {
            "paired_ratios": ratios,
            "geometric_ratio": float(np.exp(logs.mean())),
            "ci95": interval,
            "excludes_parity": interval[0] > 1,
        }
    result = {
        "complete": True,
        "cases": cases,
        "timing": timing,
        "identities": identities,
        "exact_comparisons": 6,
        "scope": "One CPU node, three counterbalanced pairs of fresh Python source-identity processes; no decoding, model loading or optimizer timing.",
        "limitations": [
            "Shared storage cache uncontrolled.",
            "Sampled process-tree RSS/PSS includes child imports but excludes benchmark parent.",
            "No full cold preparation measurement; historical cold requirement remains unmet.",
            "Serial remains default; parallel hashing is optional and shared by optimizer arms.",
        ],
    }
    (ROOT / "source_hash_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
