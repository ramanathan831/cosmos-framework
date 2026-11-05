"""Complete the locked same-workflow preparation series and historical composition."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_screen import read_raw

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-seconds", type=int, required=True)
    args = parser.parse_args()
    old = json.loads((ROOT.parent / "quantized_optimizers_round4/cold_preparation_summary.json").read_text())
    raw = read_raw(ROOT / "cold_replacement_raw.json.gz")
    receipt = json.loads(raw["trial0/preparation_receipt.json"])
    reference = old["replications"][1]["receipt"]
    for key in (
        "workers",
        "threads_per_worker",
        "exact_raw_prepared_rows",
        "independently_replayed_unique_targets",
        "total_verified_rows",
        "raw_native_calls",
        "reference_exact_unique_targets",
        "reference_corpus_sha256",
        "corpus_sha256",
        "payload_bytes",
        "cuda_initialized",
        "source_paths",
        "decoder_threads",
        "source_sha256",
    ):
        assert receipt[key] == reference[key], key
    assert hashlib.sha256(raw["trial0/manifest.json"].encode()).hexdigest() == reference["corpus_sha256"]
    assert raw["exit-code-38042206.txt"].strip() == "0"
    seconds = [2193, 2166, args.allocation_seconds]
    assert receipt["total_seconds_with_independent_exact_replay"] <= seconds[-1] <= 3000
    historical = json.loads((ROOT.parent / "quantized_optimizers_round3/corrected_product/summary.json").read_text())
    times = np.empty((3, 3, 2))
    for i, seed in enumerate((17, 29, 43)):
        for restart in range(3):
            for j, arm in enumerate(("adamw", "adamw_bnb_8bit")):
                r = next(
                    x
                    for x in historical["runs"]
                    if x["seed"] == seed and x["arm"] == arm and x["design"] == f"target_restart{restart}"
                )
                assert r["eligible"]
                times[i, restart, j] = r["prepared_delivery_seconds"]

    # Each preparation cost is common to the matched optimizer pair. Keep the
    # three timing restarts inside their quality-seed cluster when resampling.
    def ratios(preparation):
        p = np.asarray(preparation)[None, None, :]
        return ((times[:, :, 0, None] + p) / (times[:, :, 1, None] + p)).mean(axis=1)

    crossed = ratios(seconds)
    rng = np.random.default_rng(2026)
    draws = [float(crossed[np.ix_(rng.integers(0, 3, 3), rng.integers(0, 3, 3))].mean()) for _ in range(10000)]
    bounds = np.quantile(draws, [0.025, 0.975]).tolist()
    output = {
        "complete": True,
        "uninterrupted_replications": 3,
        "allocation_seconds": seconds,
        "new_receipt": receipt,
        "all_input_target_reference_checks_exact": True,
        "composition": {
            "mean_ratio": float(crossed.mean()),
            "ci95": bounds,
            "paired_seed_by_preparation_ratios": crossed.tolist(),
            "method": "Crossed paired-seed/preparation percentile bootstrap, 10000 draws, RNG2026; three timing restarts retained inside each seed. Three seeds and three preparations are small samples; no independent-data claim.",
            "speed_lower_bound_above_1_05": bounds[0] > 1.05,
            "scope": "Historical round-three prepared target deliveries plus the three uninterrupted full CPU preparations. This is a composed clock with preparation variability, not directly measured cold GPU training, empty storage caches, or a new optimizer comparison. Historical single-preparation composition remains recorded.",
        },
        "reliability_sensitivity": {
            "failed_allocation_seconds": 2361,
            "recovery_and_correctness_recheck_seconds": 54,
            "uninterrupted_successes_over_admitted_full_attempts": "3/4; not an estimated production failure probability",
            "total_series_seconds_including_failure_and_recovery": sum(seconds) + 2415,
            "seconds_per_qualifying_uninterrupted_timing_including_all_costs": (sum(seconds) + 2415) / 3,
            "composed_ratio_charging_equal_share_of_failed_attempt_and_recovery": float(
                ratios([s + 2415 / 3 for s in seconds]).mean()
            ),
            "recovered_corpus_is_not_a_fourth_timing_replication": True,
        },
        "cache_and_concurrency": "New output directories and unchanged producer/workers; shared Lustre/cache state not flushed. Replacement CPU preparation overlapped owned GPU development/loading allocations. Raw node, affinity, source/archive identities and every failed attempt remain visible.",
    }
    (ROOT / "cold_preparation_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "new_receipt"}))


if __name__ == "__main__":
    main()
