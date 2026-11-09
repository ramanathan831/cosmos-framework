"""Summarize the prospectively paired slice; do not substitute inner timers."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--inner-only", action="store_true")
    args = parser.parse_args()
    raw = json.loads(gzip.decompress(args.archive.read_bytes()))
    cases = {}
    for arm in ("control", "candidate"):
        for index in range(3):
            name = f"trial{arm}{index}"
            r = json.loads(raw[name + "/preparation_receipt.json"])
            assert r["exact_raw_prepared_rows"] == r["reference_exact_unique_targets"] == 15
            assert r["independently_replayed_unique_targets"] == 15 and r["raw_native_calls"] == 30
            assert not r["cuda_initialized"] and r["workers"] == 4 and r["threads_per_worker"] == 2
            counts = {
                field: sum(x[field] for x in r["video_reuse"])
                for field in ("hits", "misses", "oversized", "source_hashes")
            }
            if arm == "candidate":
                assert counts["hits"] == 7 and counts["misses"] == 8 and counts["oversized"] == 0
                assert (
                    max(x["retained_peak_bytes"] for x in r["video_reuse"])
                    <= r["video_cache_bytes_per_worker"]
                    == 2**26
                )
            else:
                assert r["video_cache_bytes_per_worker"] == 0 and not r["video_reuse"]
            p = None if args.inner_only else json.loads(raw[name + ".process.json"])
            if p is not None:
                assert p["returncode"] == 0 and p["elapsed_seconds"] >= r["total_seconds_with_independent_exact_replay"]
            cases[name] = {"receipt": r, "process": p, "cache_counts": counts}
    identities = {
        tuple(
            c["receipt"][k] for k in ["reference_corpus_sha256", "corpus_sha256", "payload_bytes", "fixed_slice_sha256"]
        )
        for c in cases.values()
    }
    assert len(identities) == 1

    def timing(c):
        return (
            c["receipt"]["total_seconds_with_independent_exact_replay"]
            if args.inner_only
            else c["process"]["elapsed_seconds"]
        )

    pairs = [
        {
            "pair": i,
            "control_seconds": timing(cases[f"trialcontrol{i}"]),
            "candidate_seconds": timing(cases[f"trialcandidate{i}"]),
        }
        for i in range(3)
    ]
    ratios = np.array([p["control_seconds"] / p["candidate_seconds"] for p in pairs])
    rng = np.random.default_rng(20261109)
    sampled = np.exp(rng.choice(np.log(ratios), size=(10000, 3), replace=True).mean(axis=1))
    interval = np.quantile(sampled, [0.025, 0.975]).tolist()
    result = dict(
        scope="Fixed15target/8video development slice; not a full cold-workflow result.",
        timing_scope="Inner preparation only; gate not evaluated"
        if args.inner_only
        else "Whole native preparation process including imports and shutdown; shared container setup accounted separately.",
        pairs=pairs,
        paired_speed_ratios=ratios.tolist(),
        geometric_mean_ratio=float(np.exp(np.log(ratios).mean())),
        interval95=interval,
        screen_pass=None if args.inner_only else bool(interval[0] > 1),
        exact_repeated_target_comparisons=90,
        independent_native_replay_calls=90,
        identical_corpus_manifest=True,
        cases=cases,
        uncertainty_limit="Three paired timing trials on one fixed slice and one node. Questions/videos are not treated as independent timing replicates. Shared storage caches were not cleared; first/last order is counterbalanced. No full-workload speed or independent replication claim.",
    )
    output = ROOT / ("reuse_inner_summary.json" if args.inner_only else "reuse_summary.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
