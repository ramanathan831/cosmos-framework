"""Verify the three locked full corpora and compose the historical cold clock."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def read_raw(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def main():
    ledger = json.loads((ROOT / "allocation_ledger.json").read_text())
    old = read_raw(ROOT.parent / "quantized_optimizers_round5/cold_replacement_raw.json.gz")
    reference = json.loads(old["trial0/preparation_receipt.json"])
    entries = list(json.loads(old["trial0/manifest.json"])["entries"].values())
    actual_tokens = {
        key: {
            "min": min(row[key] for row in entries),
            "max": max(row[key] for row in entries),
            "sum_unique_targets": sum(row[key] for row in entries),
        }
        for key in ("sequence_length", "supervised_tokens")
    }
    source = json.loads((ROOT / "source_identity.json").read_text())
    expected_producer = source["current_changed_tool_paths"]["tools/quantized_optimizers/parallel_prepare.py"]
    expected_archive = source["full_preparation_source"]["source_archive_sha256"]
    replications = []
    for index in range(3):
        launch = json.loads((ROOT / f"full{index}_launch.json").read_text())
        job = next(row for row in ledger["jobs"] if row["job"] == launch["job"])
        assert job["state"] == "COMPLETED" and job["exit_code"] == "0:0"
        raw = read_raw(ROOT / f"full{index}_raw.json.gz")
        assert raw[f"exit-code-{launch['job']}.txt"].strip() == "0"
        receipt = json.loads(raw["trial0/preparation_receipt.json"])
        assert receipt["source_sha256"] == expected_producer
        assert launch["source_archive_sha256"] == expected_archive
        process = json.loads(raw["trial0.process.json"])
        assert process["returncode"] == 0
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
        ):
            assert receipt[key] == reference[key], key
        assert raw["trial0/manifest.json"] == old["trial0/manifest.json"]
        assert hashlib.sha256(raw["trial0/manifest.json"].encode()).hexdigest() == receipt["corpus_sha256"]
        assert receipt["fixed_slice_sha256"] is None and receipt["scope"] == "full configured sources"
        assert receipt["video_cache_bytes_per_worker"] == 2**26
        counts = {
            key: sum(row[key] for row in receipt["video_reuse"])
            for key in ("hits", "misses", "oversized", "source_hashes")
        }
        assert counts["hits"] + counts["misses"] == 562
        assert counts["misses"] >= 308 and counts["oversized"] == 0
        assert counts["source_hashes"] == counts["hits"] + 2 * counts["misses"]
        assert len(receipt["video_reuse"]) == 308
        assert max(row["retained_peak_bytes"] for row in receipt["video_reuse"]) <= 2**26
        assert (
            receipt["total_seconds_with_independent_exact_replay"]
            <= process["elapsed_seconds"]
            <= job["elapsed_seconds"]
        )
        replications.append(
            dict(
                job=job,
                source=launch["source_commit"],
                source_archive_sha256=launch["source_archive_sha256"],
                receipt=receipt,
                process=process,
                cache_counts=counts,
            )
        )
    assert len({row["source_archive_sha256"] for row in replications}) == 1

    historical = json.loads((ROOT.parent / "quantized_optimizers_round3/corrected_product/summary.json").read_text())
    times = np.empty((3, 3, 2))
    for i, seed in enumerate((17, 29, 43)):
        for restart in range(3):
            for j, arm in enumerate(("adamw", "adamw_bnb_8bit")):
                row = next(
                    x
                    for x in historical["runs"]
                    if x["seed"] == seed and x["arm"] == arm and x["design"] == f"target_restart{restart}"
                )
                assert row["eligible"]
                times[i, restart, j] = row["prepared_delivery_seconds"]
    seconds = np.array([row["job"]["elapsed_seconds"] for row in replications])
    # Same round-five analysis: preparation is common to each optimizer pair;
    # keep timing restarts within seed and resample seeds and preparations.
    p = seconds[None, None, :]
    crossed = ((times[:, :, 0, None] + p) / (times[:, :, 1, None] + p)).mean(axis=1)
    rng = np.random.default_rng(2026)
    draws = [float(crossed[np.ix_(rng.integers(0, 3, 3), rng.integers(0, 3, 3))].mean()) for _ in range(10000)]
    bounds = np.quantile(draws, [0.025, 0.975]).tolist()
    path_plan = json.loads((ROOT.parent / "quantized_optimizers_round5/critical_path_plan.json").read_text())
    target = path_plan["nonoverlapping_bucket_means"]["target"]
    fused = target["adamw"]["prepared_total"]
    eightbit = target["adamw_bnb_8bit"]["prepared_total"]
    eightbit_training = target["adamw_bnb_8bit"]["training"]
    maximum_preparation = (fused - 1.05 * (eightbit - eightbit_training)) / 0.05
    outside_processing = np.array(
        [row["job"]["elapsed_seconds"] - row["receipt"]["preparation_seconds"] for row in replications]
    )
    output = dict(
        complete=True,
        uninterrupted_replications=3,
        exact_unique_targets_per_run=562,
        logical_rows_per_run=751,
        all_input_target_reference_checks_exact=True,
        actual_unique_target_token_counts=actual_tokens,
        replications=replications,
        allocation_seconds=seconds.tolist(),
        historical_allocation_seconds=[2193, 2166, 2121],
        descriptive_historical_preparation_ratio=float(np.mean([2193, 2166, 2121]) / seconds.mean()),
        historical_comparison_limit="Different dates/source allocations, no paired full control. Descriptive ratio only; the paired runtime evidence is the separately locked development slice.",
        composition=dict(
            mean_ratio=float(crossed.mean()),
            ci95=bounds,
            paired_seed_by_preparation_ratios=crossed.tolist(),
            speed_lower_bound_above_1_05=bool(bounds[0] > 1.05),
            method="Crossed paired-seed/preparation percentile bootstrap,10000draws,RNG2026; three timing restarts retained within each quality seed. Same method as round five.",
            scope="Historical round-three prepared target deliveries plus three new complete CPU allocations. Both optimizer arms receive identical preparation cost. This is composed, not a directly measured cold GPU pipeline or a new final optimizer comparison.",
        ),
        conditional_planning_bound=dict(
            source="Unchanged round-five target critical-path decomposition, with the new measured mean full preparation cost substituted.",
            mean_preparation_seconds=float(seconds.mean()),
            ratio_if_candidate_training_cost_zero=float(
                (seconds.mean() + fused) / (seconds.mean() + eightbit - eightbit_training)
            ),
            maximum_common_preparation_for_mean_ratio_1_05_if_candidate_training_zero=maximum_preparation,
            required_further_preparation_reduction_fraction=float(1 - maximum_preparation / seconds.mean()),
            observed_allocation_seconds_outside_worker_publication_phase=outside_processing.tolist(),
            ratio_if_entire_worker_publication_phase_and_candidate_training_were_free=float(
                (outside_processing.mean() + fused) / (outside_processing.mean() + eightbit - eightbit_training)
            ),
            limit="Planning only, holding all historical nontraining costs and target crossings fixed. Not a measured speed result, confidence bound or universal impossibility proof.",
        ),
        clocks=dict(
            native_main="Setup, preparation, publication and exact full oracle verification after Python imports; worker timers are nested and are not added to wall time.",
            python_process="Imports through worker shutdown and final verification. Excludes shared container startup.",
            allocation="Complete parent Slurm elapsed time, including container/source setup and termination; queue wait excluded. Used for cold composition consistently with historical evidence.",
            memory="Sampled process-tree PSS/RSS during native main plus workers. Excludes imports/container startup; cache payload accounting is not an RSS bound.",
        ),
        legacy_counter_note="raw_native_calls counts1124native packer invocations, including254first-pass video cache hits. It does not mean1124raw decodes.",
        cache_scope="308unique source videos and562independent uncached target replays; observed first-pass hit/miss counts are reported per run, with at most254potential hits. Full hashes, deterministic ordering and native publication remain. No cache flushing; shared Lustre/storage-cache state is not controlled.",
        limitations="Three sequential runs on one configured CPU node, historically exposed data, no new quality outcomes, no inference-latency or optimizer-specific shared-preparation claim. All failures and search costs remain in the allocation ledger rather than being discarded.",
    )
    (ROOT / "cold_preparation_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "replications"}, indent=2))


if __name__ == "__main__":
    main()
