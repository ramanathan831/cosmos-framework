# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic analyzer contracts; these tests establish no GPU speed or memory result."""

import copy
import json
from pathlib import Path

import pytest

from tools.native_qlora_qualification.performance import (
    GIB,
    STAGES,
    compare_pairs,
    memory_summary,
    metric_summary,
    paired_bootstrap,
    sampled_memory,
    summarize_run,
)
from tools.native_qlora_qualification.protocol import Experiment


def metrics(steps=300, seconds=2):
    rows = [
        {
            "step": 0,
            "val/loss_avg": 1.0,
            "timer/validation_s": 3.0,
            "val/valid_token_count": 128,
            "val/valid_sequence_count": 64,
        }
    ]
    for step in range(1, steps + 1):
        row = {
            "step": step,
            "timer/e2e_step_s": seconds,
            "train/loss_avg": 0.5,
            "train/grad_norm": 1.0,
            "train/valid_token_count": 16 + step % 2,
            "train/valid_sequence_count": 8,
            "optim/update_outcome": "applied",
            "optim/optimizer_step": float(step),
            "optim/attempt_id": float(step),
            "optim/step_skipped": 0.0,
            "optim/nonfinite_skips_total": 0.0,
            "optim/nonfinite_loss": 0.0,
            "optim/nonfinite_gradient": 0.0,
            "train/preprocessing_processing_seconds": float(step),
            "train/preprocessing_wait_seconds": float(step) / 2,
        }
        if step % 50 == 0:
            row.update(
                {
                    "val/loss_avg": 0.5,
                    "timer/validation_s": 3.0,
                    "val/valid_token_count": 128,
                    "val/valid_sequence_count": 64,
                }
            )
        rows.append(row)
    return rows


def runtime():
    return {
        "name": "NVIDIA A100-SXM4-80GB",
        "device_bytes": 79 * GIB,
        "uuid": "GPU-private-uuid",
        "cpu_affinity": list(range(8)),
        "cpu_topology": [{"cpu": i, "core": i, "package": 0, "numa_nodes": [0]} for i in range(8)],
        "allocation_id": "private-allocation",
        "host_id_sha256": "a" * 64,
        "versions": {"torch": "2.11.0+cu128"},
        "runtime_precision": {"deterministic": True},
        "torch_git_version": "b" * 40,
        "torch_cuda": "12.8",
        "python": "3.12.0",
        "thread_environment": {"OMP_NUM_THREADS": "2"},
    }


def sample(t, *, used=20 * GIB, stage="training"):
    return {
        "unix_seconds": float(t),
        "device_uuid": "GPU-private-uuid",
        "device_memory_total_bytes": 80 * GIB,
        "device_memory_used_bytes": used + GIB,
        "device_compute_processes": [
            {"pid": 123, "used_gpu_memory_bytes": used, "belongs_to_run": True},
            {"pid": 456, "used_gpu_memory_bytes": GIB, "belongs_to_run": False},
        ],
        "selected_host_rss_sum_bytes": 4 * GIB,
        "selected_host_pss_sum_bytes": 3 * GIB,
        "host_pss_complete": True,
        "stage": stage,
    }


def boundary(phase, t, peak, *, pid=1, reset=True):
    return {
        "phase": phase,
        "pid": pid,
        "rank": 0,
        "unix_seconds": float(t),
        "monotonic_seconds": float(t),
        "reset_peak_after_record": reset,
        "cuda": {"device": 0, "peak_allocated_bytes": peak, "peak_reserved_bytes": peak + 1},
    }


def write(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in value) if lines else json.dumps(value))


def experiment_root(tmp_path: Path, arm="nf4", repeat=1):
    root = tmp_path / f"{arm}_{repeat}"
    spec = Experiment(f"{arm}_{repeat}", arm).as_dict()
    contract = {
        "experiment": spec,
        "action": "train-eval",
        "build": {"code_revision": "c" * 40, "wheel_sha256": "d" * 64},
        "files": [{"path": "/private/source", "sha256": str(i) * 64} for i in range(6)],
        "tools": {"run.py": "e" * 64},
    }
    write(root / "contract.json", contract)
    outcomes = [
        {"name": name, "exit_code": 0, "elapsed_seconds": 700.0 if name == "training" else 10.0} for name in STAGES
    ]
    write(
        root / "completion.json",
        {
            "completed": True,
            "failure": None,
            "experiment": spec,
            "outcomes": outcomes,
            "complete_elapsed_seconds": 770.0,
        },
    )
    write(root / "runtime.json", runtime())
    for row in outcomes:
        write(root / "stages" / row["name"] / "outcome.json", row)
    write(root / "native/run/metrics.jsonl", metrics(), lines=True)
    used = (20 if arm == "nf4" else 27) * GIB
    phases = [
        boundary("adapter_loading_start", 1, 1),
        boundary("adapter_loading_complete", 3, used),
        boundary("trainer_ready", 4, used),
        boundary("initial_validation", 7, used),
        boundary("optimizer_step_1", 9, used),
        boundary("validation_checkpoint_boundary_1", 10, used),
        boundary("checkpoint_begin", 11, used),
        boundary("checkpoint_complete", 13, used),
    ]
    write(root / "stages/training/memory.jsonl", phases, lines=True)
    write(root / "nvml-host.jsonl", [sample(t, used=used) for t in range(1, 14)], lines=True)
    identity = {
        "base": {"revision": "f" * 40},
        "processor_contract": {"identity": "same"},
        "targets": {"language.linear": [2, 2]},
        "adapter": {"rank": 16, "alpha": 32, "seed": 42},
        "initial_adapters": {"tensor": "b" * 64},
        "quantization": {"format": "nf4"} if arm == "nf4" else None,
    }
    write(root / "final-checkpoint.json", {"optimizer_step": 300, "metadata": {"identity": identity}})
    write(root / "native/run/checkpoints/iter_000300/manifest.json", {})
    write(root / "portable/checkpoint_artifacts/iter_000300/manifest.json", {})
    write(root / "scores/summary.json", {"complete": True})
    return root


def test_exact_weighted_throughput_and_nonadditive_pipeline_counters():
    rows = metrics()
    rows[2]["timer/e2e_step_s"] = 20.0
    report = metric_summary(rows, steps=300, effective_batch=8)
    whole = report["windows"]["updates_1_300"]
    assert whole["examples_per_second"] == 2400 / 618
    assert whole["supervised_tokens_per_second"] == 4950 / 618
    assert report["windows"]["updates_6_30"]["update_seconds"]["count"] == 25
    assert report["windows"]["updates_31_300"]["update_seconds"]["count"] == 270
    assert report["initial_validation_seconds"] == 3
    assert report["scheduled_validation_seconds"] == 18
    assert report["preprocessing_last_cumulative_counters"]["preprocessing_processing_seconds"] == 300
    assert whole["update_seconds"]["sum"] != 618 + 300 + 150


@pytest.mark.parametrize(
    "damage", ["float_step", "bool_step", "missing", "duplicate", "nonfinite", "skip", "wrong_batch", "attempt"]
)
def test_bad_native_updates_fail(damage):
    rows = metrics()
    if damage == "float_step":
        rows[1]["step"] = 1.0
    elif damage == "bool_step":
        rows[1]["step"] = True
    elif damage == "missing":
        rows.pop(1)
    elif damage == "duplicate":
        rows[2]["step"] = 1
    elif damage == "nonfinite":
        rows[2]["timer/e2e_step_s"] = float("nan")
    elif damage == "skip":
        rows[2]["optim/step_skipped"] = 1
    elif damage == "wrong_batch":
        rows[2]["train/valid_sequence_count"] = 4
    else:
        rows[2]["optim/attempt_id"] = 3
    with pytest.raises(ValueError):
        metric_summary(rows, steps=300, effective_batch=8)


def test_checkpoint_begin_is_prior_window_and_overall_peak_survives_unknown_boundary():
    rows = [
        boundary("optimizer_step_1", 0, 10),
        boundary("checkpoint_begin", 2, 900),
        boundary("checkpoint_complete", 5, 100),
        boundary("unknown", 6, 1000),
    ]
    report = memory_summary(rows, [sample(i) for i in range(1, 7)], runtime())
    assert report["allocator_peak_allocated_bytes"] == 1000
    assert report["phases"]["checkpoint_write"]["allocator_peak_allocated_bytes"] == 100
    assert report["phases"]["checkpoint_write"]["interval_seconds"]["sum"] == 3
    assert report["phases"]["validation_or_boundary_overhead"]["allocator_peak_allocated_bytes"] == 900
    assert report["phases"]["unclassified"]["allocator_peak_allocated_bytes"] == 1000


def test_missing_reset_or_other_process_cannot_establish_loading_interval():
    rows = [
        boundary("adapter_loading_start", 1, 5, reset=False),
        boundary("adapter_loading_complete", 4, 10),
        boundary("adapter_loading_start", 5, 3, pid=2),
        boundary("adapter_loading_complete", 6, 4, pid=1),
    ]
    assert "loading" not in memory_summary(rows, [], runtime())["phases"]


@pytest.mark.parametrize("damage", ["sentinel", "negative", "identity", "capacity", "ownership"])
def test_invalid_nvml_cannot_establish_memory_gate(damage):
    row = sample(1)
    if damage == "sentinel":
        row["device_compute_processes"][0]["used_gpu_memory_bytes"] = 2**64 - 1
    elif damage == "negative":
        row["device_compute_processes"][0]["used_gpu_memory_bytes"] = -1
    elif damage == "identity":
        row["device_uuid"] = "other"
    elif damage == "capacity":
        row["device_memory_total_bytes"] = 1
    else:
        row["device_compute_processes"][0]["belongs_to_run"] = "true"
    assert (
        sampled_memory([row], expected_uuid=runtime()["uuid"], device_bytes=runtime()["device_bytes"])[
            "gpu_memory_evidence_valid"
        ]
        is False
    )


def test_nvml_capacity_differs_from_torch_usable_memory_and_pss_gaps_are_explicit():
    rows = [sample(1), sample(10)]
    rows[1]["host_pss_complete"] = False
    report = sampled_memory(rows, expected_uuid=runtime()["uuid"], device_bytes=runtime()["device_bytes"])
    assert report["gpu_memory_evidence_valid"] is True
    assert report["owned_device_process_peak_bytes"] == 20 * GIB
    assert report["nvml_capacity_bytes"] == 80 * GIB
    assert report["host_pss_complete"] is False
    assert report["sampling_gap_seconds"]["maximum"] == 9
    assert report["peaks_are_sampled_lower_bounds"] is True


def test_complete_report_keeps_bf16_over_budget_and_contains_no_private_paths(tmp_path):
    root = experiment_root(tmp_path, "bf16")
    report = summarize_run(root)
    assert report["completed"] and report["timing_eligible"], report
    assert report["best_fit_24gib_training_budget"]["observed_peak_within_budget"] is False
    assert report["best_fit_24gib_training_budget"]["control_retained_if_over_budget"] is True
    assert report["complete_elapsed_seconds"] == 770
    assert report["orchestrator_and_verification_residual_seconds"] == 10
    assert report["artifacts"]["portable"]["logical_bytes"] > 0
    assert "/private/" not in json.dumps(report)
    assert "GPU-private-uuid" not in json.dumps(report)
    assert "private-allocation" not in json.dumps(report)


@pytest.mark.parametrize("damage", ["failed", "missing_metrics", "wrong_validation", "profile", "bad_stage"])
def test_failed_incomplete_profiled_runs_remain_unranked(tmp_path, damage):
    root = experiment_root(tmp_path)
    if damage == "failed":
        file = root / "completion.json"
        row = json.loads(file.read_text())
        row["completed"] = False
        write(file, row)
    elif damage == "missing_metrics":
        (root / "native/run/metrics.jsonl").unlink()
    elif damage == "wrong_validation":
        rows = metrics()
        del rows[50]["val/loss_avg"]
        write(root / "native/run/metrics.jsonl", rows, lines=True)
    elif damage == "profile":
        write(root / "native/profiler/window_0.json.gz", {})
    else:
        write(root / "stages/scoring/outcome.json", {"name": "scoring", "exit_code": 1, "elapsed_seconds": 1})
    report = summarize_run(root)
    assert report["timing_eligible"] is False and report["exclusions"]


def test_three_paired_trials_bootstrap_does_not_pool_updates():
    result = paired_bootstrap([1, 2, 3], [2, 4, 6])
    assert result["nf4_over_bf16"] == 0.5
    assert result["paired_bootstrap_95_interval"] == [0.5, 0.5]
    assert result["independent_units"] == 3 and result["repetitions"] == 10000
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [2, 4])


def test_pair_matching_full_cost_memory_and_three_repeat_requirement(tmp_path):
    pairs = [
        (
            summarize_run(experiment_root(tmp_path, "nf4", repeat)),
            summarize_run(experiment_root(tmp_path, "bf16", repeat)),
        )
        for repeat in range(1, 4)
    ]
    report = compare_pairs(pairs)
    assert report["inferential"], report
    assert all(row["sampled_6gib_criterion_met"] for row in report["training_memory_pairs"])
    assert report["complete_elapsed"]["nf4_over_bf16"] == 1
    assert not compare_pairs(pairs[:2])["inferential"]
    damaged = copy.deepcopy(pairs)
    damaged[1][1]["pairing"]["cpu_affinity_sha256"] = "other"
    assert not compare_pairs(damaged)["inferential"]
    damaged = copy.deepcopy(pairs)
    damaged[1][1]["pairing"]["allocation_sha256"] = None
    assert not compare_pairs(damaged)["inferential"]
    with pytest.raises(ValueError):
        compare_pairs([pairs[0], pairs[0], pairs[2]])


def test_phase_invalid_gpu_identity_has_no_numeric_peak_claim():
    rows = [boundary("checkpoint_begin", 0, 10), boundary("checkpoint_complete", 2, 20)]
    samples = [sample(1)]
    samples[0]["device_uuid"] = "wrong GPU"
    phase = memory_summary(rows, samples, runtime())["phases"]["checkpoint_write"]
    assert phase["sampled_gpu_evidence_complete"] is False
    assert phase["sampled_owned_device_process_peak_bytes"] is None
    assert "wrong_device_identity" in phase["sampled_errors"]


def test_joint_nvml_total_and_process_sentinel_is_rejected():
    row = sample(1)
    row["device_memory_total_bytes"] = 2**64 - 1
    row["device_compute_processes"][0]["used_gpu_memory_bytes"] = 2**64 - 1
    result = sampled_memory([row], expected_uuid=runtime()["uuid"], device_bytes=runtime()["device_bytes"])
    assert result["gpu_memory_evidence_valid"] is False
    assert result["owned_device_process_peak_bytes"] is None


def test_malformed_experiment_cannot_echo_private_values_in_excluded_report(tmp_path):
    root = experiment_root(tmp_path)
    path = root / "contract.json"
    contract = json.loads(path.read_text())
    contract["experiment"]["unexpected_secret_field"] = "/private/user/sentinel"
    write(path, contract)
    report = summarize_run(root)
    assert report["timing_eligible"] is False
    assert "sentinel" not in json.dumps(report)
    assert "experiment" not in report


def test_best_fit_discloses_differences_but_preserves_pair_seed_and_repeat_scope(tmp_path):
    pairs = [
        (
            summarize_run(experiment_root(tmp_path, "nf4", repeat)),
            summarize_run(experiment_root(tmp_path, "bf16", repeat)),
        )
        for repeat in range(1, 4)
    ]
    for a, _ in pairs:
        a["experiment"]["rank"] = 32
        a["pairing"]["controls_sha256"] = "different"
        a["pairing"]["adapter_sha256"] = "different"
        a["pairing"]["initial_adapters_sha256"] = "different"
    assert not compare_pairs(pairs)["inferential"]
    report = compare_pairs(pairs, kind="best_fit")
    assert report["inferential"]
    assert report["configuration_differences"][0]["rank"] == [32, 16]
    pairs[0][0]["experiment"]["seed"] = 17
    assert not compare_pairs(pairs, kind="best_fit")["inferential"]
    pairs[0][0]["experiment"]["seed"] = 42
    pairs[0][0]["experiment"]["rank"] = 64
    assert not compare_pairs(pairs, kind="best_fit")["inferential"]


def test_valid_timing_cannot_override_invalid_memory_evidence(tmp_path):
    pairs = [
        (
            summarize_run(experiment_root(tmp_path, "nf4", repeat)),
            summarize_run(experiment_root(tmp_path, "bf16", repeat)),
        )
        for repeat in range(1, 4)
    ]
    pairs[1][0]["memory_by_stage"]["training"]["sampled"]["gpu_memory_evidence_valid"] = False
    report = compare_pairs(pairs)
    assert report["inferential"]
    assert report["training_memory_pairs"][1]["sampled_6gib_criterion_met"] is None
