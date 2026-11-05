"""Apply the prospective three-restart shared-loading protocol; development only."""

import json
from pathlib import Path
from statistics import mean, stdev

from analyze_screen import read_raw

ROOT = Path(__file__).parent


def interval(values):
    assert len(values) == 3
    center = mean(values)
    width = 4.302652729911275 * stdev(values) / 3**0.5
    return {"mean": center, "ci95": [center - width, center + width], "values": values, "method": "paired t, df=2"}


def main():
    previous = read_raw(ROOT.parent / "quantized_optimizers_round4/runtime_canary_raw.json.gz")
    protocol = json.loads((ROOT / "loading_protocol.json").read_text())
    cases = []
    for restart in range(3):
        data = read_raw(ROOT / f"loading{restart}_raw.json.gz")
        expected = {f"{arm}_{mode}" for arm in ("fused", "eightbit") for mode in ("serial", "optimized")}
        if restart == 0:
            expected.update(("pinned_controls", "checksum_rejected"))
        assert set(json.loads(data["complete.json"])["cases"]) == expected
        current = {}
        for arm in ("fused", "eightbit"):
            old = json.loads(previous[arm + "_serial/scores/summary.json"])
            old_rows = [json.loads(s) for s in previous[arm + "_serial/scores/scores.jsonl"].splitlines()]
            for mode in ("serial", "optimized"):
                name = f"{arm}_{mode}"
                summary = json.loads(data[name + "/scores/summary.json"])
                attempt = json.loads(data[name + "/attempt.json"])
                checksum = json.loads(data[name + "/scores/checkpoint-read.json"])
                rows = [json.loads(s) for s in data[name + "/scores/scores.jsonl"].splitlines()]
                assert rows == old_rows and len(rows) == 32
                for key in (
                    "checkpoint_provenance",
                    "checkpoint",
                    "metrics",
                    "objective_identity",
                    "corpus_sha256",
                    "selection_sha256",
                    "processor_identity",
                    "metric_contract",
                ):
                    assert summary[key] == old[key], key
                assert checksum["sha256"] == summary["expected_checkpoint_sha256"] == protocol["checkpoints"][arm]
                assert attempt["returncode"] == 0
                memory = [json.loads(s) for s in data[name + "/memory.jsonl"].splitlines()]
                current[name] = {
                    "attempt_seconds": attempt["elapsed_seconds"],
                    "scoring_scope_seconds": summary["elapsed_seconds"],
                    "phase_seconds": summary["phase_seconds"],
                    "peak_host_pss_bytes": max(x.get("process_tree_pss") or 0 for x in memory),
                    "peak_process_gpu_bytes": max(x.get("matched_process_gpu_bytes") or 0 for x in memory),
                    "peak_device_bytes": max(x.get("device_used") or 0 for x in memory),
                    "checkpoint_sha256": checksum["sha256"],
                    "exact_rows": len(rows),
                }
        if restart == 0:
            assert "15 passed" in data["pinned_controls/process.log"]
            failure = json.loads(data["checksum_rejected/attempt.json"])
            assert failure["returncode"] != 0
            assert (
                "checkpoint SHA-256 differs from the pinned scoring identity" in data["checksum_rejected/process.log"]
            )
            assert not any(k.startswith("checksum_rejected/scores/") for k in data)
        cases.append(current)
    ratios = {
        arm: interval([c[arm + "_serial"]["attempt_seconds"] / c[arm + "_optimized"]["attempt_seconds"] for c in cases])
        for arm in ("fused", "eightbit")
    }
    ratios["overall"] = interval(
        [
            sum(c[a + "_serial"]["attempt_seconds"] for a in ("fused", "eightbit"))
            / sum(c[a + "_optimized"]["attempt_seconds"] for a in ("fused", "eightbit"))
            for c in cases
        ]
    )
    phase_ratios = {}
    for arm in ("fused", "eightbit"):
        phase_ratios[arm] = interval(
            [
                sum(
                    c[arm + "_serial"]["phase_seconds"][key]
                    for key in ("prepared_verification", "checkpoint_read_and_restore")
                )
                / sum(
                    c[arm + "_optimized"]["phase_seconds"][key]
                    for key in ("prepared_verification", "checkpoint_read_and_restore")
                )
                for c in cases
            ]
        )
    gates = {
        "384_exact_development_rows": True,
        "checkpoint_sha_provenance_objective_input_exact": True,
        "helper_and_real_cuda_fail_closed_checks": True,
        "overall_speed_lower_bound_above_1_05": ratios["overall"]["ci95"][0] > 1.05,
        "each_arm_speed_lower_bound_above_1": all(ratios[a]["ci95"][0] > 1 for a in ("fused", "eightbit")),
    }
    output = {
        "complete": True,
        "accepted": all(gates.values()),
        "gates": gates,
        "paired_speed_ratios": ratios,
        "nonoverlapping_verification_plus_checkpoint_phase_ratios": phase_ratios,
        "cases": cases,
        "scope": "Three counterbalanced timing restarts on the same A10080GB node, two immutable optimizer checkpoints, 32 development questions. Timing restarts are not new quality seeds or recording families. Shared storage cache/concurrency and startup variation remain. No optimizer end-to-end or inference-latency claim. Concurrent read/restore child timers must not be summed.",
    }
    (ROOT / "loading_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "cases"}))


if __name__ == "__main__":
    main()
