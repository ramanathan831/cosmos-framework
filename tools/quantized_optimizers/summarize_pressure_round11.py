"""Summarize round-11 GR00T pressure runs from their native window records.

Usage: python summarize_pressure_round11.py <run_dir> [<run_dir> ...] > summary.json
Each run_dir is a launcher RESULTS directory. Reports completion, per-update phase
seconds, per-rank caching-allocator peaks/fragmentation and backend-step device time,
plus the sampled learner-process GPU peak from the existing memory samplers.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GIB = 2**30


def _mean(values):
    return statistics.fmean(values) if values else None


def summarize(run: Path) -> dict:
    windows_path = next(run.glob("runs/**/windows.jsonl"), None)
    rows = [json.loads(line) for line in windows_path.read_text().splitlines()] if windows_path else []
    attempt = json.loads((run / "attempt.json").read_text()) if (run / "attempt.json").exists() else None
    summaries = sorted(run.glob("runs/**/summary-step-*.json"))
    updates = []
    for row in rows:
        ranks = row.get("cuda_allocator") or []
        updates.append(
            {
                "optimizer_step": row["optimizer_step"],
                "phase_seconds": row["phase_seconds"],
                "window_seconds": row["seconds"],
                "peak_allocated_gib": [r["after_update"]["allocated_bytes.all.peak"] / GIB for r in ranks],
                "peak_reserved_gib": [r["after_update"]["reserved_bytes.all.peak"] / GIB for r in ranks],
                "pre_update_peak_reserved_gib": [
                    r["before_update"]["reserved_bytes.all.peak"] / GIB for r in ranks if r["before_update"]
                ],
                "peak_inactive_split_gib": [r["after_update"]["inactive_split_bytes.all.peak"] / GIB for r in ranks],
                "cumulative_alloc_retries": [r["after_update"]["num_alloc_retries"] for r in ranks],
                "cumulative_ooms": [r["after_update"]["num_ooms"] for r in ranks],
                "backend_step_ms": [r.get("optimizer_backend_step_ms") for r in ranks],
            }
        )
    # Learner process = second-largest per-process GPU user on each device (the largest is
    # the separate 55 GiB reservation). Sampled, not exact.
    learner_peaks = {}
    for sampler in sorted(run.glob("memory-gpu*.jsonl")):
        per_pid = defaultdict(int)
        for line in sampler.read_text().splitlines():
            sample = json.loads(line)
            for process in sample.get("gpu_processes") or []:
                per_pid[process["pid"]] = max(per_pid[process["pid"]], process["bytes"])
        ordered = sorted(per_pid.values(), reverse=True)
        learner_peaks[sampler.stem] = ordered[1] / 2**20 if len(ordered) > 1 else None
    steady = updates[1:]  # first resumed update includes warm-up allocations
    return {
        "run": run.name,
        "native_returncode": None if attempt is None else attempt["returncode"],
        "native_seconds": None if attempt is None else attempt["native_seconds"],
        "completed_updates": [u["optimizer_step"] for u in updates],
        "final_summary": json.loads(summaries[-1].read_text()) if summaries else None,
        "mean_update_seconds_excl_first": _mean([u["phase_seconds"]["update_seconds"] for u in steady]),
        "mean_window_seconds_excl_first": _mean([u["window_seconds"] for u in steady]),
        "mean_backend_step_ms_excl_first": _mean(
            [ms for u in steady for ms in u["backend_step_ms"] if ms is not None]
        ),
        "max_peak_allocated_gib": max((v for u in updates for v in u["peak_allocated_gib"]), default=None),
        "max_peak_reserved_gib": max((v for u in updates for v in u["peak_reserved_gib"]), default=None),
        "max_peak_inactive_split_gib": max((v for u in updates for v in u["peak_inactive_split_gib"]), default=None),
        "sampled_learner_process_peak_mib": learner_peaks,
        "updates": updates,
    }


if __name__ == "__main__":
    print(json.dumps([summarize(Path(path)) for path in sys.argv[1:]], indent=1))
