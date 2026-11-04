"""Bounded deployment canaries: both optimizers try with/without recomputation.

No task-quality inference is made from two updates. Keep every OOM attempt.
Only the largest fitting microbatch for each setting is selected, at effective
batch four. A final pair tests full-model nonpaged/paged capacity at batch one,
with checkpointing enabled. The unchanged eight-hour cap includes all attempts.
"""

import json
import os
import subprocess
import time
from pathlib import Path

root = Path(os.environ["RESULTS"])
root.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
limit = int(os.environ.get("WORK_SECONDS", "1680"))
results = []
settings = [(checkpointing, arm, False) for checkpointing in (False, True) for arm in ("adamw", "adamw_bnb_8bit")]
settings += [(True, arm, True) for arm in ("adamw_bnb_8bit", "paged_adamw_8bit")]
if os.environ.get("CAPACITY_PRESSURE_ONLY", "0") == "1":
    settings = [setting for setting in settings if setting[2]]
for checkpointing, arm, pressure in settings:
    for micro in (1,) if pressure else (4, 2, 1):
        # Measured full-model pressure attempts take under two minutes each;
        # keep a four-minute reserve. The broader batch search retains eight.
        reserve = 240 if pressure else 480
        if limit - (time.monotonic() - started) < reserve:
            (root / "pending.json").write_text(
                json.dumps({"arm": arm, "checkpointing": checkpointing, "micro": micro, "native_pressure": pressure})
                + "\n"
            )
            raise SystemExit(0)
        folder = root / f"{arm}-pressure{pressure}-checkpointing{checkpointing}-micro{micro}"
        if folder.exists():
            raise RuntimeError(f"refusing to overwrite capacity evidence: {folder}")
        env = dict(
            os.environ,
            RESULTS=str(folder),
            ARM_ONLY=arm,
            GRAD_CKPT=str(checkpointing).lower(),
            MICRO_OVERRIDE=str(micro),
            ACCUM_OVERRIDE=str((1 if pressure else 4) // micro),
            TRAIN_KEYS="[]" if pressure else os.environ["TRAIN_KEYS"],
            CALIBRATING="1",
            STEPS="2",
            VAL_EVERY="2",
            VAL_SAMPLES="1",
            DEPLOYMENT="0",
            TELEMETRY="1",
        )
        run = subprocess.run(
            ["bash", str(Path(env["SOURCE"]) / "tools/quantized_optimizers/native_canary.sh")], env=env
        )
        log = (folder / (arm + ".log")).read_text()
        status = (
            "pass"
            if run.returncode == 0
            else "oom"
            if "CUDA out of memory" in log or "torch.OutOfMemoryError" in log
            else "failure"
        )
        result = {
            "arm": arm,
            "checkpointing": checkpointing,
            "micro_batch": micro,
            "accumulation": (1 if pressure else 4) // micro,
            "native_pressure": pressure,
            "train_keys": env["TRAIN_KEYS"],
            "returncode": run.returncode,
            "status": status,
            "folder": str(folder),
        }
        try:
            if status == "pass":
                files = list((folder / arm).rglob("metrics.jsonl"))
                if len(files) != 1:
                    raise RuntimeError("missing/ambiguous native metrics")
                rows = [json.loads(line) for line in files[0].read_text().splitlines()]
                updates = [row for row in rows if "train/valid_token_count" in row]
                if rows[-1]["optim/optimizer_step"] != 2 or rows[-1]["optim/nonfinite_skips_total"]:
                    raise RuntimeError("incomplete or skipped capacity updates")
                if any(row["train/valid_token_count"] < 64 * row["train/valid_sequence_count"] for row in updates):
                    raise RuntimeError("insufficient actual supervised tokens")
                telemetry = [
                    json.loads(line) for line in (folder / (arm + ".telemetry.jsonl")).read_text().splitlines()
                ]
                forwards = [row for row in telemetry if row["kind"] == "forward" and row["grad_enabled"]]
                if not forwards or any(
                    not row.get("video_grid_thw")
                    or min(row["sequence_lengths"]) < 512
                    or any(t < 4 or h * w < 128 for t, h, w in row["video_grid_thw"])
                    for row in forwards
                ):
                    raise RuntimeError("actual media/sequence gate failed")
                result["training_seconds"] = sum(row["timer/e2e_step_s"] for row in updates)
        except Exception as error:
            result.update(status="qualification_failure", reason=str(error))
            results.append(result)
            (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            raise
        results.append(result)
        (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        if status == "failure":
            raise RuntimeError(f"non-OOM failure stops dependent canaries: {folder}")
        if status == "pass":
            break
(root / "complete.json").write_text(json.dumps({"complete": True, "seconds": time.monotonic() - started}) + "\n")
