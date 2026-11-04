"""Choose largest fitting batch under a fixed effective batch, before training.

This is an OOM capacity canary, not a claim about paging under pressure or timing.
All attempts and failures are retained. Non-OOM failures stop the comparison.
"""

import json
import os
import subprocess
from pathlib import Path

result = {}
supported = ("adamw", "adamw_bnb_8bit", "paged_adamw_8bit")
arms = os.environ.get("CAPACITY_ARMS", ",".join(supported)).split(",")
if len(arms) != len(set(arms)) or any(arm not in supported for arm in arms):
    raise ValueError("CAPACITY_ARMS must select unique supported optimizer names")
for arm in arms:
    for micro in (4, 2, 1):
        folder = Path(os.environ["RESULTS"]) / "capacity" / f"{arm}-micro{micro}"
        if folder.exists():
            raise RuntimeError(f"refusing to overwrite capacity evidence: {folder}")
        env = dict(os.environ)
        env.update(
            RESULTS=str(folder),
            ARM_ONLY=arm,
            DEPLOYMENT="0",
            CALIBRATING="1",
            MICRO_OVERRIDE=str(micro),
            ACCUM_OVERRIDE=str(4 // micro),
            STEPS="2",
            VAL_EVERY="2",
            VAL_SAMPLES="1",
            TELEMETRY="1",
        )
        run = subprocess.run(
            ["bash", str(Path(env["SOURCE"]) / "tools/quantized_optimizers/native_canary.sh")], env=env
        )
        (folder / "returncode.txt").write_text(str(run.returncode) + "\n")
        if run.returncode == 0:
            if int(env.get("VIDEO_PIXELS", "65536")) > 65536:
                telemetry = [
                    json.loads(line) for line in (folder / (arm + ".telemetry.jsonl")).read_text().splitlines()
                ]
                training = [row for row in telemetry if row["kind"] == "forward" and row["grad_enabled"]]
                if not training or any(
                    not row.get("video_grid_thw")
                    or any(t < 4 or h * w < 128 for t, h, w in row["video_grid_thw"])
                    or min(row["sequence_lengths"]) < 512
                    for row in training
                ):
                    raise RuntimeError(f"actual media/sequence shape gate failed: {folder}")
                files = list((folder / arm).rglob("metrics.jsonl"))
                rows = [json.loads(line) for line in files[0].read_text().splitlines()]
                updates = [row for row in rows if "train/valid_token_count" in row]
                if any(row["train/valid_token_count"] < 64 * row["train/valid_sequence_count"] for row in updates):
                    raise RuntimeError(f"substantive target-token canary failed: {folder}")
            result[arm] = {"micro_batch": micro, "accumulation": 4 // micro, "evidence": str(folder)}
            break
        log = (folder / (arm + ".log")).read_text()
        if "CUDA out of memory" not in log and "torch.OutOfMemoryError" not in log:
            raise RuntimeError(f"non-OOM failure; stop comparison: {folder}")
    else:
        raise RuntimeError(f"no fitting microbatch: {arm}")
Path(os.environ["CAPACITY_FILE"]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
