"""Choose largest fitting batch under a fixed effective batch, before training.

This is an OOM capacity canary, not a claim about paging under pressure or timing.
All attempts and failures are retained. Non-OOM failures stop the comparison.
"""

import json
import os
import subprocess
from pathlib import Path

result = {}
for arm in ("adamw", "adamw_bnb_8bit", "paged_adamw_8bit"):
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
            result[arm] = {"micro_batch": micro, "accumulation": 4 // micro, "evidence": str(folder)}
            break
        log = (folder / (arm + ".log")).read_text()
        if "CUDA out of memory" not in log and "torch.OutOfMemoryError" not in log:
            raise RuntimeError(f"non-OOM failure; stop comparison: {folder}")
    else:
        raise RuntimeError(f"no fitting microbatch: {arm}")
Path(os.environ["CAPACITY_FILE"]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
