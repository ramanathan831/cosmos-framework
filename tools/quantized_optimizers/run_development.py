"""Run remaining preregistered arms, one at a time, within one bounded allocation.

A completed arm is never rerun. Failed attempts are retained and stop this driver.
An allocation may end with pending arms; those still require a later allocation
within the campaign's three-job cap. This driver never submits jobs itself.
"""

import json
import os
import subprocess
import time
from pathlib import Path

root = Path(os.environ["RESULTS"])
root.mkdir(parents=True, exist_ok=True)
plan = [(17, a) for a in ("adamw", "adamw_bnb_8bit", "paged_adamw_8bit")]
plan += [(29, a) for a in ("adamw_bnb_8bit", "paged_adamw_8bit", "adamw")]
plan += [(43, a) for a in ("paged_adamw_8bit", "adamw", "adamw_bnb_8bit")]
started = time.monotonic()
for seed, arm in plan:
    output = root / f"seed{seed}" / arm
    if (output / "complete.json").exists():
        continue
    if int(os.environ.get("WORK_SECONDS", "3500")) - (time.monotonic() - started) < 600:
        (root / "pending.json").write_text(json.dumps({"next_seed": seed, "next_arm": arm}) + "\n")
        break
    if output.exists():
        raise RuntimeError(f"prior incomplete attempt requires inspection, refusing overwrite: {output}")
    env = dict(os.environ, SEED=str(seed), SEEDS="", ARM_ONLY=arm, RESULTS=str(output))
    status = subprocess.run(
        ["bash", str(Path(env["SOURCE"]) / "tools/quantized_optimizers/native_canary.sh")], env=env
    ).returncode
    if status:
        raise RuntimeError(f"arm failed with {status}: {output}")
    metrics = list((output / arm).rglob("metrics.jsonl"))
    if len(metrics) != 1:
        raise RuntimeError(f"ambiguous metrics: {output}")
    rows = [json.loads(line) for line in metrics[0].read_text().splitlines()]
    if (
        rows[-1].get("optim/optimizer_step") != int(env.get("STEPS", "40"))
        or rows[-1].get("optim/nonfinite_skips_total") != 0
    ):
        raise RuntimeError(f"incomplete/nonfinite comparison arm: {output}")
    (output / "complete.json").write_text(json.dumps({"seed": seed, "arm": arm, "metrics": str(metrics[0])}) + "\n")
else:
    (root / "pending.json").write_text(json.dumps({"complete": True}) + "\n")
