"""Finish fixed comparison cells without rerunning training or selecting outcomes."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.quantized_optimizers.run_deployment import bounded_command, write

ARMS = ("adamw", "adamw_bnb_8bit")


def verify(root, deadline, workers=1):
    def one(item):
        restart, arm = item
        output = root / f"restart{restart}_{arm}"
        checkpoint = json.loads((output / "complete.json").read_text())["checkpoint"]
        target = output / "checkpoint_digest.json"
        if not target.exists():
            bounded_command(
                [sys.executable, "-m", "tools.quantized_optimizers.checkpoint_digest", checkpoint, str(target)],
                output / f"checkpoint_digest-{os.environ['SLURM_JOB_ID']}.log",
                deadline,
            )
        return item, json.loads(target.read_text())

    items = [(restart, arm) for restart in range(3) for arm in ARMS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        identities = dict(pool.map(one, items))
    matches = {
        f"restart{restart}_{arm}_{key}": identities[0, arm][key] == identities[restart, arm][key]
        for restart in (1, 2)
        for arm in ARMS
        for key in ("model", "optim", "rng", "scheduler", "data", "optimizer_step")
    }
    write(root / "restart_parity.json", matches)
    if not all(matches.values()):
        raise RuntimeError("deterministic paired restarts changed model/optimizer/control state")


def score(root, staging, deadline):
    seed = int(os.environ["TRIAL_SEED"])
    if seed not in (17, 29, 43):
        raise ValueError("unplanned scoring seed")
    parity = json.loads((root / "restart_parity.json").read_text())
    if len(parity) != 24 or not all(parity.values()):
        raise ValueError("scoring requires all repeat-checkpoint comparisons")
    preparation = Path(os.environ["PREPARATION"])
    for arm in ("frozen", *ARMS) if seed == 17 else ARMS:
        output = root / f"final_{arm}"
        checkpoint = (
            None if arm == "frozen" else json.loads((root / f"restart0_{arm}/complete.json").read_text())["checkpoint"]
        )
        if (output / "summary.json").exists():
            summary = json.loads((output / "summary.json").read_text())
            rows = [json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()]
            selection = json.loads((preparation / "final-selection/manifest.json").read_text())
            if not summary["complete"] or summary["seed"] != seed or summary["checkpoint"] != checkpoint:
                raise ValueError("completed score identity mismatch")
            if [row["id"] for row in rows] != [row["id"] for row in selection["rows"]]:
                raise ValueError("completed score coverage mismatch")
            # A committed score precedes process-group teardown. Retain the failed
            # launcher receipt; do not repeat or select successful evaluation rows.
            continue
        if output.exists():
            output.rename(root / f"aborted_final_{arm}_before_{os.environ['SLURM_JOB_ID']}")
        command = [
            "torchrun",
            "--standalone",
            "--nproc-per-node=1",
            "-m",
            "tools.quantized_optimizers.score_dense",
            "--data",
            str(preparation / "final-selection/validation.jsonl"),
            "--selection",
            str(preparation / "final-selection/manifest.json"),
            "--output",
            str(output),
            "--seed",
            str(seed),
        ]
        if checkpoint is not None:
            command.extend(["--checkpoint", checkpoint])
        bounded_command(
            command,
            root / f"final_{arm}-{os.environ['SLURM_JOB_ID']}.log",
            deadline,
            env=dict(os.environ, SFT_PREPARED_CORPUS=str(staging)),
        )
    write(
        root / "complete.json",
        {
            "seed": seed,
            "all_training_verification_and_scores_complete": True,
            "completion_job": os.environ["SLURM_JOB_ID"],
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    verify(args.root, time.monotonic() + args.seconds, args.workers)


if __name__ == "__main__":
    main()
