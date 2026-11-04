"""Run an explicitly reserved development/evaluation command list, keeping raw costs."""

import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from tools.quantized_optimizers.run_deployment import monitor, read_model_files, write


def run(root, staging, handle, deadline):
    spec = json.loads(open(os.environ["TASK_SPEC"]).read())
    if spec["purpose"] not in {"development diagnosis", "locked comparative qualification"}:
        raise ValueError("unrecognized study purpose")
    if spec["purpose"] == "locked comparative qualification" and not spec.get("locked_before_evaluation"):
        raise ValueError("final comparisons require a protocol locked before evaluation")
    write(root / "executed_spec.json", spec)
    write(
        root / "spec_identity.json", {"sha256": hashlib.sha256(Path(os.environ["TASK_SPEC"]).read_bytes()).hexdigest()}
    )
    mapping = {"data": os.environ["SFT_DATA_ROOT"], "prepared": str(staging), "root": str(root)}
    for case in spec["cases"]:
        output = root / case["name"]
        output.mkdir(exist_ok=False)
        mapping["case"] = str(output)
        argv = [value.format_map(mapping) for value in case["argv"]]
        env = dict(os.environ, SFT_PREPARED_CORPUS=str(staging))
        env.update({key: value.format_map(mapping) for key, value in case.get("env", {}).items()})
        if "budget" in case:
            budget = dict(case["budget"])
            # A shared corpus copy/preflight is charged to each delivered run,
            # not hidden by amortizing it across the study's independent trials.
            if case.get("charge_staging", True):
                runtime = json.loads((root / f"runtime-{os.environ['SLURM_JOB_ID']}.json").read_text())
                budget["cold_offset_seconds"] = budget.get("cold_offset_seconds", 0) + runtime["staging_seconds"]
            write(output / "budget_spec.json", budget)
            env["OPTIMIZER_BUDGET_SPEC"] = str(output / "budget_spec.json")
        write(output / "command.json", {"argv": argv, "overrides": case.get("env", {}), "purpose": spec["purpose"]})
        before = time.monotonic()
        env["PRODUCT_LAUNCH_MONOTONIC"] = str(before)
        if case.get("read_model", True):
            read_model_files(output)
        with (output / "process.log").open("x") as log:
            process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
            stop = threading.Event()
            thread = threading.Thread(target=monitor, args=(handle, process, output / "memory.jsonl", stop))
            thread.start()
            try:
                code = process.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=30)
                raise
            finally:
                stop.set()
                thread.join()
        write(output / "attempt.json", {"returncode": code, "elapsed_seconds": time.monotonic() - before})
        if code and not case.get("allow_failure", False):
            raise RuntimeError(f"development case failed: {case['name']}")
        if not code and "compare_scores_to" in case:
            previous = Path(case["compare_scores_to"].format_map(mapping))
            old = [json.loads(line) for line in previous.read_text().splitlines()]
            current = [json.loads(line) for line in (output / "scores/scores.jsonl").read_text().splitlines()]
            if current != old:
                raise ValueError("direct checkpoint scoring changed per-question predictions or inputs")
            write(output / "score_parity.json", {"exact_rows": len(current), "reference": str(previous), "match": True})
        if not code and "score_checkpoint" in case:
            checkpoints = list((output / "native").rglob("model.pt"))
            if len(checkpoints) != 1:
                raise ValueError("score the single actual committed checkpoint, never an inferred state")
            scoring = case["score_checkpoint"]
            score_argv = [
                "torchrun",
                "--standalone",
                "--nproc-per-node=1",
                "-m",
                "tools.quantized_optimizers.score_dense",
                "--data",
                scoring["data"].format_map(mapping),
                "--selection",
                scoring["selection"].format_map(mapping),
                "--output",
                str(output / "score"),
                "--seed",
                str(scoring["seed"]),
                "--checkpoint",
                str(checkpoints[0].parent),
            ]
            from tools.quantized_optimizers.run_deployment import bounded_command

            score_start = time.monotonic()
            bounded_command(score_argv, output / "score.log", deadline, env)
            delivered = time.monotonic() - before
            write(
                output / "delivery.json",
                {
                    "checkpoint": str(checkpoints[0].parent),
                    "scoring_argv": score_argv,
                    "scoring_seconds": time.monotonic() - score_start,
                    "launch_to_scored_checkpoint_seconds": delivered,
                    "budget_report": json.loads((output / "diagnostics/budget.json").read_text())
                    if "budget" in case
                    else None,
                    "includes": [
                        "model verification",
                        "initialization",
                        "native training and validation",
                        "native checkpoint",
                        "fresh-process fixed-manifest scoring",
                    ],
                    "add_shared_setup_seconds_from_runtime": True,
                },
            )
    write(root / "complete.json", {"cases": [case["name"] for case in spec["cases"]]})
