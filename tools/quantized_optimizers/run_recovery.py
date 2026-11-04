"""Additional dense native40-to44 checkpoint continuation; correctness only."""

import json
import os
import sys
import time

from tools.quantized_optimizers.run_deployment import CONFIG, bounded_command, read_model_files, write


def run(root, staging, deadline):
    arm = os.environ["RECOVERY_ARM"]
    if arm not in ("adamw", "adamw_bnb_8bit"):
        raise ValueError("unsupported recovery arm")
    checkpoints = None
    results = {}
    for phase in ("reference", "resumed"):
        output = root / phase
        output.mkdir(exist_ok=False)
        env = dict(
            os.environ,
            SFT_PREPARED_CORPUS=str(staging),
            IMAGINAIRE_OUTPUT_ROOT=str(output / "native"),
            STATE_DIGEST_PATH=str(output / "state.json"),
        )
        command = [
            "torchrun",
            "--standalone",
            "--nproc-per-node=1",
            "-m",
            "tools.quantized_optimizers.prepared_canary",
            "--config",
            CONFIG,
            "--",
            f"job.name={phase}",
            "rl.seed=17",
            "rl.max_steps=44",
            "rl.save_every=40",
            f"rl.optimizer.name={arm}",
            f"rl.optimizer.impl={'fused' if arm == 'adamw' else ''}",
            f"rl.optimizer.bnb_min_8bit_size={4096 if arm == 'adamw' else 16384}",
            f"model_hf.grad_ckpt={'true' if arm == 'adamw' else 'false'}",
        ]
        if phase == "resumed":
            command.append(f"rl.resume={checkpoints[0]}")
        write(
            output / "command.json", {"argv": command, "purpose": "correctness only; fixed44update horizon and cycle40"}
        )
        started = time.monotonic()
        read_model_files(output)
        bounded_command(command, output / "train.log", deadline, env)
        results[phase] = json.loads((output / "state.json").read_text())
        metrics = list(output.rglob("metrics.jsonl"))
        if len(metrics) != 1:
            raise ValueError("ambiguous recovery metrics")
        rows = [json.loads(line) for line in metrics[0].read_text().splitlines()]
        updates = [row for row in rows if "optim/optimizer_step" in row]
        expected = list(range(1, 45)) if phase == "reference" else list(range(41, 45))
        if [int(row["optim/optimizer_step"]) for row in updates] != expected:
            raise ValueError("recovery executed unexpected update clocks")
        if any(row["optim/nonfinite_skips_total"] for row in updates):
            raise ValueError("recovery skipped an update")
        write(
            output / "complete.json",
            {"phase": phase, "elapsed_seconds": time.monotonic() - started, "updates": expected},
        )
        if phase == "reference":
            checkpoints = [p.parent for p in output.rglob("model.pt") if p.parent.name.endswith("40")]
            if len(checkpoints) != 1:
                raise ValueError(f"expected one committed checkpoint40, found{checkpoints}")
            bounded_command(
                [
                    sys.executable,
                    "-m",
                    "tools.quantized_optimizers.checkpoint_digest",
                    str(checkpoints[0]),
                    str(root / "checkpoint40_parameter.json"),
                    "--parameter",
                    results[phase]["update_probes"][0]["name"],
                ],
                root / "checkpoint40_model.log",
                deadline,
            )
            probes = {row["step"]: row for row in results[phase]["update_probes"]}
            saved = json.loads((root / "checkpoint40_parameter.json").read_text())["parameter"]
            if saved != {k: probes[40][k] for k in ("name", "sha256")}:
                raise ValueError("checkpoint40 parameter differs from the native live parameter at40")
            if probes[40]["sha256"] == probes[44]["sha256"]:
                raise ValueError("four continuation updates did not change model values")
        cursor = results[phase]["cursor_state"]["states"]["0"]
        if cursor["epoch"] != 2 or cursor["position"] != 3:
            raise ValueError("continuation did not cross the declared173-row epoch boundary")
    keys = ("model", "optimizer", "scheduler", "update", "cursor", "rng")
    matches = {key: results["reference"][key] == results["resumed"][key] for key in keys}
    matches["next_and_subsequent_live_updates"] = results["reference"]["update_probes"][-4:] == results["resumed"][
        "update_probes"
    ] and [row["step"] for row in results["resumed"]["update_probes"]] == [41, 42, 43, 44]
    write(root / "parity.json", matches)
    if not all(matches.values()):
        raise ValueError(f"fresh-process40to44recovery failed: {matches}")
    write(
        root / "complete.json",
        {
            "arm": arm,
            "matched_components": list(keys),
            "reference_updates": 44,
            "resumed_updates": 4,
            "qualified_scope": "ordinary FP32 16-block native SFT, strict kernels, prepared inputs; consolidated exact resume",
        },
    )
