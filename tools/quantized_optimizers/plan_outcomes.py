"""Expand a locked outcome protocol into existing bounded native study commands.

This does not choose budgets, tune a recipe, submit work, or inspect task scores.
All quality cases score their own committed state on the same fixed manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path

ARMS = ("adamw", "adamw_bnb_8bit")


def make_spec(protocol, seed, protocol_sha256):
    if not protocol["locked_before_evaluation"] or protocol["paired_seeds"] != [17, 29, 43]:
        raise ValueError("requires the locked three-seed protocol")
    if protocol["timing_restarts"] != 3:
        raise ValueError("retain three paired timing restarts")
    if seed not in protocol["paired_seeds"]:
        raise ValueError("unplanned seed")
    cases = []
    seed_index = protocol["paired_seeds"].index(seed)
    evaluation = protocol["evaluation"]
    if seed == 17:
        cases.append(
            {
                "name": "frozen",
                "read_model": False,
                "argv": [
                    "torchrun",
                    "--standalone",
                    "--nproc-per-node=1",
                    "-m",
                    "tools.quantized_optimizers.score_dense",
                    "--data",
                    evaluation["data"],
                    "--selection",
                    evaluation["manifest"],
                    "--output",
                    "{case}/score",
                    "--seed",
                    str(seed),
                ],
            }
        )
    cases.append(
        {
            "name": "training_exposure",
            "read_model": False,
            "argv": [
                "python",
                "-m",
                "tools.quantized_optimizers.inspect_exposure",
                "--data",
                "{data}/train-mixed.jsonl",
                "--corpus",
                "{prepared}",
                "--seed",
                str(seed),
                "--output",
                "{case}/exposure.json",
            ],
            "env": {"CUDA_VISIBLE_DEVICES": ""},
        }
    )
    designs = [("matched", None), ("training", "training"), ("end_to_end", "end_to_end")]
    designs += [(f"target_restart{restart}", "target") for restart in range(3)]
    for pair_index, (label, kind) in enumerate(designs):
        arms = ARMS[::-1] if (seed_index + pair_index) % 2 else ARMS
        for arm in arms:
            name = f"{label}_{arm}"
            argv = [
                "torchrun",
                "--standalone",
                "--nproc-per-node=1",
                "-m",
                "tools.quantized_optimizers.budgeted_sft" if kind else "tools.quantized_optimizers.task_development",
                "--config",
                "cosmos_framework/posttrain/configs/sft_optimizer_task.py",
                "--",
                f"job.name={name}",
                f"rl.seed={seed}",
                f"rl.optimizer.name={arm}",
                f"rl.optimizer.impl={'fused' if arm == 'adamw' else ''}",
                f"rl.optimizer.bnb_min_8bit_size={4096 if arm == 'adamw' else 16384}",
                f"model_hf.grad_ckpt={'true' if arm == 'adamw' else 'false'}",
                "rl.emergency_checkpoint_components=full",
            ]
            if arm == "adamw" and protocol["fused_checkpoint_modules"]:
                argv.append("model_hf.grad_ckpt_modules=[" + ",".join(protocol["fused_checkpoint_modules"]) + "]")
            case = {
                "name": name,
                "argv": argv,
                "env": {
                    "IMAGINAIRE_OUTPUT_ROOT": "{case}/native",
                    "TASK_DIAGNOSTIC_OUTPUT": "{case}/diagnostics",
                    "TASK_DEVELOPMENT_DATA": "{data}/dev-task.jsonl",
                },
                "score_checkpoint": {"data": evaluation["data"], "selection": evaluation["manifest"], "seed": seed},
                "design": label,
                "optimizer": arm,
                "seed": seed,
            }
            if kind:
                case["budget"] = dict(protocol["budgets"][kind][arm], kind=kind)
                if kind == "target":
                    case["budget"]["quality_target"] = protocol["validation_quality_target"]
            cases.append(case)
    return {
        "purpose": "locked comparative qualification",
        "locked_before_evaluation": protocol["locked_before_evaluation"],
        "protocol_sha256": protocol_sha256,
        "seed": seed,
        "prepared_verified_rows": 751,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    value = args.protocol.read_bytes()
    protocol = json.loads(value)
    args.output.mkdir(exist_ok=False, parents=True)
    for seed in protocol["paired_seeds"]:
        spec = make_spec(protocol, seed, hashlib.sha256(value).hexdigest())
        (args.output / f"seed{seed}.json").write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
