# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native GR00T artifact evaluation through the existing isolated LIBERO simulator gateway."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--sim-python", required=True)
    parser.add_argument("--cases", required=True, help="Prospectively frozen JSON list of LiberoCase mappings")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args, rest = parser.parse_known_args()
    opts = rest[1:] if rest and rest[0] == "--" else rest
    if any("=" not in item for item in opts):
        parser.error("configuration overrides require key=value")
    if os.environ.get("WORLD_SIZE") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("native GR00T rollout requires single-process torchrun with explicit allocated CUDA device")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if not Path(args.sim_python).is_file():
        raise ValueError("the configured isolated LIBERO simulator Python is unavailable")
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
        GR00T_QLORA_SPATIAL_EXECUTION_PROFILE,
        LIBERO_RUNTIME,
        LiberoCase,
        execution_protocol,
    )
    from cosmos_framework.posttrain.entrypoints.action_sft import _factory, _json, validate_config
    from cosmos_framework.posttrain.model.gr00t.action_sft import load_artifact
    from cosmos_framework.posttrain.model.gr00t.rollout import NativeArtifactRollout
    from cosmos_framework.posttrain.roles.env.sim_gateway import SimGatewayBody
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.utils.config import load_config

    cases = [LiberoCase(**row) for row in json.loads(Path(args.cases).read_text())]
    if not cases or len(set(cases)) != len(cases) or any(case.suite != "libero_spatial" for case in cases):
        raise ValueError("GR00T artifact rollout requires distinct, declared LIBERO spatial cases")
    config = load_config(args.config, ["--", *opts])
    validate_config(config)
    import torch
    import torch.distributed as dist

    enable_determinism(strict=True)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    started = time.monotonic()
    gateway = None
    try:
        output.mkdir(parents=True)
        policy = (
            _factory(config.dense_policy_factory)(config, args.checkpoint, device=device, output=output)
            if args.dense
            else load_artifact(config, args.checkpoint, device=device, output=output)
        )
        rollout = NativeArtifactRollout(policy)
        protocol = execution_protocol("libero_spatial", GR00T_QLORA_SPATIAL_EXECUTION_PROFILE)
        gateway = SimGatewayBody(
            sim="libero",
            sink="null",
            embodiment="libero_sim",
            sim_python=args.sim_python,
            sim_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
            sim_options={
                "n_lanes": 1,
                "suite": "libero_spatial",
                "execution_horizon": 8,
                "max_steps": 256,
                "execution_profile": GR00T_QLORA_SPATIAL_EXECUTION_PROFILE,
            },
        )
        probe = gateway.probe()
        result = gateway.run_evaluation(
            policy=rollout, cases=[asdict(case) for case in cases], version=0, seed=args.seed, decisions=32
        )
        episodes = result["episodes"]
        expected = {(case.task_id, case.init_state_id) for case in cases}
        actual = {(row["task_id"], row["init_state_id"]) for row in episodes}
        if (
            len(episodes) != len(cases)
            or actual != expected
            or any(
                type(row["success"]) is not bool
                or not 8 <= row["steps"] <= 256
                or row["steps"] % 8 != 0
                or (not row["success"] and row["steps"] != 256)
                for row in episodes
            )
        ):
            raise RuntimeError("native LIBERO evaluator did not complete the exact declared episode inventory")
        _json(
            output / "rollout.json",
            {
                "format": "gr00t-native-artifact-rollout-v1",
                "checkpoint": args.checkpoint,
                "dense": args.dense,
                "cases": [asdict(case) for case in cases],
                "seed": args.seed,
                "execution_profile": GR00T_QLORA_SPATIAL_EXECUTION_PROFILE,
                "protocol": protocol.identity(),
                "runtime": LIBERO_RUNTIME.identity(),
                "simulator_probe": probe,
                "result": result,
                "successes": sum(row["success"] for row in episodes),
                "episodes": len(episodes),
                "workflow_seconds": time.monotonic() - started,
            },
        )
    finally:
        if gateway is not None:
            gateway.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
