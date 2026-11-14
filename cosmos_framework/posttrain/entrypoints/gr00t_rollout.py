# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native GR00T source/artifact evaluation through the existing isolated LIBERO simulator gateway."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def load_rollout_policy(config, checkpoint=None, *, device, output, dense=False, base=False):
    """Load a declared native source or complete trained policy through its factory.

    Read and verify the committed artifact before allocating the immutable base.
    The constructed codec then checks every numerical identity and trained tensor;
    selecting the official head control must never dispatch the LoRA constructor.
    """
    if bool(checkpoint) == bool(base) or (dense and base):
        raise ValueError("choose exactly one native checkpoint or explicit --base; --dense requires a checkpoint")
    from cosmos_framework.posttrain.checkpoint.adapter import load_adapter_weights, read_adapter_metadata
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.entrypoints.action_sft import _factory
    from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec
    from cosmos_framework.posttrain.model.gr00t.qlora_source import recorded_source

    matched = "cosmos_framework.posttrain.model.gr00t.action_sft.build_policy"
    official = "cosmos_framework.posttrain.model.gr00t.official_head.build_policy"
    if config.policy_factory not in {matched, official}:
        raise ValueError("GR00T rollout requires its explicit native matched or official-head policy factory")
    mode = None
    if base and config.policy_factory == matched:
        adaptation = getattr(config.model, "backbone_adaptation", None)
        mode = getattr(adaptation, "mode", None)
        if mode not in {"bf16_frozen", "nf4_frozen"}:
            raise ValueError("GR00T source rollout requires bf16_frozen or nf4_frozen; LoRA requires an artifact")
    if dense:
        if config.policy_factory != matched or config.dense_policy_factory != (
            "cosmos_framework.posttrain.model.gr00t.dense.build_policy"
        ):
            raise ValueError("GR00T dense rollout is unavailable for the official-head control")
        return _factory(config.dense_policy_factory)(config, checkpoint, device=device, output=output)
    expected = {
        "format": "cosmos.family-backbone.v1" if config.policy_factory == matched else "cosmos.gr00t-official-head.v1",
        "base": recorded_source(),
    }
    if config.policy_factory == matched:
        expected["task_contract"] = {"family": "gr00t_n1d7"}
    manager = CheckpointManager(checkpoint_format="dcp")
    metadata = None if base else read_adapter_metadata(checkpoint, manager=manager, expected_identity=expected)
    policy = _factory(config.policy_factory)(config, device=device, output=output)
    codec = getattr(policy.net, "_cosmos_adapter_checkpoint_codec", None)
    if not isinstance(codec, AdapterCheckpointCodec):
        raise ValueError("GR00T rollout policy must expose the native complete trained-parameter checkpoint codec")
    codec.assert_live(policy.net)
    if base:
        if (
            codec.identity.get("format") != expected["format"]
            or codec.identity.get("base") != expected["base"]
            or "training_contract" in codec.identity
            or (
                config.policy_factory == matched
                and (
                    codec.identity.get("task_contract", {}).get("family") != "gr00t_n1d7"
                    or codec.identity.get("adapter", {}).get("mode") != mode
                )
            )
        ):
            raise ValueError("native GR00T source policy differs from its immutable untrained source contract")
        codec.validate_frozen_base()
    else:
        codec.validate_metadata(metadata, purpose="reload")
        load_adapter_weights(checkpoint, codec, manager=manager)
    policy.net.eval()
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument(
        "--base", action="store_true", help="Evaluate the pinned source without an adapter/head artifact"
    )
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--sim-python", required=True)
    parser.add_argument("--cases", required=True, help="Prospectively frozen JSON list of LiberoCase mappings")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--latency-warmup-calls", type=int, default=5)
    parser.add_argument("--output", required=True)
    args, rest = parser.parse_known_args()
    if args.latency_warmup_calls < 0:
        parser.error("--latency-warmup-calls must be nonnegative")
    if args.dense and args.base:
        parser.error("--dense requires --checkpoint; source evaluation does not export or fabricate an artifact")
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
    from cosmos_framework.posttrain.entrypoints.action_sft import _json, validate_config
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
        policy = load_rollout_policy(
            config, args.checkpoint, device=device, output=output, dense=args.dense, base=args.base
        )
        policy_loading_seconds = time.monotonic() - started
        rollout = NativeArtifactRollout(policy, latency_capacity=32 * len(cases))
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
                "artifact_kind": "fresh-frozen-base"
                if args.base
                else "effective-base-dense"
                if args.dense
                else "same-base-trained-policy",
                "model_identity": policy.dense_manifest["source_metadata"]["identity"]
                if args.dense
                else policy.net._cosmos_adapter_checkpoint_codec.identity,
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
                "policy_loading_seconds": policy_loading_seconds,
                "response_latency": rollout.latency_report(warmup_calls=args.latency_warmup_calls),
            },
        )
    finally:
        if gateway is not None:
            gateway.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
