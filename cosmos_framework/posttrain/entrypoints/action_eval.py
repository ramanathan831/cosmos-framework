# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native task scoring of an explicit frozen base or a complete policy artifact."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def validate_source(config, *, checkpoint, dense, base, unmerged=False):
    if bool(checkpoint) == bool(base) or ((dense or unmerged) and base) or (dense and unmerged):
        raise ValueError("choose exactly one committed checkpoint or explicit frozen --base")
    if base and config.model.backbone_adaptation.mode not in {"bf16_frozen", "nf4_frozen"}:
        raise ValueError("base evaluation requires bf16_frozen or nf4_frozen; adapters require an artifact")


def run(
    config, *, checkpoint: str | None, output: str, dense: bool = False, base: bool = False, unmerged: bool = False
):
    validate_source(config, checkpoint=checkpoint, dense=dense, base=base, unmerged=unmerged)
    import torch
    import torch.distributed as dist

    from cosmos_framework.posttrain.checkpoint.adapter import load_adapter_weights, read_adapter_metadata
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.entrypoints.action_sft import (
        _factory,
        _factory_identity,
        _json,
        _Source,
        evaluate,
        validate_config,
    )
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.utils import misc

    start = time.monotonic()
    rl = validate_config(config)
    if "LOCAL_RANK" not in os.environ or not torch.cuda.is_available():
        raise ValueError("native action evaluation requires single-process CUDA torchrun")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    enable_determinism(strict=True)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    manager = CheckpointManager(checkpoint_format="dcp")
    # Validate the commit and tensor inventory before allocating the immutable base.
    metadata = None if dense or unmerged or base else read_adapter_metadata(checkpoint, manager=manager)
    misc.set_random_seed(seed=rl.seed, by_rank=False)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"native evaluation preserves earlier results; choose a new output: {destination}")
    destination.mkdir(parents=True)
    if dense or unmerged:
        if not config.dense_policy_factory:
            raise ValueError("this native action family has no dense policy loader")
        options = {"unmerged": True} if unmerged else {}
        policy = _factory(config.dense_policy_factory)(config, checkpoint, device=device, output=destination, **options)
        metadata = (policy.unmerged_manifest if unmerged else policy.dense_manifest)["source_metadata"]
        codec = None
    else:
        policy = _factory(config.policy_factory)(config, device=device, output=destination)
        codec = policy.net._cosmos_adapter_checkpoint_codec
        if base:
            metadata = {"identity": codec.identity}
        else:
            load_adapter_weights(checkpoint, codec, manager=manager)
    source = _Source(_factory(config.data_factory)(config, "validation", policy))
    objective = _factory(config.objective_factory)(policy, config)
    callback = _factory(config.evaluation_callback) if config.evaluation_callback else None
    try:
        loaded = time.monotonic()
        metrics, rows = evaluate(policy, objective, source, config, callback)
        if codec is not None:
            codec.validate_frozen_base()
        result = {
            "format": "cosmos-native-action-evaluation-v1",
            "checkpoint": checkpoint,
            "artifact_kind": "fresh-frozen-base"
            if base
            else "dequantized-unmerged-native-lora"
            if unmerged
            else "effective-base-dense"
            if dense
            else "same-base-adapter",
            "identity": metadata["identity"],
            "scorer": _factory_identity(config),
            "source": source.identity,
            "metrics": metrics,
            "rows": rows,
            "startup_seconds": loaded - start,
            "scoring_seconds": time.monotonic() - loaded,
            "workflow_seconds": time.monotonic() - start,
        }
        _json(destination / "evaluation.json", result)
        return result
    finally:
        if objective.close is not None:
            objective.close()


def main():
    from cosmos_framework.utils.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument(
        "--base", action="store_true", help="Score the immutable frozen base without trained parameters"
    )
    parser.add_argument("--output", required=True)
    representation = parser.add_mutually_exclusive_group()
    representation.add_argument(
        "--dense", action="store_true", help="Load a family-native merged effective-base dense export"
    )
    representation.add_argument(
        "--unmerged",
        action="store_true",
        help="Load a self-contained BF16 effective-base native LoRA artifact without merging",
    )
    args, rest = parser.parse_known_args()
    opts = rest[1:] if rest and rest[0] == "--" else rest
    if any("=" not in item for item in opts):
        parser.error("configuration overrides must be key=value")
    config = load_config(args.config, ["--", *opts])
    print(
        json.dumps(
            run(
                config,
                checkpoint=args.checkpoint,
                output=args.output,
                dense=args.dense,
                base=args.base,
                unmerged=args.unmerged,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
