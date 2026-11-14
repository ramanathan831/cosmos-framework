# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rank-owned optimizer payloads inside the existing atomic DCP transaction.

Block scales and uint8 moments are serialized as opaque rank-local bytes. DCP
writes each unique rank key once; it never interprets scales as parameter shards.
This is deliberately same-topology recovery, not topology-changing conversion.
"""

from __future__ import annotations

import io
from copy import deepcopy
from traceback import clear_frames

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.utils._pytree import tree_map

from cosmos_framework.posttrain.utils.dist import run_rank_converged
from cosmos_framework.utils.optimizer_master import local_tensor


def optimizer_payload(optimizer):
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    return {
        "format": "cosmos_rank_master_v1",
        "world_size": optimizer.identity["world_size"],
        "ranks": {str(optimizer.identity["rank"]): buffer.getvalue()},
    }


def staged_optimizer(manager, path, optimizer):
    target = {"format": "", "world_size": 0, "ranks": {str(optimizer.identity["rank"]): b""}}
    try:
        if not manager.load_dcp_component(path, "optim", target):
            raise FileNotFoundError("master optimizer checkpoint is missing its rank-local component")

        def stage():
            if target["format"] != "cosmos_rank_master_v1" or target["world_size"] != optimizer.identity["world_size"]:
                raise ValueError("master optimizer checkpoint format/topology mismatch")
            saved = torch.load(
                io.BytesIO(target["ranks"][str(optimizer.identity["rank"])]), map_location="cpu", weights_only=False
            )
            return optimizer.stage_state_dict(saved)

        return run_rank_converged(stage, "master optimizer restore staging")
    finally:
        target.clear()


def restore(trainer, path):
    """A retained failure must not retain replacement moments or model snapshots."""
    try:
        return _restore(trainer, path)
    except BaseException as error:
        pending, seen = [error], set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            clear_frames(current.__traceback__)
            pending.extend(cause for cause in (current.__cause__, current.__context__) if cause is not None)
        raise


def _restore(trainer, path):
    """Validate and stage before mutation; roll back all ranks on application failure."""
    replacement = before_model = model = before_optimizer = None
    try:
        manager, optimizer, net = trainer.checkpoint_manager, trainer.opt, trainer._net()
        trainer._require_checkpoint_boundary()
        control = manager.load_component(path, "trainer")
        if control is None:
            raise ValueError("master optimizer resume requires trainer control state")
        run_rank_converged(lambda: trainer._validate_objective_identity(control), "master restore objective")
        replacement = staged_optimizer(manager, path, optimizer)
        options = StateDictOptions(cpu_offload=True)
        before_model = run_rank_converged(
            lambda: get_model_state_dict(net, options=options), "master restore model snapshot"
        )
        # CPU tensors may already alias a CPU model in tests; clone every tensor.
        before_model = tree_map(
            lambda v: v.detach().clone() if isinstance(v, torch.Tensor) else deepcopy(v), before_model
        )
        model = tree_map(lambda v: v.detach().clone() if isinstance(v, torch.Tensor) else deepcopy(v), before_model)
        if manager.has_dcp_component(path, "model"):
            if not manager.load_dcp_component(path, "model", model):
                raise FileNotFoundError("master optimizer resume requires model state")
        else:
            from cosmos_framework.posttrain.checkpoint.consolidated import fill_sharded_state_from_consolidated

            saved = manager.load_component(path, "model")
            if saved is None:
                raise FileNotFoundError("master optimizer resume requires model state")
            run_rank_converged(
                lambda: fill_sharded_state_from_consolidated(model, saved["net"], context="master model restore"),
                "master consolidated model staging",
            )

        def validate_model():
            for name, master, live in zip(optimizer.names, replacement[2], optimizer.live):
                if not torch.equal(local_tensor(model[name]).cpu(), master.detach().to(dtype=live.dtype, device="cpu")):
                    raise ValueError(f"checkpoint model/master values disagree for {name}")

        run_rank_converged(validate_model, "master/model snapshot consistency")
        before_control = trainer._trainer_state()
        before_validation = deepcopy(trainer.validation_state)
        optimizer.sync_groups()
        before_optimizer = (optimizer.masters, optimizer.backend)
        try:
            # Validate the control program first. Its mutations are rolled back if any
            # rank rejects it; model and optimizer still have their original values.
            resumed = trainer._restore_trainer_state(control)
            run_rank_converged(lambda: set_model_state_dict(net, model), "master restore model commit")
            run_rank_converged(lambda: optimizer.commit_state_dict(replacement), "master restore optimizer commit")
        except BaseException:
            run_rank_converged(lambda: set_model_state_dict(net, before_model), "master restore model rollback")
            run_rank_converged(
                lambda: optimizer.commit_state_dict(before_optimizer), "master restore optimizer rollback"
            )
            trainer._restore_trainer_state(before_control)
            trainer.validation_state = before_validation
            raise
        trainer._resumed_from_checkpoint = True
        return resumed
    finally:
        # These cells are captured by collective callbacks. Clearing frames alone
        # cannot release their contents while a retained traceback owns a callback.
        replacement = before_model = model = before_optimizer = None
