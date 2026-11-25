# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rank-owned optimizer payloads inside the existing atomic DCP transaction.

Block scales and uint8 moments are serialized as opaque rank-local bytes. DCP
writes each unique rank key once; it never interprets scales as parameter shards.
This is deliberately same-topology recovery, not topology-changing conversion.
"""

from __future__ import annotations

import hashlib
import io
import pickle
from copy import deepcopy
from traceback import clear_frames

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.utils._pytree import tree_map

from cosmos_framework.posttrain.utils.dist import run_rank_converged
from cosmos_framework.utils.optimizer_master import local_tensor

_FORMAT = "cosmos_rank_master_v3"


def _deserialize_rank_state(payload, format):
    """Decode the one rank-local payload format this module writes (``_serialize_tensor_rank_state``)."""
    if format != _FORMAT:
        raise ValueError("master optimizer checkpoint format mismatch")
    if not isinstance(payload, tuple) or len(payload) != 3:
        raise ValueError("invalid tensor master optimizer payload")
    size, expected, tensor = payload
    if (
        type(size) is not int
        or size <= 0
        or not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.uint8
        or tensor.ndim != 1
        or not tensor.is_contiguous()
        or tensor.numel() != size
    ):
        raise ValueError("invalid master optimizer byte tensor schema/length")
    content = memoryview(tensor.numpy())
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("master optimizer payload checksum/length mismatch")
    # No arbitrary unpickling on resume: tensors, Python containers/scalars, plus the minimal NumPy
    # types safe_dcp allows for checkpoint values (generator group lr/decay can be NumPy scalars).
    from cosmos_framework.checkpoint.safe_dcp import _BYTES_SAFE_GLOBALS

    try:
        with torch.serialization.safe_globals(_BYTES_SAFE_GLOBALS):
            return torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(f"refusing to load master optimizer payload: {error}") from error


def _serialize_tensor_rank_state(state):
    # A tuple remains one opaque DCP leaf. Its uint8 tensor uses torch.save's
    # raw storage records, avoiding protocol-2 UTF-8 expansion of Python bytes.
    # frombuffer retains the private exported view/BytesIO owner. Nothing in this
    # serialized snapshot aliases live parameters, moments or managed buffers.
    buffer = io.BytesIO()
    torch.save(state, buffer)
    view = buffer.getbuffer()
    tensor = torch.frombuffer(view, dtype=torch.uint8)
    return len(view), hashlib.sha256(view).hexdigest(), tensor


def optimizer_payload(optimizer):
    return {
        "format": _FORMAT,
        "world_size": optimizer.identity["world_size"],
        "ranks": {str(optimizer.identity["rank"]): _serialize_tensor_rank_state(optimizer.state_dict())},
    }


def staged_optimizer(manager, path, optimizer):
    target = {"format": "", "world_size": 0, "ranks": {str(optimizer.identity["rank"]): b""}}
    try:
        if not manager.load_dcp_component(path, "optim", target):
            raise FileNotFoundError("master optimizer checkpoint is missing its rank-local component")

        def stage():
            if target["world_size"] != optimizer.identity["world_size"]:
                raise ValueError("master optimizer checkpoint format/topology mismatch")
            saved = _deserialize_rank_state(target["ranks"][str(optimizer.identity["rank"])], target["format"])
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


def _stage_restore(trainer, path):
    """Read a complete replacement without mutating any live training state."""
    replacement = before_model = model = None
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
        return control, replacement, before_model, model
    finally:
        replacement = before_model = model = None


def _restore(trainer, path):
    """Validate one stable generation before mutation; roll back application failures."""
    replacement = before_model = model = before_optimizer = None
    try:
        manager, optimizer, net = trainer.checkpoint_manager, trainer.opt, trainer._net()
        # The lifecycle also binds provenance around trainer+cursor loading. That
        # outer check alone is too late for transactional restore: a mutation
        # during reads would be detected after the new model/moments are live.
        (control, replacement, before_model, model), _ = manager.load_with_stable_provenance(
            path, lambda: _stage_restore(trainer, path)
        )
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
