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
from copy import deepcopy
from traceback import clear_frames

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.utils._pytree import tree_map

from cosmos_framework.posttrain.utils.dist import run_rank_converged
from cosmos_framework.utils.optimizer_master import local_tensor

_CHUNK_BYTES = 64 * 1024 * 1024


def _serialize_rank_state(state):
    # DCP serializes non-tensor leaves with pickle protocol 2. A single bytes
    # object can exceed its 4-GiB string limit on real action-model shards. A
    # tuple of bounded bytes remains one opaque leaf, without that size limit.
    with io.BytesIO() as buffer:
        torch.save(state, buffer)
        size = buffer.tell()
        buffer.seek(0)
        chunks, digest = [], hashlib.sha256()
        while chunk := buffer.read(_CHUNK_BYTES):
            chunks.append(chunk)
            digest.update(chunk)
    return size, digest.hexdigest(), tuple(chunks)


def _deserialize_rank_state(payload, format):
    if format == "cosmos_rank_master_v1":
        if not isinstance(payload, bytes):
            raise ValueError("invalid legacy master optimizer payload")
        content = payload
    elif format == "cosmos_rank_master_v2":
        if not isinstance(payload, tuple) or len(payload) != 3:
            raise ValueError("invalid chunked master optimizer payload")
        size, expected, chunks = payload
        if not isinstance(chunks, tuple) or not chunks or any(not isinstance(c, bytes) or not c for c in chunks):
            raise ValueError("invalid master optimizer chunks")
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(chunk)
        if sum(map(len, chunks)) != size or digest.hexdigest() != expected:
            raise ValueError("master optimizer payload checksum/length mismatch")
        content = b"".join(chunks)
    elif format == "cosmos_rank_master_v3":
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
    else:
        raise ValueError("master optimizer checkpoint format mismatch")
    return torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)


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
        "format": "cosmos_rank_master_v3",
        "world_size": optimizer.identity["world_size"],
        "ranks": {str(optimizer.identity["rank"]): _serialize_tensor_rank_state(optimizer.state_dict())},
    }


def staged_optimizer(manager, path, optimizer):
    target = {"format": "", "world_size": 0, "ranks": {str(optimizer.identity["rank"]): b""}}
    try:
        if optimizer.identity["world_size"] == 1:
            from cosmos_framework.posttrain.checkpoint.adapter import validate_dcp_target_inventory

            # The opaque payload is deliberately one non-tensor DCP entry. Refuse
            # additional/missing entries before DCP can partially fill the target.
            validate_dcp_target_inventory(path, "optim", target)
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
        manager, optimizer, net = trainer.checkpoint_manager, trainer.opt, trainer._checkpoint_state_net()
        trainer._require_checkpoint_boundary()
        codec = trainer._adapter_checkpoint_codec()
        if codec is not None:
            # Repeat inside stable provenance; early admission alone cannot bind
            # metadata, RNG/data and model tensors to one immutable generation.
            trainer._validate_adapter_resume_state(path)
            codec.validate_frozen_base()
        control = manager.load_component(path, "trainer")
        if control is None:
            raise ValueError("master optimizer resume requires trainer control state")
        run_rank_converged(lambda: trainer._validate_objective_identity(control), "master restore objective")
        replacement = staged_optimizer(manager, path, optimizer)
        if codec is not None and trainer.lr_scheduler is not None:
            if control["lr_scheduler"]["_last_lr"] != [g["lr"] for g in replacement[1].param_groups]:
                raise ValueError("checkpoint scheduler and optimizer learning rates disagree")
        options = StateDictOptions(cpu_offload=True)
        before_model = run_rank_converged(
            lambda: get_model_state_dict(net, options=options), "master restore model snapshot"
        )
        # CPU tensors may already alias a CPU model in tests; clone every tensor.
        before_model = tree_map(
            lambda v: v.detach().clone() if isinstance(v, torch.Tensor) else deepcopy(v), before_model
        )
        model = tree_map(lambda v: v.detach().clone() if isinstance(v, torch.Tensor) else deepcopy(v), before_model)
        if codec is not None:
            from cosmos_framework.posttrain.checkpoint.adapter import validate_dcp_target_inventory

            validate_dcp_target_inventory(path, "model", model)
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
        manager, optimizer, net = trainer.checkpoint_manager, trainer.opt, trainer._checkpoint_state_net()
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
            codec = trainer._adapter_checkpoint_codec()
            if codec is not None:
                codec.assert_live(trainer._net(), optimizer)
                codec.validate_frozen_base()
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
