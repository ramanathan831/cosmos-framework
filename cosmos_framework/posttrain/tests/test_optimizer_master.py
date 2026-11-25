# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU numerical/ownership checks; CUDA/FSDP gates run separately."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.config.build import build_optimizer
from cosmos_framework.posttrain.config.rl import OptimizerConfig


def model():
    net = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        net.weight.fill_(1)
    net.register_parameter("tied", net.weight)
    net.register_parameter("unused", torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16)))
    net.register_parameter("frozen", torch.nn.Parameter(torch.ones(3), requires_grad=False))
    return net


def make(net, warmup=0):
    return build_optimizer(
        net, OptimizerConfig(lr=1e-4, weight_decay=0, warmup_steps=warmup, update_policy="fp32_master")
    )


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    elif isinstance(a, np.ndarray):  # NumPy RNG state needs array equality.
        assert np.array_equal(a, b)
    else:
        assert a == b


def test_master_retains_sub_bf16_updates_and_exact_membership():
    net = model()
    opt = make(net)
    reference = torch.nn.Parameter(net.weight.detach().float().clone())
    control = torch.optim.AdamW([reference], lr=1e-4, weight_decay=0)
    assert opt.param_groups[0]["params"][0] is net.weight
    assert opt.names == ("weight", "unused")
    for _ in range(40):
        net.weight.grad = torch.ones_like(net.weight)
        reference.grad = torch.ones_like(reference)
        opt.step()
        control.step()
        assert torch.equal(opt.masters[0], reference)
        assert torch.equal(net.weight, reference.bfloat16())
    assert (net.weight < 1).all()  # Direct BF16 updates of this size would vanish.
    assert not opt.state.get(net.unused)
    assert torch.equal(net.frozen, torch.ones(3))


def test_master_resume_next_updates_and_rejected_state_is_transactional():
    net = model()
    opt = make(net)
    for _ in range(7):
        net.weight.grad = torch.full_like(net.weight, 0.5)
        opt.step()
    saved = opt.state_dict()
    other = model()
    other.load_state_dict(net.state_dict())
    restored = make(other)
    restored.load_state_dict(saved)
    for _ in range(4):
        net.weight.grad = torch.full_like(net.weight, -0.3)
        other.weight.grad = net.weight.grad.clone()
        opt.step()
        restored.step()
        assert_equal(opt.state_dict(), restored.state_dict())
        assert torch.equal(net.weight, other.weight)
    before = restored.state_dict()
    for corruption in ("dtype", "topology", "missing"):
        bad = deepcopy(saved)
        if corruption == "dtype":
            bad["masters"] = (bad["masters"][0].bfloat16(), *bad["masters"][1:])
        elif corruption == "topology":
            bad["identity"]["world_size"] = 2
        else:
            del bad["masters"]
        with pytest.raises(ValueError):
            restored.load_state_dict(bad)
        assert_equal(before, restored.state_dict())


def test_nonfinite_does_not_advance_master_moments_or_model():
    net = model()
    opt = make(net)
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    before, weights = opt.state_dict(), deepcopy(net.state_dict())
    net.unused.grad = torch.full_like(net.unused, float("nan"))
    with pytest.raises(Exception, match="nonfinite"):
        opt.step()
    assert_equal(before, opt.state_dict())
    assert_equal(weights, net.state_dict())


def test_native_joint_evaluation_reports_actual_initialized_storage():
    from cosmos_framework.posttrain.config.rl import JointPPOConfig
    from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
    from cosmos_framework.posttrain.tests.helpers.ppo_continuous import GaussianPolicy

    policy = GaussianPolicy()
    recipe = JointPPOConfig()
    recipe.optimizer = OptimizerConfig(update_policy="fp32_master")
    learner = JointPPOTrainer(policy, recipe)
    before = learner.evaluation_state()
    assert before["optimizer_storage"]["moment_bytes"] == 0
    for parameter in learner.opt.live:
        parameter.grad = torch.ones_like(parameter)
    learner.opt.step()
    state = learner.evaluation_state()
    assert state["optimizer_storage"]["moment_bytes"] == 8 * sum(p.numel() for p in learner.opt.live)
    assert state["optimizer_storage"]["master_bytes"] == 0  # FP32 fixture aliases live parameters.
    assert state == learner.evaluation_state()
    assert state["parameters"] != before["parameters"]
    assert learner.optimizer_memory()["fp32_gradient_alias_bytes"] > 0
    assert learner.optimizer_memory()["model_gradient_bytes"] == learner.optimizer_memory()["fp32_gradient_alias_bytes"]
    learner.opt.zero_grad()
    assert learner.optimizer_memory()["fp32_gradient_alias_bytes"] == 0
    assert learner.optimizer_memory()["model_gradient_bytes"] == 0
    assert learner.evaluation_state() == state  # Scratch gradients are not durable recovery state.


def test_fp32_storage_is_already_the_master_and_rejects_rebinding():
    net = model().float()
    opt = make(net)
    assert opt.masters[0].data_ptr() == net.weight.data_ptr()
    assert opt.state_report()["master_bytes"] == 0
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    saved = opt.state_dict()
    net.weight.data = net.weight.detach().clone()
    with pytest.raises(Exception, match="storage changed"):
        opt.step()
    # Reload rebinds the final live storage, with no second FP32 master allocation.
    opt.load_state_dict(saved)
    assert opt.masters[0].data_ptr() == net.weight.data_ptr()
    assert opt.state_report()["master_bytes"] == 0


def test_dcp_rank_payload_round_trip_and_rejects_damaged_or_unknown_payloads(tmp_path):
    import torch.distributed.checkpoint as dcp

    from cosmos_framework.posttrain.checkpoint import master_optimizer as checkpoint

    net = model()
    opt = make(net)
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    payload = checkpoint.optimizer_payload(opt)
    assert payload["format"] == "cosmos_rank_master_v3"
    assert payload["ranks"]["0"][2].dtype == torch.uint8
    dcp.save(payload, checkpoint_id=tmp_path)
    # A bytes placeholder must restore the tuple as one opaque DCP leaf.
    target = {"format": "", "world_size": 0, "ranks": {"0": b""}}
    dcp.load(target, checkpoint_id=tmp_path)
    saved = checkpoint._deserialize_rank_state(target["ranks"]["0"], target["format"])
    assert_equal(saved, opt.state_dict())
    size, digest, tensor = target["ranks"]["0"]
    assert set(dcp.FileSystemReader(tmp_path).read_metadata().state_dict_metadata) == {
        "format",
        "world_size",
        "ranks.0",
    }  # DCP never interprets internal scales or masters as model shards.
    changed = tensor.clone()
    changed[0] ^= 1
    for broken in (tensor[:-1], tensor.float(), tensor.reshape(1, -1), changed):
        with pytest.raises(ValueError, match="schema/length|checksum/length"):
            checkpoint._deserialize_rank_state((size, digest, broken), "cosmos_rank_master_v3")
    # Only the format this module writes is accepted.
    for format in ("cosmos_rank_master_v1", "cosmos_rank_master_v2", ""):
        with pytest.raises(ValueError, match="format mismatch"):
            checkpoint._deserialize_rank_state(target["ranks"]["0"], format)
    # The retained serialized byte tensor owns a stable independent snapshot.
    net.weight.grad = torch.full_like(net.weight, -0.3)
    opt.step()
    assert_equal(checkpoint._deserialize_rank_state(target["ranks"]["0"], "cosmos_rank_master_v3"), saved)


@pytest.mark.parametrize("format", ["dcp", "consolidated"])
def test_native_trainer_save_restore_master_transaction(tmp_path, format, monkeypatch):
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.checkpoint import CheckpointManager
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    def trainer():
        net = model()
        objective = Objective(
            lambda model, batch: (model.net(batch[0]).float().square().mean(), {}), reduction=Reduction.MEAN
        )
        from types import SimpleNamespace

        policy = SimpleNamespace(net=net)
        from cosmos_framework.posttrain.config.build import build_lr_scheduler

        optimizer = make(net, warmup=4)
        return LLMTrainer(
            policy,
            objective,
            optimizer,
            lr_scheduler=build_lr_scheduler(optimizer, optimizer.config),
            checkpoint_manager=CheckpointManager(checkpoint_format=format, save_async=False),
        )

    original = trainer()
    # Use the native trainer counter/finite/clipping path, not manual optimizer updates.
    batch = [torch.ones(2, 4, dtype=torch.bfloat16)]
    original.train(batch)
    path = str(tmp_path / "iter_000001")
    original.save(path, data_state={"cursor": 2})
    restored = trainer()
    restored.load(path)
    assert_equal(original.opt.state_dict(), restored.opt.state_dict())
    for _ in range(3):
        original.train(batch)
        restored.train(batch)
        assert_equal(original.opt.state_dict(), restored.opt.state_dict())

    before_opt = restored.opt.state_dict()
    before_net = deepcopy(restored.model.net.state_dict())
    before_control = restored._trainer_state()
    # A rank-local metadata rejection must precede every model/control mutation.
    from cosmos_framework.posttrain.checkpoint.master_optimizer import (
        _deserialize_rank_state,
        _serialize_tensor_rank_state,
    )

    reader = restored.checkpoint_manager.load_dcp_component

    def corrupted(path, name, state, **kwargs):
        found = reader(path, name, state, **kwargs)
        if name == "optim":
            saved = _deserialize_rank_state(state["ranks"]["0"], state["format"])
            del saved["masters"]
            state["ranks"]["0"] = _serialize_tensor_rank_state(saved)
        return found

    with monkeypatch.context() as context:
        context.setattr(restored.checkpoint_manager, "load_dcp_component", corrupted)
        with pytest.raises(Exception, match="identity"):
            restored.load(path)
    assert_equal(before_opt, restored.opt.state_dict())
    assert_equal(before_net, restored.model.net.state_dict())
    # Application failure after both model and optimizer have changed must restore
    # the old values, scheduler and clocks, rather than leaving a mixed checkpoint.
    commit = restored.opt.commit_state_dict
    calls = 0
    import gc
    import weakref

    staged_backends = []
    create_backend = restored.opt._backend

    def track_backend(masters):
        backend = create_backend(masters)
        staged_backends.append(weakref.ref(backend))
        return backend

    def fail_once(staged):
        nonlocal calls
        commit(staged)
        calls += 1
        if calls == 1:
            raise RuntimeError("injected post-commit failure")

    with monkeypatch.context() as context:
        context.setattr(restored.opt, "commit_state_dict", fail_once)
        context.setattr(restored.opt, "_backend", track_backend)
        with pytest.raises(Exception, match="injected") as failure:
            restored.load(path)
    assert failure.value is not None  # Keeping the error must not keep staged CUDA allocations alive.
    gc.collect()
    assert staged_backends and all(reference() is None for reference in staged_backends)
    assert_equal(before_opt, restored.opt.state_dict())
    assert_equal(before_net, restored.model.net.state_dict())
    assert_equal(before_control, restored._trainer_state())

    # A valid-but-replaced generation during reads must be rejected before the
    # staged model, optimizer and clocks become live, including direct load().
    control_file = tmp_path / "iter_000001" / "trainer.pt"
    original_control_bytes = control_file.read_bytes()

    def mutated_generation(path, name, state, **kwargs):
        found = reader(path, name, state, **kwargs)
        if name == "optim":
            control_file.write_bytes(original_control_bytes + b"changed generation")
        return found

    with monkeypatch.context() as context:
        context.setattr(restored.checkpoint_manager, "load_dcp_component", mutated_generation)
        with pytest.raises(Exception, match="changed while it was being loaded"):
            restored.checkpoint_manager.load_checkpoint(path, lambda: restored.load(path))
    assert_equal(before_opt, restored.opt.state_dict())
    assert_equal(before_net, restored.model.net.state_dict())
    assert_equal(before_control, restored._trainer_state())


def test_fused_control_memory_counts_initialized_step_tensors():
    net = model()
    optimizer = make(net)
    assert optimizer.state_report()["metadata_bytes"] == 0
    net.weight.grad = torch.ones_like(net.weight)
    optimizer.step()
    counters = [state["step"] for state in optimizer.backend.state.values()]
    assert len(counters) == 1  # The unused parameter must not invent a counter.
    expected = sum(value.numel() * value.element_size() for value in counters)
    assert expected > 0
    assert optimizer.state_report()["metadata_bytes"] == expected


def test_warm_start_load_after_construction_reseeds_masters_before_the_first_step():
    """Trainers may build the optimizer before a model-only (warm-start/EMA) load."""
    net = model()
    opt = make(net)
    loaded = torch.full_like(net.weight, 0.25)
    net.load_state_dict({**net.state_dict(), "weight": loaded})  # model-only load, optimizer untouched
    reference = torch.nn.Parameter(loaded.float().clone())
    control = torch.optim.AdamW([reference], lr=1e-4, weight_decay=0)
    for _ in range(3):
        net.weight.grad = torch.ones_like(net.weight)
        reference.grad = torch.ones_like(reference)
        opt.step()
        control.step()
        # The update starts from the loaded weights, never from the stale construction-time copy.
        assert torch.equal(opt.masters[0], reference)
        assert torch.equal(net.weight, reference.to(net.weight.dtype))


def test_model_only_load_under_initialized_moments_is_refused_without_mutation():
    net = model()
    opt = make(net)
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    before = deepcopy(opt.state_dict())
    with torch.no_grad():
        net.weight.fill_(3)
    opt._verify_masters = True  # as after any re-binding; a later load must not be silently overwritten
    net.weight.grad = torch.ones_like(net.weight)
    with pytest.raises(Exception, match="model weights changed under an initialized"):
        opt.step()
    assert_equal(before, opt.state_dict())
    assert torch.equal(net.weight, torch.full_like(net.weight, 3))
