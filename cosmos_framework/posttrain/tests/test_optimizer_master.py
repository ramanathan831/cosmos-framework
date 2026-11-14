# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU numerical/ownership checks; CUDA/FSDP gates run separately."""

from copy import deepcopy

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
    import io

    reader = restored.checkpoint_manager.load_dcp_component

    def corrupted(path, name, state, **kwargs):
        found = reader(path, name, state, **kwargs)
        if name == "optim":
            saved = torch.load(io.BytesIO(state["ranks"]["0"]), weights_only=False)
            del saved["masters"]
            buffer = io.BytesIO()
            torch.save(saved, buffer)
            state["ranks"]["0"] = buffer.getvalue()
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

    def fail_once(staged):
        nonlocal calls
        commit(staged)
        calls += 1
        if calls == 1:
            raise RuntimeError("injected post-commit failure")

    with monkeypatch.context() as context:
        context.setattr(restored.opt, "commit_state_dict", fail_once)
        with pytest.raises(Exception, match="injected"):
            restored.load(path)
    assert_equal(before_opt, restored.opt.state_dict())
    assert_equal(before_net, restored.model.net.state_dict())
    # NumPy RNG arrays require their own equality helper.
    from tools.quantized_optimizers.master_distributed import equal

    equal(before_control, restored._trainer_state())
