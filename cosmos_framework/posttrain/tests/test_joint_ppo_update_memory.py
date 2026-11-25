# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dead or diagnostic tensors must not occupy accelerator memory across a Joint PPO update.

Regression for the fixed 55 GiB GR00T pressure continuation (round 10): the nonpaged
eight-bit learner ran out of memory in backward while (a) a full pre-update clone of every
local model shard and (b) a consumed FP32 master-gradient copy were device-resident.
"""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.config.build import build_optimizer
from cosmos_framework.posttrain.config.rl import JointPPOConfig, OptimizerConfig
from cosmos_framework.posttrain.roles.trainer import joint_ppo
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer


def _bf16_net(device="cpu", width=4):
    net = torch.nn.Sequential(torch.nn.Linear(width, width, bias=False), torch.nn.Linear(width, 3, bias=False))
    return net.to(device=device, dtype=torch.bfloat16)


@pytest.mark.L0
@pytest.mark.CPU
def test_master_step_releases_consumed_fp32_staging_but_keeps_model_gradients():
    net = _bf16_net()
    opt = build_optimizer(net, OptimizerConfig(lr=1e-3, weight_decay=0, update_policy="fp32_master"))
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()
    report = opt.state_report()
    # Before the fix the FP32 copy (4 bytes/element) survived until the next window.
    assert report["gradient_staging_bytes"] == 0
    assert all(parameter.grad is not None for parameter in net.parameters())
    assert all(master.grad is None for master in opt.masters)
    # A second step stages afresh; the update arithmetic is unchanged.
    before = [master.clone() for master in opt.masters]
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, opt.masters))


@pytest.mark.L0
@pytest.mark.CPU
def test_fp32_alias_gradient_is_not_released_by_master_step():
    net = _bf16_net().float()
    opt = build_optimizer(net, OptimizerConfig(lr=1e-3, weight_decay=0, update_policy="fp32_master"))
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()
    assert opt.state_report()["fp32_gradient_alias_bytes"] > 0


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("host", [False, True])
def test_pre_update_snapshot_is_an_independent_copy(host):
    net = _bf16_net()
    owner = SimpleNamespace(model=SimpleNamespace(net=net))
    with JointPPOTrainer._parameter_state(owner, clone=True, host=host) as before:
        assert before and all(value.device.type == "cpu" for value in before.values())
        live = {p.data_ptr() for p in net.parameters()}
        assert not live & {value.data_ptr() for value in before.values()}
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.add_(1)
        assert all(
            not torch.equal(value, parameter.detach()) for value, parameter in zip(before.values(), net.parameters())
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_allocator_receipt_is_absent_off_device_and_update_still_reports_it(monkeypatch):
    from cosmos_framework.posttrain.tests.helpers.ppo_continuous import (
        GaussianPolicy,
        LocalWindowGateway,
        collect_scored,
        make_collector,
    )

    assert joint_ppo._cuda_allocator_window(_bf16_net(), reset=True) is None
    windows = []
    real_window = joint_ppo._cuda_allocator_window
    monkeypatch.setattr(
        joint_ppo, "_cuda_allocator_window", lambda net, reset: windows.append(reset) or real_window(net, reset=reset)
    )
    for update_policy in ("parameter", "fp32_master"):
        windows.clear()
        recipe = JointPPOConfig(num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1)
        recipe.optimizer.update_policy = update_policy
        model = GaussianPolicy()
        collector = make_collector(model, LocalWindowGateway(), recipe, 0, 2)
        learner = JointPPOTrainer(model, recipe)
        receipt = learner.prepare_ppo_targets(collect_scored(collector, 0))
        result = learner.train_batch(receipt["batch_id"])
        # Default runs keep the upstream receipt: no allocator reset, gather or key.
        assert ("cuda_allocator" in result) is (update_policy == "fp32_master")
        assert result.get("cuda_allocator") is None
        # Default runs never touch (or reset) the process-wide CUDA peak counters.
        assert windows == ([] if update_policy == "parameter" else [True, False])
        assert result["parameter_update_l2"]["actor"] > 0 and result["parameter_update_l2"]["value"] > 0
        collector.close()


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pre_update_snapshot_occupies_no_device_memory_during_the_update():
    device = torch.device("cuda")
    net = _bf16_net(device, width=2048)  # 2 x 8 MiB of BF16 parameters
    owner = SimpleNamespace(model=SimpleNamespace(net=net))
    parameter_bytes = sum(p.numel() * p.element_size() for p in net.parameters())
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    original = [p.detach().clone() for p in net.parameters()]
    with JointPPOTrainer._parameter_state(owner, clone=True, host=True) as before:
        held = torch.cuda.memory_allocated(device) - baseline - parameter_bytes  # minus `original`
        assert all(value.device.type == "cpu" and value.is_pinned() for value in before.values())
        # The asynchronous copies are stream-ordered before later in-place updates.
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.add_(1)
        restored = [value.to(device, non_blocking=True) for value in before.values()]
        assert all(torch.equal(a, b) for a, b in zip(restored, original))
    # Before the fix `held` equalled every local shard (the clone was device-resident).
    assert held < parameter_bytes // 4
    # The default (non-master) path keeps the historical device-resident clone.
    with JointPPOTrainer._parameter_state(owner, clone=True) as before:
        assert all(value.is_cuda for value in before.values())


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_master_step_returns_fp32_staging_to_the_device_allocator():
    device = torch.device("cuda")
    net = _bf16_net(device, width=2048)
    opt = build_optimizer(net, OptimizerConfig(lr=1e-3, weight_decay=0, update_policy="fp32_master"))
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()  # first step initializes moments
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    torch.cuda.synchronize(device)
    before_step = torch.cuda.memory_allocated(device)
    opt.step()
    torch.cuda.synchronize(device)
    staging_bytes = sum(p.numel() * 4 for p in net.parameters())
    assert torch.cuda.memory_allocated(device) - before_step < staging_bytes // 4
    receipt = joint_ppo._cuda_allocator_window(net, reset=True)
    assert receipt is not None
    assert {"inactive_split_bytes.all.current", "reserved_bytes.all.peak", "device_free_bytes"} <= set(receipt)


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_backend_step_device_time_is_recorded_without_changing_arithmetic():
    device = torch.device("cuda")
    nets = [_bf16_net(device, width=512) for _ in range(2)]
    nets[1].load_state_dict(nets[0].state_dict())
    opts = [build_optimizer(net, OptimizerConfig(lr=1e-3, weight_decay=0, update_policy="fp32_master")) for net in nets]
    for net, opt in zip(nets, opts):
        for parameter in net.parameters():
            parameter.grad = torch.full_like(parameter, 0.5)
        opt.step()
    assert joint_ppo._backend_step_milliseconds(opts[0]) > 0
    for left, right in zip(nets[0].parameters(), nets[1].parameters()):
        assert torch.equal(left, right)


@pytest.mark.L0
@pytest.mark.CPU
def test_backend_step_time_is_absent_on_cpu():
    net = _bf16_net()
    opt = build_optimizer(net, OptimizerConfig(lr=1e-3, weight_decay=0, update_policy="fp32_master"))
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()
    assert joint_ppo._backend_step_milliseconds(opt) is None
