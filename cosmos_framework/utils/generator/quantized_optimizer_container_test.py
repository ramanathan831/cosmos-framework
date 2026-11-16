# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared eight-bit AdamW through the generator/VFM ``OptimizersContainer``.

CPU checks cover the legacy route and fail-before-construction rules; CUDA checks
exercise the real bitsandbytes backend, grouping identity and DCP exact resume.
The two-rank FSDP2 gate lives in ``quantized_optimizer_container_fsdp_test.py``.
"""

from copy import deepcopy

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.utils.generator.optimizer import build_lr_scheduler, build_optimizer

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes eight-bit AdamW requires CUDA")
KWARGS = dict(
    lr=5e-5,
    betas=[0.9, 0.99],
    eps=1e-8,
    weight_decay=0.05,
    fused=True,
    keys_to_select=["gen", "llm2action"],
    lr_multipliers={"llm2action": 5.0},
)


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = torch.nn.Linear(256, 512)  # 131k weight elements: quantized
        self.llm2action = torch.nn.Linear(512, 16)  # 8k: quantized, 5x LR
        self.frozen = torch.nn.Linear(16, 16)

    def forward(self, x):
        return self.frozen(self.llm2action(self.gen(x)))


class _Model(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        torch.manual_seed(0)
        self.net = _Net().to(device)


def _groups(container):
    return [
        (
            group["lr"],
            group["weight_decay"],
            group.get("betas"),
            group.get("eps"),
            [id(p) for p in group["params"]],
        )
        for optimizer in container
        for group in optimizer.param_groups
    ]


def test_legacy_route_and_defaults_are_unchanged():
    model = _Model()
    container = build_optimizer(model, "adamw", **deepcopy(KWARGS))
    assert not container.rank_local_checkpoint
    assert [type(o) for o in container] == [torch.optim.AdamW]
    assert not model.net.frozen.weight.requires_grad
    assert {g["lr"] for g in container.optimizers[0].param_groups} == {5e-5, 2.5e-4}


@pytest.mark.parametrize("option", [{"amsgrad": True}, {"master_weights": True}, {"capturable": True}])
def test_unsupported_options_fail_before_backend_construction(option, monkeypatch):
    import cosmos_framework.utils.quantized_optimizer as backend

    monkeypatch.setattr(backend, "require_backend", lambda **_: pytest.fail("backend constructed"))
    with pytest.raises(ValueError, match="does not support optimizer options"):
        build_optimizer(_Model(), "adamw_bnb_8bit", **deepcopy(KWARGS), **option)


def test_cpu_selection_fails_explicitly_without_fallback():
    with pytest.raises((ValueError, RuntimeError)):
        build_optimizer(_Model(), "adamw_bnb_8bit", **deepcopy(KWARGS))


def _step(model, container, scheduler, seed):
    torch.manual_seed(seed)
    x = torch.randn(4, 256, device="cuda")
    model.net(x).square().mean().backward()
    container.step()
    scheduler.step()
    container.zero_grad()


def _scheduler(container):
    return build_lr_scheduler(
        container,
        lr_scheduler_type="LambdaLinear",
        cycle_lengths=[100],
        f_max=[1.0],
        f_min=[0.0],
        f_start=[1e-6],
        warm_up_steps=[5],
        verbosity_interval=0,
    )


@CUDA
@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
def test_eight_bit_reaches_backend_with_identical_groups(name):
    legacy_model, model = _Model("cuda"), _Model("cuda")
    legacy = build_optimizer(legacy_model, "adamw", **deepcopy(KWARGS))
    container = build_optimizer(model, name, **deepcopy(KWARGS))
    assert container.rank_local_checkpoint and len(container) == 1
    optimizer = container.optimizers[0]
    assert type(optimizer.backend).__name__ == "CosmosAdamW8bit"
    assert optimizer.backend.is_paged == (name == "paged_adamw_8bit")
    names = {id(p): n for n, p in model.net.named_parameters()}
    legacy_names = {id(p): n for n, p in legacy_model.net.named_parameters()}
    as_names = lambda groups, table: [(*g[:4], [table[i] for i in g[4]]) for g in groups]  # noqa: E731
    assert as_names(_groups(container), names) == as_names(_groups(legacy), legacy_names)
    # FP32 FSDP/unsharded storage is its own master: aliased, never copied.
    assert all(m.data_ptr() == p.data_ptr() for m, p in zip(optimizer.masters, optimizer.live))
    frozen = model.net.frozen.weight.detach().clone()
    before = model.net.gen.weight.detach().clone()
    scheduler = _scheduler(container)
    _step(model, container, scheduler, 1)
    assert not torch.equal(before, model.net.gen.weight)
    assert torch.equal(frozen, model.net.frozen.weight)
    report = container.state_report()
    assert report["quantized_elements"] == sum(p.numel() for p in optimizer.live if p.numel() >= 4096)
    assert report["master_bytes"] == 0 and report["moment_bytes"] > 0
    assert optimizer.backend.state[optimizer.masters[0]]["state1"].dtype == torch.uint8
    # The scheduler-owned LR in effect at the update reaches the backend on every step.
    used = [g["lr"] for g in optimizer.param_groups]
    assert used[0] != 5e-5 * 1e-6  # warmup has advanced
    _step(model, container, scheduler, 2)
    assert [g["lr"] for g in optimizer.backend.param_groups] == used


@CUDA
def test_dcp_exact_resume_and_transactional_rejection(tmp_path, monkeypatch):
    # Memory reporting runs on the first step and must not alter resume identity.
    monkeypatch.setenv("COSMOS_OPTIMIZER_STATE_REPORT", "1")
    model = _Model("cuda")
    container = build_optimizer(model, "adamw_bnb_8bit", **deepcopy(KWARGS))
    scheduler = _scheduler(container)
    for seed in range(3):
        _step(model, container, scheduler, seed)
    dcp.save({"model": model.state_dict()}, checkpoint_id=tmp_path / "model")
    dcp.save(container.state_dict(), checkpoint_id=tmp_path / "optim")
    dcp.save(scheduler.state_dict(), checkpoint_id=tmp_path / "scheduler")
    for seed in range(3, 6):
        _step(model, container, scheduler, seed)
    expected = {n: p.detach().clone() for n, p in model.net.named_parameters()}

    resumed = _Model("cuda")
    fresh = build_optimizer(resumed, "adamw_bnb_8bit", **deepcopy(KWARGS))
    fresh_scheduler = _scheduler(fresh)
    model_state = {"model": resumed.state_dict()}
    dcp.load(model_state, checkpoint_id=tmp_path / "model")
    resumed.load_state_dict(model_state["model"])
    sched_state = fresh_scheduler.state_dict()
    dcp.load(sched_state, checkpoint_id=tmp_path / "scheduler")
    fresh_scheduler.load_state_dict(sched_state)

    # Rejected restores leave optimizer state untouched on every path.
    template = fresh.checkpoint_load_template()
    dcp.load(template, checkpoint_id=tmp_path / "optim", planner=dcp.DefaultLoadPlanner(allow_partial_load=False))
    pristine = (fresh.optimizers[0].masters, fresh.optimizers[0].backend)
    for broken in (
        dict(template, world_size=2),
        dict(template, ranks={}),
        dict(template, format="cosmos_rank_master_v0"),
    ):
        with pytest.raises(ValueError):
            fresh.load_state_dict(broken)
        assert (fresh.optimizers[0].masters, fresh.optimizers[0].backend) == pristine
    restored_bias = resumed.net.gen.bias.detach().clone()
    with torch.no_grad():
        resumed.net.gen.bias.add_(1)
    with pytest.raises(ValueError, match="disagree"):
        fresh.load_state_dict(template)
    assert (fresh.optimizers[0].masters, fresh.optimizers[0].backend) == pristine
    with torch.no_grad():
        resumed.net.gen.bias.copy_(restored_bias)

    fresh.load_state_dict(template)
    for seed in range(3, 6):
        _step(resumed, fresh, fresh_scheduler, seed)
    for name, parameter in resumed.net.named_parameters():
        assert torch.equal(parameter, expected[name]), name

    # A full-precision (legacy) optimizer checkpoint is not silently accepted.
    legacy = build_optimizer(_Model("cuda"), "adamw", **deepcopy(KWARGS))
    dcp.save(legacy.state_dict(), checkpoint_id=tmp_path / "legacy")
    from torch.distributed.checkpoint.api import CheckpointException

    with pytest.raises(CheckpointException, match="Missing key"):
        dcp.load(
            fresh.checkpoint_load_template(),
            checkpoint_id=tmp_path / "legacy",
            planner=dcp.DefaultLoadPlanner(allow_partial_load=False),
        )
