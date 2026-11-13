# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real native factory/entry routing, with CPU backend doubles only at the CUDA boundary."""

import sys
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.tests.helpers.optimizer_backend import backend_capture  # noqa: F401
from cosmos_framework.utils.optimizer_master import MasterWeightOptimizer


@pytest.mark.parametrize("name", ["adamw", "adamw_bnb_8bit", "paged_adamw_8bit"])
def test_alpagym_actual_entry_constructs_after_setup(name, monkeypatch, backend_capture):
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.entrypoints.alpagym_clrl import _build_trainer, alpagym_optimizer_config

    class Model:
        def __init__(self, path, dtype):
            self.net = torch.nn.Linear(4, 4, dtype=dtype)
            self.before_setup = self.net.weight
            self.net.register_parameter("frozen", torch.nn.Parameter(torch.zeros(1), requires_grad=False))

        def setup(self, world_size):
            self.width = world_size
            self.net.weight = torch.nn.Parameter(self.net.weight.detach().clone())

    bundle = SimpleNamespace(install_runtime_bridge=lambda: None, build_model_inputs=lambda config: None)
    modules = {
        "alpagym_runtime.policies.registry": SimpleNamespace(get_policy_bundle=lambda _: bundle),
        "alpagym_runtime.posttrain.model": SimpleNamespace(AlpamayoTrainModel=Model),
        "alpagym_runtime.posttrain.objective": SimpleNamespace(
            make_alpagym_objective=lambda *a, **kw: Objective(
                lambda m, b: (m.net(b).sum(), {}), reduction=Reduction.MEAN
            )
        ),
    }
    for key, value in modules.items():
        monkeypatch.setitem(sys.modules, key, value)
    train = SimpleNamespace(
        optm_lr=1e-4,
        optm_warmup_steps=4,
        train_policy=SimpleNamespace(
            grpo_ratio_clip_low=0.2, grpo_ratio_clip_high=0.2, kl_beta=0, reference_reset_interval=0
        ),
    )
    config = SimpleNamespace(
        cosmos=SimpleNamespace(train=train),
        policy=SimpleNamespace(model=SimpleNamespace(kind="alpamayo", path="unallocated")),
    )
    selected = None if name == "adamw" else alpagym_optimizer_config(train, name=name, update_policy="fp32_master")
    trainer = _build_trainer(config, 3, optimizer=selected)
    params = [p for g in trainer.opt.param_groups for p in g["params"]]
    assert {id(p) for p in params} == {id(p) for p in trainer.model.net.parameters() if p.requires_grad}
    assert all(p is not trainer.model.before_setup for p in params)
    group = trainer.opt.param_groups[0]
    assert (group["lr"], group["eps"], group["betas"], group["weight_decay"]) == (2.5e-5, 1e-6, (0.9, 0.999), 0.01)
    assert trainer.max_grad_norm == 1.0 and trainer.n_mini == 3 and trainer.skip_zero_grad
    if name == "adamw":
        assert type(trainer.opt) is torch.optim.AdamW and not backend_capture
    else:
        assert isinstance(trainer.opt, MasterWeightOptimizer)
        assert backend_capture[0][2].name == name
