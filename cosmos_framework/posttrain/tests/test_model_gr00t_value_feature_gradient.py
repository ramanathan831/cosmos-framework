# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in value-feature gradient isolation for GR00T Joint PPO."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.model.gr00t.ppo import Gr00tPPOHeads

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _heads(value_feature_gradient: bool) -> tuple[Gr00tPPOHeads, nn.Linear]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        head = nn.Linear(2, 2)
        head.config = SimpleNamespace(
            action_horizon=40, max_action_dim=132, backbone_embedding_dim=2, input_embedding_dim=2
        )
        head.num_inference_timesteps = 4
        trunk = nn.Linear(2, 2)  # stands in for shared actor features (vlln / state_encoder)
        heads = Gr00tPPOHeads(head, score_horizon=8, value_feature_gradient=value_feature_gradient)
        heads.action_head.value_head.float()
    return heads, trunk


def _value_grads(value_feature_gradient: bool):
    heads, trunk = _heads(value_feature_gradient)
    x = torch.ones(3, 4, 2)
    vl, state = trunk(x), trunk(torch.ones(3, 1, 2))
    values = heads._values(vl, state)
    values.sum().backward()
    head_grad = [p.grad for p in heads.action_head.value_head.parameters()]
    return values.detach(), trunk.weight.grad, head_grad


def test_native_default_routes_value_gradient_into_shared_features():
    _, trunk_grad, head_grad = _value_grads(True)
    assert trunk_grad is not None and trunk_grad.abs().sum() > 0
    assert all(g is not None for g in head_grad)


def test_isolated_value_features_keep_values_and_train_only_the_value_head():
    native_values, _, _ = _value_grads(True)
    values, trunk_grad, head_grad = _value_grads(False)
    torch.testing.assert_close(values, native_values, rtol=0, atol=0)
    assert trunk_grad is None
    assert all(g is not None and g.abs().sum() > 0 for g in head_grad)


def test_flag_identity_is_legacy_stable_and_validated():
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_model_config

    native = make_model_config("checkpoint")
    assert "value_feature_gradient" not in native.identity()
    isolated = make_model_config("checkpoint")
    isolated.value_feature_gradient = False
    assert isolated.identity()["value_feature_gradient"] is False
    assert isolated.identity_hash() != native.identity_hash()
    head = _heads(True)[0].action_head
    with pytest.raises(ValueError, match="boolean"):
        Gr00tPPOHeads(head, score_horizon=8, value_feature_gradient="no")
