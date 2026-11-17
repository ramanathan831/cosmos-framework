# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in KL anchor to a frozen initial policy for GR00T Joint PPO."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.algorithm.joint_ppo.objective import make_joint_objective
from cosmos_framework.posttrain.algorithm.joint_ppo.records import TARGET_KEY
from cosmos_framework.posttrain.model.gr00t.ppo import flow_sde_transition, selected_transition_kl
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.tests.helpers.batched_joint_ppo import batch_recipe
from cosmos_framework.posttrain.tests.helpers.fixed_episode import FixedEpisodePolicy, collect_fixed

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_transition_kl_matches_gaussian_kl_and_ignores_deterministic_and_unscored_cells():
    torch.manual_seed(0)
    latent, velocity, reference = torch.randn(3, 40, 132), torch.randn(3, 40, 132), torch.randn(3, 40, 132)
    index = torch.tensor([0, 1, 3])
    mean, std = flow_sde_transition(latent, velocity, index, stochastic=True)
    reference_mean, reference_std = flow_sde_transition(latent, reference, index, stochastic=True)
    torch.testing.assert_close(std, reference_std)  # std depends only on the denoise index
    kl = selected_transition_kl(mean, reference_mean, std, score_horizon=8)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(mean[:, :8, :7].double(), std[:, :8, :7].double()),
        torch.distributions.Normal(reference_mean[:, :8, :7].double(), std[:, :8, :7].double()),
    ).sum(dim=(1, 2))
    torch.testing.assert_close(kl.double(), expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(selected_transition_kl(mean, mean, std, score_horizon=8), torch.zeros(3))
    changed = reference_mean.clone()
    changed[:, 8:, :] += 5
    changed[:, :, 7:] += 5
    torch.testing.assert_close(selected_transition_kl(mean, changed, std, score_horizon=8), kl)
    deterministic_mean, zero_std = flow_sde_transition(latent, velocity, index, stochastic=False)
    assert torch.equal(
        selected_transition_kl(deterministic_mean, reference_mean, zero_std, score_horizon=8), torch.zeros(3)
    )


def test_gr00t_reference_kl_flag_is_validated_and_identity_stable():
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_model_config

    native = make_model_config("checkpoint")
    assert native.ppo_reference_kl_beta == 0.0 and "ppo_reference_kl_beta" not in native.identity()
    anchored = replace(native, ppo_reference_kl_beta=0.05)
    assert anchored.identity()["ppo_reference_kl_beta"] == 0.05
    assert anchored.identity_hash() != native.identity_hash()
    for bad in (-1.0, float("nan"), float("inf"), True, "0.1"):
        with pytest.raises(ValueError, match="ppo_reference_kl_beta"):
            replace(native, ppo_reference_kl_beta=bad)


class _AnchoredPolicy(FixedEpisodePolicy):
    """Adds a differentiable squared-distance 'KL' to a frozen copy of the initial parameters."""

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta
        self.ppo_reference = object()  # marks a model that carries a frozen reference
        self.initial = [p.detach().clone() for p in self.net.parameters()]

    def score_ppo_decision(self, step):
        scores = super().score_ppo_decision(step)
        kl = sum(((p - q) ** 2).sum() for p, q in zip(self.net.parameters(), self.initial, strict=True)).reshape(1)
        return replace(scores, reference_kl=kl, reference_kl_beta=self.beta)


def _objective(recipe):
    return make_joint_objective(
        policy_clip=recipe.policy_clip,
        value_clip=recipe.value_clip,
        dual_clip=recipe.dual_clip,
        huber_delta=recipe.huber_delta,
        loss_reduction=recipe.loss_reduction,
        microbatch=recipe.microbatch,
        identity=recipe.identity(),
    )


def test_objective_adds_weighted_reference_kl_to_joint_loss_only():
    recipe = batch_recipe()
    plain = FixedEpisodePolicy()
    learner = JointPPOTrainer(plain, recipe)
    receipt = learner.prepare_ppo_targets(collect_fixed(plain, recipe))
    samples = list(learner._ppo_cache.select(receipt["batch_id"], learner._ppo_cache.episode_ids(receipt["batch_id"])))
    valid = [s for s in samples if s.transitions[0].algo_extra[TARGET_KEY].loss_mask][:3]
    objective = _objective(recipe)
    base_loss, base_metrics = objective(plain, valid)
    assert "ppo_reference_kl" not in base_metrics  # native models report no reference metric

    anchored = _AnchoredPolicy(beta=0.5)
    anchored.net.load_state_dict(plain.net.state_dict())
    anchored.initial = [p.detach().clone() for p in anchored.net.parameters()]
    with torch.no_grad():
        for parameter in anchored.net.parameters():
            parameter.add_(0.1)
    shifted = FixedEpisodePolicy()
    shifted.net.load_state_dict(anchored.net.state_dict())
    shifted_loss, shifted_metrics = objective(shifted, valid)
    loss, metrics = objective(anchored, valid)
    kl = sum(float(((p - q) ** 2).sum()) for p, q in zip(anchored.net.parameters(), anchored.initial, strict=True))
    assert float(metrics["ppo_reference_kl"]) == pytest.approx(kl, rel=1e-5)
    assert float(metrics["ppo_policy_loss"]) == pytest.approx(float(shifted_metrics["ppo_policy_loss"]), rel=1e-6)
    # The penalty is weighted like the surrogate: mean over decisions of beta * kl * loss_weight.
    weight = sum(s.transitions[0].algo_extra[TARGET_KEY].loss_weight for s in valid) / len(valid)
    torch.testing.assert_close(loss, shifted_loss + 0.5 * kl * weight, rtol=1e-5, atol=1e-6)
    loss.backward()
    assert any(p.grad is not None and bool(p.grad.any()) for p in anchored.net.parameters())
    # beta = 0 reports the KL but leaves the optimized loss unchanged.
    reporter = _AnchoredPolicy(beta=0.0)
    reporter.net.load_state_dict(anchored.net.state_dict())
    reporter.initial = anchored.initial
    report_loss, report_metrics = objective(reporter, valid)
    torch.testing.assert_close(report_loss, shifted_loss, rtol=1e-6, atol=1e-7)
    assert float(report_metrics["ppo_reference_kl"]) == pytest.approx(kl, rel=1e-5)
    learner.clear_batch(receipt["batch_id"])


def _head():
    head = nn.Linear(2, 2)
    head.config = SimpleNamespace(action_horizon=40, max_action_dim=132, backbone_embedding_dim=2, input_embedding_dim=2)
    head.num_inference_timesteps = 4
    return head


def test_transition_only_requires_trace():
    from cosmos_framework.posttrain.model.gr00t.ppo import Gr00tPPOHeads

    heads = Gr00tPPOHeads(_head(), score_horizon=8)
    with pytest.raises(ValueError, match="requires a trace"):
        heads(SimpleNamespace(), None, transition_only=True)
