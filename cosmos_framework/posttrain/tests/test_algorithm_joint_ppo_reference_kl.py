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
    head.config = SimpleNamespace(
        action_horizon=40, max_action_dim=132, backbone_embedding_dim=2, input_embedding_dim=2
    )
    head.num_inference_timesteps = 4
    return head


def test_transition_only_requires_trace():
    from cosmos_framework.posttrain.model.gr00t.ppo import Gr00tPPOHeads

    heads = Gr00tPPOHeads(_head(), score_horizon=8)
    with pytest.raises(ValueError, match="requires a trace"):
        heads(SimpleNamespace(), None, transition_only=True)


def test_trainer_window_reports_reference_kl_only_for_anchored_models():
    recipe = batch_recipe()
    for model, expected in ((FixedEpisodePolicy(), False), (_AnchoredPolicy(beta=0.5), True)):
        learner = JointPPOTrainer(model, recipe)
        receipt = learner.prepare_ppo_targets(collect_fixed(model, recipe))
        update = learner.train_batch(receipt["batch_id"])
        assert all(("reference_kl" in item) is expected for item in update["optimizer_metrics"])
        assert ("reference_kl" in update["metrics"]) is expected
        if expected:
            # First minibatch: policy equals the frozen copy, so KL is exactly zero.
            assert update["optimizer_metrics"][0]["reference_kl"] == 0.0
        learner.clear_batch(receipt["batch_id"])
        if expected:
            # After an applied step the policy has moved away from its frozen copy.
            receipt = learner.prepare_ppo_targets(collect_fixed(model, recipe))
            moved = learner.train_batch(receipt["batch_id"])
            assert moved["optimizer_metrics"][0]["reference_kl"] > 0
            learner.clear_batch(receipt["batch_id"])


def test_batched_scoring_keeps_the_reference_kl_anchor():
    """A BatchedDecisionPPOPolicy's per-decision KL and beta reach the loss like the per-decision path."""
    from cosmos_framework.posttrain.tests.helpers.batched_joint_ppo import BatchedFixedEpisodePolicy

    class BatchedAnchored(BatchedFixedEpisodePolicy):
        def __init__(self, beta):
            super().__init__()
            self.beta = beta
            self.ppo_reference = object()

        def score_ppo_decisions(self, steps):
            scores = super().score_ppo_decisions(steps)
            kl = sum(((p - q) ** 2).sum() for p, q in zip(self.net.parameters(), self.initial, strict=True))
            # One (here identical) KL per scored decision, [B], as the protocol declares.
            return replace(scores, reference_kl=kl.expand(len(steps)), reference_kl_beta=self.beta)

    recipe = batch_recipe()
    plain = FixedEpisodePolicy()
    learner = JointPPOTrainer(plain, recipe)
    receipt = learner.prepare_ppo_targets(collect_fixed(plain, recipe))
    samples = list(learner._ppo_cache.select(receipt["batch_id"], learner._ppo_cache.episode_ids(receipt["batch_id"])))
    valid = [s for s in samples if s.transitions[0].algo_extra[TARGET_KEY].loss_mask][: recipe.microbatch]
    assert len(valid) == recipe.microbatch  # fills the batch: no padding template needed
    objective = _objective(recipe)

    per_decision = _AnchoredPolicy(beta=0.5)
    per_decision.net.load_state_dict(plain.net.state_dict())
    per_decision.initial = [p.detach().clone() for p in per_decision.net.parameters()]
    with torch.no_grad():
        for parameter in per_decision.net.parameters():
            parameter.add_(0.1)
    batched = BatchedAnchored(beta=0.5)
    batched.net.load_state_dict(per_decision.net.state_dict())
    batched.initial = per_decision.initial
    expected_loss, expected_metrics = objective(per_decision, valid)
    loss, metrics = objective(batched, valid)
    assert batched.scored_batches  # the batched path ran
    assert float(metrics["ppo_reference_kl"]) > 0
    assert float(metrics["ppo_reference_kl"]) == pytest.approx(float(expected_metrics["ppo_reference_kl"]), rel=1e-5)
    torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)
    learner.clear_batch(receipt["batch_id"])


@pytest.mark.parametrize("ppo_reference", [True, False])
def test_frozen_reference_is_built_only_where_the_loss_is_computed(monkeypatch, tmp_path, ppo_reference):
    """The learner builds the KL anchor; rollout rebuilds (ppo_reference=False) hold no second head."""
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_model_config
    from cosmos_framework.posttrain.model.gr00t import policy as policy_module
    from cosmos_framework.posttrain.model.gr00t import source as source_module

    class Gr00t(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = _head()
            self.backbone = nn.Identity()

    monkeypatch.setenv("GR00T_DIT_SDPA_MODE", "math")
    monkeypatch.setattr(
        source_module, "load_ppo_source", lambda config, load, publish=None: ((Gr00t(), object()), {"source": "x"})
    )
    monkeypatch.setattr(policy_module, "parallelize", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "gr00t_shard_plan", lambda precision: None)
    monkeypatch.setattr("cosmos_framework.posttrain.model.ppo.prepare_deterministic_ppo_update", lambda net: None)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.model.ppo.prepare_deterministic_ppo_forward", lambda net, **kwargs: None
    )
    config = replace(make_model_config("checkpoint"), ppo_reference_kl_beta=0.1)
    policy = policy_module.Gr00tN1d7TrainPolicy(
        config, device="cpu", save_cfg_dir=tmp_path, ppo=True, ppo_reference=ppo_reference
    )
    assert (policy.ppo_reference is not None) is ppo_reference
    if ppo_reference:
        assert policy.ppo_reference.action_head is not policy.gr00t.action_head
        assert not any(p.requires_grad for p in policy.ppo_reference.parameters())
