# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Training regularization preserves hard CE, masking and accumulated updates."""

import copy

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM
from cosmos_framework.posttrain.algorithm.sft import make_objective
from cosmos_framework.posttrain.config.sft import SFTConfig
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker, Policy, sample_batch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("exponent", [0.0, 1.0])
def test_smoothed_gradient_matches_upstream_with_masks_and_dummy_rows(exponent, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(19)
    packer = MemoryPacker(padding=3)
    trajectories = sample_batch(packer)
    for trajectory in trajectories:
        trajectory.transitions[0].algo_extra["validation_weight"] = 1.0
    trajectories.append(packer.add([1, 2, 3, 4], [-100, 2, 3, 4], weight=0.0))
    policy = Policy().to(device).train()
    objective = make_objective(packer, ce_exponent=exponent, label_smoothing=0.01)
    actual, metrics = objective(policy, trajectories)
    samples = [packer.sft_process_sample(t.transitions[0].targets) for t in trajectories]
    batch = packer.sft_collate_fn(samples, packer.sft_compute_max_len(samples))
    labels = batch["labels"][:, 1:].to(device).clone()
    labels[-1] = -100
    valid = labels != -100
    logits = policy.last_logits.detach().clone().requires_grad_(True)
    per_token = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), labels.flatten(), reduction="none", label_smoothing=0.01
    ).reshape(labels.shape)
    counts = valid.sum(dim=1).float()
    divisor = counts[counts > 0].pow(1 - exponent).sum()
    expected = (per_token.sum(dim=1) / counts.clamp(min=1).pow(exponent)).sum() / divisor
    hard = F.cross_entropy(logits[:, :-1][valid], labels[valid])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(metrics["sft_loss"], hard)
    assert metrics[OBJECTIVE_WEIGHT_SUM] == divisor.item()
    assert metrics["sft_valid_tokens"] == 5 and metrics["sft_valid_sequences"] == 2
    assert objective.adapt_metrics(metrics, "train")["train/label_smoothing"] == 0.01
    actual.backward()
    expected.backward()
    torch.testing.assert_close(policy.last_logits.grad, logits.grad)
    assert torch.count_nonzero(policy.last_logits.grad[-1]) == 0
    assert torch.count_nonzero(policy.last_logits.grad[:, -1]) == 0


def test_evaluation_and_disabled_identity_are_unchanged():
    packer = MemoryPacker()
    batch = sample_batch(packer)
    policy = Policy().eval()
    ordinary = make_objective(packer, ce_exponent=1)
    smoothed = make_objective(packer, ce_exponent=1, label_smoothing=0.01)
    a, am = ordinary(policy, batch)
    b, bm = smoothed(policy, batch)
    assert torch.equal(a, b) and torch.equal(am["sft_loss"], bm["sft_loss"])
    assert smoothed.adapt_metrics(bm, "val")["val/label_smoothing"] == 0.0
    assert ordinary.identity == make_objective(packer, ce_exponent=1, label_smoothing=0).identity
    assert "label_smoothing" not in ordinary.identity
    assert smoothed.identity["label_smoothing"] == {
        "epsilon": 0.01,
        "distribution": "uniform_vocabulary",
        "training_only": True,
    }
    assert smoothed.identity != make_objective(packer, ce_exponent=1, label_smoothing=0.02).identity


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("exponent", [0.0, 1.0])
def test_variable_length_accumulation_matches_full_batch_update(exponent, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(13)
    packer = MemoryPacker()
    batch = sample_batch(packer)
    policy = Policy().to(device)
    reference = copy.deepcopy(policy)
    objective = make_objective(packer, ce_exponent=exponent, label_smoothing=0.01)
    trainer = LLMTrainer(policy, objective, torch.optim.SGD(policy.parameters(), lr=0.1), grad_accum=len(batch))
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    expected, metrics = objective(reference, batch)
    expected.backward()
    optimizer.step()
    for row in batch:
        trainer.train([row])
    assert trainer.iteration() == 1 and trainer._denom_at_last_step == metrics[OBJECTIVE_WEIGHT_SUM]
    for a, b in zip(policy.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("value", [-0.1, 1.0, float("nan"), float("inf"), True, "0.01"])
def test_invalid_config_and_objective_fail_early(value):
    with pytest.raises(ValueError, match="label_smoothing"):
        SFTConfig(label_smoothing=value)
    with pytest.raises(ValueError, match="label_smoothing"):
        make_objective(MemoryPacker(), label_smoothing=value)


def test_native_cli_and_original_order_defaults():
    from cosmos_framework.utils.config import load_config

    config = load_config("cosmos_framework/posttrain/configs/sft_optimizer_task.py", ["--", "rl.label_smoothing=0.01"])
    assert config.rl.label_smoothing == 0.01
    assert config.rl.data.interleave_window == 0 and config.rl.data.interleave_id_prefix == ""
    assert SFTConfig().label_smoothing == 0.0


def test_exact_resume_rejects_changed_regularization():
    packer = MemoryPacker()
    policy = Policy()
    objective = make_objective(packer, label_smoothing=0.01)
    trainer = LLMTrainer(policy, objective, torch.optim.AdamW(policy.parameters()))
    trainer._validate_objective_identity({"objective_identity": objective.identity})
    for epsilon in (0.0, 0.02):
        other = make_objective(packer, label_smoothing=epsilon)
        with pytest.raises(RuntimeError, match="identity"):
            trainer._validate_objective_identity({"objective_identity": other.identity})


def test_native_validation_publishes_only_hard_ce_and_counts():
    from types import SimpleNamespace

    from cosmos_framework.posttrain.control.validation.lifecycle import ValidationTrigger
    from cosmos_framework.posttrain.entrypoints.sft import _evaluate
    from cosmos_framework.posttrain.roles.trainer.validation.sft import evaluate_sft_batch

    packer = MemoryPacker()
    batch = sample_batch(packer)
    policy = Policy().eval()
    _, expected = make_objective(packer)(policy, batch)
    policy.train()
    trainer = LLMTrainer(
        policy,
        make_objective(packer, label_smoothing=0.01),
        torch.optim.AdamW(policy.parameters()),
        objective_evaluator=evaluate_sft_batch,
    )
    source = SimpleNamespace(
        validation_batch_count=lambda *args: 1,
        get_validation_batch=lambda *args: batch,
        source_identity=lambda: "unchanged-validation-fixture",
    )
    rng_before = torch.get_rng_state().clone()
    report = _evaluate(
        {"validation": source, "policy": trainer},
        SFTConfig(label_smoothing=0.01),
        0,
        frozenset({ValidationTrigger.BEFORE_TRAINING}),
    )
    assert report.metrics["val/loss_avg"] == expected["sft_loss"].item()
    assert report.metrics["val/valid_token_count"] == 5
    assert report.metrics["val/valid_sequence_count"] == 2
    assert set(report.metrics) == {
        "val/loss_avg",
        "val/valid_token_count",
        "val/valid_sequence_count",
        "timer/validation_s",
    }
    assert all(module.training for module in policy.modules())
    assert torch.equal(torch.get_rng_state(), rng_before)
