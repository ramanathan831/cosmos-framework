"""Batch attribution retains native causal masks, padding, weighting and cleanup."""

import pytest
import torch

from cosmos_framework.posttrain.algorithm.sft import make_objective
from cosmos_framework.posttrain.roles.trainer.validation.sft import evaluate_sft_batch
from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker, Policy, sample_batch
from tools.quantized_optimizers.score_dense import evaluate_substantive_batch


@pytest.mark.parametrize("exponent", [0.0, 0.975, 1.0])
def test_native_batch_reduction_and_dummy_rows(exponent):
    packer = MemoryPacker(padding=3)
    rows = sample_batch(packer)
    for row in rows:
        row.transitions[0].algo_extra["validation_weight"] = 1.0
    rows.append(packer.add([1, 2, 3, 4], [-100, 2, 3, 4], weight=0.0))
    policy = Policy()
    policy.net.train()
    policy.net.dropout.eval()  # restore deliberately mixed modes
    objective = make_objective(packer, ce_exponent=exponent)
    expected = evaluate_sft_batch(policy, objective, rows)
    before = torch.get_rng_state().clone()
    versions = [p._version for p in policy.parameters()]
    result = evaluate_substantive_batch(policy, packer, objective, rows)
    assert result["native_loss"] == float(expected["sft_loss"])
    assert result["native_tokens"] == expected["sft_valid_tokens"] == 5
    assert [r["tokens"] for r in result["rows"]] == [1, 4, 0]
    assert result["rows"][0]["target_token_ids"] == [3]
    assert result["rows"][1]["target_token_ids"] == [5, 7, 8, 9]
    assert result["rows"][2]["token_losses"] == []
    assert torch.equal(before, torch.get_rng_state())
    assert policy.net.training and not policy.net.dropout.training
    assert versions == [p._version for p in policy.parameters()]
    assert not policy.net._forward_hooks


def test_failure_restores_collation_hooks_and_modes(monkeypatch):
    packer, policy = MemoryPacker(), Policy()
    rows = sample_batch(packer)
    objective = make_objective(packer)
    collate = packer.sft_collate_fn

    def fail(*args, **kwargs):
        raise RuntimeError("injected native failure")

    monkeypatch.setattr(policy.net, "forward", fail)
    with pytest.raises(RuntimeError, match="injected native failure"):
        evaluate_substantive_batch(policy, packer, objective, rows)
    assert packer.sft_collate_fn == collate
    assert not policy.net._forward_hooks
    assert policy.net.training
