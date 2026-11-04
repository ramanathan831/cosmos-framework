"""Real HF per-layer checkpointing preserves forward values and parameter gradients."""

import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from cosmos_framework.posttrain.model.hf import HFModel, checkpoint_modules

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def model():
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_dropout=0.0,
            _attn_implementation="eager",
            use_cache=False,
        )
    )


def test_selective_checkpoint_has_exact_values_and_gradients():
    torch.manual_seed(17)
    ordinary = model().train()
    selected = copy.deepcopy(ordinary)
    wrapper = HFModel.__new__(HFModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = selected
    wrapper.grad_ckpt_modules = ("model.layers.1",)
    wrapper.apply_gradient_checkpointing()
    assert [layer.gradient_checkpointing for layer in selected.model.layers] == [False, True, False]
    calls = []
    selected.model.layers[1].register_forward_pre_hook(lambda *_: calls.append(True))
    ids = torch.tensor([[1, 2, 3, 4]])
    a = ordinary(input_ids=ids).logits
    b = selected(input_ids=ids).logits
    assert torch.equal(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    assert len(calls) == 2  # forward plus genuine backward recomputation
    for (name, pa), (_, pb) in zip(ordinary.named_parameters(), selected.named_parameters(), strict=True):
        assert torch.equal(pa.grad, pb.grad), name


@pytest.mark.parametrize(
    "names, message",
    [
        (("model.layers.99",), "unmatched"),
        (("model.layers.0", "model.layers.0"), "unique"),
        (("model",), "GradientCheckpointingLayer"),
    ],
)
def test_bad_selective_modules_fail(names, message):
    with pytest.raises(ValueError, match=message):
        checkpoint_modules(model(), names)
