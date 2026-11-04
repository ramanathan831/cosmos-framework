"""Complete model restoration avoids redundant pretrained I/O without changing values."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from cosmos_framework.posttrain.model.hf import HFModel

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def saved_model(path, tied=False):
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=tied,
            _attn_implementation="eager",
            attention_dropout=0.0,
        )
    )
    model.save_pretrained(path)


def wrapper(path):
    return HFModel(str(path), dtype=torch.float32, attn_implementation="eager", mixed_precision_mode="autocast")


@pytest.mark.parametrize("tied", [False, True])
def test_complete_state_preserves_outputs_buffers_and_ties_without_pretrained_reads(tmp_path, monkeypatch, tied):
    # Keep this test CPU-only even when a GPU runner imports it.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    saved_model(tmp_path, tied)
    baseline = wrapper(tmp_path)
    baseline.setup(None)
    with torch.no_grad():
        for p in baseline.net.parameters():
            p.add_(0.01)
    state = {k: v.clone() for k, v in baseline.net.state_dict().items()}
    for path in tmp_path.glob("*.safetensors"):
        path.unlink()
    restored = wrapper(tmp_path)
    restored.setup(None, initial_state=state)
    baseline.net.eval()
    restored.net.eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    assert torch.equal(baseline.net(input_ids=ids).logits, restored.net(input_ids=ids).logits)
    assert set(dict(baseline.net.named_buffers())) == set(dict(restored.net.named_buffers()))
    for name, value in baseline.net.named_buffers():
        assert torch.equal(value, dict(restored.net.named_buffers())[name])
    if tied:
        assert restored.net.get_input_embeddings().weight is restored.net.get_output_embeddings().weight


@pytest.mark.parametrize("invalid", ["missing", "extra", "dtype", "shape", "sharded"])
def test_partial_or_incompatible_state_refused_before_materialization(tmp_path, monkeypatch, invalid):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    saved_model(tmp_path)
    baseline = wrapper(tmp_path)
    baseline.setup(None)
    state = dict(baseline.net.state_dict())
    key = next(iter(state))
    if invalid == "missing":
        state.pop(key)
    elif invalid == "extra":
        state["unrelated"] = torch.ones(1)
    elif invalid == "dtype":
        state[key] = state[key].to(torch.bfloat16)
    elif invalid == "shape":
        state[key] = state[key].flatten()
    restored = wrapper(tmp_path)
    with pytest.raises(ValueError, match="direct full HF state"):
        restored.setup(object() if invalid == "sharded" else None, initial_state=state)
    assert all(p.device.type == "meta" for p in restored.net.parameters())
