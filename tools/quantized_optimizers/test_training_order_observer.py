import copy
import json

import pytest
import torch

from cosmos_framework.posttrain.algorithm.sft import make_objective
from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state, restore_rng_state
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker, Policy, sample_batch
from tools.quantized_optimizers.state_digest import digest
from tools.quantized_optimizers.training_order_observer import check_collation, install, sample_indices, tensor_stats


@pytest.mark.parametrize("elements", [1, 17, 2**24 + 2, 3072 * 8192, 2**31 + 13])
def test_integer_sampling_has_exact_in_range_endpoints(elements):
    indices = sample_indices(elements, "cpu")
    assert indices.dtype == torch.int64
    assert len(indices) == min(elements, 256)
    assert indices[0] == 0 and indices[-1] == elements - 1
    assert len(indices.unique()) == len(indices)
    assert bool(((indices >= 0) & (indices < elements)).all())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_large_actual_tensor_sampling_avoids_float_endpoint_rounding(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("accelerator test runs inside the admitted CUDA diagnostic")
    elements = 3072 * 8192
    # This was the real observer failure: float32 rounds n-1 up to n.
    assert int(torch.linspace(0, elements - 1, 256, dtype=torch.float32)[-1]) == elements
    value = torch.zeros(elements, device=device)
    value[-1] = 1
    before = digest(collect_rng_state())
    stats = tensor_stats(value)
    assert stats["samples"][-1] == stats["l2"] == stats["max_abs"] == 1
    assert stats["elements"] == elements
    assert before == digest(collect_rng_state())


def test_observer_preserves_multistep_updates_rng_and_modes(tmp_path):
    torch.manual_seed(7)
    packer = MemoryPacker()
    batch = sample_batch(packer)
    a = Policy()
    b = copy.deepcopy(a)
    plain = LLMTrainer(a, make_objective(packer, ce_exponent=1), torch.optim.AdamW(a.parameters(), lr=1e-3))
    watched = LLMTrainer(b, make_objective(packer, ce_exponent=1), torch.optim.AdamW(b.parameters(), lr=1e-3))
    install(watched, packer, tmp_path, write_full=True)
    for _ in range(15):
        rng = collect_rng_state()
        x = plain.train(batch)
        after = digest(collect_rng_state())
        restore_rng_state(rng)
        y = watched.train(batch)
        assert after == digest(collect_rng_state())
        assert x == y
        assert digest(a.state_dict()) == digest(b.state_dict())
        assert digest(plain.opt.state_dict()) == digest(watched.opt.state_dict())
    observations = [json.loads(x) for x in (tmp_path / "training_observations.jsonl").read_text().splitlines()]
    assert [x["step"] for x in observations] == [1, 13, 14, 15]
    assert all(x["rng_before"] == x["rng_after"] for x in observations)


def test_swapped_multimodal_segment_is_rejected():
    from cosmos_framework.posttrain.data.packer.sft import QwenSFTDataPacker, SFTPolicyInput

    packer = object.__new__(QwenSFTDataPacker)
    from types import SimpleNamespace

    packer.max_length, packer.tokenizer = 20, SimpleNamespace(pad_token_id=0)
    samples = [
        SFTPolicyInput([1, 2, 3], [-100, 2, 3], {"pixel_values_videos": torch.full((2, 3), float(i))}) for i in range(2)
    ]
    batch = packer.sft_collate_fn(samples, 3)
    check_collation(samples, batch)
    batch["pixel_values_videos"] = batch["pixel_values_videos"].flip(0)
    with pytest.raises(AssertionError):
        check_collation(samples, batch)
