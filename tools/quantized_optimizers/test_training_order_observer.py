import copy
import json

import pytest
import torch

from cosmos_framework.posttrain.algorithm.sft import make_objective
from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state, restore_rng_state
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker, Policy, sample_batch
from tools.quantized_optimizers.state_digest import digest
from tools.quantized_optimizers.training_order_observer import check_collation, install


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
