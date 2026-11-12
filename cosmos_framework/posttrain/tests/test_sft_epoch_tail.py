"""Exact exposure and continuation of explicitly partial SFT epoch batches."""

import copy

import pytest

from cosmos_framework.posttrain.config.sft import SFTConfig, SFTDatasetConfig
from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
from cosmos_framework.posttrain.roles.env.sft import SFTDatasetEnv
from cosmos_framework.posttrain.tests.test_interleaved_cursor import Dataset


@pytest.mark.parametrize("count", [1, 2, 4, 7])
def test_complete_passes_and_wrap_compatibility(count):
    data = Dataset(346)
    partial = IndexedCursor(data, shuffle=True, seed=43, cursor_resume="strict")
    original = IndexedCursor(data, shuffle=True, seed=43, cursor_resume="strict")
    for _ in range(2):
        batches = [partial.next_batch_indices(count, stop_at_epoch=True) for _ in range((346 + count - 1) // count)]
        flat = sum(batches, [])
        assert len(flat) == 346 and sorted(flat) == list(range(346))
        assert flat == [original.next_index() for _ in range(346)]
    assert partial.state_dict() == original.state_dict()


@pytest.mark.parametrize("batches", [0, 85, 86, 87, 88, 173, 174])
def test_next_inputs_and_state_resume_at_tail(batches):
    data = Dataset(346)
    control = SFTDatasetEnv(data, seed=43, epoch_boundary="partial")
    for _ in range(batches):
        control.get_samples(4)
    saved = copy.deepcopy(control.state_dict())
    resumed = SFTDatasetEnv(data, seed=43, epoch_boundary="partial")
    assert resumed.load_state_dict(saved)
    for _ in range(8):
        a, b = control.get_samples(4), resumed.get_samples(4)
        assert [t.id for t in a] == [t.id for t in b]
        assert [t.transitions[0].targets for t in a] == [t.transitions[0].targets for t in b]
    assert control.state_dict() == resumed.state_dict()


def test_policy_is_explicit_and_rejects_unqualified_execution():
    assert SFTDatasetEnv(Dataset()).ordering_identity() is None
    assert SFTDatasetEnv(Dataset(), epoch_boundary="partial").ordering_identity() == {
        "epoch_boundary": "partial_v1",
        "order": None,
    }
    with pytest.raises(ValueError, match="single-rank"):
        SFTDatasetEnv(Dataset(8), world_size=2, epoch_boundary="partial")
    with pytest.raises(ValueError, match="single-rank"):
        SFTDatasetEnv(Dataset(), training=False, epoch_boundary="partial")
    with pytest.raises(ValueError, match="grad_accum"):
        SFTConfig(data=SFTDatasetConfig(epoch_boundary="partial"), grad_accum=2)
    with pytest.raises(ValueError, match="epoch_boundary"):
        SFTDatasetConfig(epoch_boundary="drop")


def test_tail_is_real_two_row_batch_without_padding():
    env = SFTDatasetEnv(Dataset(346), epoch_boundary="partial")
    sizes = [len(env.get_samples(4)) for _ in range(174)]
    assert sizes == ([4] * 86 + [2]) * 2
    assert env.cursor.consumed == 692
