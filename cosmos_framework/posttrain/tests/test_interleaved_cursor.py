# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Interleaving preserves exposure, default cursors, tails and exact continuation."""

import copy
import hashlib
import json
import random

import pytest
import torch

from cosmos_framework.posttrain.config.sft import SFTDatasetConfig
from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
from cosmos_framework.posttrain.roles.env.interleaved_cursor import InterleavedCursor
from cosmos_framework.posttrain.roles.env.sft import SFTDatasetEnv


class Dataset:
    def __init__(self, count=47):
        self.ids = [("mcq:" if i % 3 else "text:") + str(i) for i in range(count)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return {
            "id": self.ids[index],
            "messages": [{"role": "user", "content": str(index)}, {"role": "assistant", "content": "answer"}],
        }

    def cursor_fingerprint(self):
        return hashlib.sha256(json.dumps(self.ids).encode()).hexdigest()


def cursor(dataset, **kwargs):
    return InterleavedCursor(
        dataset, id_prefix="mcq:", window=20, shuffle=True, seed=43, cursor_resume="strict", **kwargs
    )


def take(source, count):
    return [source.next_index() for _ in range(count)]


def test_per_window_and_epoch_membership_and_within_kind_order():
    data = Dataset()
    candidate = cursor(data)
    control = IndexedCursor(data, shuffle=True, seed=43, cursor_resume="strict")
    rng = random.getstate()
    for _ in range(3):
        a, b = take(candidate, len(data)), take(control, len(data))
        assert sorted(a) == list(range(len(data)))
        for start in range(0, len(data), 20):
            x, y = a[start : start + 20], b[start : start + 20]
            assert sorted(x) == sorted(y)
            for group in (False, True):
                assert [i for i in x if data.ids[i].startswith("mcq:") == group] == [
                    i for i in y if data.ids[i].startswith("mcq:") == group
                ]
    assert random.getstate() == rng


@pytest.mark.parametrize("position", [0, 1, 19, 20, 44, 46, 47, 48, 94, 99])
def test_restore_next_inputs_through_tail_and_epoch(position):
    data = Dataset()
    reference = cursor(data)
    take(reference, position)
    state = reference.state_dict()
    restored = cursor(data)
    assert restored.load_state_dict(copy.deepcopy(state))
    assert take(reference, 101) == take(restored, 101)
    assert reference.state_dict() == restored.state_dict()


@pytest.mark.parametrize("change", ["version", "window", "prefix", "groups", "position", "dataset"])
def test_invalid_restore_is_atomic(change):
    data = Dataset()
    reference = cursor(data)
    take(reference, 23)
    saved = reference.state_dict()
    target = cursor(data)
    take(target, 7)
    before = target.state_dict()
    if change == "version":
        saved["version"] = 1
    if change == "window":
        saved["interleave"]["window"] = 10
    if change == "prefix":
        saved["interleave"]["id_prefix"] = "other:"
    if change == "groups":
        saved["interleave"]["groups_sha256"] = "bad"
    if change == "position":
        saved["position"] = 1000
    if change == "dataset":
        saved["dataset_fingerprint"] = "bad"
    with pytest.raises(ValueError):
        target.load_state_dict(saved)
    assert target.state_dict() == before


def test_cross_policy_restore_rejected_and_legacy_state_unchanged():
    data = Dataset()
    ordinary = IndexedCursor(data, shuffle=True, seed=43, cursor_resume="strict")
    saved = ordinary.state_dict()
    assert saved["version"] == 1 and "interleave" not in saved
    with pytest.raises(ValueError, match="same interleaving"):
        cursor(data).load_state_dict(saved)
    with pytest.raises(ValueError, match="version"):
        ordinary.load_state_dict(cursor(data).state_dict())
    assert ordinary.state_dict() == saved


def test_next_inputs_and_adam_updates_match_after_interruption():
    data = Dataset(23)
    reference = cursor(data)
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    def update(source, net, opt):
        indices = take(source, 4)
        x = torch.tensor(indices, dtype=torch.float64).reshape(-1, 1) / 23
        opt.zero_grad()
        loss = (net(x) - x.square()).square().mean()
        loss.backward()
        opt.step()
        return indices, copy.deepcopy(net.state_dict()), copy.deepcopy(opt.state_dict())

    for _ in range(5):
        update(reference, model, optimizer)
    saved = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), reference.state_dict())
    expected = [update(reference, model, optimizer) for _ in range(9)]
    restored = cursor(data)
    restored.load_state_dict(saved[2])
    net = torch.nn.Linear(1, 1, dtype=torch.float64)
    net.load_state_dict(saved[0])
    opt = torch.optim.AdamW(net.parameters(), lr=0.01)
    opt.load_state_dict(saved[1])
    actual = [update(restored, net, opt) for _ in range(9)]
    for a, b in zip(actual, expected, strict=True):
        assert a[0] == b[0]
        assert all(torch.equal(a[1][k], b[1][k]) for k in a[1])
        for key, state in a[2]["state"].items():
            assert all(torch.equal(value, b[2]["state"][key][name]) for name, value in state.items())


@pytest.mark.parametrize("prefix,window", [("", 20), ("mcq:", 0), ("mcq:", 1), ("mcq:", True), ("mcq:", 2.0), (3, 20)])
def test_typed_configuration_rejects_invalid_options(prefix, window):
    with pytest.raises((TypeError, ValueError)):
        SFTDatasetConfig(interleave_id_prefix=prefix, interleave_window=window)


def test_validation_and_distributed_modes_rejected():
    with pytest.raises(ValueError, match="one data rank"):
        cursor(Dataset(), shard_count=2)
    with pytest.raises(ValueError, match="validation order"):
        SFTDatasetEnv(Dataset(), training=False, interleave_id_prefix="mcq:", interleave_window=20)
    data = Dataset()
    data.ids = ["text:" + str(i) for i in range(len(data))]
    with pytest.raises(ValueError, match="both matching"):
        cursor(data)


def test_native_cli_policy_and_typed_restore_identity():
    from cosmos_framework.utils.config import load_config

    config = load_config(
        "cosmos_framework/posttrain/configs/sft_optimizer_task.py",
        [
            "--",
            "rl.data.interleave_id_prefix=mcq:",
            "rl.data.interleave_window=20",
        ],
    )
    assert config.rl.data.interleave_id_prefix == "mcq:" and config.rl.data.interleave_window == 20
    candidate = cursor(Dataset())
    saved = candidate.state_dict()
    saved["interleave"]["window"] = 20.0
    with pytest.raises(ValueError, match="same interleaving"):
        candidate.load_state_dict(saved)
    identity = candidate.ordering_identity()
    identity["window"] = 8
    assert candidate.ordering_identity()["window"] == 20
