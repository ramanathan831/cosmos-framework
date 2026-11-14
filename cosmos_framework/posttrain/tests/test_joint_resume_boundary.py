# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native RL control files must agree before any worker/model restoration."""

from copy import deepcopy

import pytest
import torch

from cosmos_framework.posttrain.checkpoint import CheckpointManager
from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.entrypoints.joint_ppo import validate_resume_boundary
from cosmos_framework.posttrain.roles.env.ppo_window import PPOWindowCollector
from cosmos_framework.posttrain.tests.helpers.joint_ppo import GaussianEnvironment


@pytest.mark.parametrize("corruption", [None, "negative", "boolean", "reset", "missing", "topology", "version", "partial"])
def test_collection_and_learner_boundary_validation_is_read_only(tmp_path, corruption):
    recipe = JointPPOConfig(num_envs=4, window_decisions=2, global_minibatch=4, microbatch=1, max_steps=8)
    environment = GaussianEnvironment()
    collectors = [PPOWindowCollector(recipe, rank, 2, environment=environment) for rank in range(2)]
    control = {"publication_version": 2, "optimizer_step": 4, "accum": 0}
    data = {"collectors": {str(c.replica): dict(c.state_dict(), next_window=2) for c in collectors}}
    state = data["collectors"]["1"]
    if corruption == "negative":
        state["next_window"] = -1
    elif corruption == "boolean":
        state["next_window"] = True
    elif corruption == "reset":
        state["reset_on_resume"] = False
    elif corruption == "missing":
        del state["reset_on_resume"]
    elif corruption == "topology":
        del data["collectors"]["1"]
    elif corruption == "version":
        state["next_window"] = 3
    elif corruption == "partial":
        control["accum"] = 1
    torch.save(control, tmp_path / "trainer.pt")
    torch.save(data, tmp_path / "data.pt")
    before = deepcopy([c.state_dict() for c in collectors])
    manager = CheckpointManager()
    if corruption is None:
        validate_resume_boundary(manager, str(tmp_path), recipe, environment, 2)
    else:
        with pytest.raises(ValueError):
            validate_resume_boundary(manager, str(tmp_path), recipe, environment, 2)
    assert before == [c.state_dict() for c in collectors]
    assert control == torch.load(tmp_path / "trainer.pt", weights_only=False)
    assert data == torch.load(tmp_path / "data.pt", weights_only=False)


def test_invalid_cursor_does_not_change_live_collector():
    recipe = JointPPOConfig(num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1)
    collector = PPOWindowCollector(recipe, 0, 2, environment=GaussianEnvironment())
    before = collector.state_dict()
    for malformed in (-1, True, 1.5):
        with pytest.raises(ValueError):
            collector.load_state_dict(dict(before, next_window=malformed))
        assert collector.state_dict() == before
