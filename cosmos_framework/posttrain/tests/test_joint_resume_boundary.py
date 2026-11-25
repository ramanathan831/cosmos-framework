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


@pytest.mark.parametrize(
    "corruption", [None, "negative", "boolean", "reset", "missing", "topology", "version", "partial"]
)
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


def test_collector_restore_binds_generation_before_cursor_mutation(tmp_path, monkeypatch):
    recipe = JointPPOConfig(num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1)
    collector = PPOWindowCollector(recipe, 0, 2, environment=GaussianEnvironment())
    saved = dict(collector.state_dict(), next_window=2)
    for name in ("model", "optim", "trainer"):
        torch.save({}, tmp_path / f"{name}.pt")
    torch.save({"collectors": {"0": saved}}, tmp_path / "data.pt")
    manager = CheckpointManager()
    collector.restore_checkpoint(manager, str(tmp_path))
    assert collector.state_dict() == saved
    provenance = collector.loaded_checkpoint_provenance()
    assert provenance == manager.checkpoint_provenance(str(tmp_path))
    provenance["signature"] = "caller mutation"
    assert collector.loaded_checkpoint_provenance() != provenance
    before = collector.state_dict(), collector.loaded_checkpoint_provenance()
    read = manager.load_component

    def mutate(path, name):
        result = read(path, name)
        with (tmp_path / "trainer.pt").open("ab") as handle:
            handle.write(b"changed generation")
        return result

    monkeypatch.setattr(manager, "load_component", mutate)
    with pytest.raises(RuntimeError, match="changed while it was being loaded"):
        collector.restore_checkpoint(manager, str(tmp_path))
    assert (collector.state_dict(), collector.loaded_checkpoint_provenance()) == before


def test_pending_publication_mount_preflight_is_fail_closed_and_skips_same_run(tmp_path, monkeypatch):
    import errno

    from cosmos_framework.posttrain.checkpoint import external_copy
    from cosmos_framework.posttrain.entrypoints.joint_ppo import preflight_pending_publication

    output = tmp_path / "original"
    checkpoint = output / "checkpoints" / "iter_000004"
    checkpoint.mkdir(parents=True)
    torch.save(
        {"optimizer_step": 4, "validation": {"progress": {"pending_step": 4}, "publication": None}},
        checkpoint / "trainer.pt",
    )
    before = (checkpoint / "trainer.pt").read_bytes()
    calls = []

    def unsupported(source, target):
        calls.append((source, target))
        raise OSError(errno.ENOTSUP, "injected unsupported mount")

    monkeypatch.setattr(external_copy, "_rename_directory_noreplace", unsupported)
    from cosmos_framework.posttrain.control.checkpointing import checkpoint_iteration_dir

    assert checkpoint_iteration_dir(str(output), 4) == str(checkpoint)  # the coordinator's path rule
    manager = CheckpointManager()
    preflight_pending_publication(manager, str(checkpoint), str(output))
    assert calls == []
    with pytest.raises(OSError, match="use the original run output directory") as failure:
        preflight_pending_publication(manager, str(checkpoint), str(tmp_path / "new"))
    assert failure.value.errno == errno.ENOTSUP and len(calls) == 1
    assert list((tmp_path / "new/checkpoints").iterdir()) == []
    assert (checkpoint / "trainer.pt").read_bytes() == before


def test_one_budget_rule_for_resume_preflight_and_restored_counters():
    from types import SimpleNamespace

    from cosmos_framework.posttrain.entrypoints.joint_ppo import is_budget_boundary

    recipe = SimpleNamespace(decisions=128, global_minibatch=32, max_steps=8)  # four updates per window
    assert is_budget_boundary(recipe, 0, 0) and is_budget_boundary(recipe, 8, 2)
    for step, version in ((4, 2), (12, 3), (-4, -1), (True, 0), (4.0, 1)):
        assert not is_budget_boundary(recipe, step, version)
