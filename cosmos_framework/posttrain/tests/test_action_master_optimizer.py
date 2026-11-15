# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real CPU fused-master/DCP transactions; eight-bit CUDA qualification is separate."""

from __future__ import annotations

import copy
import gc
import os
import random
import subprocess
import sys
import weakref

import numpy as np
import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.checkpoint.adapter import publish_adapter_artifact, read_adapter_metadata
from cosmos_framework.posttrain.checkpoint.master_optimizer import (
    _deserialize_rank_state,
    _serialize_tensor_rank_state,
    _validate_update_controls,
)
from cosmos_framework.posttrain.config.action_sft import ACTION_MASTER_OPTIMIZER_PROFILE, ActionSFTConfig
from cosmos_framework.posttrain.config.build import build_optimizer
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.entrypoints.action_sft import _Source, validate_config, validate_resume_data
from cosmos_framework.posttrain.model.checkpoint import build_adapter_checkpoint_codec
from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.tests.test_action_sft import Dataset, assert_state_equal
from cosmos_framework.posttrain.tests.test_checkpoint_adapter import FrozenBase
from cosmos_framework.posttrain.utils.rank_state import collect_rank_state
from cosmos_framework.utils.config import load_config

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class TinyAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = FrozenBase(4, 4, bias=False).requires_grad_(False)
        self.base.register_buffer("packed", torch.arange(8, dtype=torch.uint8))
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        self.head = nn.Linear(4, 2)
        self.unused = nn.Parameter(torch.ones(2))
        self._cosmos_adapter_parameters = {n: p for n, p in self.named_parameters() if p.requires_grad}
        self._cosmos_adapter_metadata = {
            "format": "cosmos.family-backbone.v1",
            "task_contract": {"family": "gr00t_n1d7"},
        }
        self._cosmos_action_optimizer_profile = ACTION_MASTER_OPTIMIZER_PROFILE
        codec = build_adapter_checkpoint_codec(self, identity={"base": "cpu-transaction-fixture"})
        codec.bind_training_identity({"schedule": "fixed-fixture", "seed": 17})
        self._cosmos_adapter_checkpoint_codec = codec

    def forward(self, x):
        return self.head(self.base(x) + self.lora_B(self.lora_A(x)))


def make_trainer(seed=17, evaluation_every=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = TinyAction()
    opt = OptimizerConfig(impl="fused", update_policy="fp32_master", lr=0.01)
    optimizer = build_optimizer(net, opt)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (1 + step))

    def objective(model, batch):
        loss = model(torch.randn(2, 4)).square().mean()
        return loss, {"loss": float(loss.detach())}

    trainer = LLMTrainer(
        net, Objective(objective, reduction=Reduction.MEAN), optimizer, checkpoint_format="dcp", lr_scheduler=scheduler
    )
    rl = ActionSFTConfig(
        optimizer=opt,
        optimizer_profile=ACTION_MASTER_OPTIMIZER_PROFILE,
        seed=seed,
        evaluation_every=evaluation_every,
        max_steps=300,
    )
    source = _Source(Dataset())
    cursor = IndexedCursor(source, shuffle=True, seed=seed, cursor_resume="strict", label="action SFT")
    trainer._adapter_resume_data_validator = lambda data, state: validate_resume_data(source, rl, data, state)
    return trainer, cursor


def step(trainer, cursor):
    cursor.next_index()
    trainer.train([None])


def save(trainer, cursor, root, history=None):
    path = root / "checkpoints" / f"iter_{trainer._iter:06d}"
    trainer.save(
        str(path),
        data_state=collect_rank_state(
            {
                "format": "action-sft-state-v1",
                "cursor": cursor.state_dict(),
                "evaluation": [{"step": 0, "metrics": {"loss": 0.25}}] if history is None else history,
            }
        ),
    )
    return path


def snapshot(trainer):
    return {
        "model": {n: p.detach().clone() for n, p in trainer._adapter_checkpoint_codec().parameters.items()},
        "optimizer": copy.deepcopy(trainer.opt.state_dict()),
        "trainer": copy.deepcopy(trainer._trainer_state()),
    }


@pytest.mark.parametrize("completed", [0, 3])
@pytest.mark.parametrize("live_steps", [0, 1])
def test_native_view_dcp_lazy_and_trained_recovery_matches_next_update(tmp_path, completed, live_steps):
    source, cursor = make_trainer()
    frozen = source._net().base.weight.detach().clone()
    packed = source._net().base.packed.clone()
    for _ in range(completed):
        step(source, cursor)
    assert all(p.data_ptr() == m.data_ptr() for p, m in zip(source.opt.live, source.opt.masters))
    assert source._net().unused not in source.opt.state
    checkpoint = save(source, cursor, tmp_path)
    before = snapshot(source)
    assert bool(source.opt.state) == bool(completed)  # No save-time lazy moment creation.
    step(source, cursor)
    expected = snapshot(source)
    restored, restored_cursor = make_trainer()
    for _ in range(live_steps):
        step(restored, restored_cursor)
    assert restored.load(str(checkpoint)) == completed
    assert_state_equal(snapshot(restored), before)
    # A resumed production cursor restores separately, after the pre-commit probe.
    step(restored, cursor)
    assert_state_equal(snapshot(restored), expected)
    torch.testing.assert_close(source._net().base.weight, frozen, rtol=0, atol=0)
    assert torch.equal(source._net().base.packed, packed)
    assert restored._net().unused not in restored.opt.state
    metadata = read_adapter_metadata(str(checkpoint), require_resume=True)
    assert metadata["optimizer"]["class"].endswith("MasterWeightOptimizer")
    assert set(metadata["parameters"]) == set(before["model"])


def test_fused_master_ownership_preserves_direct_adamw_updates_and_default_identity():
    master, cursor = make_trainer()
    direct = TinyAction()
    with torch.no_grad():
        for left, right in zip(direct.parameters(), master._net().parameters()):
            left.copy_(right)
    direct._cosmos_action_optimizer_profile = "native_fused_all_v1"
    opt = build_optimizer(direct, OptimizerConfig(impl="fused", lr=0.01))
    assert type(opt) is torch.optim.AdamW
    assert direct._cosmos_adapter_checkpoint_codec.optimizer_identity(opt)["class"] == "torch.optim.AdamW"
    # Identical gradients and scheduler-owned LR, including an unused trainable.
    for index in range(4):
        for name, parameter in direct.named_parameters():
            if parameter.requires_grad and name != "unused":
                gradient = torch.full_like(parameter, 0.01 * (index + 1))
                parameter.grad = gradient.clone()
                dict(master._net().named_parameters())[name].grad = gradient.clone()
        opt.step()
        master.opt.step()
        for left, right in zip(direct.parameters(), master._net().parameters()):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        opt.zero_grad()
        master.opt.zero_grad()


@pytest.mark.parametrize(
    "damage",
    [
        "identity",
        "moment_dtype",
        "checksum",
        "model_master",
        "future_step",
        "negative_step",
        "fractional_step",
        "nan_moment",
    ],
)
def test_opaque_optimizer_corruption_is_rejected_before_model_mutation(tmp_path, monkeypatch, damage):
    source, cursor = make_trainer()
    step(source, cursor)
    checkpoint = save(source, cursor, tmp_path)
    live, cursor = make_trainer()
    step(live, cursor)
    before = snapshot(live)
    original = live.checkpoint_manager.load_dcp_component

    def load(path, name, target):
        result = original(path, name, target)
        if name == "optim":
            payload = target["ranks"]["0"]
            if damage == "checksum":
                payload[2][0] ^= 1
            else:
                state = _deserialize_rank_state(payload, target["format"])
                if damage == "identity":
                    state["identity"]["rank"] = 1
                elif damage == "moment_dtype":
                    item = next(iter(state["backend"]["state"].values()))
                    item["exp_avg"] = item["exp_avg"].bfloat16()
                elif damage in ("future_step", "negative_step", "fractional_step", "nan_moment"):
                    item = next(iter(state["backend"]["state"].values()))
                    if damage == "nan_moment":
                        item["exp_avg"].fill_(float("nan"))
                    else:
                        item["step"].fill_({"future_step": 999, "negative_step": -1, "fractional_step": 0.5}[damage])
                else:
                    state["masters"][0].add_(1)
                target["ranks"]["0"] = _serialize_tensor_rank_state(state)
        return result

    monkeypatch.setattr(live.checkpoint_manager, "load_dcp_component", load)
    with pytest.raises(ValueError):
        live.load(str(checkpoint))
    assert_state_equal(snapshot(live), before)


@pytest.mark.parametrize("clock", [0, 1, 3])
def test_opaque_controls_accept_lazy_and_late_parameter_updates(clock):
    # A codec schema test, not a bitsandbytes kernel test. Native dtype/mapping
    # checks still run in the real backend after this control admission.
    state = {
        "backend": {
            "state": {
                0: {},
                1: {"step": clock, "state1": torch.arange(8, dtype=torch.uint8), "absmax1": torch.ones(2)},
                2: {"step": torch.tensor(float(clock)), "exp_avg": torch.zeros(4)},
            }
        }
    }
    _validate_update_controls(state, optimizer_step=3)
    _validate_update_controls({"backend": {"state": {}}}, optimizer_step=0)


@pytest.mark.parametrize("clock", [True, -1, 4, 0.5, torch.tensor(float("nan")), torch.tensor(float("inf"))])
def test_opaque_controls_reject_invalid_adam_or_bnb_clocks(clock):
    with pytest.raises(ValueError, match="clock"):
        _validate_update_controls({"backend": {"state": {0: {"step": clock}}}}, optimizer_step=3)


@pytest.mark.parametrize("name", ["absmax1", "absmax2", "state1", "state2", "qmap1"])
def test_opaque_controls_reject_nonfinite_floating_moments_and_scales(name):
    state = {"backend": {"state": {0: {"step": 1, name: torch.tensor([0.0, float("inf")])}}}}
    with pytest.raises(ValueError, match="nonfinite"):
        _validate_update_controls(state, optimizer_step=3)


@pytest.mark.parametrize("corruption", ["cursor", "history", "rng", "scheduler", "lr", "metadata"])
def test_precommit_rejections_do_not_mutate_live_state(tmp_path, monkeypatch, corruption):
    source, cursor = make_trainer()
    for _ in range(3):
        step(source, cursor)
    checkpoint = save(source, cursor, tmp_path)
    live, cursor = make_trainer()
    step(live, cursor)
    before = snapshot(live)
    original = live.checkpoint_manager.load_component

    def load(path, name):
        value = copy.deepcopy(original(path, name))
        if name == "trainer":
            if corruption == "rng":
                value["rng"][0]["torch"] = torch.ones(2, dtype=torch.float32)
            elif corruption == "scheduler":
                value["lr_scheduler"]["last_epoch"] = 9
            elif corruption == "lr":
                value["lr_scheduler"]["_last_lr"] = [0.987]
        if name == "data":
            if corruption == "cursor":
                value["states"][0]["cursor"]["position"] = 0
            elif corruption == "history":
                value["states"][0]["evaluation"] = [{"step": 4, "metrics": {"loss": 0.1}}]
        if name == "adapter" and corruption == "metadata":
            value["identity"]["base"] = "wrong-base"
        return value

    if corruption == "metadata":
        path = checkpoint / "model_metadata.pt"
        metadata = torch.load(path, weights_only=True)
        metadata["identity"]["base"] = "wrong-base"
        torch.save(metadata, path)
    monkeypatch.setattr(live.checkpoint_manager, "load_component", load)
    with pytest.raises((ValueError, RuntimeError)):
        live.load(str(checkpoint))
    assert_state_equal(snapshot(live), before)


@pytest.mark.parametrize("history_steps", [[], [2, 4], [0, 4], [0, 2]])
def test_missing_initial_or_scheduled_evaluation_refuses_resume_before_mutation(tmp_path, history_steps):
    source, cursor = make_trainer(evaluation_every=2)
    for _ in range(4):
        step(source, cursor)
    history = [{"step": update, "metrics": {"loss": 0.25}} for update in history_steps]
    checkpoint = save(source, cursor, tmp_path, history)
    live, cursor = make_trainer(evaluation_every=2)
    step(live, cursor)
    before = snapshot(live)
    with pytest.raises(ValueError, match="evaluation history is missing required steps"):
        live.load(str(checkpoint))
    assert_state_equal(snapshot(live), before)


@pytest.mark.parametrize(
    "completed,cadence,history_steps",
    [(0, 10, [0]), (3, 2, [0, 2]), (3, 2, [0, 1, 2, 3]), (100, 128, [0]), (100, 128, [0, 33, 100])],
)
def test_history_admits_initial_save_between_scores_and_extra_boundaries(tmp_path, completed, cadence, history_steps):
    source, cursor = make_trainer(evaluation_every=cadence)
    for _ in range(completed):
        step(source, cursor)
    history = [{"step": update, "metrics": {"loss": 0.25}} for update in history_steps]
    checkpoint = save(source, cursor, tmp_path, history)
    before = snapshot(source)
    live, _ = make_trainer(evaluation_every=cadence)
    assert live.load(str(checkpoint)) == completed
    assert_state_equal(snapshot(live), before)


@pytest.mark.parametrize("step", [-1, 301, 10**30, True])
def test_resume_horizon_refusal_precedes_cursor_and_history_expansion(step):
    source = _Source(Dataset())
    rl = ActionSFTConfig(max_steps=300)
    state = collect_rank_state({"format": "action-sft-state-v1", "cursor": None, "evaluation": []})
    with pytest.raises(ValueError, match="outside the declared training horizon"):
        validate_resume_data(source, rl, state, {"optimizer_step": step})


@pytest.mark.parametrize("step", [0, 300])
def test_resume_horizon_includes_initial_and_final_completed_updates(step):
    source = _Source(Dataset())
    rl = ActionSFTConfig(max_steps=300, evaluation_every=100)
    cursor = IndexedCursor(source, shuffle=True, seed=rl.seed, cursor_resume="strict", label="action SFT")
    for _ in range(step):
        cursor.next_index()
    state = collect_rank_state(
        {
            "format": "action-sft-state-v1",
            "cursor": cursor.state_dict(),
            "evaluation": [{"step": update, "metrics": {"loss": 0.25}} for update in range(0, step + 1, 100)],
        }
    )
    validate_resume_data(source, rl, state, {"optimizer_step": step})


def test_commit_failure_rolls_back_and_releases_replacement_without_gc(tmp_path, monkeypatch):
    source, cursor = make_trainer()
    for _ in range(3):
        step(source, cursor)
    checkpoint = save(source, cursor, tmp_path)
    live, cursor = make_trainer()
    step(live, cursor)
    before = snapshot(live)
    refs = []
    original = live.opt.commit_state_dict

    def fail(staged):
        original(staged)
        if len(staged) == 3:
            refs.append(weakref.ref(staged[1]))
            raise RuntimeError("injected post-optimizer commit failure")

    monkeypatch.setattr(live.opt, "commit_state_dict", fail)
    enabled = gc.isenabled()
    gc.disable()
    try:
        with pytest.raises(RuntimeError, match="injected") as retained_error:
            live.load(str(checkpoint))
        assert retained_error.value is not None
        assert refs and all(ref() is None for ref in refs)
        assert_state_equal(snapshot(live), before)
    finally:
        if enabled:
            gc.enable()


def test_opaque_serialization_retains_uint8_moments_and_fp32_scales_and_rejects_corruption():
    state = {"moments": torch.arange(16, dtype=torch.uint8), "scales": torch.randn(3), "unused": {}}
    payload = _serialize_tensor_rank_state(state)
    assert_state_equal(_deserialize_rank_state(payload, "cosmos_rank_master_v3"), state)
    payload[2][0] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        _deserialize_rank_state(payload, "cosmos_rank_master_v3")


def test_fresh_portable_reload_does_not_require_training_optimizer(tmp_path):
    trainer, cursor = make_trainer()
    for _ in range(2):
        step(trainer, cursor)
    checkpoint = save(trainer, cursor, tmp_path)
    artifact = publish_adapter_artifact(str(checkpoint), str(tmp_path / "portable" / "checkpoints" / "iter_000002"))
    expected = trainer._net()(torch.ones(1, 4)).detach()
    output = tmp_path / "output.pt"
    script = """
import sys, torch
sys.modules['bitsandbytes'] = None
from cosmos_framework.posttrain.tests.test_action_master_optimizer import TinyAction, make_trainer
from cosmos_framework.posttrain.checkpoint.adapter import load_adapter_weights
from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
torch.manual_seed(17)
net = TinyAction()
load_adapter_weights(sys.argv[1], net._cosmos_adapter_checkpoint_codec, manager=CheckpointManager())
resumed, _ = make_trainer()
assert resumed.load(sys.argv[3]) == 2
torch.testing.assert_close(net(torch.ones(1,4)), resumed._net()(torch.ones(1,4)), rtol=0, atol=0)
torch.save(net(torch.ones(1,4)).detach(), sys.argv[2])
"""
    subprocess.run(
        [sys.executable, "-c", script, str(artifact), str(output), str(checkpoint)],
        check=True,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    torch.testing.assert_close(torch.load(output, weights_only=True), expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["bf16_lora", "nf4_lora"])
def test_explicit_recipe_preserves_selected_base_and_separates_optimizer(mode):
    config = load_config(
        "cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora_bnb8.py",
        ["--", f"model.backbone_adaptation.mode={mode}"],
    )
    rl = validate_config(config)
    assert (rl.optimizer.name, rl.optimizer.update_policy, rl.optimizer.impl) == ("adamw_bnb_8bit", "fp32_master", "")
    assert rl.optimizer.lr == 1e-5
    baseline = load_config("cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py", ["--"])
    assert (baseline.rl.optimizer.name, baseline.rl.optimizer.update_policy) == ("adamw", "parameter")


@pytest.mark.parametrize(
    "override",
    [
        "model.backbone_adaptation.mode=nf4_frozen",
        "model.backbone_adaptation.rank=64",
        "model.backbone_adaptation.train_action_head=false",
        "rl.grad_accum=2",
        "rl.save_async=true",
        "model.embodiment_tag=other",
        "rl.optimizer.name=adamw_bnb_8bit_paged",
        "rl.optimizer.update_policy=parameter",
    ],
)
def test_unqualified_modes_fail_at_config_admission(override):
    with pytest.raises((ValueError, RuntimeError)):
        config = load_config("cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora_bnb8.py", ["--", override])
        validate_config(config)


def test_eight_bit_runtime_preflight_is_explicit_and_does_not_affect_default(monkeypatch):
    from cosmos_framework.posttrain.entrypoints.action_sft import _preflight_optimizer_runtime
    from cosmos_framework.utils import quantized_optimizer

    def missing(*, paged):
        assert paged is False
        raise ImportError("fixture: missing backend")

    monkeypatch.setattr(quantized_optimizer, "require_backend", missing)
    _preflight_optimizer_runtime(ActionSFTConfig())  # The default never imports/probes bnb kernels.
    rl = ActionSFTConfig(
        optimizer=OptimizerConfig(name="adamw_bnb_8bit", update_policy="fp32_master"),
        optimizer_profile=ACTION_MASTER_OPTIMIZER_PROFILE,
    )
    with pytest.raises(ImportError, match="native_action/README.md.*No optimizer fallback"):
        _preflight_optimizer_runtime(rl)
