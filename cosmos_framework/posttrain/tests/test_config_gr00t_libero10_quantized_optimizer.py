# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""LIBERO-10 quantized-optimizer recipe composes native suite bindings with the 4-GPU optimizer profile."""

import json

import pytest

from cosmos_framework.posttrain.config.config_joint_ppo import validate_joint_ppo_config
from cosmos_framework.posttrain.embodied.sims.libero.protocol import LIBERO_PROTOCOLS, fixed_manifest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

QUANTIZED = "cosmos_framework/posttrain/configs/gr00t_n1d7_libero10_quantized_optimizer.py"
SPATIAL = "cosmos_framework/posttrain/configs/gr00t_n1d7_libero_quantized_optimizer.py"


def _checkpoint(tmp_path):
    from cosmos_framework.posttrain.model.gr00t.build import CHECKPOINT_REVISION

    checkpoint = tmp_path / CHECKPOINT_REVISION / "libero_10"
    checkpoint.mkdir(parents=True)
    for name, value in {
        "config.json": {"action_horizon": 40, "max_action_dim": 132, "num_inference_timesteps": 4},
        "processor_config.json": {
            "processor_kwargs": {"modality_configs": {"libero_sim": {"action": {"delta_indices": list(range(16))}}}}
        },
        "statistics.json": {},
        "embodiment_id.json": {},
        "model.safetensors.index.json": {},
    }.items():
        (checkpoint / name).write_text(json.dumps(value))
    return checkpoint


def _load(path, tmp_path, profile, *extra, checkpoint=None):
    from cosmos_framework.utils.config import load_config

    for split in ("train", "eval"):
        fixed_manifest(split, "libero_10", profile if split == "eval" else "legacy_100").write(
            tmp_path / f"{split}.json"
        )
    return load_config(
        path,
        [
            "--",
            f"model.checkpoint_path={checkpoint or _checkpoint(tmp_path)}",
            "environment.config.sim_python=python",
            f"environment.config.eval_profile={profile}",
            f"environment.config.train_manifest={tmp_path / 'train.json'}",
            f"environment.config.eval_manifest={tmp_path / 'eval.json'}",
            *extra,
        ],
    )


@pytest.mark.parametrize("profile", ["legacy_100", "aligned_200"])
def test_libero10_quantized_recipe_binds_suite_profile_and_optimizer_policy(tmp_path, profile):
    from cosmos_framework.utils.lazy_config import instantiate

    config = _load(QUANTIZED, tmp_path, profile)
    validate_joint_ppo_config(config, 4)
    environment = config.environment.config
    protocol = LIBERO_PROTOCOLS["libero_10"]
    assert environment.suite == "libero_10" and environment.eval_profile == profile
    assert config.model.execution_horizon == environment.execution_horizon == protocol.execution_horizon == 8
    assert environment.max_episode_steps == protocol.max_episode_steps == 720
    recipe = instantiate(config.rl)
    assert (recipe.num_envs, recipe.window_decisions, recipe.global_minibatch, recipe.microbatch) == (4, 16, 64, 2)
    assert recipe.optimizer.name == "adamw_bnb_8bit"
    assert recipe.optimizer.update_policy == "fp32_master"
    assert config.policy.verify_frozen_state is True and config.model.backbone_from_checkpoint is True
    # Native recipe hyperparameters are unchanged relative to the spatial quantized recipe.
    assert recipe.optimizer.lr == 5e-6 and recipe.value_lr == 1e-4
    assert list(recipe.optimizer.betas) == [0.9, 0.95] and recipe.optimizer.eps == 1e-8
    assert recipe.optimizer.weight_decay == 0.01 and recipe.optimizer.grad_clip_norm == 1.0


def test_libero10_quantized_recipe_selects_fused_control_and_paged(tmp_path):
    from cosmos_framework.utils.lazy_config import instantiate

    for name, impl in (("adamw", "fused"), ("paged_adamw_8bit", "")):
        directory = tmp_path / name
        directory.mkdir()
        config = _load(QUANTIZED, directory, "legacy_100", f"rl.optimizer.name={name}", f"rl.optimizer.impl={impl}")
        validate_joint_ppo_config(config, 4)
        recipe = instantiate(config.rl)
        assert (recipe.optimizer.name, recipe.optimizer.impl, recipe.optimizer.update_policy) == (
            name,
            impl,
            "fp32_master",
        )


def test_libero10_quantized_recipe_rejects_spatial_checkpoint_and_geometry(tmp_path):
    spatial = tmp_path / "other" / "libero_spatial"
    spatial.mkdir(parents=True)
    config = _load(QUANTIZED, tmp_path, "legacy_100", checkpoint=spatial)
    with pytest.raises(ValueError, match="libero_10"):
        validate_joint_ppo_config(config, 4)
    (tmp_path / "geometry").mkdir()
    config = _load(QUANTIZED, tmp_path / "geometry", "legacy_100", "model.execution_horizon=16")
    with pytest.raises(ValueError, match="action geometry"):
        validate_joint_ppo_config(config, 4)


def test_spatial_quantized_recipe_is_unchanged():
    from cosmos_framework.utils.config import load_config

    config = load_config(SPATIAL, [])
    assert config.environment.config.suite == "libero_spatial"
    assert config.environment.config.eval_profile == "legacy_100"
    assert config.model.execution_horizon == 16
    assert config.job.name == "gr00t_n1d7_libero_quantized_optimizer"


def test_libero10_native_profile_keeps_native_window_minibatch_and_hyperparameters(tmp_path):
    from cosmos_framework.posttrain.config.rl import JointPPOConfig
    from cosmos_framework.utils.lazy_config import instantiate

    native = "cosmos_framework/posttrain/configs/gr00t_n1d7_libero10_quantized_optimizer_native.py"
    config = _load(native, tmp_path, "aligned_200", "rl.optimizer.name=adamw", "rl.optimizer.impl=fused")
    validate_joint_ppo_config(config, 8)
    recipe, default = instantiate(config.rl), JointPPOConfig()
    assert (recipe.num_envs, recipe.window_decisions, recipe.global_minibatch, recipe.microbatch) == (
        default.num_envs,
        default.window_decisions,
        default.global_minibatch,
        default.microbatch,
    ) == (8, 16, 32, 2)
    assert recipe.decisions // recipe.global_minibatch == 4
    assert recipe.optimizer.update_policy == "fp32_master" and recipe.optimizer.impl == "fused"
    for name in ("lr", "betas", "eps", "weight_decay", "grad_clip_norm"):
        assert getattr(recipe.optimizer, name) == getattr(default.optimizer, name)
    assert recipe.value_lr == default.value_lr


def test_libero10_native_value_feature_isolation_override(tmp_path):
    native = "cosmos_framework/posttrain/configs/gr00t_n1d7_libero10_quantized_optimizer_native.py"
    config = _load(native, tmp_path, "aligned_200", "model.value_feature_gradient=false")
    validate_joint_ppo_config(config, 8)
    assert config.model.value_feature_gradient is False
