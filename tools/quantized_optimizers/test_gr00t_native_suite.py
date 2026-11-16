"""Suite selection in the native GR00T launcher: Spatial is pinned, LIBERO-10 binds its own sources."""

import json
import sys
from types import SimpleNamespace

import pytest

from tools.quantized_optimizers.gr00t_native import SUITE_CONFIGS, model_directory, native_command


def _args(**changes):
    values = dict(
        suite="libero_spatial",
        eval_profile="legacy_100",
        model_dir="",
        optimizer="adamw_bnb_8bit",
        resume="",
        updates=8,
        seed=17,
        checkpoint_every=4,
        validation_every=4,
        evaluate_only=False,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_spatial_command_is_unchanged(tmp_path):
    assets = tmp_path / "assets"
    model = model_directory(_args(), assets)
    command, overrides = native_command(_args(), model, tmp_path / "manifests")
    # Exact pre-change round-ten command for the same inputs.
    assert command == [
        sys.executable,
        "-m",
        "cosmos_framework.posttrain.entrypoints.joint_ppo",
        "--config",
        "cosmos_framework/posttrain/configs/gr00t_n1d7_libero_quantized_optimizer.py",
    ]
    assert overrides == [
        f"model.checkpoint_path={assets / 'model'}",
        "rl.max_steps=8",
        "rl.save_every=4",
        "rl.keep_optim_last=0",
        "rl.keep_model_last=0",
        "rl.validation.every_steps=4",
        "rl.seed=17",
        "rl.run_ts=integration",
        "rl.optimizer.name=adamw_bnb_8bit",
        "rl.optimizer.impl=",
        "job.name=gr00t-master-adamw_bnb_8bit",
    ]
    _, resumed = native_command(_args(optimizer="adamw", resume="/c/iter_000004"), model, tmp_path)
    assert resumed[-3:] == ["rl.optimizer.impl=fused", "job.name=gr00t-master-adamw", "rl.resume=/c/iter_000004"]


def test_spatial_rejects_model_override(tmp_path):
    with pytest.raises(ValueError, match="only for LIBERO-10"):
        model_directory(_args(model_dir=str(tmp_path)), tmp_path)


def _libero10(tmp_path):
    from cosmos_framework.posttrain.model.gr00t.build import CHECKPOINT_REVISION

    directory = tmp_path / CHECKPOINT_REVISION / "libero_10"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps({"action_horizon": 40, "max_action_dim": 132, "num_inference_timesteps": 4})
    )
    (directory / "processor_config.json").write_text(
        json.dumps(
            {"processor_kwargs": {"modality_configs": {"libero_sim": {"action": {"delta_indices": list(range(16))}}}}}
        )
    )
    return directory


def test_libero10_binds_pinned_checkpoint_profile_and_manifests(tmp_path):
    directory = _libero10(tmp_path)
    args = _args(suite="libero_10", eval_profile="aligned_200", model_dir=str(directory), optimizer="adamw")
    model = model_directory(args, tmp_path / "assets")
    assert model == directory
    manifests = tmp_path / "manifests"
    command, overrides = native_command(args, model, manifests)
    assert command[-1] == SUITE_CONFIGS["libero_10"]
    assert f"model.checkpoint_path={directory}" in overrides
    assert "job.name=gr00t-libero10-master-adamw" in overrides
    assert overrides[-3:] == [
        "environment.config.eval_profile=aligned_200",
        f"environment.config.train_manifest={manifests / 'libero_10-train.json'}",
        f"environment.config.eval_manifest={manifests / 'libero_10-eval.json'}",
    ]


def test_libero10_requires_pinned_snapshot_directory(tmp_path):
    with pytest.raises(ValueError, match="requires --model-dir"):
        model_directory(_args(suite="libero_10"), tmp_path)
    wrong = tmp_path / "other" / "libero_10"
    wrong.mkdir(parents=True)
    with pytest.raises(ValueError, match="pinned snapshot revision"):
        model_directory(_args(suite="libero_10", model_dir=str(wrong)), tmp_path)
    directory = _libero10(tmp_path)
    (directory / "config.json").write_text(
        json.dumps({"action_horizon": 16, "max_action_dim": 132, "num_inference_timesteps": 4})
    )
    with pytest.raises(ValueError, match="40x132"):
        model_directory(_args(suite="libero_10", model_dir=str(directory)), tmp_path)
