"""Complete-checkpoint construction is explicit and cannot accept missing frozen weights."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_model_config
from cosmos_framework.posttrain.model.gr00t.policy import _load_local_ppo_model

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_checkpoint_construction_changes_identity_without_changing_legacy():
    config = make_model_config("source")
    original = asdict(config)
    original.pop("backbone_buffer_dtype")
    original.pop("backbone_from_checkpoint")
    assert config.identity() == original
    assert replace(config, backbone_from_checkpoint=True).identity() != original


@pytest.mark.parametrize("bad_keys", [None, "missing_keys", "unexpected_keys", "mismatched_keys"])
def test_checkpoint_loader_preserves_strict_restore(monkeypatch, bad_keys):
    from cosmos_framework.posttrain.model.gr00t import materialize

    model = torch.nn.Linear(2, 2)
    model.action_head = SimpleNamespace(mask_token=None)
    calls = []

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return model, {bad_keys: ["backbone.model.weight"]} if bad_keys else {}

    monkeypatch.setattr(materialize, "checkpoint_model_class", lambda: Loader)
    config = SimpleNamespace(
        **dict.fromkeys(
            [
                "tune_llm",
                "tune_visual",
                "tune_projector",
                "tune_diffusion_model",
                "tune_vlln",
                "state_dropout_prob",
                "backbone_trainable_params_fp32",
                "load_bf16",
            ],
            False,
        )
    )
    pipeline = SimpleNamespace(
        config=SimpleNamespace(model=config, training=SimpleNamespace(start_from_checkpoint="full-source")),
        transformers_loading_kwargs={"local_files_only": True},
    )
    if bad_keys:
        with pytest.raises(RuntimeError, match="Checkpoint weight mismatch"):
            _load_local_ppo_model(pipeline, backbone_config="exact-metadata")
    else:
        assert _load_local_ppo_model(pipeline, backbone_config="exact-metadata") is model
    assert calls[0][0] == "full-source"
    assert calls[0][1]["backbone_config"] == "exact-metadata"
    assert calls[0][1]["use_flash_attention"] is False
    assert calls[0][1]["local_files_only"] is True
