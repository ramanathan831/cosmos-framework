# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Runtime-only config attributes on experiment-loaded (struct OmegaConf) configs."""

from types import SimpleNamespace

import attrs
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

from cosmos_framework.inference.inference import set_runtime_config_attr


@attrs.define(slots=False)
class _ModelConfig:
    max_action_dim: int = 64


def test_struct_experiment_config_accepts_runtime_attr_without_unlocking():
    config = OmegaConf.structured(_ModelConfig)
    with pytest.raises(ConfigAttributeError):
        config._auto_camera_fps = 20  # the server failure this helper fixes
    set_runtime_config_attr(config, "_auto_camera_fps", 20)
    assert config._auto_camera_fps == 20
    with pytest.raises(ConfigAttributeError):
        config.other_new_key = 1  # struct mode is restored afterwards


def test_plain_config_object():
    config = SimpleNamespace()
    set_runtime_config_attr(config, "_auto_camera_fps", None)
    assert config._auto_camera_fps is None
