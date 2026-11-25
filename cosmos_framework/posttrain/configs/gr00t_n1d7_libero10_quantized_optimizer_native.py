# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T-N1.7 / LIBERO-10 Joint PPO on its native profile with explicitly owned FP32 update masters.

Unlike ``gr00t_n1d7_libero10_quantized_optimizer`` (the reduced four-GPU
LIBERO-Spatial window/minibatch profile), this keeps every native LIBERO-10 /
Joint PPO default: eight environments (one per GPU on an eight-GPU node),
sixteen-decision windows, global minibatch 32 (four updates per 128-decision
window), native learning rates, betas, epsilon, decay and clipping. Only the
optimizer ownership policy and frozen-state verification are added.
"""

import os
from dataclasses import replace

from cosmos_framework.posttrain.configs.gr00t_n1d7_libero10_ppo import make_config as make_libero10_config


def make_config():
    config = make_libero10_config()
    config.rl.optimizer = replace(config.rl.optimizer, name="adamw_bnb_8bit", impl="", update_policy="fp32_master")
    config.policy.verify_frozen_state = True
    config.model.backbone_from_checkpoint = True
    config.environment.config.sim_pythonpath = os.environ.get("POSTTRAIN_SIM_PYTHONPATH")
    config.job.name = "gr00t_n1d7_libero10_quantized_optimizer_native"
    config.job.wandb_mode = "disabled"
    return config
