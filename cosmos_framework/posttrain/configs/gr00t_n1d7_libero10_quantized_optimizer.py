# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native four-GPU GR00T-N1.7 / LIBERO-10 Joint PPO with explicitly owned FP32 update masters.

This composes the unchanged native LIBERO-10 bindings (suite protocol, manifests,
pinned ``libero_10`` checkpoint geometry) with the same four-GPU window/minibatch
profile and optimizer ownership policy as the LIBERO-Spatial quantized-optimizer
recipe. Select ``rl.optimizer.name=adamw rl.optimizer.impl=fused`` for the
compatible FP32-state control, or ``paged_adamw_8bit`` for managed state.

LIBERO-10's native training partition uses all fifty initial states per task, so
every native LIBERO-10 evaluation profile is a training-pool evaluation.
"""

import os
from dataclasses import replace

from cosmos_framework.posttrain.configs.gr00t_n1d7_libero10_ppo import make_config as make_libero10_config


def make_config():
    config = make_libero10_config()
    # Same four-GPU profile as gr00t_n1d7_libero_public_4gpu: one environment per
    # GPU and one optimizer update per sixteen-decision window.
    config.rl.num_envs = 4
    config.rl.window_decisions = 16
    config.rl.global_minibatch = 64
    config.rl.microbatch = 2
    config.rl.optimizer = replace(config.rl.optimizer, name="adamw_bnb_8bit", impl="", update_policy="fp32_master")
    config.policy.verify_frozen_state = True
    config.model.backbone_from_checkpoint = True
    config.environment.config.sim_pythonpath = os.environ.get("POSTTRAIN_SIM_PYTHONPATH")
    config.job.name = "gr00t_n1d7_libero10_quantized_optimizer"
    config.job.wandb_mode = "disabled"
    return config
