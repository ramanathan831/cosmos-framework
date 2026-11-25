# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native four-GPU GR00T Joint PPO with explicitly owned FP32 update masters.

Use rl.optimizer.name=paged_adamw_8bit for managed state, or
rl.optimizer.name=adamw rl.optimizer.impl=fused for the compatible FP32-state
control. The legacy public recipe retains its original optimizer/storage policy.
Task quality and time-to-quality require the separate model-specific protocol.
"""

import os
from dataclasses import replace

from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_public_4gpu import make_config as make_base_config


def make_config():
    config = make_base_config()
    config.rl.optimizer = replace(config.rl.optimizer, name="adamw_bnb_8bit", impl="", update_policy="fp32_master")
    # Verify the unchanged backbone for both the quantized arms and their matched
    # FP32-state control. Its hashing/copy cost belongs to the product clock.
    config.policy.verify_frozen_state = True
    config.model.backbone_from_checkpoint = True
    config.environment.config.sim_pythonpath = os.environ.get("POSTTRAIN_SIM_PYTHONPATH")
    config.job.name = "gr00t_n1d7_libero_quantized_optimizer"
    return config
