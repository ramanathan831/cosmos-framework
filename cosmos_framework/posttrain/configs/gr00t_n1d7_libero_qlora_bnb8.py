# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Separate integration: native GR00T flow-BC LoRA with nonpaged eight-bit AdamW.

Base precision remains independently selected by backbone_adaptation.mode.
This recipe does not alter the established fused-AdamW family baseline.
"""

from cosmos_framework.posttrain.config.action_sft import ACTION_MASTER_OPTIMIZER_PROFILE
from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_qlora import make_config as base_config


def make_config():
    config = base_config()
    config.job.name = "gr00t_n1d7_libero_nf4_bnb8_flow"
    config.rl.optimizer_profile = ACTION_MASTER_OPTIMIZER_PROFILE
    config.rl.optimizer.name = "adamw_bnb_8bit"
    config.rl.optimizer.impl = ""
    config.rl.optimizer.update_policy = "fp32_master"
    return config
