# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Two complete346-row passes for the separately locked development comparison.

Each epoch has86 four-row updates and one real two-row tail. This recipe is not
an extension of the historical15-update qualification endpoint. Runtime launch
preflight must bind the exact346-row manifest recorded in the study protocol.
"""

from cosmos_framework.posttrain.configs.sft_optimizer_task import make_config as task_config


def make_config():
    config = task_config()
    config.rl.data.epoch_boundary = "partial"
    config.rl.max_steps = 174
    config.rl.lr_schedule.cycle_steps = 174
    config.rl.validation.every_steps = 29
    config.rl.save_every = 87
    config.rl.keep_model_last = 0
    config.rl.keep_optim_last = 0
    return config
