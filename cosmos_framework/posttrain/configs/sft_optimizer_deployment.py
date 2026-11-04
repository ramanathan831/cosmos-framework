# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Matched 16-block deployment comparison; prepared inputs and ordinary FP32 trainables."""

import os

from cosmos_framework.posttrain.configs.sft_quantized_optimizer import make_config as base_config


def make_config():
    config = base_config()
    config.rl.optimizer.train_keys = [f"layers.{index}." for index in range(16, 32)]
    config.rl.prepared_corpus = os.environ.get("SFT_PREPARED_CORPUS", "")
    config.rl.lr_schedule.kind = "cosine"
    config.rl.lr_schedule.f_start = 0.5
    config.rl.lr_schedule.f_min = 0.1
    config.rl.lr_schedule.cycle_steps = 40
    config.rl.deterministic = True
    config.rl.save_every = 40
    config.job.group = "optimizer_deployment"
    return config
