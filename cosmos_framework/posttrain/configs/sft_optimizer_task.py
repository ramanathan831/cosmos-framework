# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared WTS adaptation recipe selected on development data.

train-mixed.jsonl retains every substantive multi-question target and adds one
original single-choice question per training video. ce_exponent=1 gives equal
per-row weight to the two objectives. This is an explicit shared recipe change,
not the optimizer-isolation workload. Fifteen updates follow the observed
development learning curve; the cosine schedule retains its 40-update horizon.

Uses ordinary FP32 trainables in decoder blocks16..31 and BF16 compute. For the
fused control override rl.optimizer.name=adamw rl.optimizer.impl=fused
rl.optimizer.bnb_min_8bit_size=4096 model_hf.grad_ckpt=true.
"""

import os
from pathlib import Path

from cosmos_framework.posttrain.configs.sft_optimizer_deployment import make_config as deployment_config


def make_config():
    config = deployment_config()
    config.rl.data.path = str(Path(os.environ.get("SFT_DATA_ROOT", "/data")) / "train-mixed.jsonl")
    config.rl.optimizer.lr = 1e-5
    config.rl.ce_exponent = 1.0
    config.rl.max_steps = 15
    config.rl.save_every = 15
    config.rl.keep_optim_last = 0
    config.model_hf.grad_ckpt = False
    config.job.group = "optimizer_task"
    return config
