# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Single-device SFT; ordinary FP32 trainables and BF16 autocast, no weight quantization.

SFT_MODEL_PATH and SFT_DATA_ROOT identify immutable local artifacts. Override
rl.optimizer.name=adamw rl.optimizer.impl=fused for the high-precision control,
or rl.optimizer.name=paged_adamw_8bit for the capacity mode. This recipe is not
quality/performance qualification evidence.
"""

import os

from cosmos_framework.posttrain.configs.sft_qwen3p8_27b_multimodal import build_config


def make_config():
    config = build_config()
    config.model_hf.name_or_path = os.environ.get("SFT_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
    config.model_hf.parallelism.dp_shard = 1
    config.rl.checkpoint_format = "consolidated"
    config.rl.optimizer.name = "adamw_bnb_8bit"
    config.job.group = "quantized_optimizer"
    return config
