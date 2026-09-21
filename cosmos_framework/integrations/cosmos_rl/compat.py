# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Framework-owned model compatibility setup for optional Cosmos-RL hooks."""

import os
from functools import wraps

_INSTALLED = False


def install_runtime_extensions():
    """Install owned extensions before a native worker constructs models/data.

    Patch methods on the original classes so existing registry entries and
    imports share the same implementation. Refuse unsupported native versions.
    No installed source files are edited by this runtime operation.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    from cosmos_rl.dispatcher.data.packer import hf_vlm_data_packer
    from cosmos_rl.policy.model import hf_models
    from cosmos_rl.policy.worker import sft_worker

    from cosmos_framework.integrations.cosmos_rl.feature_cache import HFModelMethods, _ValidationVideoFeatureCache
    from cosmos_framework.integrations.cosmos_rl.packer import HFVLMDataPackerMethods, qwen_vl_process_vision_info

    updates = [(hf_models.HFModel, HFModelMethods), (hf_vlm_data_packer.HFVLMDataPacker, HFVLMDataPackerMethods)]
    for target, implementation in updates:
        for name in vars(implementation):
            if name.startswith("__"):
                continue
            if not hasattr(target, name):
                raise RuntimeError(f"Unsupported Cosmos-RL source: missing {target.__name__}.{name}")
    for target, implementation in updates:
        for name, value in vars(implementation).items():
            if not name.startswith("__"):
                setattr(target, name, value)
    hf_models._ValidationVideoFeatureCache = _ValidationVideoFeatureCache
    hf_vlm_data_packer.qwen_vl_process_vision_info = qwen_vl_process_vision_info
    original_init = sft_worker.SFTDataset.__init__

    @wraps(original_init)
    def dataset_init(self, *args, **kwargs):
        batch_threads = int(os.environ.get("COSMOS_SFT_BATCH_THREADS", "1"))
        if batch_threads < 1:
            raise ValueError("COSMOS_SFT_BATCH_THREADS must be positive")
        original_init(self, *args, **kwargs)
        self.batch_threads = batch_threads

    sft_worker.SFTDataset.__init__ = dataset_init
    _INSTALLED = True


def configure_patch_embedding():
    """Install the algebraically equivalent Qwen3-VL projection when needed."""
    from cosmos_framework.model.generator.qwen3_vl_compat import (
        _linear_patch_embed_forward,
        should_use_linear_patch_embed,
    )

    if not should_use_linear_patch_embed("auto"):
        return False
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

    Qwen3VLVisionPatchEmbed.forward = _linear_patch_embed_forward
    return True
