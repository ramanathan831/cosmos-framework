# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Construct the native PPO graph from metadata; the complete checkpoint owns weights.

Isaac 51d4c89f loads redundant base weights in its constructor. These subclasses
change construction only: all forward, action, masking and trainability methods
remain Isaac's. No global class monkeypatch or SFT training adapter is involved.
The caller must retain the strict missing/unexpected/mismatched-key checks.
"""


def checkpoint_model_class():
    # Optional model dependencies stay lazy for ordinary optimizer/SFT users.
    import torch
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
    from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
    from transformers import PreTrainedModel, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    class CheckpointBackbone(Qwen3Backbone):
        def __init__(self, config, directory):
            torch.nn.Module.__init__(self)
            metadata = Qwen3VLConfig.from_pretrained(directory, local_files_only=True)
            if not 1 <= config.select_layer <= metadata.text_config.num_hidden_layers:
                raise ValueError("GR00T select_layer is outside its exact backbone metadata")
            # No implicit attention fallback: this explicit path supports SDPA.
            if config.use_flash_attention:
                raise ValueError("checkpoint-owned GR00T construction requires use_flash_attention=False")
            metadata._attn_implementation = "sdpa"
            self.model = Qwen3VLForConditionalGeneration(metadata).eval()
            if config.load_bf16:
                self.model.to(dtype=torch.bfloat16)
            while len(self.model.language_model.layers) > config.select_layer:
                self.model.language_model.layers.pop(-1)
            self.select_layer = config.select_layer
            self.set_trainable_parameters(config.tune_llm, config.tune_visual, config.tune_top_llm_layers)
            if config.load_bf16 and config.backbone_trainable_params_fp32:
                for parameter in self.parameters():
                    if parameter.requires_grad:
                        parameter.data = parameter.data.float()
            self._reset_rotary_inv_freq()
            self._apply_vision_patch_embed_channels_last()

    class CheckpointGr00t(Gr00tN1d7):
        def __init__(self, config, *, backbone_config, transformers_loading_kwargs=None):
            PreTrainedModel.__init__(self, config)
            self.backbone = CheckpointBackbone(config, backbone_config)
            self.action_head = Gr00tN1d7ActionHead(config)
            self.collator = Gr00tN1d7DataCollator(
                model_name=backbone_config,
                model_type=config.backbone_model_type,
                transformers_loading_kwargs={"local_files_only": True},
            )

    return CheckpointGr00t
