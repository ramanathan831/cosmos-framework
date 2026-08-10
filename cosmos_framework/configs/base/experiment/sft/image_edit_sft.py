# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 generator-tower image-editing SFT recipes.

Nano and Edge update their full generator pathways; Super inherits the
generation-module LoRA policy from :mod:`vision_sft_super`.  All three share a
map-style paired-image dataloader and deterministic validation objective.
"""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.vision_sft_edge import vision_sft_edge
from cosmos_framework.configs.base.experiment.sft.vision_sft_nano import vision_sft_nano
from cosmos_framework.configs.base.experiment.sft.vision_sft_super import vision_sft_super
from cosmos_framework.data.generator.dataflow import (
    CosmosDataLoader,
    IdentityProcessor,
    MapDistributor,
    SequentialPackingBatcher,
    VFMListCollator,
)
from cosmos_framework.data.generator.local_datasets.paired_image_editing_dataset import (
    PairedImageEditingDataset,
)
from cosmos_framework.utils.lazy_config import LazyCall as L


def _paired_image_loader(split: str, *, shuffle: bool, cfg_dropout_rate: float, num_workers: int):
    return L(CosmosDataLoader)(
        distributor=L(MapDistributor)(
            dataset=L(PairedImageEditingDataset)(
                manifest_path=f"${{oc.env:DATASET_PATH}}/{split}.jsonl",
                tokenizer_config="${model.config.vlm_config.tokenizer}",
                width=848,
                height=480,
                cfg_dropout_rate=cfg_dropout_rate,
                max_caption_tokens=4096,
                dataset_name=f"image_edit_{split}",
            ),
            seed=42,
            shuffle=shuffle,
            name=split,
        ),
        processor=L(IdentityProcessor)(),
        # Two 848x480 image items already consume a meaningful amount of VAE
        # memory.  Keep a single pair per microbatch and scale globally with
        # data parallelism / gradient accumulation.
        batcher=L(SequentialPackingBatcher)(
            max_sequence_length=None,
            max_samples_per_batch=1,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            patch_spatial=2,
            sound_latent_fps=0,
            audio_sample_rate=48000,
        ),
        collator=L(VFMListCollator)(),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
    )


def _make_image_edit_recipe(base_recipe, model_name: str):
    recipe = copy.deepcopy(base_recipe)
    recipe.job.group = "generator_sft"
    recipe.job.name = f"image_edit_sft_{model_name}"
    recipe.trainer.run_validation = True
    recipe.trainer.run_validation_on_start = True
    recipe.trainer.max_val_iter = 3
    recipe.trainer.validation_iter = 100
    recipe.checkpoint.dcp_async_mode_enabled = False
    recipe.dataloader_train = _paired_image_loader(
        "train",
        shuffle=True,
        cfg_dropout_rate=0.1,
        num_workers=4,
    )
    recipe.dataloader_val = _paired_image_loader(
        "val",
        shuffle=False,
        cfg_dropout_rate=0.0,
        num_workers=2,
    )
    return recipe


image_edit_sft_nano = _make_image_edit_recipe(vision_sft_nano, "nano")
image_edit_sft_super = _make_image_edit_recipe(vision_sft_super, "super")
image_edit_sft_edge = _make_image_edit_recipe(vision_sft_edge, "edge")


cs = ConfigStore.instance()
for _name, _recipe in (
    ("image_edit_sft_nano", image_edit_sft_nano),
    ("image_edit_sft_super", image_edit_sft_super),
    ("image_edit_sft_edge", image_edit_sft_edge),
):
    cs.store(group="experiment", package="_global_", name=_name, node=_recipe)
