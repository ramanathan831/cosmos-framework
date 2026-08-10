# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Map-style paired-image dataset for Cosmos3 generator-tower SFT.

Each JSONL record describes one source/target editing pair::

    {"id": "sample-0001", "source": "images/source.png",
     "target": "images/target.png", "instruction": "Add a scratch."}

Relative image paths are resolved from the manifest directory.  The dataset
produces the same two-item vision layout as the interleaved image-editing
pipeline: the source is conditioned and the final target item is generated.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image, ImageOps

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.modalities import add_special_tokens
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate

_IMAGE_EDITING_SYSTEM_PROMPT = "You are a helpful assistant who will edit images based on the user's instructions."


class PairedImageEditingDataset(torch.utils.data.Dataset):
    """Load aligned source/target image-editing examples from a JSONL manifest."""

    def __init__(
        self,
        manifest_path: str,
        tokenizer_config: Any,
        width: int = 848,
        height: int = 480,
        cfg_dropout_rate: float = 0.0,
        max_caption_tokens: int = 4096,
        dataset_name: str = "paired_image_editing",
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Paired-image manifest not found: {self.manifest_path}")
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError(f"width and height must be positive multiples of 16, got {width}x{height}")
        if not 0.0 <= cfg_dropout_rate <= 1.0:
            raise ValueError(f"cfg_dropout_rate must be in [0, 1], got {cfg_dropout_rate}")
        if max_caption_tokens <= 0:
            raise ValueError("max_caption_tokens must be positive")

        self.width = width
        self.height = height
        self.cfg_dropout_rate = cfg_dropout_rate
        self.max_caption_tokens = max_caption_tokens
        self.dataset_name = dataset_name
        self.tokenizer_config = tokenizer_config
        self._tokenizer = None

        records: list[dict[str, Any]] = []
        with self.manifest_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                missing = {key for key in ("source", "target", "instruction") if not record.get(key)}
                if missing:
                    raise ValueError(
                        f"{self.manifest_path}:{line_number}: missing required field(s): {sorted(missing)}"
                    )
                record["id"] = str(record.get("id") or f"line-{line_number:06d}")
                records.append(record)
        if not records:
            raise ValueError(f"Paired-image manifest is empty: {self.manifest_path}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image(self, path: str) -> Path:
        image_path = Path(path).expanduser()
        if not image_path.is_absolute():
            image_path = self.manifest_path.parent / image_path
        return image_path.resolve()

    def _load_image(self, path: str) -> Image.Image:
        image_path = self._resolve_image(path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Paired-image sample not found: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return ImageOps.fit(
                image,
                (self.width, self.height),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )

    def _get_tokenizer(self):
        if self._tokenizer is None:
            processor = lazy_instantiate(self.tokenizer_config)
            tokenizer, _ = add_special_tokens(processor.tokenizer)
            self._tokenizer = tokenizer
        return self._tokenizer

    @staticmethod
    def _normalize(image: Image.Image) -> torch.Tensor:
        tensor = transforms_F.to_tensor(image)
        return transforms_F.normalize(tensor, mean=[0.5] * 3, std=[0.5] * 3)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source = self._load_image(str(record["source"]))
        target = self._load_image(str(record["target"]))
        instruction = str(record["instruction"])
        if self.cfg_dropout_rate and random.random() < self.cfg_dropout_rate:
            instruction = ""

        text_ids = tokenize_caption(
            instruction,
            self._get_tokenizer(),
            system_prompt=_IMAGE_EDITING_SYSTEM_PROMPT,
        )[: self.max_caption_tokens]
        image_size = torch.tensor(
            [self.height, self.width, self.height, self.width],
            dtype=torch.float32,
        )
        return {
            "__key__": record["id"],
            "__url__": str(self.manifest_path),
            "images": [self._normalize(source), self._normalize(target)],
            "image_size": [image_size.clone(), image_size.clone()],
            "text_token_ids": torch.tensor(text_ids, dtype=torch.long),
            "ai_caption": instruction,
            "selected_caption_type": "editing_instruction",
            "fps": 30.0,
            "num_frames": 2,
            "dataset_name": self.dataset_name,
            "sequence_plan": SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=[],
            ),
        }
