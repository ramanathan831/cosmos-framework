# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import torch
from PIL import Image

from cosmos_framework.data.generator.local_datasets import paired_image_editing_dataset as dataset_module


class _Tokenizer:
    special_tokens_map = {}

    def add_tokens(self, _tokens):
        return None

    def convert_tokens_to_ids(self, token):
        return {"<|vision_start|>": 10, "<|vision_end|>": 11}[token]

    def apply_chat_template(self, conversations, **_kwargs):
        assert conversations[0]["role"] == "system"
        assert conversations[-1]["content"] == "Add a scratch."
        return [1, 2, 3]


class _Processor:
    tokenizer = _Tokenizer()


def test_paired_image_editing_dataset(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (64, 32), color=(10, 20, 30)).save(source)
    Image.new("RGB", (64, 32), color=(30, 20, 10)).save(target)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "pair-1",
                "source": source.name,
                "target": target.name,
                "instruction": "Add a scratch.",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(dataset_module, "lazy_instantiate", lambda _config: _Processor())

    dataset = dataset_module.PairedImageEditingDataset(
        manifest_path=str(manifest),
        tokenizer_config={},
        width=64,
        height=32,
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["__key__"] == "pair-1"
    assert sample["text_token_ids"].tolist() == [1, 2, 3]
    assert len(sample["images"]) == 2
    assert all(tuple(image.shape) == (3, 32, 64) for image in sample["images"])
    assert all(image.dtype == torch.float32 for image in sample["images"])
    assert sample["sequence_plan"].condition_frame_indexes_vision == []


def test_paired_image_editing_dataset_rejects_bad_resolution(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{"source":"a.png","target":"b.png","instruction":"x"}\n')
    try:
        dataset_module.PairedImageEditingDataset(
            manifest_path=str(manifest),
            tokenizer_config={},
            width=63,
            height=32,
        )
    except ValueError as error:
        assert "multiples of 16" in str(error)
    else:
        raise AssertionError("invalid resolution was accepted")
