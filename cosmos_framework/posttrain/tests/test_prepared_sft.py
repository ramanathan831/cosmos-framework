# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepared inputs preserve values/ownership and reject stale source identities."""

import hashlib
import json
import random
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.config.sft import SFTConfig
from cosmos_framework.posttrain.data.packer.prepared_sft import (
    PreparedSFTCorpus,
    attach_prepared_corpus,
    prepare_corpus,
    target_from_row,
    target_key,
)
from cosmos_framework.posttrain.data.packer.sft import SFTPolicyInput

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class Packer:
    max_length = 4096
    processor_kwargs = {}
    tokenizer = SimpleNamespace(chat_template="fixed")

    def validation_semantics_identity(self):
        return {"max_length": self.max_length, "processor_kwargs": self.processor_kwargs}

    def sft_process_sample(self, target):
        return SFTPolicyInput(
            [2, 3, 4], [-100, 3, 4], {"pixel_values_videos": torch.arange(12, dtype=torch.float32).reshape(2, 6)}
        )


class Dataset:
    def __init__(self, media):
        self.media_hashes = {str(media): hashlib.sha256(media.read_bytes()).hexdigest()}
        self.row = {
            "messages": [
                {"role": "user", "content": [{"type": "video", "video": str(media)}]},
                {"role": "assistant", "content": "an answer"},
            ]
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.row

    def cursor_fingerprint(self):
        return "toy-source"


@pytest.fixture
def corpus(tmp_path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fixture media")
    data, packer, root = Dataset(media), Packer(), tmp_path / "prepared"
    prepare_corpus(packer, [data], root)
    return packer, data, root, media


def test_values_mutation_isolation_and_rng(corpus):
    packer, data, root, _ = corpus
    target = target_from_row(data[0])
    expected = packer.sft_process_sample(target)
    before = torch.get_rng_state().clone(), random.getstate()
    bound = attach_prepared_corpus(packer, str(root), [data])
    first = packer.sft_process_sample(target)
    first.input_ids[0] = 99
    first.mm_kwargs["pixel_values_videos"].zero_()
    second = packer.sft_process_sample(target)
    assert second.input_ids == expected.input_ids
    assert second.labels == expected.labels
    assert torch.equal(second.mm_kwargs["pixel_values_videos"], expected.mm_kwargs["pixel_values_videos"])
    reopened = PreparedSFTCorpus(Packer(), root, [data]).process(target)
    assert torch.equal(reopened.mm_kwargs["pixel_values_videos"], expected.mm_kwargs["pixel_values_videos"])
    assert torch.equal(torch.get_rng_state(), before[0]) and random.getstate() == before[1]
    assert bound.hits == 2


@pytest.mark.parametrize("change", ["media", "payload", "settings", "target"])
def test_bound_corpus_refuses_changes(corpus, change):
    packer, data, root, media = corpus
    target = target_from_row(data[0])
    bound = PreparedSFTCorpus(packer, root, [data])
    bound.process(target)
    if change == "media":
        media.write_bytes(b"mutated media")
    elif change == "payload":
        (root / f"{target_key(target)}.pt").write_bytes(b"broken payload")
    elif change == "settings":
        packer.max_length = 1234
    else:
        target.messages[-1]["content"] = "a different target"
    with pytest.raises(ValueError):
        bound.process(target)


def test_unverified_payload_digest_and_source_refused(corpus):
    packer, data, root, _ = corpus
    manifest = json.loads((root / "manifest.json").read_text())
    key = target_key(target_from_row(data[0]))
    manifest["entries"][key]["sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest mismatch"):
        PreparedSFTCorpus(packer, root, [data]).process(target_from_row(data[0]))
    manifest["sources"] = ["another-source"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source identity"):
        PreparedSFTCorpus(packer, root, [data])


def test_random_preprocessor_rejected_without_manifest_or_rng_change(tmp_path):
    class RandomPacker(Packer):
        def sft_process_sample(self, target):
            random.random()
            torch.rand(1)
            return super().sft_process_sample(target)

    media = tmp_path / "media"
    media.write_bytes(b"bytes")
    before = random.getstate(), torch.get_rng_state().clone()
    root = tmp_path / "prepared"
    with pytest.raises(ValueError, match="without RNG consumption"):
        prepare_corpus(RandomPacker(), [Dataset(media)], root)
    assert not (root / "manifest.json").exists()
    assert random.getstate() == before[0] and torch.equal(torch.get_rng_state(), before[1])


def test_prepared_path_typed_and_default_preserved():
    assert asdict(SFTConfig())["prepared_corpus"] == ""
    with pytest.raises(TypeError, match="path string"):
        SFTConfig(prepared_corpus=True)
