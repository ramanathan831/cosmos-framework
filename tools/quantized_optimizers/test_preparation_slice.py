"""A development subset must not weaken the production corpus binding checks."""

import hashlib
import json

import pytest

from cosmos_framework.posttrain.data.packer.prepared_sft import (
    PreparedSFTCorpus,
    prepare_corpus,
    target_from_row,
    target_key,
)
from cosmos_framework.posttrain.tests.test_prepared_sft import Dataset, Packer
from tools.quantized_optimizers.parallel_prepare import verify_reference_slice


@pytest.mark.parametrize("damage", [None, "manifest", "payload", "media", "selection", "processor"])
def test_pinned_slice_reference(tmp_path, damage):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"media")
    dataset, packer = Dataset(media), Packer()
    root = tmp_path / "reference"
    manifest = prepare_corpus(packer, [dataset], root)
    corpus = PreparedSFTCorpus(packer, root, [dataset])
    target = target_from_row(dataset[0])
    key = target_key(target)
    plan = dict(
        reference=str(root),
        reference_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        targets=[dict(target=key)],
    )
    if damage == "manifest":
        (root / "manifest.json").write_text(json.dumps({**manifest, "extra": 1}))
    elif damage == "payload":
        payload = root / (key + ".pt")
        payload.write_bytes(payload.read_bytes() + b"damaged")
    elif damage == "media":
        media.write_bytes(b"other")
    elif damage == "selection":
        plan["targets"] = []
    elif damage == "processor":
        manifest = {**manifest, "packer": {}}
    if damage:
        with pytest.raises(ValueError):
            verify_reference_slice(plan, manifest, corpus, {key: target})
    else:
        verify_reference_slice(plan, manifest, corpus, {key: target})
