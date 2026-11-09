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


def test_worker_timers_include_identity_and_preserve_saved_values(tmp_path, monkeypatch):
    from tools.quantized_optimizers import parallel_prepare as module

    clock = [0.0]

    class TimedPacker(Packer):
        def sft_process_sample(self, target):
            clock[0] += 2
            return super().sft_process_sample(target)

        def validation_semantics_identity(self):
            clock[0] += 3
            return super().validation_semantics_identity()

    monkeypatch.setattr(module, "_PACKER", TimedPacker(), raising=False)
    monkeypatch.setattr(module, "_DIRECTORY", tmp_path, raising=False)
    monkeypatch.setattr(module, "_DECODER_THREADS", None, raising=False)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    target = target_from_row({"messages": [{"role": "assistant", "content": "answer"}]})
    receipt = module.process_target(target)
    assert receipt["seconds"] == sum(receipt["phase_seconds"].values()) == 7
    assert receipt["phase_seconds"] == dict(
        first_sample=2, independent_replay=2, serialize_reload_compare=0, processor_identity=3
    )


@pytest.mark.parametrize("fail", [False, True])
def test_video_timers_preserve_identity_ownership_and_restore_inheritance(monkeypatch, fail):
    from tools.quantized_optimizers import parallel_prepare as module

    clock = [0.0]

    class BaseVideo:
        def _decode_and_sample_videos(self, value):
            clock[0] += 2
            if fail:
                raise ValueError("native decoder failure")
            return value

        def preprocess(self, value):
            result = self._decode_and_sample_videos(value)
            clock[0] += 1
            return result

    class Video(BaseVideo):
        pass

    processor, other, value = Video(), Video(), object()
    settings = dict(processor.__dict__)
    methods = dict(Video.__dict__)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    if fail:
        with pytest.raises(ValueError, match="native decoder failure"):
            with module.video_stage_times(processor):
                processor.preprocess(value)
    else:
        with module.video_stage_times(processor) as stages:
            assert processor.preprocess(value) is value
            assert other.preprocess(value) is value
            assert processor.__dict__ == settings
        assert stages == {
            "preprocess": {"calls": 1, "seconds": 3},
            "_decode_and_sample_videos": {"calls": 1, "seconds": 2},
        }
    assert dict(Video.__dict__) == methods
    assert processor.__dict__ == settings
