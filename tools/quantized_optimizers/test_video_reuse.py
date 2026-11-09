"""Ownership, identity and failure tests; real native parity is a separate CPU run."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.data.datasets.sft import sha256_file
from tools.quantized_optimizers.video_reuse import VideoReuse, retained_bytes


@dataclass
class Metadata:
    fps: float
    frames_indices: np.ndarray


class Qwen3VLVideoProcessor:
    def __init__(self):
        self.scale = 1
        self.calls = 0

    def to_dict(self):
        return dict(scale=self.scale)

    def __call__(self, videos, **kwargs):
        self.calls += 1
        return dict(
            pixel_values_videos=torch.arange(12).reshape(3, 4) * self.scale,
            video_grid_thw=torch.tensor([[1, 2, 3]]),
            video_metadata=[Metadata(25.0, np.array([0, 3, 6]))],
        )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda name: "5.8.1" if name == "transformers" else "pinned")
    path = tmp_path / "video.mp4"
    path.write_bytes(b"source")
    original = Qwen3VLVideoProcessor()
    cache = VideoReuse(original, path, sha256_file(str(path)), 64 * 1024)
    return path, original, cache


def test_caller_cannot_corrupt_reused_tensors_or_metadata(setup):
    path, original, cache = setup
    args = dict(videos=[str(path)], num_frames=3, return_tensors="pt")
    first = cache(**args)
    first["pixel_values_videos"].zero_()
    first.pop("video_metadata")
    second = cache(**args)
    assert second["pixel_values_videos"].sum() == 66
    second["video_metadata"][0].fps = 0
    second["video_metadata"][0].frames_indices.fill(9)
    third = cache(**args)
    assert third["video_metadata"][0].fps == 25
    assert third["video_metadata"][0].frames_indices.tolist() == [0, 3, 6]
    assert original.calls == 1
    assert cache.stats["hits"] == 2
    assert 0 < cache.stats["retained_peak_bytes"] <= cache.limit_bytes


def test_sampling_and_processor_settings_are_part_of_identity(setup):
    path, original, cache = setup
    cache(videos=[str(path)], num_frames=3)
    cache(videos=[str(path)], num_frames=4)
    original.scale = 2
    result = cache(videos=[str(path)], num_frames=4)
    assert result["pixel_values_videos"].sum() == 132
    assert original.calls == 3 and cache.stats["hits"] == 0


@pytest.mark.parametrize("same_stat", [False, True])
def test_source_mutation_rejected_even_on_hit(setup, monkeypatch, same_stat):
    path, original, cache = setup
    cache(videos=[str(path)])
    path.write_bytes(b"damage")
    if same_stat:
        monkeypatch.setattr("tools.quantized_optimizers.video_reuse.stamp", lambda _: cache.source_stamp)
    with pytest.raises(ValueError, match="source"):
        cache(videos=[str(path)])
    assert original.calls == 1


def test_native_failure_and_scope_restore_original_processor(setup):
    path, original, cache = setup
    owner = SimpleNamespace(video_processor=original)
    with pytest.raises(RuntimeError, match="native failure"):
        with cache.installed(owner):
            assert owner.video_processor.to_dict() == original.to_dict()
            raise RuntimeError("native failure")
    assert owner.video_processor is original
    assert "video_processor" not in original.__dict__


def test_oversized_output_is_explicitly_uncached(setup):
    path, original, cache = setup
    cache.limit_bytes = 1
    for _ in range(2):
        cache(videos=[str(path)])
    assert cache.stats["oversized"] == 2
    assert cache.value is None and cache.stats["retained_peak_bytes"] == 0
    assert original.calls == 2


def test_unknown_or_live_buffers_fail():
    with pytest.raises(ValueError, match="detached"):
        retained_bytes(torch.ones(3, requires_grad=True))
    with pytest.raises(ValueError, match="unsupported"):
        retained_bytes(object())
    with pytest.raises(ValueError, match="owned"):
        retained_bytes(np.arange(5)[1:])


def test_hit_cannot_bypass_independent_native_target_replay(setup, monkeypatch, tmp_path):
    from cosmos_framework.posttrain.data.packer.prepared_sft import target_from_row
    from cosmos_framework.posttrain.tests.test_prepared_sft import Packer
    from tools.quantized_optimizers import parallel_prepare as module

    path, original, cache = setup

    class VideoPacker(Packer):
        processor = SimpleNamespace(video_processor=original)

        def sft_process_sample(self, target):
            value = self.processor.video_processor(videos=[str(path)])
            sample = super().sft_process_sample(target)
            sample.mm_kwargs = {"pixels": value["pixel_values_videos"]}
            return sample

    monkeypatch.setattr(module, "_PACKER", VideoPacker(), raising=False)
    monkeypatch.setattr(module, "_DIRECTORY", tmp_path, raising=False)
    monkeypatch.setattr(module, "_DECODER_THREADS", None, raising=False)
    target = target_from_row({"messages": [{"role": "assistant", "content": "text"}]})
    module.process_target(target, cache)
    module.process_target(target, cache)
    assert original.calls == 3  # One reused first pass, two independent replays.
    assert cache.stats["hits"] == 1
    assert module._PACKER.processor.video_processor is original
