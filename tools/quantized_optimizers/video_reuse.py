"""Bounded, worker-local reuse of deterministic Qwen video intermediates.

Only the first preparation of a target may use this cache. The caller must replay
each complete target with the original processor. Text, labels and masks are never
cached here. Source hashes and processor settings are checked on every lookup.
"""

import importlib.metadata
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import is_dataclass
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.posttrain.data.datasets.sft import sha256_file
from cosmos_framework.posttrain.data.packer.prepared_sft import stamp
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256


def single_video(target):
    media = []
    for message in target.messages:
        content = message.get("content")
        for part in content if isinstance(content, list) else ():
            if part.get("type") in ("image", "video"):
                if part["type"] != "video" or not isinstance(part.get("video"), str):
                    raise ValueError("video reuse requires exactly one local video and no images")
                media.append(str(Path(part["video"]).resolve(strict=True)))
    if len(media) != 1:
        raise ValueError("video reuse requires exactly one local video and no images")
    return media[0]


def retained_bytes(value):
    """Count owned CPU storage plus Python containers; reject unknown buffers."""
    objects, storage = set(), set()

    def size(item):
        if id(item) in objects:
            return 0
        objects.add(id(item))
        overhead = sys.getsizeof(item)
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu" or item.requires_grad or item.layout != torch.strided:
                raise ValueError("video reuse requires detached strided CPU tensors")
            buffer = item.untyped_storage()
            key = buffer.data_ptr()
            extra = 0 if key in storage else buffer.nbytes()
            storage.add(key)
            return overhead + extra
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject or item.base is not None:
                raise ValueError("video reuse requires owned numeric NumPy arrays")
            return overhead  # NumPy includes owned storage in __sizeof__.
        if isinstance(item, Mapping):
            return overhead + sum(size(k) + size(v) for k, v in item.items())
        if isinstance(item, (list, tuple)):
            return overhead + sum(map(size, item))
        if is_dataclass(item) and not isinstance(item, type):
            return overhead + size(vars(item))
        if item is None or isinstance(item, (str, bytes, bool, int, float, np.generic)):
            return overhead
        raise ValueError(f"unsupported video cache value: {type(item).__name__}")

    return size(value)


class VideoReuse:
    def __init__(self, original, source, digest, limit_bytes):
        if type(limit_bytes) is not int or limit_bytes <= 0:
            raise ValueError("video reuse requires a positive byte limit")
        self.original = original
        self.source = Path(source).resolve(strict=True)
        self.digest = digest
        self.source_stamp = stamp(self.source)
        self.limit_bytes = limit_bytes
        self.versions = {name: importlib.metadata.version(name) for name in ("transformers", "torch", "torchcodec")}
        if self.versions["transformers"] != "5.8.1" or type(original).__name__ != "Qwen3VLVideoProcessor":
            raise ValueError("video reuse requires inspected Qwen3VLVideoProcessor from Transformers 5.8.1")
        self.key = self.value = None
        self.stats = dict(hits=0, misses=0, oversized=0, retained_peak_bytes=0, source_hashes=0)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def verify_source(self):
        if stamp(self.source) != self.source_stamp:
            raise ValueError("video reuse source mutated")
        digest = sha256_file(str(self.source))
        self.stats["source_hashes"] += 1
        if digest != self.digest or stamp(self.source) != self.source_stamp:
            raise ValueError("video reuse source digest mismatch")

    def __call__(self, videos=None, **kwargs):
        if (
            not isinstance(videos, (list, tuple))
            or len(videos) != 1
            or not isinstance(videos[0], str)
            or Path(videos[0]).resolve(strict=True) != self.source
        ):
            raise ValueError("video reuse received a different source or unsupported media layout")
        self.verify_source()
        key = canonical_json_sha256(
            dict(
                source=str(self.source),
                sha256=self.digest,
                stamp=self.source_stamp,
                videos=videos,
                kwargs=kwargs,
                processor=self.original.to_dict(),
                implementation=f"{type(self.original).__module__}.{type(self.original).__qualname__}",
                versions=self.versions,
            ),
            context="video reuse identity",
        )
        if key == self.key:
            self.stats["hits"] += 1
            # Qwen consumes metadata with pop() and may fill in fps. No caller may
            # mutate the retained copy or another target's buffers/metadata.
            return deepcopy(self.value)
        self.stats["misses"] += 1
        value = self.original(videos=videos, **kwargs)
        self.verify_source()
        size = retained_bytes(value)  # Reject unsupported buffer ownership before copying.
        self.key = self.value = None
        if size <= self.limit_bytes:
            snapshot = deepcopy(value)
            size = retained_bytes(snapshot)
            if size > self.limit_bytes:
                raise ValueError("video reuse copy exceeded its byte limit")
            self.key, self.value = key, snapshot
            self.stats["retained_peak_bytes"] = max(self.stats["retained_peak_bytes"], size)
        else:
            self.stats["oversized"] += 1
        return value

    @contextmanager
    def installed(self, processor):
        if processor.video_processor is not self.original:
            raise ValueError("video reuse processor ownership changed")
        processor.video_processor = self
        try:
            yield
        finally:
            processor.video_processor = self.original
