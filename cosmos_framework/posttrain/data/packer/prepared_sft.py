# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Immutable CPU inputs produced by the native SFT packer, with unchanged collation.

Preparation is an explicit offline operation. A missing or incompatible entry is
an error, never a fallback that changes timing or stochastic preprocessing semantics.
The corpus owns no model, optimizer, cursor, worker or RNG state.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.posttrain.data.datasets.sft import sha256_file
from cosmos_framework.posttrain.data.packer.sft import SFTPolicyInput
from cosmos_framework.posttrain.schema import SFTTarget
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256

FORMAT = "cosmos.prepared-sft.v1"


def target_key(target: SFTTarget) -> str:
    return canonical_json_sha256(asdict(target), context="prepared SFT target")


def target_from_row(row: dict[str, Any]) -> SFTTarget:
    return SFTTarget(**{k: row[k] for k in ("messages", "tools", "supervise", "opener") if k in row})


def stamp(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _settings(packer: Any) -> str:
    tokenizer, processor = getattr(packer, "tokenizer", None), getattr(packer, "processor", None)
    return canonical_json_sha256(
        {
            "max_length": packer.max_length,
            "processor_kwargs": packer.processor_kwargs,
            "mask_think_opener": getattr(packer, "mask_think_opener", False),
            "tokenizer_template": getattr(tokenizer, "chat_template", None),
            "processor_template": getattr(processor, "chat_template", None),
            "media": {
                key: component.to_dict()
                for key in ("image_processor", "video_processor")
                if (component := getattr(processor, key, None)) is not None
            },
        },
        context="prepared SFT runtime settings",
    )


def deterministic_sample(packer: Any, target: SFTTarget) -> SFTPolicyInput:
    """Preserve CPU RNG and reject packers that consume it, rather than cache randomness."""
    py, np_state, pt = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        sample = packer.sft_process_sample(target)
        after = np.random.get_state()
        if (
            py != random.getstate()
            or np_state[0] != after[0]
            or not np.array_equal(np_state[1], after[1])
            or np_state[2:] != after[2:]
            or not torch.equal(pt, torch.get_rng_state())
        ):
            raise ValueError("prepared SFT requires deterministic preprocessing without RNG consumption")
        if not isinstance(sample, SFTPolicyInput):
            raise TypeError("prepared SFT requires native SFTPolicyInput samples")
        for tensor in (sample.mm_kwargs or {}).values():
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.requires_grad:
                raise ValueError("prepared SFT media must be detached CPU tensors")
        return sample
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(pt)


def prepare_corpus(packer: Any, datasets: list[Any], output: str | Path) -> dict[str, Any]:
    """Publish a manifest only after all entries and source-media checks succeed."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    media: dict[str, str] = {}
    for dataset in datasets:
        if not hasattr(dataset, "media_hashes"):
            raise ValueError("prepared SFT currently requires local JSONL sources with media hashes")
        for name, digest in dataset.media_hashes.items():
            if name in media and media[name] != digest:
                raise ValueError("conflicting media identities")
            media[name] = digest
    stamps = {name: stamp(Path(name)) for name in media}
    for name, digest in media.items():
        if sha256_file(name) != digest or stamp(Path(name)) != stamps[name]:
            raise ValueError("source media changed before corpus preparation")
    entries = {}
    identity = deepcopy(packer.validation_semantics_identity())
    for dataset in datasets:
        for index in range(len(dataset)):
            target = target_from_row(dataset[index])
            key = target_key(target)
            if key in entries:
                continue
            sample = deterministic_sample(packer, target)
            buffer = io.BytesIO()
            torch.save(asdict(sample), buffer)
            value = buffer.getvalue()
            (root / f"{key}.pt").write_bytes(value)
            entries[key] = {
                "sha256": hashlib.sha256(value).hexdigest(),
                "bytes": len(value),
                "sequence_length": len(sample.input_ids),
                "supervised_tokens": sum(label != -100 for label in sample.labels[1:]),
            }
    if identity != packer.validation_semantics_identity() or any(stamp(Path(n)) != s for n, s in stamps.items()):
        raise ValueError("source/processor changed during corpus preparation")
    manifest = {
        "format": FORMAT,
        "packer": identity,
        "media": media,
        "sources": sorted({d.cursor_fingerprint() for d in datasets}),
        "entries": entries,
    }
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    temporary.replace(root / "manifest.json")
    return manifest


class PreparedSFTCorpus:
    def __init__(
        self, packer: Any, directory: str | Path, datasets: list[Any], *, verification_workers: int = 4
    ) -> None:
        if type(verification_workers) is not int or not 1 <= verification_workers <= 4:
            raise ValueError("prepared source verification requires one to four workers")
        self.root, self.packer = Path(directory).resolve(), packer
        value = (self.root / "manifest.json").read_bytes()
        self.identity = hashlib.sha256(value).hexdigest()
        self.manifest = json.loads(value)
        if self.manifest.get("format") != FORMAT or canonical_json_sha256(
            self.manifest["packer"], context="prepared processor"
        ) != canonical_json_sha256(packer.validation_semantics_identity(), context="live processor"):
            raise ValueError("prepared SFT format/processor identity mismatch")
        self.settings = _settings(packer)
        media = {}
        for dataset in datasets:
            if dataset.cursor_fingerprint() not in self.manifest["sources"]:
                raise ValueError("prepared SFT source identity mismatch")
            for name, digest in getattr(dataset, "media_hashes", {}).items():
                if name in media and media[name] != digest:
                    raise ValueError("conflicting prepared SFT source media identities")
                media[name] = digest

        def verify_media(item):
            name, digest = item
            path, before = Path(name), stamp(Path(name))
            if self.manifest["media"].get(name) != digest or sha256_file(name) != digest or stamp(path) != before:
                raise ValueError("prepared SFT source media mismatch")
            return name, before

        # Each source still receives a complete byte hash. A stat-only cache
        # would miss same-size rewrites on filesystems with coarse timestamps.
        self.verification_workers = min(verification_workers, max(1, len(media)))
        if self.verification_workers == 1:
            self.media_stamps = dict(map(verify_media, media.items()))
        else:
            with ThreadPoolExecutor(max_workers=self.verification_workers) as pool:
                self.media_stamps = dict(pool.map(verify_media, media.items()))
        for name, before in self.media_stamps.items():
            if stamp(Path(name)) != before:
                raise ValueError("prepared SFT source media mismatch")
        self.entry_stamps = {}
        for key, entry in self.manifest["entries"].items():
            if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
                raise ValueError("invalid prepared SFT entry key")
            path = self.root / f"{key}.pt"
            if path.resolve().parent != self.root or path.stat().st_size != entry["bytes"]:
                raise ValueError("invalid prepared SFT entry path/size")
            self.entry_stamps[key] = stamp(path)
        for dataset in datasets:
            if any(target_key(target_from_row(dataset[i])) not in self.entry_stamps for i in range(len(dataset))):
                raise ValueError("prepared SFT corpus does not cover the requested source")
        self.hits = 0

    def process(self, target: SFTTarget) -> SFTPolicyInput:
        if _settings(self.packer) != self.settings:
            raise ValueError("prepared SFT processor settings changed")
        for message in target.messages:
            content = message.get("content")
            for part in content if isinstance(content, list) else ():
                if isinstance(part, dict) and part.get("type") in ("image", "video"):
                    name = part[part["type"]]
                    if (
                        name not in self.media_stamps
                        or stamp(Path(name)) != self.media_stamps[name]
                        or sha256_file(name) != self.manifest["media"][name]
                    ):
                        raise ValueError("prepared SFT media changed or was not bound")
        key = target_key(target)
        entry = self.manifest["entries"].get(key)
        if entry is None:
            raise ValueError("target is missing from prepared SFT corpus; no runtime fallback")
        path = self.root / f"{key}.pt"
        if stamp(path) != self.entry_stamps[key]:
            raise ValueError("prepared SFT entry changed")
        # Container/remote filesystems can retain the same stat timestamps for
        # same-size rewrites. A stat-only fast path cannot establish immutability.
        if sha256_file(str(path)) != entry["sha256"] or stamp(path) != self.entry_stamps[key]:
            raise ValueError("prepared SFT entry digest mismatch")
        # Each load has private mmap storage. Consumers cannot mutate future hits.
        sample = SFTPolicyInput(**torch.load(path, map_location="cpu", weights_only=True, mmap=True))
        if len(sample.input_ids) != entry["sequence_length"] or len(sample.labels) != len(sample.input_ids):
            raise ValueError("prepared SFT sample shape mismatch")
        self.hits += 1
        return sample


def attach_prepared_corpus(
    packer: Any, directory: str, datasets: list[Any], *, verification_workers: int = 4
) -> PreparedSFTCorpus:
    corpus = PreparedSFTCorpus(packer, directory, datasets, verification_workers=verification_workers)
    packer.sft_process_sample = corpus.process
    return corpus
