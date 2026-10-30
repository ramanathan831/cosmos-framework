# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepare measured native SFT runtime fixtures; synthetic extensions are not quality data.

All media remain real local files. Long examples preserve the original question
and answer, then add a reproducible textual ledger and a supervised rendering of
that supplied ledger. Counts are fitted with the real processor, never by padding
or truncating token IDs. The output manifest includes actual grids, tensor bytes,
labels and sequence lengths for every row, plus microbatch padding statistics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from worker_memory import file_hash, json_hash, tree_hash, write_json

PROTOCOL = "cosmos.native-qlora-runtime-workloads.v1"
VARIANTS = {
    "short_video8": {"frames": 8, "pixels_per_frame": 65_536, "long": False},
    "long4k_video8": {"frames": 8, "pixels_per_frame": 65_536, "long": True},
    "short_video16_large": {"frames": 16, "pixels_per_frame": 262_144, "long": False},
    "long4k_video16_large": {"frames": 16, "pixels_per_frame": 262_144, "long": True},
}


def processor_options(frames: int, pixels_per_frame: int) -> dict[str, Any]:
    if (
        type(frames) is not int
        or frames < 2
        or frames % 2
        or type(pixels_per_frame) is not int
        or pixels_per_frame < 4096
    ):
        raise ValueError("frames must be positive/even and per-frame budget at least4096")
    return {
        "images_kwargs": {"size": {"shortest_edge": 65_536, "longest_edge": 262_144}},
        "videos_kwargs": {
            "num_frames": frames,
            "fps": None,
            "size": {"shortest_edge": frames * 4096, "longest_edge": frames * pixels_per_frame},
        },
    }


def long_row(original: dict[str, Any], count: int, ordinal: int) -> dict[str, Any]:
    if type(count) is not int or count < 1:
        raise ValueError("ledger count must be positive")
    row = copy.deepcopy(original)
    messages = row["messages"]
    if [m["role"] for m in messages if m["role"] != "system"] != ["user", "assistant"]:
        raise ValueError("runtime ledger extension requires one user/assistant exchange")
    gold = messages[-1]["content"]
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError("runtime fixture requires an original text answer")
    records, sentences = [], []
    for index in range(count):
        identifier = ordinal * 1000 + index + 1
        lane, speed, delay = (index + ordinal) % 4 + 1, (index * 7 + ordinal) % 61, (index * 3 + ordinal) % 19
        records.append(f"{identifier},{lane},{speed},{delay}")
        sentences.append(
            f"Record {identifier}: the supplied lane is {lane}, speed is {speed} units, and delay is {delay} seconds."
        )
    instruction = (
        "This is an explicitly synthetic runtime exercise. Answer the original video question first. "
        "Then render every supplied ledger record in order as 'Record ID: the supplied lane is LANE, "
        "speed is SPEED units, and delay is DELAY seconds.' The ledger is synthetic text, not a claim "
        "about the video. Include all records; do not abbreviate. This additional format supersedes "
        "the earlier instruction to return only a letter.\nLedger columns: ID,LANE,SPEED,DELAY\n" + "\n".join(records)
    )
    user = next(message for message in messages if message["role"] == "user")
    if not isinstance(user["content"], list):
        raise ValueError("runtime fixture requires structured real-video user content")
    user["content"].append({"type": "text", "text": instruction})
    if messages[0]["role"] == "system":
        messages[0]["content"] = "Follow the user's complete task and reproduce supplied records accurately."
    messages[-1]["content"] = gold + "\n" + "\n".join(sentences)
    row["id"] = "runtime-" + hashlib.sha256(f"{original.get('id')}:{ordinal}:{count}".encode()).hexdigest()
    return row


def fit_long_row(
    original: dict[str, Any],
    ordinal: int,
    process: Callable,
    *,
    minimum: int = 3840,
    maximum: int = 4096,
    minimum_supervised: int = 1024,
) -> tuple[dict[str, Any], Any, int]:
    """Find a complete ledger fitting the measured interval; never trim a target or swallow decoder errors."""
    low, high, accepted = 1, 512, None
    while low <= high:
        count = (low + high) // 2
        row = long_row(original, count, ordinal)
        try:
            sample = process(row)
        except ValueError as error:
            if "SFT sequence length" in str(error) and "exceeds max_length=" in str(error):
                high = count - 1
                continue
            raise
        length = len(sample.input_ids)
        if length > maximum:
            high = count - 1
        else:
            accepted = (row, sample, count)
            low = count + 1
    if accepted is None:
        raise ValueError("media and one complete ledger record do not fit the sequence budget")
    row, sample, count = accepted
    supervised = sum(label != -100 for label in sample.labels[1:])
    if len(sample.input_ids) < minimum or supervised < minimum_supervised:
        raise ValueError("measured long fixture missed its sequence or supervised-token lower bound")
    return row, sample, count


def select_rows(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    from cosmos_framework.posttrain.data.datasets.sft import normalize_row

    if count < 1:
        raise ValueError("selection count must be positive")
    candidates, seen = [], set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = normalize_row(json.loads(line), path.parent, label=f"runtime source row {line_number}")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError("source rows require distinct nonempty string IDs")
            seen.add(identifier)
            parts = [
                part
                for message in row["messages"]
                if isinstance(message["content"], list)
                for part in message["content"]
                if part.get("type") in {"video", "image"}
            ]
            if len(parts) != 1 or parts[0]["type"] != "video":
                raise ValueError("runtime source must contain exactly one real video per row")
            priority = json_hash([seed, identifier])
            candidates.append((priority, identifier, line_number, row))
    if len(candidates) < count:
        raise ValueError("source cannot satisfy the requested fixed selection")
    selected = sorted(sorted(candidates)[:count], key=lambda item: item[2])
    return [item[-1] for item in selected]


def media_paths(rows: list[dict[str, Any]]) -> set[str]:
    return {
        part["video"]
        for row in rows
        for message in row["messages"]
        if isinstance(message["content"], list)
        for part in message["content"]
        if part.get("type") == "video"
    }


def measure_sample(sample: Any, packer: Any) -> dict[str, Any]:
    batch = packer.sft_collate_fn([sample], packer.sft_compute_max_len([sample]), -100)
    valid = (
        (batch["labels"][:, 1:] != -100)
        & batch["attention_mask"][:, 1:].bool()
        & batch["attention_mask"][:, :-1].bool()
    )
    grid = batch.get("video_grid_thw")
    if grid is None or grid.shape != (1, 3) or not batch["pixel_values_videos"].numel():
        raise ValueError("runtime fixture lost its actual video payload/grid")
    thw = [int(value) for value in grid[0]]
    return {
        "sequence_length": len(sample.input_ids),
        "supervised_tokens": int(valid.sum()),
        "video_grid_thw": thw,
        "processed_temporal_slots": thw[0] * 2,
        "processed_height": thw[1] * 16,
        "processed_width": thw[2] * 16,
        "merged_visual_tokens": thw[0] * thw[1] * thw[2] // 4,
        "sample_sha256": tree_hash(sample),
        "batch_sha256": tree_hash(batch),
        "tensors": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "logical_bytes": value.numel() * value.element_size(),
            }
            for key, value in batch.items()
        },
    }


def padding_summary(lengths: list[int], microbatch: int) -> dict[str, int]:
    if microbatch < 1 or len(lengths) % microbatch or any(length < 1 for length in lengths):
        raise ValueError("lengths must form complete positive-length microbatches")
    padded = sum(max(lengths[start : start + microbatch]) * microbatch for start in range(0, len(lengths), microbatch))
    return {
        "microbatch": microbatch,
        "unpadded_tokens": sum(lengths),
        "padded_tokens": padded,
        "padding_tokens": padded - sum(lengths),
    }


def prepare(args: argparse.Namespace) -> None:
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.data.packer.sft_preprocess import build_sft_preprocessing
    from cosmos_framework.posttrain.schema import SFTTarget

    output = Path(args.output).absolute()
    if output.exists():
        raise FileExistsError("runtime workload output must be a fresh directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = {"train": Path(args.source_train).resolve(), "validation": Path(args.source_validation).resolve()}
    source_hashes = {key: file_hash(path) for key, path in sources.items()}
    selected = {
        split: select_rows(path, args.rows if split == "train" else args.validation_rows, args.seed)
        for split, path in sources.items()
    }
    if {row["id"] for row in selected["train"]} & {row["id"] for row in selected["validation"]}:
        raise ValueError("selected training and validation IDs overlap")
    media = {path: file_hash(Path(path)) for rows in selected.values() for path in media_paths(rows)}
    if {media[path] for path in media_paths(selected["train"])} & {
        media[path] for path in media_paths(selected["validation"])
    }:
        raise ValueError("selected training and validation video contents overlap")
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL,
        "runtime_stress_only": True,
        "quality_evidence": False,
        "seed": args.seed,
        "source_sha256": source_hashes,
        "media_sha256": media,
        "build_receipt_sha256": file_hash(Path(args.build_receipt)),
        "script_sha256": file_hash(Path(__file__)),
        "variants": {},
    }
    with tempfile.TemporaryDirectory(prefix=".runtime-workloads-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for name, variant in VARIANTS.items():
            directory = staging / name
            directory.mkdir()
            options = processor_options(variant["frames"], variant["pixels_per_frame"])
            packer = build_sft_packer(
                args.model_path,
                max_length=4096,
                processor_kwargs=options,
                mask_think_opener=False,
                trust_remote_code=False,
            )
            cached = build_sft_preprocessing(
                packer,
                name_or_path=args.model_path,
                cache_bytes=2**30,
                prefetch=False,
                media_hashes=media,
                max_pending=8,
            )

            def process(row):
                return cached.sft_process_sample(
                    SFTTarget(
                        row["messages"],
                        tools=row.get("tools"),
                        supervise=row.get("supervise", "all_assistant"),
                        opener=row.get("opener", "self"),
                    )
                )

            reports, files = {}, {}
            try:
                for split, rows in selected.items():
                    records = []
                    with (directory / f"{split}.jsonl").open("x") as stream:
                        for ordinal, original in enumerate(rows):
                            if variant["long"]:
                                row, sample, ledger = fit_long_row(original, ordinal, process)
                            else:
                                row, ledger = copy.deepcopy(original), 0
                                sample = process(row)
                            measured = measure_sample(sample, packer)
                            records.append(
                                {
                                    "source_row_sha256": json_hash(original),
                                    "output_row_sha256": json_hash(row),
                                    "ledger_records": ledger,
                                    **measured,
                                }
                            )
                            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                    files[split] = file_hash(directory / f"{split}.jsonl")
                    reports[split] = {
                        "rows": records,
                        "padding": [padding_summary([row["sequence_length"] for row in records], b) for b in (1, 2, 4)],
                    }
                policy = {
                    "max_length": 4096,
                    "processor_kwargs": options,
                    "microbatches": [{"microbatch": b, "grad_accum": 8 // b} for b in (1, 2, 4)],
                    "device_process_budget_bytes": 24 * 1024**3,
                }
                details = {
                    "policy": policy,
                    "processor_identity_sha256": json_hash(packer.validation_semantics_identity()),
                    "files": files,
                    "measurements": reports,
                    "frame_indices_and_timestamps": "not preserved by native packer; temporal slots are not unique source frames",
                }
                write_json(directory / "manifest.json", details)
                manifest["variants"][name] = {"manifest_sha256": file_hash(directory / "manifest.json"), **policy}
            finally:
                cached.close()
        if any(file_hash(path) != source_hashes[split] for split, path in sources.items()):
            raise ValueError("source JSONL changed while preparing runtime workloads")
        if any(file_hash(Path(path)) != digest for path, digest in media.items()):
            raise ValueError("source media changed while preparing runtime workloads")
        write_json(staging / "manifest.json", manifest)
        os.rename(staging, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-train", "source-validation", "model-path", "build-receipt", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--validation-rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.rows, args.validation_rows) < 4 or args.rows % 4 or args.validation_rows % 4:
        parser.error("split sizes must be positive multiples of4")
    prepare(args)


if __name__ == "__main__":
    main()
