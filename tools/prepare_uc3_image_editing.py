#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Adapt the AnomalyGen UC3 archive to Cosmos3 paired image-editing SFT.

The source archive is unpaired.  This adapter preserves real defect pixels by
alpha-compositing each anomaly/mask exemplar onto clean images.  Clean-image
identities and defect exemplars are split before the cross product so the
validation set does not leak either source identity or defect mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Archive member escapes extraction root: {member.filename}")
        bundle.extractall(destination)


def _fit(image: Image.Image, size: tuple[int, int], *, mask: bool = False) -> Image.Image:
    return ImageOps.fit(
        image,
        size,
        method=Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _load_specs(dataset_root: Path) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    with (dataset_root / "defect_spec.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            spec = json.loads(line)
            defect_type = spec["defect_type"].split("+", 1)[-1]
            specs[defect_type] = spec
    return specs


def _discover_defects(dataset_root: Path, specs: dict[str, dict]) -> dict[str, list[tuple[Path, Path]]]:
    phone = dataset_root / "Phone"
    result: dict[str, list[tuple[Path, Path]]] = {}
    for defect_type in sorted(specs):
        anomaly_dir = phone / "anomaly_image" / defect_type
        mask_dir = phone / "mask" / defect_type
        pairs = []
        for anomaly_path in sorted(anomaly_dir.glob("*.png")):
            mask_path = mask_dir / f"{anomaly_path.stem}_mask.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"No mask for anomaly image: {anomaly_path}")
            pairs.append((anomaly_path, mask_path))
        if len(pairs) < 2:
            raise ValueError(f"Need at least two exemplars for {defect_type}, found {len(pairs)}")
        result[defect_type] = pairs
    return result


def _instruction(defect_type: str, spec: dict) -> str:
    roi = str(spec["roi_prompt_defect_location"]).rstrip(".")
    return (
        f"Add a realistic {defect_type} defect to {roi}. "
        "Preserve the phone, framing, lighting, and background outside the defect."
    )


def _make_target(
    clean_path: Path,
    anomaly_path: Path,
    mask_path: Path,
    output_path: Path,
    size: tuple[int, int],
    feather_radius: float,
) -> None:
    with (
        Image.open(clean_path) as clean_image,
        Image.open(anomaly_path) as anomaly_image,
        Image.open(mask_path) as mask_image,
    ):
        clean = _fit(clean_image.convert("RGB"), size)
        anomaly = _fit(anomaly_image.convert("RGB"), size)
        alpha = _fit(mask_image.convert("L"), size, mask=True)
        if feather_radius > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        Image.composite(anomaly, clean, alpha).save(output_path, format="PNG", optimize=True)


def prepare(archive: Path, output_dir: Path, width: int, height: int, feather_radius: float) -> dict:
    if not archive.is_file():
        raise FileNotFoundError(f"UC3 archive not found: {archive}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError(f"width and height must be positive multiples of 16, got {width}x{height}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = output_dir / "raw"
    with tempfile.TemporaryDirectory(prefix="uc3-extract-") as temp_dir:
        temporary_root = Path(temp_dir)
        _safe_extract(archive, temporary_root)
        candidates = sorted(temporary_root.glob("*/defect_spec.jsonl"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one UC3 dataset root, found {len(candidates)}")
        shutil.copytree(candidates[0].parent, raw_root / candidates[0].parent.name)

    dataset_root = raw_root / "UC3_data"
    specs = _load_specs(dataset_root)
    defects = _discover_defects(dataset_root, specs)
    clean_images = sorted((dataset_root / "Phone" / "clean_image").glob("*.png"))
    if len(clean_images) < 5:
        raise ValueError(f"Need at least five clean images, found {len(clean_images)}")

    # The provided UC3 archive has 20 clean images and five exemplars for each
    # of three defects.  Split identities first: 16/4 clean, 4/1 per defect.
    val_clean_count = max(1, len(clean_images) // 5)
    train_clean = clean_images[:-val_clean_count]
    val_clean = clean_images[-val_clean_count:]
    train_defects = {name: pairs[:-1] for name, pairs in defects.items()}
    val_defects = {name: [pairs[-1]] for name, pairs in defects.items()}

    target_root = output_dir / "targets"
    (target_root / "train").mkdir(parents=True)
    (target_root / "val").mkdir(parents=True)
    size = (width, height)

    manifests: dict[str, list[dict]] = {"train": [], "val": []}
    for split, clean_split, defect_split in (
        ("train", train_clean, train_defects),
        ("val", val_clean, val_defects),
    ):
        for clean_path in clean_split:
            for defect_type, exemplar_pairs in sorted(defect_split.items()):
                for anomaly_path, mask_path in exemplar_pairs:
                    sample_id = f"{split}-{clean_path.stem}-{defect_type}-{anomaly_path.stem}"
                    target_path = target_root / split / f"{sample_id}.png"
                    _make_target(
                        clean_path,
                        anomaly_path,
                        mask_path,
                        target_path,
                        size,
                        feather_radius,
                    )
                    manifests[split].append(
                        {
                            "id": sample_id,
                            "source": clean_path.relative_to(output_dir).as_posix(),
                            "target": target_path.relative_to(output_dir).as_posix(),
                            "instruction": _instruction(defect_type, specs[defect_type]),
                            "defect_type": f"Phone+{defect_type}",
                            "defect_image": anomaly_path.relative_to(output_dir).as_posix(),
                            "mask": mask_path.relative_to(output_dir).as_posix(),
                        }
                    )

    for split, records in manifests.items():
        _write_jsonl(output_dir / f"{split}.jsonl", records)

    # Fifteen fixed, held-out-clean inference cases: one for every real UC3
    # anomaly/mask exemplar.  Both Cosmos3 and AnomalyGen can consume this plan.
    comparison_records: list[dict] = []
    exemplar_index = 0
    for defect_type, exemplar_pairs in sorted(defects.items()):
        for anomaly_path, mask_path in exemplar_pairs:
            clean_path = val_clean[exemplar_index % len(val_clean)]
            comparison_records.append(
                {
                    "id": f"compare-{defect_type}-{anomaly_path.stem}",
                    "source": clean_path.relative_to(output_dir).as_posix(),
                    "instruction": _instruction(defect_type, specs[defect_type]),
                    "defect_type": f"Phone+{defect_type}",
                    "real_reference": anomaly_path.relative_to(output_dir).as_posix(),
                    "mask": mask_path.relative_to(output_dir).as_posix(),
                    "seed": 42 + exemplar_index,
                }
            )
            exemplar_index += 1
    _write_jsonl(output_dir / "comparison.jsonl", comparison_records)

    summary = {
        "source_archive": str(archive.resolve()),
        "source_archive_sha256": _sha256(archive),
        "resolution": {"width": width, "height": height},
        "feather_radius": feather_radius,
        "clean_split": {"train": len(train_clean), "val": len(val_clean)},
        "defect_exemplar_split": {
            "train": sum(len(items) for items in train_defects.values()),
            "val": sum(len(items) for items in val_defects.values()),
        },
        "samples": {
            "train": len(manifests["train"]),
            "val": len(manifests["val"]),
            "comparison": len(comparison_records),
        },
        "manifest_sha256": {name: _sha256(output_dir / f"{name}.jsonl") for name in ("train", "val", "comparison")},
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--feather-radius", type=float, default=2.0)
    args = parser.parse_args()
    summary = prepare(args.archive, args.output_dir, args.width, args.height, args.feather_radius)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
