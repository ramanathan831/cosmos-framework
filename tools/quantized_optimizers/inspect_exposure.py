"""Reconstruct exact native cursor exposure and inspect the immutable CPU payloads."""

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.posttrain.data.datasets.sft import JSONLSFTDataset
from cosmos_framework.posttrain.data.packer.prepared_sft import target_from_row, target_key
from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = JSONLSFTDataset(str(args.data))
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if dataset.cursor_fingerprint() not in manifest["sources"]:
        raise ValueError("exposure source identity mismatch")
    cursor = IndexedCursor(dataset, shuffle=True, seed=args.seed, cursor_resume="strict", label="SFT source")
    rows = []
    for index in range(160):
        row_index = cursor.next_index()
        row = dataset[row_index]
        key = target_key(target_from_row(row))
        sample = torch.load(args.corpus / f"{key}.pt", map_location="cpu", weights_only=True, mmap=True)
        mm = sample["mm_kwargs"]
        rows.append(
            {
                "update": index // 4 + 1,
                "row_index": row_index,
                "id": row.get("id"),
                "entry": key,
                "sequence_length": len(sample["input_ids"]),
                "supervised_tokens": sum(x != -100 for x in sample["labels"][1:]),
                "media_shapes": {k: list(v.shape) for k, v in mm.items()},
                "video_grid_thw": mm["video_grid_thw"].tolist(),
            }
        )
    args.output.write_text(json.dumps({"rows": rows, "cursor": cursor.state_dict()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
