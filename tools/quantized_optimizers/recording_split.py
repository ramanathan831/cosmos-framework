"""Remove training overlap with fixed evaluation recording/video components.

Recording identity excludes scenario, trial and camera. Exact video SHA256 joins
otherwise distinct recording families. Evaluation bytes are preserved unchanged.
This repairs within-campaign separation; it does not erase prior inspection.
"""

import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def media_key(path):
    # Lustre fs11/fsw aliases refer to the same immutable source. Retain the
    # wts_train/wts_eval directory to avoid conflating different source files.
    return path.split("/data/", 1)[1]


def components(inventory):
    parent, owners, families = {}, {}, {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for row in inventory:
        match = re.fullmatch(r"([0-9]{8})_([0-9]+)_.+", row["id"])
        if not match:
            raise ValueError("unknown recording identity; reviewed grouping required")
        datetime.datetime.strptime(match[1], "%Y%m%d")
        family = match[1] + "_" + match[2]
        sha = row["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("exact media SHA256 required")
        find(family)
        if sha in owners:
            a, b = sorted((find(family), find(owners[sha])))
            parent[b] = a
        else:
            owners[sha] = family
        key = media_key(row["path"])
        if key in families and families[key] != (family, sha):
            raise ValueError("ambiguous media inventory identity")
        families[key] = (family, sha)
    members = {}
    for family in parent:
        members.setdefault(find(family), []).append(family)
    hashes = {k: digest(json.dumps(sorted(v), separators=(",", ":")).encode()) for k, v in members.items()}
    return {
        key: {"group": hashes[find(family)], "recording": family, "sha256": sha}
        for key, (family, sha) in families.items()
    }


def row_component(row, inventory):
    media = [
        c["video"]
        for m in row["messages"]
        if isinstance(m["content"], list)
        for c in m["content"]
        if c.get("type") == "video"
    ]
    if len(media) != 1:
        raise ValueError("this workload requires exactly one identified video per row")
    return inventory[media_key(media[0])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--prior-train", type=Path)
    parser.add_argument("--prior-validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite a prior split")
    inventory = components([json.loads(line) for line in args.inventory.read_text().splitlines()])
    train_lines = args.train.read_text().splitlines(keepends=True)
    train = [json.loads(line) for line in train_lines]
    validation = [json.loads(line) for line in args.validation.read_text().splitlines()]
    heldout = {row_component(row, inventory)["group"] for row in validation}
    allowed = None
    if args.prior_train:
        allowed = {
            row_component(json.loads(line), inventory)["group"] for line in args.prior_train.read_text().splitlines()
        }
    if args.prior_validation:
        prior_heldout = {
            row_component(json.loads(line), inventory)["group"]
            for line in args.prior_validation.read_text().splitlines()
        }
        if heldout - prior_heldout:
            raise ValueError("validation contains recordings outside the approved prior selection")
    retained, removed = [], []
    for line, row in zip(train_lines, train, strict=True):
        info = row_component(row, inventory)
        reason = (
            "overlaps_evaluation"
            if info["group"] in heldout
            else "outside_prior_train_groups"
            if allowed is not None and info["group"] not in allowed
            else None
        )
        if reason:
            removed.append({"id": row["id"], "reason": reason, **info})
        else:
            retained.append(line)
    if not retained or not validation:
        raise ValueError("group separation left an empty split")
    kept = [json.loads(line) for line in retained]
    groups = {"train": sorted({row_component(row, inventory)["group"] for row in kept}), "validation": sorted(heldout)}
    assert not set(groups["train"]) & set(groups["validation"])
    args.output.mkdir(parents=True)
    (args.output / "train.jsonl").write_text("".join(retained))
    (args.output / "validation.jsonl").write_bytes(args.validation.read_bytes())
    manifest = {
        "purpose": "User-approved reused data, recording-disjoint within campaign; prior inspection retained",
        "input_sha256": {
            "train": digest(args.train.read_bytes()),
            "validation": digest(args.validation.read_bytes()),
            "inventory": digest(args.inventory.read_bytes()),
        },
        "output_sha256": {s: digest((args.output / (s + ".jsonl")).read_bytes()) for s in ("train", "validation")},
        "rows": {"train_before": len(train), "train_after": len(kept), "validation": len(validation)},
        "removed_training_rows": removed,
        "connected_groups": groups,
        "overlap_after": [],
        "grouping": "date + recording identifier, joined by exact video SHA256 across the full inventory; trial/camera/scenario removed",
        "validation_bytes_unchanged": True,
    }
    manifest["prior_selection_sha256"] = {
        key: digest(path.read_bytes())
        for key, path in (("train", args.prior_train), ("validation", args.prior_validation))
        if path
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"rows": manifest["rows"], "removed": len(removed), "groups": {s: len(g) for s, g in groups.items()}}
        )
    )


if __name__ == "__main__":
    main()
