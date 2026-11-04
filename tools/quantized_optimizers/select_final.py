"""Select reused QLoRA evaluation rows outside campaign train/development groups.

Selection uses recording identities and a fixed hash seed, never task outcomes.
Prior inspection of the source evaluation pool remains part of its provenance.
"""

import argparse
import json
from pathlib import Path

from tools.quantized_optimizers.recording_split import components, digest, row_component


def select(lines, inventory, excluded, *, count=16, seed=2026):
    groups = {}
    for line in lines:
        row = json.loads(line)
        group = row_component(row, inventory)["group"]
        if group not in excluded:
            groups.setdefault(group, []).append((row["id"], line))
    if len(groups) < count:
        raise ValueError(f"need {count} eligible recording groups; found {len(groups)}")

    def rank(value):
        return digest(f"{seed}:{value}".encode())

    chosen = sorted(groups, key=rank)[:count]
    return [min(groups[g], key=lambda item: rank(item[0]))[1] for g in chosen], chosen, len(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-selection", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite a final selection")
    manifest = json.loads(args.campaign_manifest.read_text())
    excluded = set(manifest["connected_groups"]["train"] + manifest["connected_groups"]["validation"])
    prior_groups = sorted(
        {group for path in args.exclude_selection for group in json.loads(path.read_text())["selected_groups"]}
    )
    excluded.update(prior_groups)
    inventory = components([json.loads(line) for line in args.inventory.read_text().splitlines()])
    selected, groups, available = select(args.validation.read_text().splitlines(keepends=True), inventory, excluded)
    args.output.mkdir(parents=True)
    target = args.output / "validation.jsonl"
    target.write_text("".join(selected))
    output = {
        "purpose": "Final campaign scoring on user-approved reused QLoRA evaluation; NOT historically untouched",
        "selection": "16 recording components ranked by SHA256('2026:'+group), one row per group ranked by SHA256('2026:'+id); no outcomes used",
        "excluded": {**manifest["connected_groups"], **({"previous_final": prior_groups} if prior_groups else {})},
        "eligible_group_count": available,
        "selected_groups": groups,
        "rows": [{"id": json.loads(line)["id"], **row_component(json.loads(line), inventory)} for line in selected],
        "input_sha256": {
            "inventory": digest(args.inventory.read_bytes()),
            "full_qlora_validation": digest(args.validation.read_bytes()),
            "campaign_manifest": digest(args.campaign_manifest.read_bytes()),
            **{f"previous_final_{i}": digest(path.read_bytes()) for i, path in enumerate(args.exclude_selection)},
        },
        "output_sha256": digest(target.read_bytes()),
        "overlap_with_train_or_development": [],
        "final_selection_seed": 2026,
    }
    (args.output / "manifest.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"rows": len(selected), "groups": len(groups), "eligible_groups": available}))


if __name__ == "__main__":
    main()
