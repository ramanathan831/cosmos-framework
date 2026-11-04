"""Lock more questions per eligible recording without inspecting task outcomes.

Retain every previous final question; add three metadata-hash selected questions
per group. Historical exposure remains explicit. The two earlier 16-row summaries
remain separate secondary comparisons, not overwritten by the expanded manifest.
"""

import argparse
import json
from pathlib import Path

from tools.quantized_optimizers.recording_split import components, digest, row_component


def main():
    p = argparse.ArgumentParser()
    for name in ("inventory", "full_eval", "development_manifest", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--previous-selection", action="append", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("new evaluation output required")
    inventory = components([json.loads(line) for line in args.inventory.read_text().splitlines()])
    development = json.loads(args.development_manifest.read_text())
    excluded = set(development["connected_groups"]["train"] + development["connected_groups"]["development"])
    previous = [json.loads(p.read_text()) for p in args.previous_selection]
    fixed_ids = {row["id"] for manifest in previous for row in manifest["rows"]}
    groups = {}
    for line in args.full_eval.read_text().splitlines():
        row = json.loads(line)
        group = row_component(row, inventory)["group"]
        if group not in excluded:
            groups.setdefault(group, []).append(row)
    chosen = []
    for group in sorted(groups):
        fixed = [row for row in groups[group] if row["id"] in fixed_ids]
        extra = sorted(
            (row for row in groups[group] if row["id"] not in fixed_ids),
            key=lambda row: digest(("r3-final-2026:" + row["id"]).encode()),
        )
        if len(fixed) > 4 or len(fixed) + len(extra) < 4:
            raise ValueError("each recording must support exactly four selected questions")
        chosen.extend(fixed + extra[: 4 - len(fixed)])
    if not fixed_ids <= {row["id"] for row in chosen}:
        raise ValueError("a historical final row was lost")
    args.output.mkdir(parents=True)
    blob = "".join(json.dumps(row) + "\n" for row in chosen).encode()
    (args.output / "validation.jsonl").write_bytes(blob)
    manifest = {
        "purpose": "Authorized reused-pool comparative benchmark, historically exposed; no independent replication claim",
        "selection": "All32 recording groups outside repaired train/development; four questions per group, retaining both previous fixed selections and choosing remaining rows by SHA256(r3-final-2026:id). No score consulted.",
        "output_sha256": digest(blob),
        "selected_groups": sorted(groups),
        "rows": [{"id": row["id"], **row_component(row, inventory)} for row in chosen],
        "historical_selections": [
            {"sha256": digest(p.read_bytes()), "ids": [row["id"] for row in old["rows"]]}
            for p, old in zip(args.previous_selection, previous, strict=True)
        ],
        "input_sha256": {
            str(p): digest(p.read_bytes()) for p in [args.inventory, args.full_eval, args.development_manifest]
        },
        "statistical_units": "Paired seeds and recording groups; four questions do not count as four independent recordings.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"groups": len(groups), "questions": len(chosen), "historical_rows_retained": len(fixed_ids)}))


if __name__ == "__main__":
    main()
