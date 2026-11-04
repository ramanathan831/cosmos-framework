"""Build a repaired development comparison without selecting from task outcomes.

The substantive numbered-answer objective is retained. A separately named mixed
recipe adds one original multiple-choice example per training video. Final scoring
continues to use the previous fixed manifest and is never consulted for tuning.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from tools.quantized_optimizers.prepare_development import decode_annotation
from tools.quantized_optimizers.recording_split import components, row_component


def key(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def repair(rows, annotations):
    lookup = {}
    for row in annotations:
        if row["phase"] != "full_video":
            continue
        question, options, letter = decode_annotation(row)
        qa = lookup.setdefault(row["video"], {})
        if question in qa and qa[question] != options[letter]:
            raise ValueError("conflicting annotation answers")
        qa[question] = options[letter]
    result = []
    for row in rows:
        qa = sorted(lookup[row["id"]].items())[:24]
        expected = "\n".join(f"{i + 1}. {q}" for i, (q, _) in enumerate(qa))
        if row["messages"][1]["content"][1]["text"] != expected:
            raise ValueError("substantive question selection changed")
        fixed = copy.deepcopy(row)
        fixed["messages"][-1]["content"] = "\n".join(f"{i + 1}. {a}" for i, (_, a) in enumerate(qa))
        result.append(fixed)
    return result


def main():
    p = argparse.ArgumentParser()
    for name in (
        "old_train",
        "old_dev",
        "train_annotations",
        "eval_annotations",
        "full_eval",
        "inventory",
        "fixed_final",
        "output",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("new output directory required")
    inventory = components(read(args.inventory))
    old_train, old_dev = read(args.old_train), read(args.old_dev)
    train_annotations = json.loads(args.train_annotations.read_text())
    train = repair(old_train, train_annotations)
    dev = repair(old_dev, json.loads(args.eval_annotations.read_text()))
    train_groups = {row_component(row, inventory)["group"] for row in train}
    dev_groups = {row_component(row, inventory)["group"] for row in dev}
    if train_groups & dev_groups:
        raise ValueError("recording separation failed")
    by_video = {}
    for row in train_annotations:
        if row["phase"] == "full_video":
            by_video.setdefault(row["video"], []).append(row)
    mixed = []
    for row in train:
        annotation = min(by_video[row["id"]], key=lambda r: key("r3-mixed:" + r["id"]))
        _, _, letter = decode_annotation(annotation)
        question = annotation["conversations"][0]["value"].replace("\\n", "\n").replace("<video>", "").strip()
        footer = "Answer with the option's letter from the given choices directly."
        if not question.endswith(footer):
            question += "\n\n" + footer
        task = {
            "id": "mcq:" + annotation["id"],
            "messages": [
                {"role": "system", "content": "Answer the multiple-choice question with its letter."},
                {
                    "role": "user",
                    "content": [copy.deepcopy(row["messages"][1]["content"][0]), {"type": "text", "text": question}],
                },
                {"role": "assistant", "content": letter},
            ],
        }
        mixed.extend([row, task])
    eval_groups = {}
    for row in read(args.full_eval):
        group = row_component(row, inventory)["group"]
        eval_groups.setdefault(group, []).append(row)
    dev_task = []
    for group in sorted(dev_groups):
        dev_task.extend(sorted(eval_groups[group], key=lambda r: key("r3-dev:" + r["id"]))[:2])
    final = read(args.fixed_final)
    final_groups = {row_component(row, inventory)["group"] for row in final}
    if final_groups & (train_groups | dev_groups):
        raise ValueError("fixed final overlaps train/development")
    args.output.mkdir(parents=True)
    files = {
        "train.jsonl": train,
        "train-mixed.jsonl": mixed,
        "validation.jsonl": dev,
        "dev-task.jsonl": dev_task,
        "fixed-final.jsonl": final,
    }
    records = {}
    for name, rows in files.items():
        blob = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (args.output / name).write_bytes(blob)
        records[name] = {"rows": len(rows), "sha256": hashlib.sha256(blob).hexdigest()}
    selection = {
        "output_sha256": records["dev-task.jsonl"]["sha256"],
        "rows": [{"id": row["id"], **row_component(row, inventory)} for row in dev_task],
        "purpose": "Development only, two metadata-hash selected tasks per approved development recording group",
    }
    (args.output / "dev-task-manifest.json").write_text(json.dumps(selection, indent=2) + "\n")
    manifest = {
        "files": records,
        "source_sha256": {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in vars(args)
            if name != "output"
        },
        "repaired_target_rows": {
            "train": sum(a != b for a, b in zip(train, old_train, strict=True)),
            "development": sum(a != b for a, b in zip(dev, old_dev, strict=True)),
        },
        "connected_groups": {
            "train": sorted(train_groups),
            "development": sorted(dev_groups),
            "fixed_final": sorted(final_groups),
        },
        "eligible_eval_groups_outside_train_development": sorted(set(eval_groups) - train_groups - dev_groups),
        "historical_exposure": "Approved reused QLoRA pool; fixed final retains prior inspection and campaign scoring. No untouched claim.",
        "mixture": "Equal row counts substantive numbered answer-text and original single-question letter task; both optimizers share any selected recipe.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "files": records,
                "repaired_targets": manifest["repaired_target_rows"],
                "eligible_groups": len(manifest["eligible_eval_groups_outside_train_development"]),
            }
        )
    )


if __name__ == "__main__":
    main()
