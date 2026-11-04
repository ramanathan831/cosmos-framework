"""Ground-truth multi-question WTS development rows; never an untouched test set.

Reuse only recording groups already present in the prior train/validation split.
Decode existing option letters into their annotated answer text, without generating
rationales or repeating questions in targets. Full-video questions avoid ambiguous
phase alignment. Conflicting duplicate labels fail closed.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from tools.quantized_optimizers.recording_split import components, row_component


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--validation-raw", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--media-inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace a previous development dataset")
    inventory = components([json.loads(line) for line in args.media_inventory.read_text().splitlines()])
    groups, connected = {}, {}
    for split in ("train", "validation"):
        groups[split] = set()
        connected[split] = set()
        for line in (args.prior / (split + ".jsonl")).read_text().splitlines():
            connected[split].add(row_component(json.loads(line), inventory)["group"])
            for m in json.loads(line)["messages"]:
                if isinstance(m["content"], list):
                    for c in m["content"]:
                        if c.get("type") == "video":
                            groups[split].add(c["video"].split("/videos/", 1)[1].split("/")[0])
    if connected["train"] & connected["validation"]:
        raise ValueError("prior recording/video components overlap; use recording_split.py to remove training overlap")
    selected = defaultdict(dict)
    roots = {"train": args.raw, "validation": args.validation_raw}
    for split, raw in roots.items():
        for row in json.loads((raw / "annotations.json").read_text()):
            video = row["video"]
            if video.split("/")[0] not in groups[split] or row["phase"] != "full_video":
                continue
            lookup = {"messages": [{"content": [{"type": "video", "video": str(raw / "videos" / video)}]}]}
            if row_component(lookup, inventory)["group"] not in connected[split]:
                continue  # normal_trimmed contains multiple independent recordings
            question = row["conversations"][0]["value"].replace("\\n", "\n").replace("<video>", "").strip()
            matches = list(re.finditer(r"(?:^|\n)\s*([A-Z]):\s*", question))
            options = {
                m.group(1): question[
                    m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(question)
                ].strip()
                for i, m in enumerate(matches)
            }
            answer = options[row["conversations"][1]["value"].strip()]
            question = question[: matches[0].start()].strip()
            if question in selected[(split, video)] and selected[(split, video)][question] != answer:
                raise ValueError(f"conflicting full-video labels for {video}")
            selected[(split, video)][question] = answer
    manifest = {
        "purpose": "reused development groups, NOT final evaluation",
        "source_sha256": {s: digest(r / "annotations.json") for s, r in roots.items()},
        "prior_sha256": {s: digest(args.prior / (s + ".jsonl")) for s in groups},
        "groups": {s: sorted(g) for s, g in groups.items()},
        "connected_groups": {s: sorted(g) for s, g in connected.items()},
        "media_inventory_sha256": digest(args.media_inventory),
        "rows": {},
    }
    rendered = {}
    for split, allowed in groups.items():
        result = []
        for (source_split, video), qa in sorted(selected.items()):
            if source_split != split:
                continue
            if video.split("/")[0] not in allowed or len(qa) < 8:
                continue
            qa = sorted(qa.items())[:24]
            result.append(
                {
                    "id": video,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Answer each numbered question about the video with a numbered answer. Use the answer text, not option letters.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": str(roots[split] / "videos" / video)},
                                {"type": "text", "text": "\n".join(f"{i + 1}. {q}" for i, (q, a) in enumerate(qa))},
                            ],
                        },
                        {"role": "assistant", "content": "\n".join(f"{i + 1}. {a}" for i, (q, a) in enumerate(qa))},
                    ],
                }
            )
        if not result:
            raise ValueError(f"no substantive rows in {split}")
        rendered[split] = "".join(json.dumps(r) + "\n" for r in result)
        manifest["rows"][split] = {
            "count": len(result),
            "sha256": hashlib.sha256(rendered[split].encode()).hexdigest(),
            "answer_words": [len(r["messages"][-1]["content"].split()) for r in result],
        }
    args.output.mkdir(parents=True)
    for split, content in rendered.items():
        (args.output / (split + ".jsonl")).write_text(content)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
