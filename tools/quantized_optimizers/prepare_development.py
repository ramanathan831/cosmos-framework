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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--validation-raw", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace a previous development dataset")
    groups = {}
    for split in ("train", "validation"):
        groups[split] = set()
        for line in (args.prior / (split + ".jsonl")).read_text().splitlines():
            for m in json.loads(line)["messages"]:
                if isinstance(m["content"], list):
                    for c in m["content"]:
                        if c.get("type") == "video":
                            groups[split].add(c["video"].split("/videos/", 1)[1].split("/")[0])
    if groups["train"] & groups["validation"]:
        raise ValueError("prior recording groups overlap; cannot manufacture independence")
    selected = defaultdict(dict)
    roots = {"train": args.raw, "validation": args.validation_raw}
    for split, raw in roots.items():
        for row in json.loads((raw / "annotations.json").read_text()):
            video = row["video"]
            if video.split("/")[0] not in groups[split] or row["phase"] != "full_video":
                continue
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
    args.output.mkdir(parents=True)
    manifest = {
        "purpose": "reused development groups, NOT final evaluation",
        "source_sha256": {s: digest(r / "annotations.json") for s, r in roots.items()},
        "prior_sha256": {s: digest(args.prior / (s + ".jsonl")) for s in groups},
        "groups": {s: sorted(g) for s, g in groups.items()},
        "rows": {},
    }
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
        dest = args.output / (split + ".jsonl")
        dest.write_text("".join(json.dumps(r) + "\n" for r in result))
        manifest["rows"][split] = {
            "count": len(result),
            "sha256": digest(dest),
            "answer_words": [len(r["messages"][-1]["content"].split()) for r in result],
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
