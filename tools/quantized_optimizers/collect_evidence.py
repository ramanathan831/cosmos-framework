"""Collect bounded textual evidence without copying model/optimizer checkpoints.

Run on the machine holding RESULTS, then copy the single compressed output.
Each input retains its relative name and content; receipt hashes detect changes.
Only read the campaign directory you own. No SSH configuration is inspected.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    files, receipt = {}, {}
    suffixes = {".json", ".jsonl", ".yaml", ".txt", ".log", ".csv", ".err", ".start", ".end", ".passed", ".xml"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes or path.resolve() == args.output.resolve():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"evidence symlink escapes campaign root: {path}")
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError(f"oversized evidence requires explicit review: {path}")
        value = path.read_bytes()
        name = str(path.relative_to(root))
        files[name] = value.decode("utf-8")
        receipt[name] = {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    files["_collection.json"] = json.dumps({"root": str(root), "files": receipt}, sort_keys=True)
    blob = gzip.compress(json.dumps(files, sort_keys=True).encode(), mtime=0)
    args.output.write_bytes(blob)
    print(
        json.dumps({"files": len(receipt), "compressed_bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
    )


if __name__ == "__main__":
    main()
