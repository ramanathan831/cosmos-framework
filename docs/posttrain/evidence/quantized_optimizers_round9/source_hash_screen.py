"""Three counterbalanced source-identity pairs; no decoding or cold pipeline claim."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def child(args):
    from cosmos_framework.posttrain.data.datasets.sft import JSONLSFTDataset
    from cosmos_framework.posttrain.data.packer.prepared_sft import target_from_row, target_key
    from cosmos_framework.posttrain.utils.identity import canonical_json_sha256

    root = Path(os.environ["SFT_DATA_ROOT"])
    start = time.monotonic()
    identities = []
    phases = []
    for name, count in (("train-mixed.jsonl", 346), ("validation.jsonl", 56), ("dev-task.jsonl", 32)):
        phase = time.monotonic()
        dataset = JSONLSFTDataset(str(root / name), media_hash_workers=args.workers)
        phases.append(time.monotonic() - phase)
        assert len(dataset) == count
        identities.append(
            {
                "name": name,
                "rows": len(dataset),
                "source_sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                "fingerprint": dataset.cursor_fingerprint(),
                "normalized_rows_sha256": canonical_json_sha256(dataset.rows, context="screen rows"),
                "ordered_media": list(dataset.media_hashes.items()),
                "target_keys": [target_key(target_from_row(dataset[i])) for i in range(len(dataset))],
                "bytes_hashed": sum(Path(p).stat().st_size for p in dataset.media_hashes),
            }
        )
    result = {
        "complete": True,
        "workers": args.workers,
        "identity_seconds": sum(phases),
        "phase_seconds": phases,
        "inner_seconds": time.monotonic() - start,
        "identities": identities,
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def parent(args):
    import psutil

    if args.output.exists():
        raise ValueError("fresh output directory required")
    args.output.mkdir(parents=True)
    reference = None
    for pair, order in enumerate(((1, 4), (4, 1), (1, 4))):
        for workers in order:
            name = f"pair{pair}-workers{workers}"
            output = args.output / f"{name}.json"
            command = [sys.executable, __file__, "--workers", str(workers), "--output", str(output)]
            start = time.monotonic()
            peak = {"rss_bytes": 0, "pss_bytes": 0, "processes": 0, "samples": 0}
            with (args.output / f"{name}.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                tracked = psutil.Process(process.pid)
                while process.poll() is None:
                    rss = pss = 0
                    try:
                        tree = [tracked, *tracked.children(recursive=True)]
                    except psutil.NoSuchProcess:
                        tree = []
                    for member in tree:
                        try:
                            memory = member.memory_full_info()
                            rss += memory.rss
                            pss += memory.pss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    peak["rss_bytes"] = max(peak["rss_bytes"], rss)
                    peak["pss_bytes"] = max(peak["pss_bytes"], pss)
                    peak["processes"] = max(peak["processes"], len(tree))
                    peak["samples"] += 1
                    time.sleep(0.1)
            elapsed = time.monotonic() - start
            receipt = {
                "returncode": process.returncode,
                "elapsed_seconds": elapsed,
                "peak_process_tree_memory": peak,
                "command": command,
            }
            (args.output / f"{name}.process.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if process.returncode:
                raise RuntimeError(f"{name} failed: {process.returncode}")
            result = json.loads(output.read_text())
            if reference is None:
                reference = result["identities"]
            if result["identities"] != reference:
                raise ValueError("source rows, target identities or complete media hashes differ")
    (args.output / "complete.txt").write_text("complete\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, choices=[1, 4])
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only screen")
    child(options) if options.workers else parent(options)
