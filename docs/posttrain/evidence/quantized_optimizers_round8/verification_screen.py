"""Counterbalanced post-publication verification only, never a cold preparation."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from cosmos_framework.posttrain.data.datasets.sft import JSONLSFTDataset
from cosmos_framework.posttrain.data.packer.prepared_sft import PreparedSFTCorpus, target_from_row, target_key
from tools.quantized_optimizers.parallel_prepare import make_packer, process_memory
from tools.quantized_optimizers.prepare_corpus import same
from tools.quantized_optimizers.verify_published_corpus import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tensors", "bytes"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh result required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU only")
    torch.set_num_threads(8)
    begin = time.monotonic()
    with process_memory() as memory:
        config = "cosmos_framework/posttrain/configs/sft_optimizer_deployment.py"
        rl, packer = make_packer(config)
        root = Path(os.environ["SFT_DATA_ROOT"])
        paths = [
            rl.data.path,
            rl.validation.data.path,
            *[str(root / x) for x in ["train-mixed.jsonl", "dev-task.jsonl", "fixed-final.jsonl"]],
            os.environ["FINAL_DATA"],
        ]
        datasets = [JSONLSFTDataset(x) for x in paths]
        targets = {target_key(target_from_row(d[i])): target_from_row(d[i]) for d in datasets for i in range(len(d))}
        candidate = Path(os.environ["CANDIDATE_CORPUS"])
        reference = Path(os.environ["REFERENCE_CORPUS"])
        expected = "ea2e64893ab3d7da0f4c8be33001ae99c96dfd2c54fe293449c29255cd1f2d70"
        for directory in (candidate, reference):
            if hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest() != expected:
                raise ValueError("pinned full manifest mismatch")
        phase = time.monotonic()
        corpus = PreparedSFTCorpus(packer, candidate, datasets)
        oracle = PreparedSFTCorpus(packer, reference, datasets)
        binding = time.monotonic() - phase
        compare_start = time.monotonic()
        if args.mode == "tensors":
            for target in targets.values():
                if not same(asdict(corpus.process(target)), asdict(oracle.process(target))):
                    raise ValueError("native tensor comparison failed")
            proof = {"exact_payloads_per_corpus": len(targets)}
        else:
            proof = verify(candidate, reference, expected)
        comparison = time.monotonic() - compare_start
        result = {
            "complete": True,
            "mode": args.mode,
            "logical_rows": sum(map(len, datasets)),
            "targets": len(targets),
            "manifest_sha256": expected,
            "setup_seconds": phase - begin,
            "binding_seconds": binding,
            "comparison_seconds": comparison,
            "verification_seconds": time.monotonic() - phase,
            "total_inner_seconds": time.monotonic() - begin,
            "proof": proof,
            "peak_memory": dict(memory),
            "affinity": sorted(os.sched_getaffinity(0)),
            "scope": "Fresh verifier process on existing qualified full corpora; shared storage cache uncontrolled; no decode, no cold preparation.",
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
