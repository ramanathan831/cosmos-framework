"""Build and exactly verify prepared inputs without loading weights or using CUDA."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from cosmos_framework.posttrain.data.datasets.sft import JSONLSFTDataset
from cosmos_framework.posttrain.data.packer.prepared_sft import (
    PreparedSFTCorpus,
    deterministic_sample,
    prepare_corpus,
    target_from_row,
    target_key,
)
from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
from cosmos_framework.posttrain.entrypoints.sft import sft_policy_source, validate_config_structure
from cosmos_framework.utils.config import load_config


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-data", action="append", default=[])
    parser.add_argument("--reuse-corpus", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(8)
    config = load_config(args.config, [])
    rl = validate_config_structure(config)
    model, trust = sft_policy_source(config)
    packer = build_sft_packer(
        model,
        max_length=rl.max_length,
        processor_kwargs=rl.processor_kwargs,
        trust_remote_code=trust,
        mask_think_opener=rl.mask_think_opener,
    )
    paths = [rl.data.path, rl.validation.data.path, *args.extra_data]
    datasets = [JSONLSFTDataset(path) for path in paths]
    reused = None
    native_process = packer.sft_process_sample
    if args.reuse_corpus:
        previous_receipt = json.loads((args.reuse_corpus / "preparation_receipt.json").read_text())
        if previous_receipt["cuda_initialized"] or previous_receipt["exact_raw_prepared_rows"] < 1:
            raise ValueError("reuse requires a completed native CPU parity receipt")
        reused = PreparedSFTCorpus(
            packer, args.reuse_corpus, [JSONLSFTDataset(path) for path in previous_receipt["source_paths"]]
        )

        def process(target):
            return (
                reused.process(target) if target_key(target) in reused.manifest["entries"] else native_process(target)
            )

        packer.sft_process_sample = process
    manifest = prepare_corpus(packer, datasets, args.output)
    packer.sft_process_sample = native_process
    preparation_seconds = time.perf_counter() - started
    corpus = PreparedSFTCorpus(packer, args.output, datasets)
    checked, reused_checked = 0, 0
    for dataset in datasets:
        for index in range(len(dataset)):
            target = target_from_row(dataset[index])
            if reused is not None and target_key(target) in reused.manifest["entries"]:
                expected = reused.process(target)
                reused_checked += 1
            else:
                expected = deterministic_sample(packer, target)
                checked += 1
            if not same(asdict(expected), asdict(corpus.process(target))):
                raise ValueError("raw/prepared CPU tensors differ")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU corpus preparation initialized CUDA")
    receipt = {
        "preparation_seconds": preparation_seconds,
        "total_seconds_with_independent_exact_replay": time.perf_counter() - started,
        "exact_raw_prepared_rows": checked,
        "verified_reused_rows": reused_checked,
        "total_verified_rows": checked + reused_checked,
        "reused_receipt_sha256": hashlib.sha256(
            (args.reuse_corpus / "preparation_receipt.json").read_bytes()
        ).hexdigest()
        if args.reuse_corpus
        else None,
        "corpus_sha256": corpus.identity,
        "payload_bytes": sum(v["bytes"] for v in manifest["entries"].values()),
        "cuda_initialized": False,
        "source_paths": paths,
    }
    (args.output / "preparation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
