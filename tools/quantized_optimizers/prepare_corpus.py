"""Build and exactly verify prepared inputs without loading weights or using CUDA."""

import argparse
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
    manifest = prepare_corpus(packer, datasets, args.output)
    preparation_seconds = time.perf_counter() - started
    corpus = PreparedSFTCorpus(packer, args.output, datasets)
    checked = 0
    for dataset in datasets:
        for index in range(len(dataset)):
            target = target_from_row(dataset[index])
            if not same(asdict(deterministic_sample(packer, target)), asdict(corpus.process(target))):
                raise ValueError("raw/prepared CPU tensors differ")
            checked += 1
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU corpus preparation initialized CUDA")
    receipt = {
        "preparation_seconds": preparation_seconds,
        "total_seconds_with_independent_exact_replay": time.perf_counter() - started,
        "exact_raw_prepared_rows": checked,
        "corpus_sha256": corpus.identity,
        "payload_bytes": sum(v["bytes"] for v in manifest["entries"].values()),
        "cuda_initialized": False,
        "source_paths": paths,
    }
    (args.output / "preparation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
