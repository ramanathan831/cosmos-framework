"""Parallel CPU input preparation using the unchanged native packer and format.

Workers independently replay each unique target before returning a completed file.
The native publisher still owns source checks, deterministic sampling, serialization
and atomic manifest publication. No CUDA, model weights, or scoring is involved.
"""

import argparse
import cProfile
import hashlib
import json
import multiprocessing
import os
import pstats
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
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
from cosmos_framework.posttrain.data.packer.sft import SFTPolicyInput, build_sft_packer
from cosmos_framework.posttrain.entrypoints.sft import sft_policy_source, validate_config_structure
from cosmos_framework.utils.config import load_config
from tools.quantized_optimizers.prepare_corpus import same


def make_packer(config_path):
    config = load_config(config_path, [])
    rl = validate_config_structure(config)
    source, trust = sft_policy_source(config)
    return rl, build_sft_packer(
        source,
        max_length=rl.max_length,
        processor_kwargs=rl.processor_kwargs,
        trust_remote_code=trust,
        mask_think_opener=rl.mask_think_opener,
    )


def worker_init(config_path, threads, directory, profile):
    global _PACKER, _DIRECTORY, _PROFILE
    torch.set_num_threads(threads)
    _, _PACKER = make_packer(config_path)
    _DIRECTORY = Path(directory)
    _PROFILE = profile


def worker_target(target):
    if not _PROFILE:
        return process_target(target)
    profiler = cProfile.Profile()
    result = profiler.runcall(process_target, target)
    result["profile"] = [
        {
            "function": f"{path}:{line}:{name}",
            "primitive_calls": cc,
            "calls": nc,
            "own_seconds": tt,
            "cumulative_seconds": ct,
        }
        for (path, line, name), (cc, nc, tt, ct, _) in pstats.Stats(profiler).stats.items()
    ]
    return result


def process_target(target):
    started = time.monotonic()
    first = deterministic_sample(_PACKER, target)
    path = _DIRECTORY / (target_key(target) + ".pt")
    torch.save(asdict(first), path)
    # A second native decode/resize/tokenization, not a cached-payload comparison.
    second = deterministic_sample(_PACKER, target)
    if not same(asdict(second), torch.load(path, weights_only=True, mmap=True)):
        raise ValueError("independent native CPU replay changed the prepared target")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation worker initialized CUDA")
    return {
        "path": str(path),
        "seconds": time.monotonic() - started,
        "packer": _PACKER.validation_semantics_identity(),
        "cuda_initialized": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-data", action="append", default=[])
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--sample-targets", type=int)
    parser.add_argument("--profile", action="store_true", help="Development CPU profile; not a timing trial")
    parser.add_argument("--reference-corpus", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only preparation requires explicit empty CUDA_VISIBLE_DEVICES")
    if args.output.exists():
        raise ValueError("preparation output must be new")
    started = time.monotonic()
    torch.set_num_threads(8)
    rl, packer = make_packer(args.config)
    paths = [rl.data.path, rl.validation.data.path, *args.extra_data]
    datasets = [JSONLSFTDataset(path) for path in paths]
    originals = datasets
    all_rows = {target_key(target_from_row(d[i])): d[i] for d in datasets for i in range(len(d))}
    if args.sample_targets is not None:
        if not 0 < args.sample_targets <= len(all_rows):
            raise ValueError("invalid canary target count")
        rows = [all_rows[key] for key in sorted(all_rows)[: args.sample_targets]]
        subset = args.output.with_suffix(".canary.jsonl")
        subset.write_text("".join(json.dumps(row) + "\n" for row in rows))
        datasets = [JSONLSFTDataset(str(subset))]
        paths = [str(subset)]
    targets = {target_key(target_from_row(d[i])): target_from_row(d[i]) for d in datasets for i in range(len(d))}
    identity = packer.validation_semantics_identity()
    receipts, futures = {}, {}
    with tempfile.TemporaryDirectory(prefix="native-prepare-workers-") as temporary:
        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=worker_init,
            initargs=(args.config, 8 // args.workers, temporary, args.profile),
        )
        native = packer.sft_process_sample

        def process(target):
            # Submission happens only after native prepare_corpus captured and
            # verified source identities/stamps. Failure cannot publish a manifest.
            if not futures:
                futures.update({key: pool.submit(worker_target, value) for key, value in targets.items()})
            key = target_key(target)
            result = futures[key].result()
            if result["packer"] != identity or result["cuda_initialized"]:
                raise ValueError("worker processor identity/device mismatch")
            receipts[key] = result
            return SFTPolicyInput(**torch.load(result["path"], weights_only=True, mmap=True))

        packer.sft_process_sample = process
        processing_started = time.monotonic()
        try:
            manifest = prepare_corpus(packer, datasets, args.output)
        finally:
            packer.sft_process_sample = native
            pool.shutdown(wait=True, cancel_futures=True)
        processing_seconds = time.monotonic() - processing_started
    corpus = PreparedSFTCorpus(packer, args.output, datasets)
    reference = PreparedSFTCorpus(packer, args.reference_corpus, originals)
    for target in targets.values():
        if not same(asdict(corpus.process(target)), asdict(reference.process(target))):
            raise ValueError("parallel preparation changed previously qualified native inputs")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU publisher initialized CUDA")
    result = {
        "workers": args.workers,
        "threads_per_worker": 8 // args.workers,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "preparation_seconds": processing_seconds,
        "total_seconds_with_independent_exact_replay": time.monotonic() - started,
        "exact_raw_prepared_rows": len(targets),
        "independently_replayed_unique_targets": len(targets),
        "total_verified_rows": sum(len(d) for d in datasets),
        "raw_native_calls": 2 * len(targets),
        "reference_exact_unique_targets": len(targets),
        "reference_corpus_sha256": reference.identity,
        "corpus_sha256": corpus.identity,
        "payload_bytes": sum(v["bytes"] for v in manifest["entries"].values()),
        "cuda_initialized": False,
        "source_paths": paths,
        "worker_compute_seconds_sum": sum(r["seconds"] for r in receipts.values()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "preparation_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.profile:
        combined = {}
        for receipt in receipts.values():
            for row in receipt["profile"]:
                values = combined.setdefault(
                    row["function"], dict.fromkeys(("primitive_calls", "calls", "own_seconds", "cumulative_seconds"), 0)
                )
                for key in values:
                    values[key] += row[key]
        ordered = sorted(combined.items(), key=lambda item: item[1]["cumulative_seconds"], reverse=True)
        (args.output / "worker_profile.json").write_text(
            json.dumps(
                {
                    "scope": "Summed worker profiles of independent native preparation and replay. Nested cumulative times overlap; not an uninstrumented timing result.",
                    "targets": len(receipts),
                    "functions": [{"function": name, **values} for name, values in ordered],
                },
                indent=2,
            )
            + "\n"
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
