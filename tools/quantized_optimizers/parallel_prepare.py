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
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from cosmos_framework.posttrain.data.datasets.sft import JSONLSFTDataset, sha256_file
from cosmos_framework.posttrain.data.packer.prepared_sft import (
    PreparedSFTCorpus,
    deterministic_sample,
    prepare_corpus,
    stamp,
    target_from_row,
    target_key,
)
from cosmos_framework.posttrain.data.packer.sft import SFTPolicyInput, build_sft_packer
from cosmos_framework.posttrain.entrypoints.sft import sft_policy_source, validate_config_structure
from cosmos_framework.utils.config import load_config
from tools.quantized_optimizers.prepare_corpus import same
from tools.quantized_optimizers.verify_published_corpus import verify as verify_bytes
from tools.quantized_optimizers.video_reuse import VideoReuse, single_video


@contextmanager
def process_memory():
    """Sample the preparation process tree, including spawned decoder workers."""
    done = threading.Event()
    peak = {"rss_bytes": 0, "pss_bytes": 0, "samples": 0}

    def sample():
        while not done.is_set():
            rss = pss = 0
            todo, seen = [os.getpid()], set()
            while todo:
                pid = todo.pop()
                if pid in seen:
                    continue
                seen.add(pid)
                try:
                    todo.extend(map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split()))
                    values = {
                        line.split(":")[0]: int(line.split()[1]) * 1024
                        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()[1:]
                        if line.startswith(("Rss:", "Pss:"))
                    }
                    rss += values["Rss"]
                    pss += values["Pss"]
                except (OSError, KeyError):
                    continue
            peak["rss_bytes"] = max(peak["rss_bytes"], rss)
            peak["pss_bytes"] = max(peak["pss_bytes"], pss)
            peak["samples"] += 1
            done.wait(0.5)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield peak
    finally:
        done.set()
        thread.join()


def verify_reference_slice(plan, manifest, corpus, targets):
    """Exact selected-entry oracle check; does not claim full source binding.

    A filtered JSONL intentionally has a different cursor identity. Do not weaken
    PreparedSFTCorpus's full-source contract to accommodate this development test.
    """
    reference = Path(plan["reference"])
    path = reference / "manifest.json"
    before = stamp(path)
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != plan["reference_sha256"]:
        raise ValueError("slice reference manifest checksum mismatch")
    oracle = json.loads(value)
    if oracle["format"] != manifest["format"] or oracle["packer"] != manifest["packer"]:
        raise ValueError("slice reference processor/format mismatch")
    if set(targets) != {row["target"] for row in plan["targets"]}:
        raise ValueError("slice target selection changed")
    for name, digest in manifest["media"].items():
        source_stamp = stamp(Path(name))
        if oracle["media"].get(name) != digest or sha256_file(name) != digest or stamp(Path(name)) != source_stamp:
            raise ValueError("slice reference source mismatch")
    for key, target in targets.items():
        payload = reference / f"{key}.pt"
        payload_stamp = stamp(payload)
        if sha256_file(payload) != oracle["entries"][key]["sha256"]:
            raise ValueError("slice reference payload checksum mismatch")
        if not same(asdict(corpus.process(target)), torch.load(payload, weights_only=True, mmap=True)):
            raise ValueError("slice differs from qualified native reference")
        if stamp(payload) != payload_stamp:
            raise ValueError("slice reference payload mutated")
    if stamp(path) != before or sha256_file(path) != plan["reference_sha256"]:
        raise ValueError("slice reference manifest mutated")


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


def worker_init(config_path, threads, directory, profile, decoder_threads):
    global _PACKER, _DIRECTORY, _PROFILE, _DECODER_THREADS
    torch.set_num_threads(threads)
    _, _PACKER = make_packer(config_path)
    _DIRECTORY = Path(directory)
    _PROFILE = profile
    _DECODER_THREADS = decoder_threads


@contextmanager
def capped_cpu_decoder(threads):
    """Process-local development override; restore the native decoder before replay."""
    import importlib.metadata

    import torchcodec.decoders as decoders

    if importlib.metadata.version("transformers") != "5.8.1":
        raise ValueError("decoder thread intervention requires the inspected Transformers5.8.1 source")
    original = decoders.VideoDecoder
    calls = []

    def create(*args, **kwargs):
        device = torch.device(kwargs.get("device") or torch.get_default_device())
        if device.type != "cpu" or kwargs.get("seek_mode") != "exact" or kwargs.get("num_ffmpeg_threads") != 0:
            raise ValueError("unexpected decoder contract; refuse to change device, seek or sampling")
        calls.append(1)
        return original(*args, **dict(kwargs, num_ffmpeg_threads=threads))

    decoders.VideoDecoder = create
    try:
        yield
        if not calls:
            raise ValueError("candidate did not exercise the native torchcodec video decoder")
    finally:
        decoders.VideoDecoder = original


def worker_target(target, cache=None):
    if not _PROFILE:
        return process_target(target, cache)
    profiler = cProfile.Profile()
    with video_stage_times(_PACKER.processor.video_processor) as stages:
        result = profiler.runcall(process_target, target, cache)
    result["video_stages"] = stages
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


def worker_video_group(targets, source, digest, limit_bytes):
    cache = VideoReuse(_PACKER.processor.video_processor, source, digest, limit_bytes)
    results = {target_key(target): worker_target(target, cache) for target in targets}
    # One receipt per group; retained buffers die before the next submitted group.
    next(iter(results.values()))["video_reuse"] = dict(cache.stats)
    return results


@contextmanager
def video_stage_times(processor):
    """Direct wall timers: cProfile call stacks can be inconsistent across native ops.

    Patch the class only inside this single-worker call, leaving the instance's
    serialized settings untouched. Inherited methods and failures restore exactly.
    The decode time is nested inside video preprocessing; never add those totals.
    """
    cls = type(processor)
    missing = object()
    previous = {}
    stages = {name: {"calls": 0, "seconds": 0.0} for name in ("preprocess", "_decode_and_sample_videos")}

    def measured(original, name):
        def call(instance, *args, **kwargs):
            if instance is not processor:
                return original(instance, *args, **kwargs)
            started = time.monotonic()
            try:
                return original(instance, *args, **kwargs)
            finally:
                stages[name]["calls"] += 1
                stages[name]["seconds"] += time.monotonic() - started

        return call

    try:
        for name in stages:
            original = getattr(cls, name)
            previous[name] = cls.__dict__.get(name, missing)
            setattr(cls, name, measured(original, name))
        yield stages
    finally:
        for name, original in previous.items():
            if original is missing:
                delattr(cls, name)
            else:
                setattr(cls, name, original)


def process_target(target, cache=None):
    started = time.monotonic()
    timing = None
    if _DECODER_THREADS is None:
        if cache is None:
            first = deterministic_sample(_PACKER, target)
        else:
            with cache.installed(_PACKER.processor):
                first = deterministic_sample(_PACKER, target)
        first_finished = time.monotonic()
        # Always independently decode and process with the original native
        # implementation, including cache hits. This is part of the old endpoint.
        second = deterministic_sample(_PACKER, target)
        samples_finished = time.monotonic()
        phases = {
            "first_sample": first_finished - started,
            "independent_replay": samples_finished - first_finished,
        }
    else:
        # Counterbalance decode order without consulting timings or task scores.
        order = ["candidate", "control"] if int(target_key(target)[-1], 16) % 2 else ["control", "candidate"]
        samples, seconds = {}, {}
        for mode in order:
            before = time.monotonic()
            if mode == "candidate":
                with capped_cpu_decoder(_DECODER_THREADS):
                    samples[mode] = deterministic_sample(_PACKER, target)
            else:
                samples[mode] = deterministic_sample(_PACKER, target)
            seconds[mode] = time.monotonic() - before
        first, second = samples["candidate"], samples["control"]
        timing = {"target": target_key(target), "order": order, "seconds": seconds}
        samples_finished = time.monotonic()
        phases = {"decoder_pair_samples": samples_finished - started}
    path = _DIRECTORY / (target_key(target) + ".pt")
    torch.save(asdict(first), path)
    # Both are independent native decodes; control uses the original thread setting.
    if not same(asdict(second), torch.load(path, weights_only=True, mmap=True)):
        raise ValueError("independent native CPU replay changed the prepared target")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation worker initialized CUDA")
    serialized = time.monotonic()
    identity = _PACKER.validation_semantics_identity()
    finished = time.monotonic()
    phases.update(serialize_reload_compare=serialized - samples_finished, processor_identity=finished - serialized)
    return {
        "path": str(path),
        "seconds": finished - started,
        "packer": identity,
        "cuda_initialized": False,
        "decoder_pair": timing,
        "phase_seconds": phases,
    }


def main(memory=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-data", action="append", default=[])
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--sample-targets", type=int)
    parser.add_argument("--profile", action="store_true", help="Development CPU profile; not a timing trial")
    parser.add_argument("--decoder-threads", type=int, choices=(1, 2), help="CPU development canary with native replay")
    parser.add_argument("--reference-corpus", type=Path, required=True)
    parser.add_argument("--fixed-slice", type=Path, help="Pinned development-only selection and oracle identity")
    parser.add_argument("--reference-verification", choices=("tensors", "bytes"), default="tensors")
    parser.add_argument(
        "--video-cache-bytes", type=int, default=0, help="Opt-in bounded first-pass video reuse per worker"
    )
    args = parser.parse_args()
    if args.video_cache_bytes < 0 or (args.video_cache_bytes and args.decoder_threads is not None):
        raise ValueError("video reuse requires a nonnegative byte limit and unchanged native decoder settings")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only preparation requires explicit empty CUDA_VISIBLE_DEVICES")
    if args.output.exists():
        raise ValueError("preparation output must be new")
    started = time.monotonic()
    torch.set_num_threads(8)
    rl, packer = make_packer(args.config)
    paths = [rl.data.path, rl.validation.data.path, *args.extra_data]
    plan = None
    if args.fixed_slice:
        if args.sample_targets is not None or args.extra_data:
            raise ValueError("fixed slice cannot be combined with another selection")
        plan = json.loads(args.fixed_slice.read_text())
        if Path(plan["reference"]).resolve() != args.reference_corpus.resolve():
            raise ValueError("slice reference path mismatch")
        if sha256_file(plan["data"]) != plan["data_sha256"] or sha256_file(plan["source"]) != plan["source_sha256"]:
            raise ValueError("slice source identity mismatch")
        paths = [plan["data"]]
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
    groups, media = {}, {}
    if args.video_cache_bytes:
        for dataset in datasets:
            for name, digest in dataset.media_hashes.items():
                resolved = str(Path(name).resolve(strict=True))
                if resolved in media and media[resolved] != digest:
                    raise ValueError("conflicting video reuse media identities")
                media[resolved] = digest
        for target in targets.values():
            source = single_video(target)
            if source not in media:
                raise ValueError("video reuse source was not bound by the native dataset")
            groups.setdefault(source, []).append(target)
    identity = packer.validation_semantics_identity()
    receipts, futures = {}, {}
    with tempfile.TemporaryDirectory(prefix="native-prepare-workers-") as temporary:
        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=worker_init,
            initargs=(args.config, 8 // args.workers, temporary, args.profile, args.decoder_threads),
        )
        native = packer.sft_process_sample

        def process(target):
            # Submission happens only after native prepare_corpus captured and
            # verified source identities/stamps. Failure cannot publish a manifest.
            if not futures:
                if args.video_cache_bytes:
                    for source, group in groups.items():
                        future = pool.submit(worker_video_group, group, source, media[source], args.video_cache_bytes)
                        futures.update({target_key(value): future for value in group})
                else:
                    futures.update({key: pool.submit(worker_target, value) for key, value in targets.items()})
            key = target_key(target)
            result = futures[key].result()
            if args.video_cache_bytes:
                result = result[key]
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
    verification_started = time.monotonic()
    corpus = PreparedSFTCorpus(packer, args.output, datasets)
    if plan:
        if args.reference_verification == "bytes":
            if set(targets) != {row["target"] for row in plan["targets"]}:
                raise ValueError("slice target selection changed")
            verify_bytes(
                args.output,
                args.reference_corpus,
                plan["reference_sha256"],
                candidate_manifest_sha256=corpus.identity,
                selected_keys=targets,
            )
        else:
            verify_reference_slice(plan, manifest, corpus, targets)
        if sha256_file(plan["data"]) != plan["data_sha256"] or sha256_file(plan["source"]) != plan["source_sha256"]:
            raise ValueError("slice source changed during preparation")
        reference_identity = plan["reference_sha256"]
    else:
        reference = PreparedSFTCorpus(packer, args.reference_corpus, originals)
        if args.reference_verification == "bytes":
            verify_bytes(args.output, args.reference_corpus, reference.identity)
        else:
            for target in targets.values():
                if not same(asdict(corpus.process(target)), asdict(reference.process(target))):
                    raise ValueError("parallel preparation changed previously qualified native inputs")
        reference_identity = reference.identity
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
        "reference_corpus_sha256": reference_identity,
        "corpus_sha256": corpus.identity,
        "payload_bytes": sum(v["bytes"] for v in manifest["entries"].values()),
        "cuda_initialized": False,
        "source_paths": paths,
        "worker_compute_seconds_sum": sum(r["seconds"] for r in receipts.values()),
        "decoder_threads": args.decoder_threads,
        "video_cache_bytes_per_worker": args.video_cache_bytes,
        "video_reuse": [r["video_reuse"] for r in receipts.values() if "video_reuse" in r],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "setup_seconds": processing_started - started,
        "verification_seconds": time.monotonic() - verification_started,
        "reference_verification": args.reference_verification,
        "process_tree_peak": dict(memory or {}),
        "fixed_slice_sha256": sha256_file(args.fixed_slice) if plan else None,
        "scope": "selected development targets only" if plan else "full configured sources",
        "worker_phase_seconds_sum": {
            key: sum(r["phase_seconds"][key] for r in receipts.values())
            for key in next(iter(receipts.values()))["phase_seconds"]
        },
    }
    (args.output / "preparation_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.decoder_threads is not None:
        (args.output / "decoder_pairs.json").write_text(
            json.dumps([receipt["decoder_pair"] for receipt in receipts.values()], indent=2) + "\n"
        )
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
                    "cprofile_caution": "Native-operator callbacks may produce inconsistent call-stack counters. Use direct monotonic stage timers for phase attribution, not cumulative cProfile percentages.",
                    "targets": len(receipts),
                    "direct_video_stages": {
                        name: {
                            field: sum(r["video_stages"][name][field] for r in receipts.values())
                            for field in ("calls", "seconds")
                        }
                        for name in ("preprocess", "_decode_and_sample_videos")
                    },
                    "functions": [{"function": name, **values} for name, values in ordered],
                },
                indent=2,
            )
            + "\n"
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    with process_memory() as memory:
        if "--profile" in sys.argv:
            profile = cProfile.Profile()
            profile.runcall(main, memory)
            output = Path(sys.argv[sys.argv.index("--output") + 1])
            (output / "parent_profile.json").write_text(
                json.dumps(
                    [
                        {"function": f"{path}:{line}:{name}", "calls": nc, "own_seconds": tt, "cumulative_seconds": ct}
                        for (path, line, name), (_, nc, tt, ct, _) in pstats.Stats(profile).stats.items()
                    ],
                    indent=2,
                )
                + "\n"
            )
        else:
            main(memory)
