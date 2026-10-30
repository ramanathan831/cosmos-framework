# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded CPU replay of native preprocessing; never trains or changes a live job.

The launcher owns the sealed runtime, CPU allocation and container. Each ``run``
starts a fresh interpreter before importing NumPy/Torch. ``blas-one`` changes only
the two BLAS environment variables; the inherited OMP setting stays unchanged.
Compare runs before interpreting memory differences. End-only GC/trim is a
diagnostic in the disposable worker, never a production allocator policy.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROTOCOL = "cosmos.native-preprocessing-memory.v1"
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM")
MALL_FIELDS = (
    "arena",
    "ordblks",
    "smblks",
    "hblks",
    "hblkhd",
    "usmblks",
    "fsmblks",
    "uordblks",
    "fordblks",
    "keepcost",
)


class Mallinfo2(ctypes.Structure):
    _fields_ = [(key, ctypes.c_size_t) for key in MALL_FIELDS]


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_topology(*, sysfs_root: Path = Path("/sys/devices/system/cpu")) -> dict[str, Any]:
    """Inspect this process's effective affinity, not the scheduler's requested CPU count."""
    affinity = sorted(os.sched_getaffinity(0))
    records, missing = [], []
    for cpu in affinity:
        directory = sysfs_root / f"cpu{cpu}"
        try:
            socket = int((directory / "topology/physical_package_id").read_text())
            core = int((directory / "topology/core_id").read_text())
        except (OSError, ValueError):
            missing.append(cpu)
            continue
        nodes = sorted(int(path.name[4:]) for path in directory.glob("node[0-9]*"))
        records.append({"logical_cpu": cpu, "socket": socket, "core": core, "numa_nodes": nodes})
    models = set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                models.add(line.partition(":")[2].strip())
    except OSError:
        pass
    return {
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "models": sorted(models),
        "affinity": affinity,
        "logical_cpu_count": len(affinity),
        "physical_cpu_count": len({(row["socket"], row["core"]) for row in records}) if not missing else None,
        "cpu_records": records,
        "unavailable_cpus": missing,
    }


def validate_cpu_topology(topology: dict[str, Any], expected: int | None) -> None:
    if expected is None:
        return
    if type(expected) is not int or expected < 1:
        raise ValueError("expected physical CPUs must be a positive integer")
    if topology["unavailable_cpus"] or topology["physical_cpu_count"] != expected:
        raise ValueError(
            f"effective CPU affinity requires exactly {expected} physical cores; "
            f"observed {topology['physical_cpu_count']} physical / {topology['logical_cpu_count']} logical"
        )


def policy_environment(environ: dict[str, str], policy: str) -> dict[str, str]:
    result = dict(environ)
    if policy == "blas-one":
        result.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    elif policy != "inherited":
        raise ValueError("unknown thread policy")
    result.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return result


def tree_hash(value: Any) -> str:
    """Hash exact tensor bytes, dtype/shape, metadata and sequence types in bounded chunks."""
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def tag(item: Any) -> None:
        encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu" or item.requires_grad:
                raise ValueError("only detached CPU tensors may enter diagnostic fingerprints")
            tag(["tensor", str(item.dtype), list(item.shape)])
            flat = item.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
            for offset in range(0, flat.size, 4 * 1024**2):
                digest.update(memoryview(flat[offset : offset + 4 * 1024**2]))
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("object arrays cannot be fingerprinted")
            tag(["ndarray", str(item.dtype), list(item.shape)])
            flat = np.ascontiguousarray(item).reshape(-1).view(np.uint8)
            for offset in range(0, flat.size, 4 * 1024**2):
                digest.update(memoryview(flat[offset : offset + 4 * 1024**2]))
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            tag(["dataclass", type(item).__qualname__])
            for field in dataclasses.fields(item):
                tag(field.name)
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            tag(["mapping", len(item)])
            for key in sorted(item):
                tag(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            tag([type(item).__name__, len(item)])
            for child in item:
                visit(child)
        elif isinstance(item, np.generic):
            visit(item.item())
        elif item is None or isinstance(item, (str, bool, int, float)):
            tag([type(item).__name__, item])
        elif isinstance(item, bytes):
            tag(["bytes", len(item)])
            digest.update(item)
        else:
            raise TypeError(f"unsupported fingerprint value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def rng_hash() -> str:
    import random

    import numpy as np
    import torch

    return tree_hash((random.getstate(), np.random.get_state(), torch.get_rng_state()))


def process_memory(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": pid}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, raw = line.partition(":")
            if key in {"VmRSS", "VmHWM"}:
                result[key] = int(raw.split()[0]) * 1024
            elif key == "Threads":
                result[key] = int(raw)
            elif key in {"Cpus_allowed_list", "Mems_allowed_list"}:
                result[key] = raw.strip()
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            key, _, raw = line.partition(":")
            if raw.strip().endswith("kB"):
                result[key] = int(raw.split()[0]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        result["unavailable"] = True
    return result


def descendants(pid: int) -> list[int]:
    found = [pid]
    try:
        for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split():
            found.extend(descendants(int(child)))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return found


def worker_snapshot(label: str) -> dict[str, Any]:
    import torch

    from cosmos_framework.posttrain.data.packer import sft_preprocess

    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("diagnostic worker must remain CPU-only")
    cache = sft_preprocess._WORKER_CACHE
    if cache is None:
        raise RuntimeError("native preprocessing worker is not initialized")
    stats = cache.stats()
    if any(item["bytes"] > item["limit"] for item in stats.values()):
        raise RuntimeError("cache exceeded its retained-entry byte budget")
    libc = ctypes.CDLL(None)
    allocator: dict[str, Any] = {"available": False}
    if hasattr(libc, "mallinfo2") and hasattr(libc, "gnu_get_libc_version"):
        libc.gnu_get_libc_version.restype = ctypes.c_char_p
        libc.mallinfo2.argtypes, libc.mallinfo2.restype = [], Mallinfo2
        state = libc.mallinfo2()
        allocator = {
            "available": True,
            "glibc": libc.gnu_get_libc_version().decode(),
            "mallinfo2": {key: getattr(state, key) for key in MALL_FIELDS},
        }
    thread_names = {}
    for task in Path("/proc/self/task").iterdir():
        try:
            name = (task / "comm").read_text().strip()
            thread_names[name] = thread_names.get(name, 0) + 1
        except FileNotFoundError:
            pass
    maps = Path("/proc/self/maps").read_text().splitlines()
    libraries = sorted(
        {
            Path(line.split()[-1]).name
            for line in maps
            if any(word in line.lower() for word in ("blas", "mkl", "libgomp", "libiomp", "avcodec"))
        }
    )
    return {
        "label": label,
        "memory": process_memory(os.getpid()),
        "allocator": allocator,
        "cache": stats,
        "torch_threads": torch.get_num_threads(),
        "thread_names": thread_names,
        "native_libraries": libraries,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
        "fd_count": len(list(Path("/proc/self/fd").iterdir())),
        "cuda_initialized": False,
        "rng_sha256": rng_hash(),
        "cpu_topology": cpu_topology(),
    }


def worker_end_diagnostic() -> dict[str, Any]:
    import gc

    from cosmos_framework.posttrain.data.packer import sft_preprocess

    cache = sft_preprocess._WORKER_CACHE
    assert cache is not None

    def live():
        return tree_hash((dict(cache.samples.entries), dict(cache.videos.entries)))

    original, rng = live(), rng_hash()
    before = worker_snapshot("before_gc")
    collected = gc.collect()
    after_gc = worker_snapshot("after_gc")
    libc = ctypes.CDLL(None)
    trim = None
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim.argtypes, libc.malloc_trim.restype = [ctypes.c_size_t], ctypes.c_int
        trim = libc.malloc_trim(0)
    after_trim = worker_snapshot("after_end_only_trim")
    if live() != original or rng_hash() != rng or not before["cache"] == after_gc["cache"] == after_trim["cache"]:
        raise RuntimeError("end diagnostic changed live cache contents, counters or RNG")
    return {
        "before": before,
        "after_gc": after_gc,
        "after_trim": after_trim,
        "gc_collected": collected,
        "malloc_trim_return": trim,
        "cache_sha256": original,
        "cache_and_rng_unchanged": True,
    }


def compare_reports(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("comparison requires at least two completed replays")
    reports = [json.loads((path / "result.json").read_text()) for path in paths]
    required = (
        "build_receipt_sha256",
        "source_fingerprint",
        "processor_identity_sha256",
        "row_output_sha256",
        "row_order_sha256",
        "data_state_sha256",
        "rng_before_sha256",
        "rng_after_sha256",
        "rows",
        "seed",
        "cache_bytes",
        "window_rows",
        "consumer_delay_seconds",
        "config",
        "override",
        "versions",
    )
    for path, report in zip(paths, reports, strict=True):
        if report.get("protocol") != PROTOCOL or report.get("complete") is not True:
            raise ValueError("incomplete or incompatible diagnostic report")
        if report["rng_before_sha256"] != report["rng_after_sha256"]:
            raise ValueError("replay changed main RNG")
        if not report["worker_rng_unchanged"] or not report["end_diagnostic"]["cache_and_rng_unchanged"]:
            raise ValueError("worker RNG/cache check failed")
        for key in required:
            if report[key] != reports[0][key]:
                raise ValueError(f"replays differ in {key}")
        records = path / "rows.jsonl"
        if file_hash(records) != report["row_output_sha256"]:
            raise ValueError("row records differ from receipt")
    return {
        "protocol": PROTOCOL,
        "passed": True,
        "semantic_and_tensor_bytes_exact": True,
        "inputs": [
            {
                "result_sha256": file_hash(path / "result.json"),
                "policy": report["thread_policy"],
                "repeat": report["repeat"],
                "peak_worker_pss_bytes": report["peak_worker_pss_bytes"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
            for path, report in zip(paths, reports, strict=True)
        ],
        "compared_fields": list(required),
        "causality_requires_matched_cpu_placement": True,
    }


def run_child(args: argparse.Namespace) -> None:
    topology = cpu_topology()
    validate_cpu_topology(topology, args.expected_physical_cpus)
    import importlib.metadata
    import random

    import numpy as np
    import torch

    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.data.packer.sft_preprocess import build_sft_preprocessing
    from cosmos_framework.posttrain.roles.env.sft import build_sft_env
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.utils.lazy_config import instantiate

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU replay inherited CUDA initialization")
    out = Path(args.output)
    config = load_config(args.config, ["--", *args.override])
    rl = instantiate(config.rl)
    if rl.seed != args.seed:
        raise ValueError("CLI seed must match the native recipe")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = build_sft_env(rl.data, args.seed)
    packer = build_sft_packer(
        args.model_path,
        max_length=rl.max_length,
        processor_kwargs=rl.processor_kwargs,
        mask_think_opener=rl.mask_think_opener,
        trust_remote_code=False,
    )
    pipeline = build_sft_preprocessing(
        packer,
        name_or_path=args.model_path,
        cache_bytes=args.cache_bytes,
        prefetch=True,
        media_hashes=data.dataset.media_hashes,
        max_pending=8,
    )
    stop, peaks = threading.Event(), {"Pss": 0, "Rss": 0}

    def monitor() -> None:
        with (out / "host-memory.jsonl").open("x") as stream:
            while not stop.is_set():
                rows = [process_memory(pid) for pid in descendants(os.getpid())]
                totals = {key: sum(row.get(key, 0) for row in rows) for key in peaks}
                for key in peaks:
                    peaks[key] = max(peaks[key], totals[key])
                stream.write(json.dumps({"unix_seconds": time.time(), "processes": rows, "sum": totals}) + "\n")
                stream.flush()
                if totals["Rss"] > args.max_rss_gib * 1024**3:
                    os._exit(97)  # Outer process terminates this entire owned process group.
                stop.wait(1)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    snapshots, order = [], hashlib.sha256()
    before, started, next_snapshot = rng_hash(), time.monotonic(), 0
    try:
        with (out / "rows.jsonl").open("x") as records:
            for offset in range(0, args.rows, 8):
                trajectories = data.get_samples(8)
                targets = [row.transitions[0].targets for row in trajectories]
                pipeline.prepare_window(targets)
                for trajectory, target in zip(trajectories, targets, strict=True):
                    sample = pipeline.sft_process_sample(target)
                    batch = pipeline.sft_collate_fn([sample], pipeline.sft_compute_max_len([sample]), -100)
                    if not batch.get("pixel_values_videos", torch.empty(0)).numel():
                        raise RuntimeError("WTS replay requires actual video payloads")
                    valid = (
                        (batch["labels"][:, 1:] != -100)
                        & batch["attention_mask"][:, 1:].bool()
                        & batch["attention_mask"][:, :-1].bool()
                    )
                    record = {
                        "row_identity_sha256": hashlib.sha256(trajectory.id.encode()).hexdigest(),
                        "sample_sha256": tree_hash(sample),
                        "batch_sha256": tree_hash(batch),
                        "sequence_length": len(sample.input_ids),
                        "supervised_tokens": int(valid.sum()),
                        "tensor_shapes": {key: list(value.shape) for key, value in batch.items()},
                    }
                    records.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                    order.update((trajectory.id + "\n").encode())
                    del sample, batch, valid
                    if args.consumer_delay_seconds:
                        time.sleep(args.consumer_delay_seconds)
                pipeline.finish_window()
                if rng_hash() != before:
                    raise RuntimeError("CPU replay changed main RNG")
                done = offset + 8
                if done >= next_snapshot or done == args.rows:
                    row = pipeline._executor.submit(worker_snapshot, f"rows_{done}").result(timeout=60)
                    validate_cpu_topology(row["cpu_topology"], args.expected_physical_cpus)
                    row.update(rows=done, elapsed_seconds=time.monotonic() - started)
                    snapshots.append(row)
                    with (out / "snapshots.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps({"rows": done, "worker_pss_bytes": row["memory"].get("Pss")}), flush=True)
                    next_snapshot = (done // 100 + 1) * 100
                del trajectories, targets
        end = pipeline._executor.submit(worker_end_diagnostic).result(timeout=180)
        worker_rng = {row["rng_sha256"] for row in snapshots}
        if len(worker_rng) != 1 or end["before"]["rng_sha256"] not in worker_rng:
            raise RuntimeError("worker RNG changed across preprocessing windows")
        result = {
            "protocol": PROTOCOL,
            "complete": True,
            "training_performed": False,
            "cuda_initialized": False,
            "output": str(out.resolve()),
            "config": args.config,
            "override": args.override,
            "build_receipt_sha256": file_hash(Path(args.build_receipt)),
            "rows": args.rows,
            "seed": args.seed,
            "repeat": args.repeat,
            "thread_policy": args.thread_policy,
            "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
            "parent_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_topology": topology,
            "expected_physical_cpus": args.expected_physical_cpus,
            "source_fingerprint": data.source_fingerprint,
            "processor_identity_sha256": json_hash(packer.validation_semantics_identity()),
            "row_order_sha256": order.hexdigest(),
            "row_output_sha256": file_hash(out / "rows.jsonl"),
            "data_state_sha256": tree_hash(data.state_dict()),
            "rng_before_sha256": before,
            "rng_after_sha256": rng_hash(),
            "worker_rng_unchanged": True,
            "cache_bytes": args.cache_bytes,
            "window_rows": 8,
            "consumer_delay_seconds": args.consumer_delay_seconds,
            "end_diagnostic": end,
            "peak_worker_pss_bytes": max(row["memory"].get("Pss", 0) for row in snapshots),
            "sampled_tree_peaks_bytes": peaks,
            "elapsed_seconds": time.monotonic() - started,
            "versions": {
                key: importlib.metadata.version(key) for key in ("torch", "numpy", "transformers", "torchcodec")
            },
            "limitations": [
                "No model or concurrent GPU training; CPU replay is a separate experiment.",
                "Boundary worker samples and 1Hz process-tree samples are lower bounds.",
                "Thread environment is an intervention, not proof of decoder or allocator attribution.",
                "RSS/PSS excludes unmapped filesystem page cache; HWM is lifetime, not phase-local.",
                "End-only trim is diagnostic; no production allocator change is qualified.",
            ],
        }
        write_json(out / "result.json", result)
    finally:
        pipeline.close()
        stop.set()
        watcher.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("config", "model-path", "data-root", "build-receipt", "output"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--thread-policy", choices=("inherited", "blas-one"), required=True)
    run.add_argument("--rows", type=int, default=1000)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--repeat", type=int, default=0)
    run.add_argument("--cache-bytes", type=int, default=2**30)
    run.add_argument("--consumer-delay-seconds", type=float, default=0.0)
    run.add_argument("--max-rss-gib", type=float, default=24.0)
    run.add_argument("--timeout-seconds", type=int, default=3300)
    run.add_argument(
        "--expected-physical-cpus", type=int, help="Refuse wrong effective affinity before importing Torch"
    )
    run.add_argument("--override", action="append", default=[])
    run.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    compare = sub.add_parser("compare")
    compare.add_argument("--run", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        write_json(args.output, compare_reports(args.run))
        return
    if args.rows < 8 or args.rows % 8 or args.cache_bytes < 0 or args.repeat < 0 or args.timeout_seconds < 1:
        parser.error("rows must be a positive multiple of 8; cache/repeat nonnegative; timeout positive")
    if (
        not all(math.isfinite(value) and value >= 0 for value in (args.consumer_delay_seconds, args.max_rss_gib))
        or args.max_rss_gib == 0
    ):
        parser.error("delay must be finite/nonnegative and RSS bound positive")
    if args.child:
        run_child(args)
        return
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    topology = cpu_topology()
    write_json(output / "cpu-topology.json", topology)
    validate_cpu_topology(topology, args.expected_physical_cpus)
    env = policy_environment(dict(os.environ), args.thread_policy)
    env.update(SFT_MODEL_PATH=args.model_path, SFT_DATA_ROOT=args.data_root, PYTHONHASHSEED=str(args.seed))
    write_json(
        output / "launch.json",
        {
            "argv": sys.argv,
            "script_sha256": file_hash(Path(__file__)),
            "inherited_thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
            "effective_thread_environment": {key: env.get(key) for key in THREAD_KEYS},
        },
    )
    with (output / "child.log").open("x") as log:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=args.timeout_seconds)
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        write_json(output / "exit.json", {"exit_code": code})
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
