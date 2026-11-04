"""Bounded, fail-closed paired native deployment trials. Does not submit jobs."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pynvml

CONFIG = "cosmos_framework/posttrain/configs/sft_optimizer_deployment.py"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bounded_command(command, log_path, deadline, env=None):
    remaining = deadline - time.monotonic()
    if remaining < 30:
        raise RuntimeError("work deadline leaves no time for dependent verification")
    with log_path.open("x") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
    if result.returncode:
        raise RuntimeError(f"dependent verification failed: {log_path}")


def monitor(handle, process, output, stop):
    """External samples; no torch/CUDA context, hooks or model synchronization."""
    with output.open("x") as stream:
        while not stop.is_set():
            row = {"monotonic": time.monotonic(), "time_ns": time.time_ns()}
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                row.update(
                    device_used=memory.used,
                    device_total=memory.total,
                    gpu_processes=[{"pid": p.pid, "bytes": p.usedGpuMemory} for p in gpu_processes],
                    utilization=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                    power_mw=pynvml.nvmlDeviceGetPowerUsage(handle),
                    sm_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                )
                todo, seen, rss, pss = [process.pid], set(), 0, 0
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
                        pass
                row.update(process_tree_pids=sorted(seen), process_tree_rss=rss, process_tree_pss=pss)
                matched = [p.usedGpuMemory for p in gpu_processes if p.pid in seen]
                row["matched_process_gpu_bytes"] = sum(matched) if matched else None
            except Exception as error:
                row["sample_error"] = str(error)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            stop.wait(0.25)


def launch(root, name, arm, seed, *, prepared, canary, handle, deadline):
    output = root / name
    output.mkdir(exist_ok=False)
    env = dict(
        os.environ,
        IMAGINAIRE_OUTPUT_ROOT=str(output / "native"),
        SFT_PREPARED_CORPUS=prepared,
        STATE_DIGEST_PATH=str(output / "state.json"),
    )
    command = [
        "torchrun",
        "--standalone",
        "--nproc-per-node=1",
        "-m",
        "tools.quantized_optimizers.prepared_canary" if canary else "cosmos_framework.posttrain.entrypoints.sft",
        "--config",
        CONFIG,
        "--",
        f"job.name={name}",
        f"rl.seed={seed}",
        f"rl.optimizer.name={arm}",
        f"rl.optimizer.impl={'fused' if arm == 'adamw' else ''}",
        f"rl.optimizer.bnb_min_8bit_size={4096 if arm == 'adamw' else 16384}",
        f"model_hf.grad_ckpt={'true' if arm == 'adamw' else 'false'}",
    ]
    if canary:
        command.extend(["rl.max_steps=4", "rl.save_every=0", "rl.validation.every_steps=4"])
    write(output / "command.json", {"argv": command, "prepared_corpus": prepared})
    start = time.monotonic()
    start_ns = time.time_ns()
    with (output / "train.log").open("x") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        stop = threading.Event()
        thread = threading.Thread(target=monitor, args=(handle, process, output / "memory.jsonl", stop))
        thread.start()
        try:
            status = process.wait(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.terminate()
            status = process.wait(timeout=20)
            raise RuntimeError("allocation work deadline reached") from None
        finally:
            stop.set()
            thread.join()
    elapsed = time.monotonic() - start
    write(
        output / "attempt.json",
        {
            "returncode": status,
            "elapsed_seconds": elapsed,
            "start_time_ns": start_ns,
            "seed": seed,
            "arm": arm,
            "canary_instrumented": canary,
        },
    )
    if status:
        raise RuntimeError(f"native arm failed: {output}")
    paths = list(output.rglob("metrics.jsonl"))
    if len(paths) != 1:
        raise RuntimeError("expected exactly one metrics file")
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    if (
        rows[-1].get("optim/optimizer_step") != (4 if canary else 40)
        or rows[-1].get("optim/nonfinite_skips_total") != 0
    ):
        raise RuntimeError("incomplete or skipped-update arm")
    checkpoint = None
    if not canary:
        checkpoints = list(output.rglob("iter_000000040"))
        if not checkpoints:
            checkpoints = [p.parent for p in output.rglob("model.pt") if "iter_" in p.parent.name]
        if len(checkpoints) != 1:
            raise RuntimeError(f"expected one final committed checkpoint: {checkpoints}")
        checkpoint = str(checkpoints[0])
    write(
        output / "complete.json",
        {"seed": seed, "arm": arm, "metrics": str(paths[0]), "checkpoint": checkpoint, "elapsed_seconds": elapsed},
    )
    return output


def main():
    root = Path(os.environ["RESULTS"])
    mode = os.environ["MODE"]
    if mode not in ("canary", "trials"):
        raise ValueError("MODE must be canary or trials")
    started = time.monotonic()
    deadline = started + int(os.environ["WORK_SECONDS"])
    preparation = Path(os.environ["PREPARATION"])
    receipt = json.loads((preparation / "prepared-corpus/preparation_receipt.json").read_text())
    if receipt["cuda_initialized"] or receipt["exact_raw_prepared_rows"] != 245:
        raise ValueError("missing complete CPU parity receipt")
    # Slurm can leave JOB_GPUS empty on the batch shell while binding the srun
    # device correctly. Resolve CUDA's actual device in an exiting preflight
    # child; the external monitor itself never creates a CUDA context.
    device = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import json,torch; assert torch.cuda.device_count()==1, 'expected exactly one CUDA-visible GPU'; "
                "p=torch.cuda.get_device_properties(0); print(json.dumps({'name':p.name,'uuid':str(p.uuid)}))",
            ],
            text=True,
            timeout=45,
        )
    )
    gpu_uuid = device["uuid"]
    if not gpu_uuid.startswith("GPU-"):
        gpu_uuid = "GPU-" + gpu_uuid
    os.environ["MONITOR_GPU_UUID"] = gpu_uuid
    write(root / f"device-preflight-{os.environ['SLURM_JOB_ID']}.json", device)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(os.environ["MONITOR_GPU_UUID"])
    if pynvml.nvmlDeviceGetName(handle) != "NVIDIA A100-SXM4-80GB":
        raise ValueError("trial requires the prespecified A100 80GB")
    staging = Path(tempfile.mkdtemp(prefix="prepared-sft-")) / "corpus"
    shutil.copytree(preparation / "prepared-corpus", staging)
    stage_seconds = time.monotonic() - started
    write(
        root / f"runtime-{os.environ['SLURM_JOB_ID']}.json",
        {
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "bitsandbytes")},
            "node": os.environ["SLURM_JOB_NODELIST"],
            "gpu_uuid": os.environ["MONITOR_GPU_UUID"],
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "source_archive": os.environ["SOURCE_ARCHIVE"],
            "source_archive_sha256": hashlib.sha256(Path(os.environ["SOURCE_ARCHIVE"]).read_bytes()).hexdigest(),
            "prepared_corpus_sha256": receipt["corpus_sha256"],
            "staging_seconds": stage_seconds,
            "preparation_cpu_seconds": receipt["total_seconds_with_independent_exact_replay"],
        },
    )
    if mode == "canary":
        outputs = []
        for name, arm, corpus in (
            ("raw_fused", "adamw", ""),
            ("prepared_fused", "adamw", str(staging)),
            ("prepared_8bit", "adamw_bnb_8bit", str(staging)),
        ):
            outputs.append(launch(root, name, arm, 17, prepared=corpus, canary=True, handle=handle, deadline=deadline))
            if len(outputs) == 2:
                a, b = [json.loads((path / "state.json").read_text()) for path in outputs]
                keys = ("model", "optimizer", "scheduler", "update", "cursor", "rng")
                matches = {key: a[key] == b[key] for key in keys}
                write(root / "parity.json", matches)
                if not all(matches.values()):
                    raise RuntimeError(f"raw/prepared native parity failed: {matches}")
        write(root / "complete.json", {"canary": "passed", "elapsed_seconds": time.monotonic() - started})
    else:
        seed = int(os.environ["TRIAL_SEED"])
        if seed not in (17, 29, 43):
            raise ValueError("unplanned seed")
        bounded_command(
            [
                sys.executable,
                "-m",
                "tools.quantized_optimizers.inspect_exposure",
                "--data",
                str(Path(os.environ["SFT_DATA_ROOT"]) / "train.jsonl"),
                "--corpus",
                str(staging),
                "--seed",
                str(seed),
                "--output",
                str(root / "exposure.json"),
            ],
            root / "exposure.log",
            deadline,
        )
        for restart in range(3):
            arms = ("adamw", "adamw_bnb_8bit")
            if (restart + (0 if seed == 17 else 1 if seed == 29 else 2)) % 2:
                arms = arms[::-1]
            for arm in arms:
                if deadline - time.monotonic() < 240:
                    raise RuntimeError("insufficient reserved time for next run; retain incomplete cells")
                launch(
                    root,
                    f"restart{restart}_{arm}",
                    arm,
                    seed,
                    prepared=str(staging),
                    canary=False,
                    handle=handle,
                    deadline=deadline,
                )
        write(root / "training_complete.json", {"seed": seed, "elapsed_seconds": time.monotonic() - started})
        identities = {}
        for restart in range(3):
            for arm in ("adamw", "adamw_bnb_8bit"):
                output = root / f"restart{restart}_{arm}"
                checkpoint = json.loads((output / "complete.json").read_text())["checkpoint"]
                bounded_command(
                    [
                        sys.executable,
                        "-m",
                        "tools.quantized_optimizers.checkpoint_digest",
                        checkpoint,
                        str(output / "checkpoint_digest.json"),
                    ],
                    output / "checkpoint_digest.log",
                    deadline,
                )
                identities[restart, arm] = json.loads((output / "checkpoint_digest.json").read_text())
        matches = {
            f"restart{restart}_{arm}_{key}": identities[0, arm][key] == identities[restart, arm][key]
            for restart in (1, 2)
            for arm in ("adamw", "adamw_bnb_8bit")
            for key in ("model", "optim", "rng", "scheduler", "data", "optimizer_step")
        }
        write(root / "restart_parity.json", matches)
        if not all(matches.values()):
            raise RuntimeError("deterministic paired restarts changed model/optimizer/control state")
        # Fixed restart0 only; selection was sealed before preparation or training.
        for arm in ("frozen", "adamw", "adamw_bnb_8bit") if seed == 17 else ("adamw", "adamw_bnb_8bit"):
            command = [
                "torchrun",
                "--standalone",
                "--nproc-per-node=1",
                "-m",
                "tools.quantized_optimizers.score_dense",
                "--data",
                str(preparation / "final-selection/validation.jsonl"),
                "--selection",
                str(preparation / "final-selection/manifest.json"),
                "--output",
                str(root / f"final_{arm}"),
                "--seed",
                str(seed),
            ]
            if arm != "frozen":
                checkpoint = json.loads((root / f"restart0_{arm}/complete.json").read_text())["checkpoint"]
                command.extend(["--checkpoint", checkpoint])
            bounded_command(
                command, root / f"final_{arm}.log", deadline, env=dict(os.environ, SFT_PREPARED_CORPUS=str(staging))
            )
        write(root / "complete.json", {"seed": seed, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
