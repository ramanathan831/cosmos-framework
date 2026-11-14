"""Run the actual four-GPU Joint PPO entry with existing approved LIBERO assets.

This first eight-window run is an integration gate, not task-quality selection.
Native evaluation, source validation, rollout and checkpoint paths stay enabled.
"""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=("adamw", "adamw_bnb_8bit", "paged_adamw_8bit"), default="adamw")
    parser.add_argument("--resume", default="")
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("updates must be positive")
    root, output = Path(os.environ["RESOURCE_ROOT"]), Path(os.environ["RESULTS"])
    expected = {"numpy": "1.26.4", "transformers": "4.57.6", "diffusers": "0.35.1", "bitsandbytes": "0.49.2"}
    actual = {name: importlib.metadata.version(name) for name in expected}
    (output / "dependency-check.json").write_text(json.dumps({"expected": expected, "actual": actual}, indent=2))
    if actual != expected:
        raise RuntimeError("action runtime differs from its explicit dependency pins")
    assets = root / "cosmos-family-adapters-20261030/gr00t-public-assets-v1"
    cache = Path(os.environ["HF_HOME"]) / "hub/models--nvidia--Cosmos-Reason2-2B"
    revision = "9ce19a195e423419c349abfc86fd07178b230561"
    (cache / "snapshots").mkdir(parents=True)
    (cache / "refs").mkdir()
    (cache / "snapshots" / revision).symlink_to(assets / "backbone", target_is_directory=True)
    (cache / "refs/main").write_text(revision)
    config = Path(os.environ["LIBERO_CONFIG_PATH"])
    config.mkdir()
    shutil.copyfile(
        "/localhome/local-rarunachalam/cosmos-adapter-all-family-2node-20261111/"
        "libero-renderer-v1/installed-attempt02/config/config.yaml",
        config / "config.yaml",
    )
    (config / ".config_lock").touch()
    Path(os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"]).write_text(
        json.dumps({"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}})
    )
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import fixed_manifest

    manifests = Path(os.environ["LIBERO_MANIFEST_ROOT"])
    manifests.mkdir()
    for split in ("train", "eval"):
        fixed_manifest(split, "libero_spatial", "legacy_100").write(manifests / f"libero_spatial-{split}.json")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cosmos_framework.posttrain.model.gr00t.preflight",
            "--checkpoint",
            str(assets / "model"),
            "--output",
            str(output / "model-preflight.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            os.environ["POSTTRAIN_SIM_PYTHON"],
            "-m",
            "cosmos_framework.posttrain.embodied.sims.libero.preflight",
            "--train-manifest",
            str(manifests / "libero_spatial-train.json"),
            "--eval-manifest",
            str(manifests / "libero_spatial-eval.json"),
            "--output",
            str(output / "sim-preflight.json"),
        ],
        check=True,
    )
    command = [
        sys.executable,
        "-m",
        "cosmos_framework.posttrain.entrypoints.joint_ppo",
        "--config",
        "cosmos_framework/posttrain/configs/gr00t_n1d7_libero_quantized_optimizer.py",
    ]
    overrides = [
        f"model.checkpoint_path={assets / 'model'}",
        f"rl.max_steps={args.updates}",
        "rl.save_every=4",
        "rl.keep_optim_last=0",
        "rl.keep_model_last=0",
        "rl.validation.every_steps=4",
        "rl.seed=17",
        "rl.run_ts=integration",
        f"rl.optimizer.name={args.optimizer}",
        f"rl.optimizer.impl={'fused' if args.optimizer == 'adamw' else ''}",
        f"job.name=gr00t-master-{args.optimizer}",
    ]
    if args.resume:
        overrides.append(f"rl.resume={args.resume}")
    (output / "command.json").write_text(json.dumps(command + ["--", *overrides], indent=2))
    subprocess.run(command + ["--validate-manifest", "--", *overrides], check=True)
    import pynvml

    from tools.quantized_optimizers.run_deployment import monitor

    pynvml.nvmlInit()
    before = time.monotonic()
    with (output / "native.log").open("x") as log:
        process = subprocess.Popen(command + ["--", *overrides], stdout=log, stderr=subprocess.STDOUT)
        stop = threading.Event()
        monitors = []
        for index, visible in enumerate(os.environ["CUDA_VISIBLE_DEVICES"].split(",")):
            handle = (
                pynvml.nvmlDeviceGetHandleByIndex(int(visible))
                if visible.isdigit()
                else pynvml.nvmlDeviceGetHandleByUUID(visible)
            )
            thread = threading.Thread(target=monitor, args=(handle, process, output / f"memory-gpu{index}.jsonl", stop))
            thread.start()
            monitors.append(thread)
        try:
            code = process.wait()
        finally:
            stop.set()
            for thread in monitors:
                thread.join()
            pynvml.nvmlShutdown()
    (output / "attempt.json").write_text(json.dumps({"returncode": code, "native_seconds": time.monotonic() - before}))
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
