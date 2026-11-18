"""Run the actual four-GPU Joint PPO entry with existing approved LIBERO assets.

This first eight-window run is an integration gate, not task-quality selection.
Native evaluation, source validation, rollout and checkpoint paths stay enabled.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


@contextmanager
def workflow_memory(output):
    """Sample the launcher and all children, including preflight and restoration."""
    import pynvml

    from tools.quantized_optimizers.run_deployment import monitor

    pynvml.nvmlInit()
    stop, threads = threading.Event(), []
    (output / "memory-scope.json").write_text(
        json.dumps({"root_pid": os.getpid(), "scope": "launcher_and_all_children_before_preflight_through_delivery"})
    )
    try:
        for index, visible in enumerate(os.environ["CUDA_VISIBLE_DEVICES"].split(",")):
            handle = (
                pynvml.nvmlDeviceGetHandleByIndex(int(visible))
                if visible.isdigit()
                else pynvml.nvmlDeviceGetHandleByUUID(visible)
            )
            thread = threading.Thread(
                target=monitor,
                args=(handle, SimpleNamespace(pid=os.getpid()), output / f"memory-gpu{index}.jsonl", stop),
            )
            thread.start()
            threads.append(thread)
        yield
    finally:
        stop.set()
        for thread in threads:
            thread.join()
        pynvml.nvmlShutdown()


def checked_phase(command, output, phase, **kwargs):
    """Sequential launcher phases; native_seconds contains its own nested timers."""
    started, start_ns, passed = time.monotonic(), time.time_ns(), False
    try:
        subprocess.run(command, check=True, **kwargs)
        passed = True
    finally:
        with (output / "launcher-phases.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {"phase": phase, "start_ns": start_ns, "seconds": time.monotonic() - started, "passed": passed}
                )
                + "\n"
            )


SUITE_CONFIGS = {
    "libero_spatial": "cosmos_framework/posttrain/configs/gr00t_n1d7_libero_quantized_optimizer.py",
    "libero_10": "cosmos_framework/posttrain/configs/gr00t_n1d7_libero10_quantized_optimizer.py",
}
# LIBERO-10 on its unchanged native Joint PPO profile (8 envs, 4 updates per 128-decision window).
LIBERO10_NATIVE_CONFIG = "cosmos_framework/posttrain/configs/gr00t_n1d7_libero10_quantized_optimizer_native.py"


def model_directory(args, assets):
    """Spatial keeps its curated approved copy; LIBERO-10 must name the pinned snapshot subfolder."""
    if args.suite == "libero_spatial":
        if args.model_dir:
            raise ValueError("--model-dir is only for LIBERO-10; Spatial uses its approved curated copy")
        return assets / "model"
    if not args.model_dir:
        raise ValueError("LIBERO-10 requires --model-dir <snapshot>/<revision>/libero_10")
    from cosmos_framework.posttrain.model.gr00t.build import _validate_libero10_checkpoint

    directory = Path(args.model_dir)
    _validate_libero10_checkpoint(str(directory))
    return directory


def source_probe(assets, output, model=None):
    """Load the real native model on CPU before admitting expensive GPU setup."""
    from collections import Counter

    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_model_config
    from cosmos_framework.posttrain.model.gr00t.policy import BF16_PRECISION, build_official_gr00t_model, load_processor
    from cosmos_framework.posttrain.model.gr00t.source import load_ppo_source

    directory = output / "model-config"
    directory.mkdir()

    def load(source):
        model = build_official_gr00t_model(
            str(source.checkpoint),
            save_cfg_dir=directory,
            revision=source.backbone_revision,
            precision=BF16_PRECISION,
            local_files_only=True,
            ppo=True,
            backbone_config=str(source.backbone),
        )
        processor = load_processor(str(source.checkpoint), model_name=str(source.processor))
        return model, processor

    before = time.monotonic()
    model_dir = assets / "model" if model is None else model
    config = make_model_config(str(model_dir), 8 if model is not None else 16)
    config.backbone_from_checkpoint = True
    (model, processor), identity = load_ppo_source(config, load)
    # Every persisted tensor must come from the approved full checkpoint. This
    # probe does not accept a successful constructor as proof of materialization.
    import torch
    from safetensors import safe_open

    state = model.state_dict()
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    checked = set()
    for filename in sorted(set(weight_map.values())):
        with safe_open(model_dir / filename, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                expected = shard.get_tensor(name)
                actual = state[name]
                if actual.is_meta or not torch.equal(actual, expected.to(actual.dtype)):
                    raise RuntimeError(f"native GR00T checkpoint tensor differs: {name}")
                checked.add(name)
    optional = {name for name in state if "mask_token" in name}
    if set(state) - checked - optional:
        raise RuntimeError("native GR00T has unrestored persistent state")
    for name, value in model.named_buffers():
        if value.is_meta or (value.is_floating_point() and not torch.isfinite(value).all()):
            raise RuntimeError(f"native GR00T has invalid derived buffer: {name}")
    inventory = Counter()
    for _, parameter in model.named_parameters():
        inventory[f"{'trainable' if parameter.requires_grad else 'frozen'}_{parameter.dtype}"] += parameter.numel()
    (output / "source-probe.json").write_text(
        json.dumps(
            {
                "scope": "CPU native model/processor source load only; no training or GPU qualification",
                "seconds": time.monotonic() - before,
                "source": identity,
                "parameter_elements": dict(inventory),
                "processor": type(processor).__name__,
                "exact_checkpoint_tensors": len(checked),
                "construction": "complete-checkpoint-sdpa-v1",
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=("adamw", "adamw_bnb_8bit", "paged_adamw_8bit"), default="adamw")
    parser.add_argument("--resume", default="")
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--validation-every", type=int, default=4)
    parser.add_argument("--co-resident-gib", type=int, default=0, help="Synthetic paging stress only; GiB per GPU")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--suite", choices=tuple(SUITE_CONFIGS), default="libero_spatial")
    parser.add_argument("--eval-profile", default="legacy_100")
    parser.add_argument("--model-dir", default="", help="LIBERO-10 only: pinned snapshot libero_10 directory")
    parser.add_argument("--recipe-profile", choices=("public_4gpu", "native"), default="public_4gpu")
    parser.add_argument(
        "--override", action="append", default=[], help="LIBERO-10 only: predeclared recipe override, e.g. rl.optimizer.lr=1e-6"
    )
    args = parser.parse_args()
    if min(args.updates, args.checkpoint_every, args.validation_every) < 1:
        parser.error("updates and checkpoint/validation intervals must be positive")
    if args.co_resident_gib < 0:
        parser.error("co-resident GiB must be nonnegative")
    if args.evaluate_only and not args.resume:
        parser.error("--evaluate-only requires --resume")
    if args.suite == "libero_spatial" and (
        args.eval_profile != "legacy_100" or args.model_dir or args.recipe_profile != "public_4gpu"
    ):
        parser.error("LIBERO-Spatial keeps its legacy_100 profile and approved model copy")
    root, output = Path(os.environ["RESOURCE_ROOT"]), Path(os.environ["RESULTS"])
    if args.source_only:
        run(args, root, output)
    else:
        with workflow_memory(output):
            run(args, root, output)


def native_command(args, model, manifests):
    """Native Joint PPO command and overrides; Spatial output is pinned by a golden test."""
    suite = args.suite
    command = [
        sys.executable,
        "-m",
        "cosmos_framework.posttrain.entrypoints.joint_ppo",
        "--config",
        LIBERO10_NATIVE_CONFIG if getattr(args, "recipe_profile", "public_4gpu") == "native" else SUITE_CONFIGS[suite],
    ]
    if args.evaluate_only:
        command.append("--evaluate-only")
    overrides = [
        f"model.checkpoint_path={model}",
        f"rl.max_steps={args.updates}",
        f"rl.save_every={args.checkpoint_every}",
        "rl.keep_optim_last=0",
        "rl.keep_model_last=0",
        f"rl.validation.every_steps={args.validation_every}",
        f"rl.seed={args.seed}",
        "rl.run_ts=integration",
        f"rl.optimizer.name={args.optimizer}",
        f"rl.optimizer.impl={'fused' if args.optimizer == 'adamw' else ''}",
        f"job.name={'gr00t' if suite == 'libero_spatial' else 'gr00t-libero10'}-master-{args.optimizer}",
    ]
    if suite != "libero_spatial":
        overrides += [
            f"environment.config.eval_profile={args.eval_profile}",
            f"environment.config.train_manifest={manifests / f'{suite}-train.json'}",
            f"environment.config.eval_manifest={manifests / f'{suite}-eval.json'}",
        ]
    extra = list(getattr(args, "override", []) or [])
    if extra and suite == "libero_spatial":
        raise ValueError("--override is only for LIBERO-10 development trials")
    for item in extra:
        key = item.split("=", 1)[0]
        allowed = key.startswith(("rl.", "environment.config.")) or key in ("model.value_feature_gradient", "model.ppo_reference_kl_beta")
        if "=" not in item or not allowed or key.startswith("rl.resume"):
            raise ValueError(f"unsupported recipe override {item!r}")
    overrides += extra
    if args.resume:
        overrides.append(f"rl.resume={args.resume}")
    return command, overrides


def run(args, root, output):
    # Allocation identity is needed before interpreting timing differences across
    # scheduler-selected nodes. This records no environment or credential values.
    cpu_models = sorted(
        {
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        }
    )
    (output / "host-placement.json").write_text(
        json.dumps(
            {
                "hostname": platform.node(),
                "kernel": platform.release(),
                "architecture": platform.machine(),
                "cpu_models": cpu_models,
                "launcher_cpu_affinity": sorted(os.sched_getaffinity(0)),
                "numa_cpu_lists": {
                    path.parent.name: path.read_text().strip()
                    for path in sorted(Path("/sys/devices/system/node").glob("node*/cpulist"))
                },
                "scope": "allocation launcher; individual native worker affinity may differ",
            },
            indent=2,
        )
    )
    expected = {"numpy": "1.26.4", "transformers": "4.57.6", "diffusers": "0.39.0", "bitsandbytes": "0.49.2"}
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
    model = model_directory(args, assets)
    if args.source_only:
        source_probe(assets, output, None if args.suite == "libero_spatial" else model)
        return
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
    suite = args.suite
    for split in ("train", "eval"):
        # Training partitions are profile-independent except shared_500, which binds both splits.
        profile = args.eval_profile if split == "eval" or args.eval_profile == "shared_500" else "legacy_100"
        fixed_manifest(split, suite, profile).write(manifests / f"{suite}-{split}.json")
    checked_phase(
        [
            sys.executable,
            "-m",
            "cosmos_framework.posttrain.model.gr00t.preflight",
            "--checkpoint",
            str(model),
            "--output",
            str(output / "model-preflight.json"),
        ],
        output,
        "model_preflight",
    )
    checked_phase(
        [
            os.environ["POSTTRAIN_SIM_PYTHON"],
            "-m",
            "cosmos_framework.posttrain.embodied.sims.libero.preflight",
            "--train-manifest",
            str(manifests / f"{suite}-train.json"),
            "--eval-manifest",
            str(manifests / f"{suite}-eval.json"),
            "--output",
            str(output / "sim-preflight.json"),
        ],
        output,
        "simulator_preflight",
        env={**os.environ, "PYTHONPATH": os.environ["POSTTRAIN_SIM_PYTHONPATH"]},
    )
    command, overrides = native_command(args, model, manifests)
    (output / "command.json").write_text(json.dumps(command + ["--", *overrides], indent=2))
    checked_phase(command + ["--validate-manifest", "--", *overrides], output, "manifest_validation")
    from tools.quantized_optimizers.device_reservation import ResidentDeviceMemory

    with ResidentDeviceMemory(args.co_resident_gib, output) as reservation:
        before = time.monotonic()
        with (output / "native.log").open("x") as log:
            process = subprocess.Popen(command + ["--", *overrides], stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    reservation.check()
                    try:
                        code = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                reservation.check()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        native_seconds = time.monotonic() - before
    (output / "attempt.json").write_text(json.dumps({"returncode": code, "native_seconds": native_seconds}))
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
