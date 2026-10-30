# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Measure native training through portable publication and sealed-split evaluation.

This is an experiment orchestrator, not a training implementation. Every optimizer
update runs in cosmos_framework.posttrain.entrypoints.sft under torchrun. Invoke
inside a single-GPU scheduler allocation with the sealed posttrain environment.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from processes import OwnedProcessTree
from protocol import Experiment, file_sha256, write_json


def verify_build(build: Path) -> dict:
    receipt = json.loads((build / "receipt.json").read_text())
    wheel = build / receipt["wheel_file"]
    if file_sha256(wheel) != receipt["wheel_sha256"]:
        raise ValueError("sealed wheel digest differs")
    import cosmos_framework

    if Path(cosmos_framework.__file__).resolve().parent.parent != (build / "site").resolve():
        raise ValueError("framework must resolve to the sealed wheel overlay")
    count = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("cosmos_framework/") and not name.endswith("/"):
                if archive.read(name) != (build / "site" / name).read_bytes():
                    raise ValueError(f"installed package differs from sealed wheel: {name}")
                count += 1
    if count != receipt["package_files"]:
        raise ValueError("sealed package inventory differs")
    for name, expected in {"torch": "2.11.0+cu128", "transformers": "5.8.1", "bitsandbytes": "0.49.2"}.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"unsupported runtime version for {name}")
    return receipt


def verify_kernel_gate(path: Path, receipt: dict) -> dict:
    gate = json.loads(path.read_text())
    if gate.get("completed") is not True or gate.get("build") != receipt:
        raise ValueError("training requires the completed real-kernel gate for this exact build")
    for label, path_key, digest_key, count in (
        ("real CUDA", "junit_xml", "junit_sha256", 14),
        ("higher-precision multimodal", "higher_precision_junit_xml", "higher_precision_junit_sha256", 1),
    ):
        xml_path, expected_digest = gate.get(path_key), gate.get(digest_key)
        if not isinstance(xml_path, str) or not xml_path or not isinstance(expected_digest, str):
            raise ValueError(f"kernel gate is missing its {label} XML identity")
        try:
            junit = Path(xml_path)
            if file_sha256(junit) != expected_digest:
                raise ValueError(f"kernel gate {label} XML differs from its receipt")
            cases = list(ET.parse(junit).getroot().iter("testcase"))
        except (OSError, ET.ParseError) as error:
            raise ValueError(f"kernel gate {label} XML is unavailable or invalid") from error
        if len(cases) != count or any(
            case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")
        ):
            raise ValueError(f"kernel gate requires all {count} {label} tests without skips or failures")
    return gate


def preflight(output: Path) -> None:
    """Executed in a disposable process so the orchestrator retains no CUDA context."""
    import torch

    from cosmos_framework.posttrain.model.quantization.loader import runtime_precision_policy
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enable_determinism(strict=True)
    props = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (8, 0):
        raise ValueError("qualification requires one visible SM80 accelerator")
    if "A100" not in props.name or props.total_memory < 79_000_000_000:
        raise ValueError("this qualification is bounded to A100 80GB")
    topology = []
    for cpu in sorted(os.sched_getaffinity(0)):
        directory = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        topology.append(
            {
                "cpu": cpu,
                "package": int((directory / "topology/physical_package_id").read_text()),
                "core": int((directory / "topology/core_id").read_text()),
                "numa_nodes": sorted(int(path.name.removeprefix("node")) for path in directory.glob("node[0-9]*")),
            }
        )
    physical_cores = len({(row["package"], row["core"]) for row in topology})
    if physical_cores != 8:
        raise ValueError(f"controlled qualification requires eight physical CPU cores; got {physical_cores}")
    write_json(
        output,
        {
            "name": props.name,
            "device_bytes": props.total_memory,
            "uuid": "GPU-" + str(uuid.UUID(str(props.uuid).removeprefix("GPU-"))),
            "runtime_precision": runtime_precision_policy(),
            "torch_git_version": torch.version.git_version,
            "torch_cuda": torch.version.cuda,
            "python": sys.version,
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "bitsandbytes",
                    "torchcodec",
                    "torchvision",
                    "safetensors",
                    "accelerate",
                )
            },
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_topology": topology,
            "physical_cpu_cores": physical_cores,
            "allocation_id": os.environ.get("SLURM_JOB_ID"),
            "host_id_sha256": hashlib.sha256(os.uname().nodename.encode()).hexdigest(),
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM")
            },
        },
    )


def kernel_gate(args: argparse.Namespace) -> None:
    output = Path(args.output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    receipt = verify_build(Path(args.build))
    stages = Stages(output, Path(args.time_binary))
    stages.run(
        "preflight",
        [sys.executable, str(Path(__file__).resolve()), "preflight", "--output", str(output / "runtime.json")],
    )
    junit = output / "gpu-tests.xml"
    tests = Path(args.build) / "site/cosmos_framework/posttrain/tests"
    stages.run(
        "real_cuda_kernels",
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "-c",
            "/dev/null",
            "-q",
            f"--junitxml={junit}",
            "-o",
            f"cache_dir={output / 'pytest-cache'}",
            *[
                str(tests / name)
                for name in (
                    "test_model_qwen3_vl_qlora_gpu.py",
                    "test_native_adapter_execution_gpu.py",
                    "test_sft_selected_logits_gpu.py",
                )
            ],
        ],
    )
    gate = {
        "completed": True,
        "build": receipt,
        "junit_xml": str(junit),
        "junit_sha256": file_sha256(junit),
        "outcomes": stages.outcomes,
    }
    # Verify before publication: an XML containing skips cannot become a gate receipt.
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    if len(cases) != 14 or any(case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")):
        raise RuntimeError("expected all 14 real CUDA tests without skips")
    cpu_junit = output / "higher-precision-tests.xml"
    stages.run(
        "higher_precision_multimodal_regression",
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "-c",
            "/dev/null",
            "-q",
            f"--junitxml={cpu_junit}",
            "-o",
            f"cache_dir={output / 'pytest-cache'}",
            str(tests / "test_entrypoints_sft.py") + "::test_real_qwen_multimodal_recipe_cpu",
        ],
        env_extra={"CUDA_VISIBLE_DEVICES": ""},
    )
    cpu_cases = list(ET.parse(cpu_junit).getroot().iter("testcase"))
    if len(cpu_cases) != 1 or any(
        case.find(tag) is not None for case in cpu_cases for tag in ("skipped", "failure", "error")
    ):
        raise RuntimeError("existing higher-precision multimodal regression must pass without skips")
    gate.update(higher_precision_junit_xml=str(cpu_junit), higher_precision_junit_sha256=file_sha256(cpu_junit))
    write_json(output / "completion.json", gate)


class Stages:
    def __init__(self, output: Path, time_binary: Path, *, timeout_seconds: int = 7200):
        self.output, self.time_binary, self.timeout = output, time_binary, timeout_seconds
        self.outcomes: list[dict] = []
        self.work = output / "work"
        self.work.mkdir()
        (output / "stages").mkdir()
        self.marker = output / "current-stage.txt"
        self.marker.write_text("initialization\n")

    def run(self, name: str, command: list[str], *, env_extra: dict | None = None) -> None:
        stage = self.output / "stages" / name
        stage.mkdir()
        self.marker.write_text(name + "\n")
        write_json(stage / "command.json", command)
        env = dict(os.environ, COSMOS_QLORA_METRICS_FILE=str(stage / "memory.jsonl"), **(env_extra or {}))
        started = time.monotonic()
        with (stage / "output.log").open("x") as stream:
            child = subprocess.Popen(
                [str(self.time_binary), "-v", "-o", str(stage / "time.txt"), *command],
                cwd=self.work,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            tree = OwnedProcessTree(child.pid)
            try:
                status = child.wait(timeout=self.timeout)
            except BaseException:
                tree.cleanup()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                write_json(stage / "interrupted.json", {"elapsed_seconds": time.monotonic() - started})
                raise
            finally:
                tree.cleanup()
        result = {"name": name, "exit_code": status, "elapsed_seconds": time.monotonic() - started}
        write_json(stage / "outcome.json", result)
        self.outcomes.append(result)
        print(json.dumps(result), flush=True)
        if status:
            raise RuntimeError(f"stage {name} exited {status}; failed evidence retained")


def final_checkpoint(output: Path, experiment: Experiment, receipt: dict) -> Path:
    from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager

    manager = CheckpointManager(checkpoint_format="dcp")
    candidates = list((output / "native").rglob(f"checkpoints/iter_{experiment.steps:06d}"))
    if len(candidates) != 1:
        raise ValueError("expected exactly one own-run final checkpoint")
    checkpoint = candidates[0]
    # read_adapter_metadata verifies the atomic publication/manifest contract.
    metadata = read_adapter_metadata(str(checkpoint), require_resume=True, manager=manager)
    controls = manager.load_component(str(checkpoint), "trainer")
    if any(controls[key] != experiment.steps for key in ("optimizer_step", "iter", "attempt_id")):
        raise ValueError("training did not complete all scheduled updates")
    if controls["accum"] or controls["nonfinite_skips_total"] or controls.get("zero_grad_skips_total", 0):
        raise ValueError("final checkpoint has partial accumulation or skipped updates")
    validation = controls["validation"]
    if validation["progress"]["pending_step"] is not None or validation["publication"] is not None:
        raise ValueError("final validation/publication is unfinished")
    identity = metadata["identity"]
    for key in ("code_revision", "wheel_sha256"):
        if identity["training_contract"][key] != receipt[key]:
            raise ValueError("checkpoint build identity differs")
    if (identity["quantization"] is not None) != (experiment.arm == "nf4"):
        raise ValueError("checkpoint base format differs from experiment")
    write_json(
        output / "final-checkpoint.json",
        {
            "checkpoint": str(checkpoint),
            "optimizer_step": experiment.steps,
            "metadata": metadata,
            "provenance": manager.checkpoint_provenance(str(checkpoint)),
        },
    )
    return checkpoint


def execute(args: argparse.Namespace) -> None:
    experiment = Experiment.read(args.spec)
    output = Path(args.output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = verify_build(Path(args.build))
    verify_kernel_gate(Path(args.kernel_gate), receipt)
    if os.environ.get("PYTHONHASHSEED") != str(experiment.seed):
        raise ValueError("PYTHONHASHSEED must match the experiment before process start")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic cuBLAS environment must precede process start")
    os.environ.update(
        COSMOS_TRAINING_CODE_REVISION=receipt["code_revision"],
        COSMOS_TRAINING_WHEEL_SHA256=receipt["wheel_sha256"],
        IMAGINAIRE_OUTPUT_ROOT=str(output / "native"),
        SFT_MODEL_PATH=args.model,
        SFT_DATA_ROOT=args.data_root,
    )
    os.environ.pop("COSMOS_SFT_PROFILE_DIR", None)
    common = experiment.overrides()
    script = Path(__file__).resolve()
    torchrun = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1"]
    provenance_files = [
        Path(args.spec),
        Path(args.build) / "receipt.json",
        Path(args.split_manifest),
        Path(args.score_data),
        Path(args.data_root) / "train.jsonl",
        Path(args.data_root) / "validation.jsonl",
        Path(args.kernel_gate),
    ]
    write_json(
        output / "contract.json",
        {
            "protocol": "cosmos.native-qlora-experiment.v1",
            "experiment": experiment.as_dict(),
            "action": args.action,
            "split": args.split,
            "build": receipt,
            "overrides": common,
            "files": [{"path": str(p), "sha256": file_sha256(p)} for p in provenance_files],
            "tools": {p.name: file_sha256(p) for p in script.parent.glob("*.py")},
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "checkpoint": args.checkpoint,
            "preflight_measurement": "separately timed before external sampler",
        },
    )
    stages = Stages(output, Path(args.time_binary))
    sampler = None
    passed = False
    failure = None
    try:
        stages.run("preflight", [sys.executable, str(script), "preflight", "--output", str(output / "runtime.json")])
        runtime = json.loads((output / "runtime.json").read_text())
        os.environ["COSMOS_QLORA_GPU_UUID"] = runtime["uuid"]
        with (output / "sampler.log").open("x") as log:
            sampler = subprocess.Popen(
                [
                    sys.executable,
                    str(script.parent / "monitor.py"),
                    str(os.getpid()),
                    str(output / "nvml-host.jsonl"),
                    str(stages.marker),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        if args.action == "train-eval":
            if checkpoint is not None:
                raise ValueError("fresh experiment cannot inherit another run's checkpoint")
            stages.run(
                "training",
                [
                    *torchrun,
                    "-m",
                    "cosmos_framework.posttrain.entrypoints.sft",
                    "--config",
                    experiment.recipe,
                    "--",
                    *common,
                    "rl.resume=off",
                ],
            )
            checkpoint = final_checkpoint(output, experiment, receipt)
        integration = "cosmos_framework.posttrain.tests.qlora_integration"
        probe_overrides = [part for item in common for part in ("--override", item)]
        if checkpoint is not None:
            portable_iter = output / "portable" / "checkpoints" / f"iter_{experiment.steps:06d}"
            stages.run(
                "portable_publication",
                [
                    sys.executable,
                    "-m",
                    "cosmos_framework.posttrain.checkpoint.adapter",
                    "--source",
                    str(checkpoint),
                    "--output-iter",
                    str(portable_iter),
                ],
            )
            portable = output / "portable" / "checkpoint_artifacts" / portable_iter.name
            if not portable.is_dir():
                raise ValueError("portable adapter publication missing")
            for name, location in (("dcp", checkpoint), ("portable", portable)):
                command = [
                    *torchrun,
                    "-m",
                    integration,
                    "probe",
                    "--config",
                    experiment.recipe,
                    "--checkpoint",
                    str(location),
                    "--data",
                    str(Path(args.data_root) / "validation.jsonl"),
                    "--output",
                    str(output / "probes" / name),
                    "--require-updates",
                    *probe_overrides,
                ]
                if name == "portable":
                    command += ["--image", str(output / "probes/dcp/video_first_frame.png")]
                stages.run("probe_" + name, command)
            stages.run(
                "reload_exact",
                [
                    sys.executable,
                    "-m",
                    integration,
                    "compare-probes",
                    "--reference",
                    str(output / "probes/dcp"),
                    "--candidate",
                    str(output / "probes/portable"),
                    "--atol",
                    "0",
                    "--rtol",
                    "0",
                ],
            )
        score = [
            *torchrun,
            str(script.parent / "score.py"),
            "--config",
            experiment.recipe,
            "--data",
            args.score_data,
            "--output",
            str(output / "scores"),
            "--split-manifest",
            args.split_manifest,
            "--split-manifest-sha256",
            file_sha256(args.split_manifest),
            "--split",
            args.split,
            "--all-rows",
            *probe_overrides,
        ]
        if args.selection_receipt:
            score += [
                "--selection-receipt",
                args.selection_receipt,
                "--selection-receipt-sha256",
                file_sha256(args.selection_receipt),
            ]
        if checkpoint is not None:
            score += ["--checkpoint", str(checkpoint)]
        stages.run("scoring", score)
        if sampler.poll() is not None:
            raise RuntimeError("external sampler failed; complete resource evidence unavailable")
        passed = True
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        if sampler is not None:
            sampler.terminate()
            try:
                sampler.wait(timeout=15)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
        write_json(
            output / "completion.json",
            {
                "completed": passed,
                "failure": failure,
                "experiment": experiment.as_dict(),
                "outcomes": stages.outcomes,
                "complete_elapsed_seconds": time.monotonic() - started,
                "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "quality_qualified": False,
                "quality_gate_requires_external_paired_analysis": True,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("preflight")
    command.add_argument("--output", required=True)
    command = commands.add_parser("kernel-gate")
    for name in ("output", "build", "time-binary"):
        command.add_argument("--" + name, required=True)
    command = commands.add_parser("experiment")
    for name in (
        "spec",
        "output",
        "build",
        "model",
        "data-root",
        "score-data",
        "split-manifest",
        "time-binary",
        "kernel-gate",
    ):
        command.add_argument("--" + name, required=True)
    command.add_argument("--action", choices=("train-eval", "evaluate", "frozen"), default="train-eval")
    command.add_argument(
        "--split", choices=("development", "final", "regression", "untouched_diagnostic"), default="development"
    )
    command.add_argument("--checkpoint")
    command.add_argument("--selection-receipt")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(Path(args.output))
    elif args.command == "kernel-gate":
        kernel_gate(args)
    else:
        if (args.action == "evaluate") != bool(args.checkpoint):
            parser.error("evaluate requires a checkpoint; other actions forbid one")
        execute(args)


if __name__ == "__main__":
    main()
