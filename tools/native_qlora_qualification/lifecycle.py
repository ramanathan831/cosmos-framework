# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Qualify native adapter interruption, exact continuation and portable reload.

Run inside an allocated, sealed single-GPU runtime. This tool submits no jobs and
contains no training loop: every update runs through the native SFT entrypoint.
Failure attempts and logs remain in the new output directory. Training horizon,
schedule and validation controls are identical within the four-update comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.native_qlora_qualification.processes import OwnedProcessTree
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    # Direct script invocation must not put the checkout root ahead of the sealed
    # installed cosmos_framework package. Load only this dependency-light sibling.
    import importlib.util

    _process_spec = importlib.util.spec_from_file_location(
        "native_qlora_processes", Path(__file__).with_name("processes.py")
    )
    _process_module = importlib.util.module_from_spec(_process_spec)
    _process_spec.loader.exec_module(_process_module)
    OwnedProcessTree = _process_module.OwnedProcessTree

INTEGRATION = "cosmos_framework.posttrain.tests.qlora_integration"
NATIVE = "cosmos_framework.posttrain.entrypoints.sft"
FIXED = {
    "rl.max_steps": 4,
    "rl.lr_schedule.cycle_steps": 4,
    "rl.optimizer.warmup_steps": 1,
    "rl.save_every": 1,
    "rl.keep_optim_last": 4,
    "rl.keep_model_last": 4,
    "rl.validation.enabled": True,
    "rl.validation.before_training": True,
    "rl.validation.every_steps": 2,
    "rl.validation.max_samples": 4,
    "rl.validation.save_checkpoint": True,
    "rl.deterministic": True,
    "rl.save_async": False,
    "rl.checkpoint_format": "dcp",
    "rl.checkpoint_store.backend": "local",
}
RESERVED = {"job.name", "job.group", "job.project", "rl.run_ts", "rl.resume", "rl.init_from"}
NUMERICAL_METRICS = (
    "train/loss_avg",
    "train/grad_norm",
    "train/learning_rate",
    "train/valid_token_count",
    "train/valid_sequence_count",
    "optim/optimizer_step",
    "optim/attempt_id",
    "optim/nonfinite_gradient",
    "optim/nonfinite_loss",
    "optim/nonfinite_skips_total",
    "optim/nonfinite_skips_consecutive",
    "optim/step_skipped",
    "optim/finite_gradient_check_enabled",
)


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def controlled_overrides(requested: list[str]) -> list[str]:
    """Refuse hidden identity/cadence changes instead of shadowing conflicting dotlists."""
    result, seen = [], set()
    for item in requested:
        key, separator, raw = item.partition("=")
        if not separator or key.startswith("+") or key in seen:
            raise ValueError(f"overrides require unique, explicit native keys: {key!r}")
        seen.add(key)
        if key in RESERVED:
            raise ValueError(f"qualification owns output and resume override {key!r}")
        if key in FIXED:
            try:
                value = json.loads(raw)
            except ValueError:
                value = raw
            if type(value) is not type(FIXED[key]) or value != FIXED[key]:
                raise ValueError(f"qualification fixes {key}={FIXED[key]!r}; conflicting override refused")
        else:
            result.append(item)
    return result + [f"{key}={json.dumps(value)}" for key, value in FIXED.items()]


def compare_next_metrics(reference: Path, candidate: Path, step: int) -> dict[str, Any]:
    def one(path: Path) -> dict[str, Any]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        selected = [row for row in rows if row.get("step") == step and "train/loss_avg" in row]
        if len(selected) != 1:
            raise ValueError(f"expected exactly one native update metric at step {step}")
        row = selected[0]
        if row.get("optim/update_outcome") != "applied":
            raise ValueError("qualification requires an applied native optimizer update")
        values = {key: row[key] for key in NUMERICAL_METRICS}
        if not all(type(v) in (int, float) and math.isfinite(v) for v in values.values()):
            raise ValueError("missing or nonfinite native numerical update metric")
        if any(values[key] != step for key in ("optim/optimizer_step", "optim/attempt_id")):
            raise ValueError("native optimizer and attempt clocks differ")
        if any(
            values[key] != 0
            for key in NUMERICAL_METRICS
            if key.startswith("optim/")
            and key not in {"optim/optimizer_step", "optim/attempt_id", "optim/finite_gradient_check_enabled"}
        ):
            raise ValueError("native continuation reported a skipped/nonfinite update")
        if values["optim/finite_gradient_check_enabled"] != 1:
            raise ValueError("finite-gradient checks must remain enabled")
        return values

    left, right = one(reference), one(candidate)
    if left != right:
        raise AssertionError("native next-update numerical metrics differ")
    return {
        "passed": True,
        "optimizer_step": step,
        "numerical_metrics": left,
        "excluded_observational_metrics": ["timer/*", "train/preprocessing_*", "train/grad_clip_fraction"],
        "scope": "Exact native numerical update metrics; cold-cache counters, timings and reporter-history clip fractions are observational. Checkpoint comparison separately verifies all optimizer/control/RNG/data state.",
    }


class Stages:
    def __init__(self, output: Path, environment: dict[str, str], timeout: int):
        self.process_capability = OwnedProcessTree.check_support()
        self.output, self.environment, self.timeout = output, environment, timeout
        (output / "stages").mkdir()
        write_json(output / "process-capability.json", self.process_capability)
        self.outcomes: list[dict[str, Any]] = []

    def run(
        self, name: str, command: list[str], *, interrupted: bool = False, interruption_receipt: Path | None = None
    ) -> Path:
        signal_dir = self.environment.get("SLURM_LOG_DIR")
        if signal_dir and (Path(signal_dir) / "SIGUSR1_RECEIVED").exists():
            raise RuntimeError("scheduler stop marker forbids beginning another qualification stage")
        folder = self.output / "stages" / name
        folder.mkdir()
        write_json(folder / "command.json", command)
        environment = {**self.environment, "COSMOS_QLORA_METRICS_FILE": str(folder / "memory.jsonl")}
        started = time.monotonic()
        child = None
        code = None
        tracker = None
        cleanup = None
        operation_error = None
        timeout = False
        try:
            with (folder / "output.log").open("x") as stream:
                child = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.STDOUT, env=environment, start_new_session=True
                )
                tracker = OwnedProcessTree(child.pid)
                try:
                    code = child.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    timeout = True
                    cleanup = tracker.cleanup()
                    tracker = None
                    code = child.wait()
                if interrupted and code != 0 and not timeout:
                    if interruption_receipt is None:
                        raise ValueError("interrupted stage requires its own rank receipt")
                    event = json.loads(interruption_receipt.read_text())
                    tracker.validate_interruption(event["process_snapshot"])
        except BaseException as error:
            operation_error = error
        finally:
            if tracker is not None:
                try:
                    cleanup = tracker.cleanup()
                except BaseException as error:
                    operation_error = error
                    cleanup = {"failed": True, "error_type": type(error).__name__}
        outcome = {
            "name": name,
            "exit_code": code,
            "elapsed_seconds": time.monotonic() - started,
            "timed_out": timeout,
            "injected_interruption_expected": interrupted,
            "owned_process_cleanup": cleanup,
        }
        write_json(folder / "outcome.json", outcome)
        self.outcomes.append(outcome)
        if operation_error is not None:
            raise RuntimeError(f"qualification stage {name!r} failed; see preserved log") from operation_error
        if timeout or (code == 0 if interrupted else code != 0):
            raise RuntimeError(f"qualification stage {name!r} returned an unexpected status; see preserved log")
        return folder


def run(args: argparse.Namespace) -> dict[str, Any]:
    from cosmos_framework.posttrain.config.build import select_policy_backend
    from cosmos_framework.posttrain.entrypoints.sft import validate_config_structure
    from cosmos_framework.posttrain.tests.qlora_integration import inspect_checkpoint
    from cosmos_framework.posttrain.utils.identity import canonical_config_value
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.utils.lazy_config import instantiate

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stages = None
    try:
        overrides = controlled_overrides(args.override)
        config = load_config(args.config, ["--", *overrides])
        rl = validate_config_structure(config)
        model = instantiate(select_policy_backend(config).model_config)
        if getattr(model, "adapter", None) is None or rl.grad_accum < 2:
            raise ValueError("recovery qualification requires native adapters and grad_accum>=2")
        receipt = {
            "code_revision": os.environ.get("COSMOS_TRAINING_CODE_REVISION", ""),
            "wheel_sha256": os.environ.get("COSMOS_TRAINING_WHEEL_SHA256", ""),
        }
        if not re.fullmatch("[a-f0-9]{40}", receipt["code_revision"]) or not re.fullmatch(
            "[a-f0-9]{64}", receipt["wheel_sha256"]
        ):
            raise ValueError("qualification requires the sealed build receipt environment")
        spec = {
            "protocol": "cosmos.native-qlora-recovery-spec.v1",
            "config": args.config,
            "requested_overrides": args.override,
            "qualification_overrides": overrides,
            "build_receipt": receipt,
            "resolved_model": canonical_config_value(model),
            "resolved_sft": canonical_config_value(rl),
            "selected": {
                "arm": "nf4" if model.quantized_base is not None else "bf16",
                "seed": rl.seed,
                "adapter_initialization_seed": model.adapter.initialization_seed,
                "lr": rl.optimizer.lr,
                "rank": model.adapter.rank,
                "alpha": model.adapter.alpha,
                "microbatch": rl.micro_batch_size,
                "grad_accum": rl.grad_accum,
                "processor_kwargs": canonical_config_value(rl.processor_kwargs),
                "quantization": canonical_config_value(model.quantized_base),
            },
            "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "exact_atol": 0,
            "exact_rtol": 0,
            "scope": "Four-update qualification; full quality training uses its independently recorded horizon and schedule.",
        }
        if args.spec and json.loads(Path(args.spec).read_text()) != spec:
            raise ValueError("immutable recovery experiment spec differs from resolved native controls")
        write_json(output / "experiment.json", spec)
        if args.plan_only:
            return {"complete": False, "plan_only": True, "experiment_sha256": digest(spec)}
        env = {**os.environ, "IMAGINAIRE_OUTPUT_ROOT": str(output / "native")}
        stages = Stages(output, env, args.stage_timeout)
        launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1", "-m"]
        roots = {}

        def native(name: str, *, mode: str | None = None, source: str = "off") -> tuple[Path, list[str]]:
            root = output / "native/qualification/recovery" / name / "canary"
            roots[name] = root
            opts = overrides + [
                "job.project=qualification",
                "job.group=recovery",
                f"job.name={name}",
                "rl.run_ts=canary",
                f"rl.resume={source}",
            ]
            if mode:
                marker = output / f"{name}-interruption.json"
                command = launcher + [
                    INTEGRATION,
                    "interrupt-native",
                    "--config",
                    args.config,
                    "--checkpoint-root",
                    str(root / "checkpoints"),
                    "--receipt",
                    str(marker),
                    "--mode",
                    mode,
                    "--after-step",
                    "2",
                ]
                command += [part for value in opts for part in ("--override", value)]
            else:
                command = launcher + [NATIVE, "--config", args.config, "--", *opts]
            stages.run(name, command, interrupted=mode is not None, interruption_receipt=marker if mode else None)
            if mode:
                event = json.loads(marker.read_text())
                expected = inspect_checkpoint(str(root / "checkpoints/iter_000002"), expected_step=2)
                if event.get("exit_code") != 86 or event.get("mode") != mode or event["checkpoint"] != expected:
                    raise ValueError("native interruption receipt/source differs from its committed checkpoint")
                if (root / "checkpoints/iter_000003").exists():
                    raise ValueError("interrupted run unexpectedly published its next update")
            return root, opts

        native("reference")
        proof = {}
        for mode in ("boundary", "accumulation"):
            root, _ = native(mode, mode=mode)
            source = str(root / "checkpoints/iter_000002")
            stages.run(
                mode + "_source_compare",
                [
                    sys.executable,
                    "-m",
                    INTEGRATION,
                    "compare-checkpoints",
                    "--reference",
                    str(roots["reference"] / "checkpoints/iter_000002"),
                    "--candidate",
                    source,
                    "--atol",
                    "0",
                    "--rtol",
                    "0",
                ],
            )
            resumed, _ = native(mode + "_resumed", source=source)
            comparisons = {}
            for step in (3, 4):
                record = inspect_checkpoint(str(resumed / f"checkpoints/iter_{step:06d}"), expected_step=step)
                folder = stages.run(
                    f"{mode}_compare_{step}",
                    [
                        sys.executable,
                        "-m",
                        INTEGRATION,
                        "compare-checkpoints",
                        "--reference",
                        str(roots["reference"] / f"checkpoints/iter_{step:06d}"),
                        "--candidate",
                        record["path"],
                        "--atol",
                        "0",
                        "--rtol",
                        "0",
                    ],
                )
                metrics = compare_next_metrics(roots["reference"] / "metrics.jsonl", resumed / "metrics.jsonl", step)
                write_json(folder / "numerical-update.json", metrics)
                comparisons[str(step)] = {"checkpoint": record, "metrics": metrics}
            proof[mode] = comparisons
        source = str(roots["accumulation_resumed"] / "checkpoints/iter_000004")
        portable_hint = output / "portable/checkpoints/iter_000004"
        stages.run(
            "publish_portable",
            [
                sys.executable,
                "-m",
                "cosmos_framework.posttrain.checkpoint.adapter",
                "--source",
                source,
                "--output-iter",
                str(portable_hint),
            ],
        )
        portable = output / "portable/checkpoint_artifacts/iter_000004"
        for kind, checkpoint in (("dcp", source), ("portable", str(portable))):
            command = launcher + [
                INTEGRATION,
                "probe",
                "--config",
                args.config,
                "--checkpoint",
                checkpoint,
                "--data",
                args.data,
                "--output",
                str(output / "probes" / kind),
                "--require-updates",
            ]
            image = (
                args.image
                if kind == "dcp"
                else str(Path(args.image).resolve())
                if args.image
                else str(output / "probes/dcp/video_first_frame.png")
            )
            if image:
                command += ["--image", image]
            command += [part for value in overrides for part in ("--override", value)]
            stages.run("probe_" + kind, command)
        stages.run(
            "compare_reload",
            [
                sys.executable,
                "-m",
                INTEGRATION,
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
        result = {
            "protocol": "cosmos.native-qlora-recovery-result.v1",
            "complete": True,
            "experiment_sha256": digest(spec),
            "build_receipt": receipt,
            "recovery": proof,
            "outcomes": stages.outcomes,
            "reload_exact": True,
            "release_quality_qualified": False,
        }
        write_json(output / "completion.json", result)
        return result
    except BaseException as error:
        write_json(
            output / "failure.json",
            {
                "complete": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "outcomes": stages.outcomes if stages else [],
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="New directory; failed attempts are never overwritten")
    parser.add_argument("--data", required=True)
    parser.add_argument("--image")
    parser.add_argument("--spec", help="Optional exact experiment.json from a reviewed identical plan")
    parser.add_argument(
        "--plan-only", action="store_true", help="Resolve and persist the immutable spec without launching"
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--stage-timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.stage_timeout < 1:
        parser.error("stage-timeout must be positive")
    result = run(args)
    print(
        json.dumps({"complete": result["complete"], "experiment_sha256": result["experiment_sha256"]}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
