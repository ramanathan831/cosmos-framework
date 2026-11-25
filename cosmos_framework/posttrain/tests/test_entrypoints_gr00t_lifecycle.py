# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T entry lifecycle with real local collection, optimization and checkpoint state."""

import gc
import json
import logging
import signal
import sys
import threading
import time
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_config
from cosmos_framework.posttrain.embodied.sims.libero.protocol import fixed_manifest
from cosmos_framework.posttrain.entrypoints import joint_ppo as gr00t_ppo
from cosmos_framework.posttrain.roles.env.ppo_window import PPOWindowCollector
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.tests.helpers.ppo_continuous import EvaluationIntegratorGateway, GaussianPolicy
from cosmos_framework.posttrain.utils.entry import RunAttempt, resolve_all
from cosmos_framework.posttrain.utils.log import logger
from cosmos_framework.posttrain.utils.report import ConsoleReporter, make_reporters
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def local_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ray

    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.loop import weight_sync
    from cosmos_framework.posttrain.utils import entry, mesh, report

    events, allocated = [], []
    config = make_config()
    config.model.checkpoint_path, config.environment.config.sim_python, config.job.wandb_mode = (
        "gaussian",
        "python",
        "disabled",
    )
    config.rl = JointPPOConfig(
        num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1, max_steps=4, save_every=0
    )
    config.rl.validation.enabled = False
    for split in ("train", "eval"):
        path = tmp_path / f"{split}.json"
        fixed_manifest(split).write(path)
        setattr(config.environment.config, f"{split}_manifest", str(path))
    attempt = {"output": tmp_path / "first", "resume": None}

    class Gateway(EvaluationIntegratorGateway):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            self.stream = (tmp_path / f"{name}-resource").open("w")

        def close(self) -> None:
            events.append(self.name)
            self.stream.close()
            super().close()

    class Collector(PPOWindowCollector):
        def close(self) -> None:
            events.append("collector")
            super().close()

    class Rollout(GaussianPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.stream = (tmp_path / "rollout-resource").open("w")

        def close(self) -> None:
            events.append("rollout")
            self.stream.close()

    class RecordingReporter(ConsoleReporter):
        def finish(self) -> None:
            events.append("reporters")
            super().finish()

    def assemble(config, recipe, output, resume, width, *, evaluation=None):
        learner = JointPPOTrainer(GaussianPolicy(), recipe)
        rollout, gateway = Rollout(), Gateway("sim")
        evaluation_enabled = recipe.validation.enabled if evaluation is None else evaluation
        evaluation_gateway = Gateway("eval_sim") if evaluation_enabled else None
        from cosmos_framework.posttrain.control.buffer import ExperienceBuffer
        from cosmos_framework.posttrain.roles.reward.worker import RewardWorker
        from cosmos_framework.utils.lazy_config import instantiate

        environment = instantiate(config.environment)
        collector = Collector(recipe, 0, 2, environment=environment)
        reward = RewardWorker([instantiate(config.reward)], training_group_size=1)
        replay = ExperienceBuffer(exact_batch_size=recipe.decisions)
        if resume:
            _, provenance = learner.checkpoint_manager.load_checkpoint(resume, lambda: learner.load(resume))
            learner.set_loaded_checkpoint_provenance(provenance)
            collector.restore_checkpoint(learner.checkpoint_manager, resume)
        if recipe.validation.enabled:
            learner.configure_validation(gr00t_ppo.validation_identity(config))
        allocated.append((learner, collector, rollout, gateway, evaluation_gateway))
        units = {
            "collector": WorkerGroup("collector", "env", body=collector),
            "rollout": WorkerGroup("rollout", "rollout", body=rollout),
            "sim": WorkerGroup("sim", "env", body=gateway),
            "policy": WorkerGroup("policy", "trainer", body=learner),
            "rm": WorkerGroup("rm", "reward", body=reward),
            "replay": WorkerGroup("replay", "buffer", body=replay),
        }
        if evaluation_gateway is not None:
            units["eval_sim"] = WorkerGroup("eval_sim", "env", body=evaluation_gateway)
        resolve_all(units["collector"].bind_rollout(units["rollout"]))
        return units

    def publish(rollout, policy, *, validation_step):
        rollout.body.net.load_state_dict(policy.body.model.net.state_dict())
        rollout.body.version = validation_step

    finalize = CheckpointLifecycle.finalize_units
    shutdown = ray.shutdown

    def finalize_units(units):
        events.append("checkpoint")
        finalize(units)

    def shutdown_ray():
        events.append("ray")
        shutdown()

    monkeypatch.setenv("WANDB_MODE", "disabled")
    # This fixture owns one local learner/collector with two environment lanes;
    # it must not advertise a second collector absent from its saved inventory.
    monkeypatch.setattr(mesh, "detect_mesh", lambda: (1, 1))
    # assemble below uses the Gaussian fixture, not the sharded GR00T model.
    config.recipe_check = None
    monkeypatch.setattr(entry, "pin_run_attempt", lambda *_: RunAttempt(str(attempt["output"]), (), attempt["resume"]))
    monkeypatch.setattr(gr00t_ppo, "assemble", assemble)
    monkeypatch.setattr(weight_sync, "weight_sync", publish)
    monkeypatch.setattr(report, "make_reporters", lambda *_args, **_kwargs: [RecordingReporter()])
    monkeypatch.setattr(CheckpointLifecycle, "finalize_units", finalize_units)
    monkeypatch.setattr(ray, "shutdown", shutdown_ray)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        yield config, attempt, events, allocated
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for _, _, rollout, gateway, evaluation_gateway in allocated:
            # Close even when a failed assertion interrupts the test itself.
            rollout.stream.close()
            gateway.stream.close()
            if evaluation_gateway is not None:
                evaluation_gateway.stream.close()


def test_window_counter_is_per_process_and_preserves_durable_window_rows(
    local_run, one_logger_sink, monkeypatch
) -> None:
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle

    config, attempt, events, allocated = local_run
    coordinators = []
    retained_telemetry = []
    build_coordinator = CheckpointLifecycle.coordinator

    def capture(*args, **kwargs):
        coordinator = build_coordinator(*args, **kwargs)
        coordinators.append(weakref.ref(coordinator))
        retained_telemetry.append(kwargs["observer"])
        return coordinator

    monkeypatch.setattr(CheckpointLifecycle, "coordinator", capture)
    first = gr00t_ppo.run(config)
    # Discard unrelated unrooted frames while deliberately keeping telemetry alive:
    # a strong checkpoint callback would still retain its coordinator through this root.
    gc.collect()
    assert len(retained_telemetry) == len(coordinators) == 1 and coordinators[0]() is None
    assert first["windows"] == 2 and first["optimizer_step"] == 4 and first["publication_version"] == 2
    counters = ("train_iterations_start", "train_iterations_end", "train_samples_start", "train_samples_end")
    assert tuple(one_logger_sink[0].store[key] for key in counters) == (
        0,
        first["publication_version"],
        0,
        first["windows"] * config.rl.decisions,
    )
    rows = [json.loads(line) for line in (attempt["output"] / "windows.jsonl").read_text().splitlines()]
    assert [row["optimizer_step"] for row in rows] == [2, 4]
    for row in rows:
        assert all(value >= 0 for value in row["phase_seconds"].values())
        assert sum(row["phase_seconds"].values()) == pytest.approx(row["seconds"])
    setup = json.loads((attempt["output"] / "native-setup.json").read_text())
    assert setup["seconds"] >= 0
    checkpoint_metrics = [frame for frame in one_logger_sink[0].metrics if "joint_ppo_checkpoint_path" in frame]
    assert checkpoint_metrics == [
        {
            "joint_ppo_checkpoint_path": str(attempt["output"] / "checkpoints/iter_000004"),
            "joint_ppo_checkpoint_optimizer_step": first["optimizer_step"],
        }
    ]
    assert events[-6:] == ["checkpoint", "reporters", "sim", "collector", "rollout", "ray"]
    attempt["resume"] = str(attempt["output"] / "checkpoints/iter_000004")
    attempt["output"] = attempt["output"].parent / "restored"
    config.rl.max_steps = 6
    second = gr00t_ppo.run(config)
    assert second["windows"] == 1 and second["optimizer_step"] == 6 and second["publication_version"] == 3
    assert len(one_logger_sink) == 2 and all(sink.finishes == 1 for sink in one_logger_sink)
    assert tuple(one_logger_sink[1].store[key] for key in counters) == (
        first["publication_version"],
        second["publication_version"],
        first["publication_version"] * config.rl.decisions,
        second["publication_version"] * config.rl.decisions,
    )
    assert all(rollout.stream.closed and gateway.stream.closed for _, _, rollout, gateway, _ in allocated)


@pytest.mark.parametrize("failure", ["counter", "loop", "collector_close"])
def test_training_cleanup_covers_early_and_late_failures(
    local_run, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    config, _, events, allocated = local_run
    assemble = gr00t_ppo.assemble

    def inject(*args, **kwargs):
        units = assemble(*args, **kwargs)
        collector, policy = units["collector"], units["policy"]
        if failure == "counter":
            policy.body.progress = lambda: {"optimizer_step": 1, "publication_version": 0}
        elif failure == "loop":

            def fail_window(version):
                raise RuntimeError("injected collection failure")

            collector.body.begin_window = fail_window
        elif failure == "collector_close":
            close = collector.body.close

            def fail_close():
                close()
                raise RuntimeError("injected close failure")

            collector.body.close = fail_close
        return units

    monkeypatch.setattr(gr00t_ppo, "assemble", inject)
    with pytest.raises((RuntimeError, ValueError)) as retained:
        gr00t_ppo.run(config)
    assert retained.value.__traceback__ is not None
    expected = ["checkpoint", "sim", "collector", "rollout", "ray"]
    if failure in ("loop", "collector_close"):
        expected.insert(1, "reporters")
    assert events[-len(expected) :] == expected
    assert all(rollout.stream.closed and gateway.stream.closed for _, _, rollout, gateway, _ in allocated)


def test_assessment_cleanup_and_complete_reentry_leave_training_untouched(local_run) -> None:
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import LiberoCase, LiberoTrial, LiberoTrialManifest
    from cosmos_framework.posttrain.tests.libero_assessment_cli import run_assessment

    config, attempt, events, allocated = local_run
    directory = attempt["output"].parent
    manifest = directory / "trials.json"
    LiberoTrialManifest("libero_spatial", (LiberoTrial(LiberoCase(3, 1), 7),)).write(manifest)
    result = run_assessment(config, manifest=str(manifest), output_dir=str(directory / "assessment"))
    assert result["complete"] and result["completed_trials"] == 1
    assert result["optimizer_step"] == 0 and result["publication_version"] == 0
    assert events[-7:] == ["reporters", "checkpoint", "eval_sim", "sim", "collector", "rollout", "ray"]
    events.clear()
    assert run_assessment(config, manifest=str(manifest), output_dir=str(directory / "assessment")) == result
    assert events == ["checkpoint", "eval_sim", "sim", "collector", "rollout", "ray"]
    assert allocated[-1][-1].evaluated == []


def test_formal_window_validation_runs_before_and_after_training_without_analysis(
    local_run, one_logger_sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.utils import mesh

    config, attempt, _, allocated = local_run
    # The local Gaussian fixture has one real collection replica and no GR00T hardware constraint.
    monkeypatch.setattr(mesh, "detect_mesh", lambda: (1, 1))
    config.recipe_check = None
    config.rl.validation.enabled = True
    config.rl.validation.save_checkpoint = False
    result = gr00t_ppo.run(config)
    directory = attempt["output"] / "evaluation"
    reports = [json.loads(path.read_text()) for path in sorted(directory.glob("step_*.json"))]
    assert [report["optimizer_step"] for report in reports] == [0, result["optimizer_step"]]
    assert all(report["training_state_unchanged"] for report in reports)
    assert all(report["metrics"]["val/episode_count"] == len(fixed_manifest("eval").cases) for report in reports)
    assert len(allocated[-1][-1].evaluated) == sum(report["metrics"]["val/episode_count"] for report in reports)
    assert not (directory / "baseline-check.json").exists()
    assert len(one_logger_sink) == 1 and one_logger_sink[0].finishes == 1
    assert one_logger_sink[0].store["validation_iterations"] == len(reports)
    assert any(frame.get("validation_iterations_time_total", 0) > 0 for frame in one_logger_sink[0].metrics)


def test_checkpoint_evaluation_restores_exact_state_without_training_or_resaving(local_run, monkeypatch):
    from cosmos_framework.posttrain.utils import mesh

    config, attempt, _, allocated = local_run
    monkeypatch.setattr(mesh, "detect_mesh", lambda: (1, 1))
    config.recipe_check = None
    config.rl.validation.enabled = True
    config.rl.validation.save_checkpoint = False
    original = gr00t_ppo.run(config)
    source = attempt["output"]
    step = original["optimizer_step"]
    before = json.loads((source / "evaluation" / f"step_{step:06d}.json").read_text())
    attempt["resume"] = str(source / "checkpoints" / f"iter_{step:06d}")
    attempt["output"] = source.parent / "evaluation-only"
    summary = gr00t_ppo.run(config, evaluate_only=True)
    artifacts = attempt["output"] / "checkpoint-evaluation" / f"iter_{step:06d}"
    assert summary["output"] == str(artifacts)
    after = json.loads((artifacts / "evaluation" / f"step_{step:06d}.json").read_text())
    assert summary["mode"] == "checkpoint_evaluation" and summary["windows"] == 0
    assert summary["optimizer_step"] == step
    assert before["learner_state"] == after["learner_state"]
    assert before["episodes"] == after["episodes"]
    assert not (attempt["output"] / "windows.jsonl").exists()
    assert not (attempt["output"] / "checkpoints").exists()
    assert allocated[-1][1].state_dict() == allocated[-2][1].state_dict()
    assert after["checkpoint_provenance"] == summary["checkpoint_provenance"]
    assert after["checkpoint_provenance"] == allocated[-1][0].loaded_checkpoint_provenance()
    # A second evaluation of the same checkpoint into the same output never overwrites the first.
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gr00t_ppo.run(config, evaluate_only=True)


def test_training_resume_receipt_matches_live_state_before_next_update(local_run, monkeypatch):
    config, attempt, _, allocated = local_run
    config.rl.validation.enabled = True
    config.rl.validation.save_checkpoint = False
    original = gr00t_ppo.run(config)
    source = attempt["output"]
    step = original["optimizer_step"]
    expected = json.loads((source / "evaluation" / f"step_{step:06d}.json").read_text())["learner_state"]
    cursor = allocated[-1][1].state_dict()
    attempt["resume"] = str(source / "checkpoints" / f"iter_{step:06d}")
    attempt["output"] = source.parent / "resume-attestation"
    config.rl.max_steps = step + 2
    attest = gr00t_ppo.record_restored_boundary
    observations = []

    def unchanged(manager, path, output, policy, collector):
        before = resolve_all(policy.evaluation_state())
        mode = policy.body.model.net.training
        attest(manager, path, output, policy, collector)
        assert resolve_all(policy.evaluation_state()) == before
        assert policy.body.model.net.training == mode
        observations.append(before)

    monkeypatch.setattr(gr00t_ppo, "record_restored_boundary", unchanged)
    result = gr00t_ppo.run(config)
    receipt = json.loads((attempt["output"] / f"resume-state-step-{step}.json").read_text())
    assert receipt["learner_state"] == observations[0] == expected
    assert receipt["data"] == {"collectors": {"0": cursor}}
    assert receipt["checkpoint_provenance"] == allocated[-1][0].loaded_checkpoint_provenance()
    assert receipt["progress"]["optimizer_step"] == step
    assert result["optimizer_step"] == step + 2 and result["windows"] == 1


@pytest.mark.parametrize("change", ["collector", "during_attestation"])
def test_resume_attestation_rejects_generation_change_without_receipt(local_run, monkeypatch, change):
    config, attempt, _, allocated = local_run
    gr00t_ppo.run(config)
    attempt["resume"] = str(attempt["output"] / "checkpoints/iter_000004")
    attempt["output"] = attempt["output"].parent / "bad-attestation"
    config.rl.max_steps = 6
    assemble = gr00t_ppo.assemble

    def inject(*args, **kwargs):
        units = assemble(*args, **kwargs)
        if change == "collector":
            units["collector"].body.loaded_checkpoint_provenance = lambda: {"invalid": True}
        else:
            observe = units["policy"].body.evaluation_state

            def mutate():
                state = observe()
                (Path(attempt["resume"]) / "generation-mutated.txt").write_text("changed")
                return state

            units["policy"].body.evaluation_state = mutate
        return units

    monkeypatch.setattr(gr00t_ppo, "assemble", inject)
    with pytest.raises(RuntimeError, match="checkpoint generation"):
        gr00t_ppo.run(config)
    assert allocated[-1][0].optimizer_step == 4
    assert not list(attempt["output"].glob("resume-state-*.json"))
    assert not (attempt["output"] / "windows.jsonl").exists()


def test_checkpoint_evaluation_rejects_source_mutation_before_publishing_scores(local_run, monkeypatch):
    from cosmos_framework.posttrain.control.validation.rollout import WindowEvaluator

    config, attempt, _, allocated = local_run
    config.rl.validation.enabled = True
    config.rl.validation.save_checkpoint = False
    original = gr00t_ppo.run(config)
    step = original["optimizer_step"]
    checkpoint = attempt["output"] / "checkpoints" / f"iter_{step:06d}"
    attempt["resume"], attempt["output"] = str(checkpoint), checkpoint.parent.parent / "rejected-evaluation"
    validation_seed = WindowEvaluator.policy_validation_seed
    control = checkpoint / "trainer.pt"
    original_bytes = control.read_bytes()

    def mutate_after_restore(self):
        # The learner is already restored. A later source generation cannot be
        # named as the checkpoint that produced these scores.
        control.write_bytes(original_bytes + b"changed during evaluation")
        return validation_seed(self)

    monkeypatch.setattr(WindowEvaluator, "policy_validation_seed", mutate_after_restore)
    try:
        with pytest.raises(RuntimeError, match="checkpoint generation changed"):
            gr00t_ppo.run(config, evaluate_only=True)
        assert not list(attempt["output"].rglob("evaluation/*.json"))
        assert not list(attempt["output"].rglob("evaluation-summary-*.json"))
        assert allocated[-1][0].optimizer_step == step
    finally:
        control.write_bytes(original_bytes)


def test_checkpoint_evaluation_rejects_missing_resume_before_model_allocation(local_run):
    config, _, _, allocated = local_run
    config.rl.validation.enabled = True
    with pytest.raises(ValueError, match="explicit resumable checkpoint"):
        gr00t_ppo.run(config, evaluate_only=True)
    assert allocated == []


@pytest.mark.parametrize("final_same_output", [False, True])
def test_pending_validation_resumes_to_new_directory_with_collector_provenance(
    local_run, monkeypatch, final_same_output
):
    from cosmos_framework.posttrain.control.validation.rollout import WindowEvaluator

    config, attempt, _, allocated = local_run
    config.rl.validation.enabled = True
    config.rl.validation.every_steps = 2
    config.rl.validation.save_checkpoint = True
    config.rl.keep_optim_last = config.rl.keep_model_last = 0
    source = attempt["output"]
    evaluate = WindowEvaluator.__call__

    interrupted_step = config.rl.max_steps if final_same_output else 2

    def interrupt(self, step, reasons):
        if step == interrupted_step:
            raise RuntimeError("interrupted before pending evaluation")
        return evaluate(self, step, reasons)

    with monkeypatch.context() as context:
        context.setattr(WindowEvaluator, "__call__", interrupt)
        with pytest.raises(RuntimeError, match="interrupted before pending evaluation"):
            gr00t_ppo.run(config)
    checkpoint = source / "checkpoints" / f"iter_{interrupted_step:06d}"
    manager = allocated[-1][0].checkpoint_manager
    source_identity = manager.checkpoint_provenance(str(checkpoint))
    attempt["resume"], attempt["output"] = (
        str(checkpoint),
        source if final_same_output else source.parent / "resumed-validation",
    )
    state_before = allocated[-1][0].evaluation_state()
    cursor_before = allocated[-1][1].state_dict()
    result = gr00t_ppo.run(config)
    assert result["optimizer_step"] == 4
    if final_same_output:
        assert result["windows"] == 0
        assert allocated[-1][0].evaluation_state() == state_before
        assert allocated[-1][1].state_dict() == cursor_before
    else:
        assert manager.checkpoint_provenance(str(checkpoint)) == source_identity
    assert allocated[-1][1].loaded_checkpoint_provenance() == source_identity
    assert allocated[-1][0].loaded_checkpoint_provenance() == source_identity
    assert (attempt["output"] / "evaluation" / "step_000002.json").is_file()
    assert (attempt["output"] / "evaluation" / "step_000004.json").is_file()


def test_online_reporting_failure_does_not_block_training(
    local_run, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import wandb

    from cosmos_framework.posttrain.utils import report

    config, _, _, allocated = local_run
    monkeypatch.setenv("WANDB_MODE", "online")
    # This test injects a W&B sink fault. A Slurm allocation must not additionally
    # auto-enable the separate OneLogger credential/startup contract.
    monkeypatch.setenv("ENABLE_ONELOGGER", "false")
    monkeypatch.setenv("WANDB_PROJECT", "cosmos3_action")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.INFO, logger="posttrain")
    attempts = []

    def fail_init(**kwargs):
        attempts.append(kwargs)
        raise wandb.errors.CommError("injected reporting authentication failure")

    monkeypatch.setattr(wandb, "init", fail_init)
    monkeypatch.setattr(report, "make_reporters", make_reporters)
    result = gr00t_ppo.run(config)
    assert result["windows"] == 2 and result["optimizer_step"] == config.rl.max_steps
    assert allocated and attempts
    assert all(attempt["mode"] == "online" for attempt in attempts)
    messages = [record.getMessage() for record in caplog.records]
    assert any("continuing without wandb" in message for message in messages)
    assert any("train/joint_loss" in message for message in messages)


@pytest.fixture
def one_logger_sink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Replace only the SDK transport; the adapter and trainer counters/timers remain real."""
    sinks = []

    def create_client(config):
        # Independent transports represent separate attempts; this does not test
        # reopening the external SDK's singleton within one production process.
        sink = SimpleNamespace(store={}, metrics=[], fail=False, send_failures=0, finishes=0)

        def log_metrics(metrics):
            if sink.fail:
                sink.send_failures += 1
                raise RuntimeError("injected telemetry send failure")
            sink.metrics.append(dict(metrics))

        def finish():
            sink.finishes += 1
            if sink.fail:
                raise RuntimeError("injected telemetry finish failure")

        sink.store_set, sink.store_get = sink.store.__setitem__, sink.store.__getitem__
        sink.store_has_key, sink.store_pop = sink.store.__contains__, sink.store.pop
        sink.log_metrics, sink.finish = log_metrics, finish
        sink.log_app_tag = lambda _tag: None
        sinks.append(sink)
        return sink

    package, core = ModuleType("one_logger"), ModuleType("one_logger.core")
    package.core, core.OneLogger = core, create_client
    monkeypatch.setitem(sys.modules, "one_logger", package)
    monkeypatch.setitem(sys.modules, "one_logger.core", core)
    netrc = tmp_path / "telemetry.netrc"
    netrc.write_text("machine api.wandb.ai login fixture password fixture-only\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("ENABLE_ONELOGGER", "true")
    monkeypatch.setenv("ONE_LOGGER_ASYNC", "true")
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    return sinks


@pytest.mark.parametrize("hook", ["__init__", "on_app_start"])
def test_telemetry_startup_failure_closes_client_before_model_setup(local_run, one_logger_sink, monkeypatch, hook):
    from cosmos_framework.utils.one_logger.one_logger_utils import OneLoggerUtils

    config, _, _, allocated = local_run
    failure = RuntimeError(f"injected telemetry {hook} failure after client creation")
    original = getattr(OneLoggerUtils, hook)

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise failure

    with monkeypatch.context() as injection:
        injection.setattr(OneLoggerUtils, hook, fail)
        with pytest.raises(RuntimeError) as retained:
            gr00t_ppo.run(config)
    assert retained.value is failure
    assert not allocated
    assert len(one_logger_sink) == 1 and one_logger_sink[0].finishes == 1
    assert gr00t_ppo.run(config)["optimizer_step"] == config.rl.max_steps
    assert len(one_logger_sink) == 2 and all(sink.finishes == 1 for sink in one_logger_sink)


def test_telemetry_ends_app_when_model_setup_fails_before_training(local_run, one_logger_sink, monkeypatch):
    config, _, _, allocated = local_run
    failure = RuntimeError("injected model setup failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(gr00t_ppo, "assemble", fail)
    with pytest.raises(RuntimeError) as retained:
        gr00t_ppo.run(config)
    assert retained.value is failure
    assert not allocated
    assert len(one_logger_sink) == 1 and one_logger_sink[0].finishes == 1
    assert any("app_finish_time" in frame for frame in one_logger_sink[0].metrics)
    assert not any("app_train_loop_finish_time" in frame for frame in one_logger_sink[0].metrics)


def test_checkpoint_failure_preserves_error_without_telemetry_success(local_run, one_logger_sink, monkeypatch):
    config, attempt, events, allocated = local_run
    config.rl.max_steps = 2
    failure = RuntimeError("injected checkpoint failure after trained window")
    assemble = gr00t_ppo.assemble

    def inject(*args, **kwargs):
        units = assemble(*args, **kwargs)
        learner = units["policy"].body

        def fail_save(*_args, **_kwargs):
            assert learner.progress() == {"optimizer_step": 2, "publication_version": 1}
            raise failure

        learner.save = fail_save
        return units

    monkeypatch.setattr(gr00t_ppo, "assemble", inject)
    with pytest.raises(RuntimeError) as retained:
        gr00t_ppo.run(config)
    assert retained.value is failure
    rows = [json.loads(line) for line in (attempt["output"] / "windows.jsonl").read_text().splitlines()]
    assert [row["optimizer_step"] for row in rows] == [2]
    assert any(frame.get("save_checkpoint_count") == 1 for frame in one_logger_sink[0].metrics)
    assert any(frame.get("save_checkpoint_sync_count") == 1 for frame in one_logger_sink[0].metrics)
    assert not any("joint_ppo_checkpoint_path" in frame for frame in one_logger_sink[0].metrics)
    assert not any("train_iterations_productive_end" in frame for frame in one_logger_sink[0].metrics)
    assert one_logger_sink[0].finishes == 1
    assert events[-6:] == ["checkpoint", "reporters", "sim", "collector", "rollout", "ray"]
    assert all(rollout.stream.closed and gateway.stream.closed for _, _, rollout, gateway, _ in allocated)


def test_training_error_survives_telemetry_send_and_finish_failures(local_run, one_logger_sink, monkeypatch):
    config, attempt, events, allocated = local_run
    failure = RuntimeError("injected second-window training failure")
    assemble = gr00t_ppo.assemble

    def inject(*args, **kwargs):
        units = assemble(*args, **kwargs)
        learner = units["policy"].body
        train_batch = learner.train_batch

        def fail_after_first_window(batch_id):
            if learner.optimizer_step == 2:
                one_logger_sink[0].fail = True
                raise failure
            return train_batch(batch_id)

        learner.train_batch = fail_after_first_window
        return units

    monkeypatch.setattr(gr00t_ppo, "assemble", inject)
    with pytest.raises(RuntimeError) as retained:
        gr00t_ppo.run(config)
    assert retained.value is failure
    rows = [json.loads(line) for line in (attempt["output"] / "windows.jsonl").read_text().splitlines()]
    assert [row["optimizer_step"] for row in rows] == [2]
    assert one_logger_sink[0].store["train_iterations"] == 1
    assert one_logger_sink[0].send_failures == one_logger_sink[0].finishes == 1
    assert events[-6:] == ["checkpoint", "reporters", "sim", "collector", "rollout", "ray"]
    assert all(rollout.stream.closed and gateway.stream.closed for _, _, rollout, gateway, _ in allocated)


@pytest.mark.parametrize("stage", ["trainer", "publication", "validation"])
def test_joint_failure_does_not_complete_telemetry_intervals(
    local_run, one_logger_sink, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from cosmos_framework.posttrain.control.loop import weight_sync
    from cosmos_framework.posttrain.control.validation.rollout import WindowEvaluator

    config, _, _, _ = local_run
    failure = RuntimeError(f"injected {stage} failure")
    assemble = gr00t_ppo.assemble

    def fail(*_args, **_kwargs):
        raise failure

    if stage == "trainer":

        def inject(*args, **kwargs):
            units = assemble(*args, **kwargs)
            monkeypatch.setattr(units["policy"].body, "train_batch", fail)
            return units

        monkeypatch.setattr(gr00t_ppo, "assemble", inject)
    elif stage == "publication":
        publish = weight_sync.weight_sync

        def fail_publication(*args, validation_step, **kwargs):
            if validation_step:
                raise failure
            return publish(*args, validation_step=validation_step, **kwargs)

        monkeypatch.setattr(weight_sync, "weight_sync", fail_publication)
    else:
        config.rl.validation.enabled = config.rl.validation.before_training = True
        monkeypatch.setattr(WindowEvaluator, "__call__", fail)

    with pytest.raises(RuntimeError) as retained:
        gr00t_ppo.run(config)
    assert retained.value is failure
    sink = one_logger_sink[0]
    assert sink.store["train_iterations_end"] == 0
    assert sink.store["train_samples_end"] == (config.rl.decisions if stage == "publication" else 0)
    if stage == "validation":
        assert sink.store["validation_iterations"] == 0
        assert not any("validation_iterations_time_total" in frame for frame in sink.metrics)
    assert sink.finishes == 1


def test_checkpoint_location_is_captured_before_later_save_and_delayed_delivery(
    local_run, one_logger_sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.lifecycle import ControllerLifecycleObserver

    config, attempt, _, _ = local_run
    config.rl.save_every = 2
    entered, release = threading.Event(), threading.Event()

    class Delayed(ControllerLifecycleObserver):
        def on_checkpoint_save_start(self, model_step: int) -> None:
            if model_step == 2:
                entered.set()
                assert release.wait(5.0)

    create = CheckpointLifecycle.coordinator

    def capture(*args, **kwargs):
        kwargs["observer"].add(Delayed())
        coordinator = create(*args, **kwargs)
        save = coordinator.save

        def release_after_second_save(model_step, **options):
            save(model_step, **options)
            if model_step == 2:
                assert entered.wait(2.0)
            else:
                release.set()

        monkeypatch.setattr(coordinator, "save", release_after_second_save)
        return coordinator

    monkeypatch.setattr(CheckpointLifecycle, "coordinator", capture)
    try:
        gr00t_ppo.run(config)
    finally:
        release.set()
    records = [frame for frame in one_logger_sink[0].metrics if "joint_ppo_checkpoint_path" in frame]
    assert records == [
        {
            "joint_ppo_checkpoint_path": str(attempt["output"] / f"checkpoints/iter_{step:06d}"),
            "joint_ppo_checkpoint_optimizer_step": step,
        }
        for step in (2, 4)
    ]


def test_partial_initialization_error_survives_bounded_pending_sdk_finish(
    local_run, one_logger_sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.control import lifecycle
    from cosmos_framework.utils.one_logger.one_logger_utils import OneLoggerUtils

    config, _, _, allocated = local_run
    failure = RuntimeError("injected partial initialization with stalled SDK finish")
    entered, release = threading.Event(), threading.Event()
    failure_started = []
    observers = []
    create = lifecycle.build_controller_observers
    initialize = OneLoggerUtils.__init__

    def capture(*args, **kwargs):
        result = create(*args, **kwargs)
        observers.append(result)
        return result

    def fail_after_client(self, sdk_config):
        initialize(self, sdk_config)
        finish = self.one_logger.finish

        def blocked_finish():
            entered.set()
            assert release.wait(5.0)
            finish()

        self.one_logger.finish = blocked_finish
        failure_started.append(time.monotonic())
        raise failure

    monkeypatch.setattr(lifecycle, "build_controller_observers", capture)
    monkeypatch.setattr(lifecycle.ControllerLifecycleObservers, "DRAIN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(OneLoggerUtils, "__init__", fail_after_client)
    try:
        with pytest.raises(RuntimeError) as retained:
            gr00t_ppo.run(config)
        assert retained.value is failure
        assert time.monotonic() - failure_started[0] < 1.0
        assert entered.is_set() and not allocated
        assert one_logger_sink[0].finishes == 0, "SDK finish remains pending after the controller returned"
    finally:
        release.set()
        assert all(observer.shutdown(3.0) for observer in observers)
    assert one_logger_sink[0].finishes == 1
