import pytest

from tools.quantized_optimizers.budget_policy import BudgetPolicy

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_training_clock_excludes_loading_and_never_substitutes_overrun_update():
    p = BudgetPolicy("training", 60, 5, 100)
    assert p.stop_reason(200) is None
    p.loop_monotonic = 200
    p.update(1, 251)
    assert p.stop_reason(254) is None
    p.update(2, 256)
    assert p.stop_reason(256)
    assert p.report(300)["training_checkpoint_eligible"]
    # Unexpected slow update is censored, not silently replaced by update1.
    q = BudgetPolicy("training", 60, 5, 100, loop_monotonic=200)
    q.update(1, 251)
    q.update(2, 262)
    assert not q.report(300)["training_checkpoint_eligible"]
    assert q.report(300)["updates"][-1]["step"] == 2


def test_product_clock_includes_setup_and_requires_delivery_cost_check():
    p = BudgetPolicy("end_to_end", 300, 90, 100, cold_offset_seconds=30, loop_monotonic=200)
    p.update(1, 260)
    assert p.stop_reason(280)
    report = p.report(390)
    assert report["process_complete_seconds"] == 320
    assert report["training_checkpoint_eligible"] and not report["process_within_budget"]
    assert report["requires_external_delivery_check"]


def test_validation_overrun_cannot_pass_training_budget_from_earlier_update():
    p = BudgetPolicy("training", 60, 10, 100, loop_monotonic=200)
    p.update(1, 249)
    p.checkpoint_boundary_monotonic = 262  # required validation ended after budget
    report = p.report(320)
    assert report["completed_training_validation_boundary_seconds"] == 62
    assert not report["training_checkpoint_eligible"]


def test_target_requires_an_observed_post_training_validation():
    p = BudgetPolicy("target", 300, 90, 100, quality_target=0.95)
    p.validation(0, 0.9, 110)
    assert p.target_hit is None
    p.validation(5, 0.96, 140)
    assert p.target_hit is None
    p.validation(10, 0.94, 160)
    assert p.target_hit == {"step": 10, "nll": 0.94, "clock_seconds": 60.0}
    assert "target" in p.stop_reason(160)


def test_invalid_budget_and_duplicate_update_are_errors():
    with pytest.raises(ValueError):
        BudgetPolicy("training", 50, 50, 0)
    p = BudgetPolicy("training", 50, 5, 0, loop_monotonic=0)
    p.update(1, 2)
    with pytest.raises(ValueError, match="duplicated"):
        p.update(1, 3)


def test_advance_budget_notice_completes_due_validation_before_stopping():
    p = BudgetPolicy("end_to_end", 390, 180, 100, required_validation_steps={0, 5})
    assert p.stop_reason(320) is None  # initial validation cannot be silently skipped
    p.validation(0, 1.9, 321)
    for step in range(1, 6):
        p.update(step, 321 + step)
    assert p.stop_reason(326) is None  # native at_step must not defer validation5
    assert not p.report(326)["training_checkpoint_eligible"]
    p.validation(5, 0.9, 340)
    assert p.stop_reason(340)
    assert p.report(345)["training_checkpoint_eligible"]


def test_training_clock_excludes_durable_save_before_required_validation():
    p = BudgetPolicy("training", 60, 10, 100, loop_monotonic=200, required_validation_steps={1})
    p.update(1, 249)
    p.checkpoint_boundary_monotonic = 249
    p.checkpoint_intervals.append((249, 299))
    assert p.elapsed(249) == 49
    p.validation(1, 0.8, 311)  # 50s save plus12s required validation
    assert p.report(315)["completed_training_validation_boundary_seconds"] == 61
    assert not p.report(315)["training_checkpoint_eligible"]


@pytest.mark.parametrize("external_stop", [False, True])
def test_native_validation_transaction_distinguishes_budget_notice_from_signal(external_stop):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.control.validation import ValidationReport, ValidationSchedule
    from cosmos_framework.posttrain.control.validation.boundary import ValidationBoundaryCoordinator
    from cosmos_framework.posttrain.control.validation.transaction import ValidationTransaction
    from cosmos_framework.posttrain.schema.checkpoint import CheckpointComponentPlan

    events = []
    budget = BudgetPolicy("end_to_end", 390, 180, 100, required_validation_steps={1})
    budget.update(1, 311)
    policy = SimpleNamespace(
        mark_validation_pending=lambda *_args: events.append("mark"),
        complete_validation=lambda _step: events.append("complete"),
    )
    checkpoints = SimpleNamespace(
        resolve_component_plan=lambda **_kwargs: CheckpointComponentPlan.from_value(None),
        save_full=lambda *_args, **_kwargs: events.append("save"),
        persist_validation_completion=lambda _step: events.append("persist"),
    )

    def evaluate(step, reasons):
        events.append("evaluate")
        budget.validation(step, 0.8, 330)
        return ValidationReport(
            model_step=step,
            sample_count=1,
            metrics={"val/loss_avg": 0.8},
            source_fingerprint="development",
            trigger_reasons=reasons,
            duration_s=12,
        )

    boundary = ValidationBoundaryCoordinator(
        schedule=ValidationSchedule(enabled=True, before_training=False, every_steps=1),
        total_steps=1,
        transaction=ValidationTransaction(policy, checkpoints, [], set()),
        evaluator=evaluate,
        should_stop=lambda: "signal SIGTERM" if external_stop else budget.stop_reason(311),
    )
    outcome = boundary.at_step(1, {"optim/optimizer_step": 1}, scheduled_checkpoint_due=True)
    if external_stop:
        assert outcome.preempted == "signal SIGTERM"
        assert events == ["mark", "save"]
        assert not budget.report(340)["training_checkpoint_eligible"]
    else:
        assert outcome.preempted is None
        assert events == ["mark", "save", "evaluate", "complete", "persist"]
        assert budget.stop_reason(330)
        assert budget.report(340)["training_checkpoint_eligible"]
