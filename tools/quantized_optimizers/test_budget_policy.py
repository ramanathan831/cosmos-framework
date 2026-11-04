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
