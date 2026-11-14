# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Manual sampling/weighting fixtures; these are not real policy-quality results."""

import copy
import json

import pytest

from tools.native_qlora_qualification import action_quality as quality


def tables(factor=1.0):
    return {
        17: {
            "task-a": {"ep-a": 1.0 * factor, "ep-b": 3.0 * factor},
            "task-b": {"ep-c": 2.0 * factor, "ep-d": 4.0 * factor},
        },
        42: {
            "task-a": {"ep-a": 5.0 * factor, "ep-b": 7.0 * factor},
            "task-b": {"ep-c": 6.0 * factor, "ep-d": 8.0 * factor},
        },
    }


def row(identifier, value, count=70):
    return {"ids": [identifier], "metrics": {quality.METRIC: {"sum": value * count, "count": count}}}


def test_windows_are_averaged_before_episode_weighting():
    groups = {"a": ("task-a", "ep-a"), **{f"b{i}": ("task-a", "ep-b") for i in range(9)}}
    result = quality.episode_means([row("a", 100), *[row(f"b{i}", 0) for i in range(9)]], groups)
    assert result == {"task-a": {"ep-a": 100, "ep-b": 0}}
    assert sum(result["task-a"].values()) / 2 == 50  # Row-natural mean would be10.


def test_equal_tasks_not_global_episode_count_weighting():
    reference = {seed: {"a": {"a1": 1.0, "a2": 1.0}, "b": {f"b{i}": 9.0 for i in range(4)}} for seed in (17, 42)}
    result = quality.compare_recipes(reference, reference, draws=30)
    assert result["reference_equal_seed_task_episode_mse"] == 5
    assert result["relative_increase_upper_one_sided95"] == 0


def test_shared_episode_indices_across_training_seeds_preserve_correlation():
    cube = [[[0.0, 100.0], [20.0, 40.0]], [[100.0, 0.0], [40.0, 20.0]]]
    # Opposite episode effects across two trained seeds cancel because both
    # seeds see the SAME episode draw. Independent per-seed draws need not.
    assert quality.draw_mean(cube, [0, 1], [0], [[0, 0]]) == 50
    assert quality.draw_mean(cube, [0, 1], [0], [[1, 1]]) == 50
    assert quality.draw_mean(cube, [0, 0], [0], [[0, 0]]) == 0  # Training-seed uncertainty remains.
    # Two selected occurrences of a task may have distinct episode draws;
    # each occurrence's draw still remains common across all seeds.
    assert quality.draw_mean(cube, [0, 1], [0, 0], [[0, 0], [1, 1]]) == 50


def test_identical_arms_have_zero_bootstrap_uncertainty_despite_large_cluster_variation():
    reference = tables()
    result = quality.compare_recipes(copy.deepcopy(reference), reference, draws=500)
    assert result["relative_increase"] == 0
    assert result["relative_increase_upper_one_sided95"] == 0


def test_manual_proportional_effect_and_seed_task_episode_means():
    result = quality.compare_recipes(tables(1.1), tables(), draws=200)
    assert result["reference_equal_seed_task_episode_mse"] == 4.5
    assert result["candidate_equal_seed_task_episode_mse"] == pytest.approx(4.95)
    assert result["relative_increase"] == pytest.approx(0.1)
    assert result["relative_increase_upper_one_sided95"] == pytest.approx(0.1)


def test_exact_manual_rng_draw_plan_and_reproducibility(monkeypatch):
    class FixedDraw:
        def __init__(self, seed):
            assert seed == 6809
            self.calls = 0

        def choices(self, values, *, k):
            self.calls += 1
            # One seed draw, one task draw, then one episode draw per task
            # occurrence. There is no inside-seed episode sampling loop.
            return {1: [0, 1], 2: [0, 1], 3: [0, 0], 4: [1, 1]}[self.calls]

    monkeypatch.setattr(quality.random, "Random", FixedDraw)
    candidate = tables()
    candidate[17]["task-a"]["ep-a"] = 9.0
    # Reference draw=(1+4+5+8)/4=4.5; candidate=(9+4+5+8)/4=6.5.
    result = quality.compare_recipes(candidate, tables(), draws=1)
    assert result["relative_increase_upper_one_sided95"] == pytest.approx(6.5 / 4.5 - 1)


@pytest.mark.parametrize("change", ["duplicate", "missing", "unknown", "denominator", "nonfinite"])
def test_bad_native_row_inventory_or_objective_reduction_fails(change):
    groups = {"a": ("t", "e"), "b": ("t", "e")}
    rows = [row("a", 1), row("b", 2)]
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "missing":
        rows.pop()
    elif change == "unknown":
        rows[1]["ids"] = ["unknown"]
    elif change == "denominator":
        rows[1]["metrics"][quality.METRIC]["count"] = 69
    else:
        rows[0]["metrics"][quality.METRIC]["sum"] = float("nan")
    with pytest.raises(ValueError):
        quality.episode_means(rows, groups, expected_count=70)


@pytest.mark.parametrize("change", ["seed", "task", "episode", "zero-reference"])
def test_missing_pairing_and_undefined_relative_statistic_fail(change):
    before, after = tables(), tables()
    if change == "seed":
        after.pop(17)
    elif change == "task":
        after[17].pop("task-a")
    elif change == "episode":
        after[42]["task-b"].pop("ep-d")
    else:
        before = tables(0)
    with pytest.raises(ValueError):
        quality.compare_recipes(after, before, draws=10)


def test_all_declared_seeds_required_before_recipe_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "load_groups", lambda *args: {"a": ("t", "e")})
    # Deliberately no completed results: do not manufacture nine runs or native
    # checkpoints merely to test admission of the analysis inventory.
    with pytest.raises(ValueError, match="all nine"):
        quality.analyze([], validation=tmp_path / "unused", manifest=tmp_path / "unused")


def rollout_case(task, initial=0, **extra):
    return {"suite": "libero_spatial", "task_id": task, "init_state_id": initial, **extra}


def test_rollout_equal_tasks_not_control_frames_or_number_of_initial_states():
    cases = [rollout_case(0), *[rollout_case(1, index) for index in range(9)]]
    candidate = [{**case, "success": case["task_id"] == 0, "steps": 256} for case in cases]
    reference = [{**case, "success": True, "steps": 8} for case in cases]
    result = quality.compare_rollout_checkpoints(candidate, reference, cases, draws=100)
    assert result["candidate_equal_task_success"] == 0.5  # Case-natural rate would be0.1.
    assert result["reference_equal_task_success"] == 1
    assert result["success_loss_percentage_points"] == 50
    assert not result["recipe_preservation_established"]
    assert not result["useful_adaptation_established"]


def test_rollout_identical_arms_cancel_task_variability_and_all_fail_is_not_usefulness():
    cases = [rollout_case(0), rollout_case(1)]
    for outcomes in ((True, False), (False, False)):
        episodes = [{**case, "success": outcome} for case, outcome in zip(cases, outcomes, strict=True)]
        result = quality.compare_rollout_checkpoints(episodes, copy.deepcopy(episodes), cases, draws=100)
        assert result["success_loss_upper_one_sided95_percentage_points"] == 0
        assert not result["recipe_preservation_established"]
        assert not result["useful_adaptation_established"]


def test_rollout_manual_paired_whole_task_draw(monkeypatch):
    class FixedDraw:
        def __init__(self, seed):
            assert seed == 6809

        def choices(self, values, *, k):
            assert values == [100, -100] and k == 2
            return [values[1], values[1]]

    monkeypatch.setattr(quality.random, "Random", FixedDraw)
    cases = [rollout_case(0), rollout_case(1)]
    before = [rollout_case(0, success=True), rollout_case(1, success=False)]
    after = [rollout_case(0, success=False), rollout_case(1, success=True)]
    result = quality.compare_rollout_checkpoints(after, before, cases, draws=1)
    assert result["success_loss_percentage_points"] == 0
    assert result["success_loss_upper_one_sided95_percentage_points"] == -100


@pytest.mark.parametrize("change", ["duplicate", "missing", "unknown", "nonboolean", "case-duplicate", "boolean-id"])
def test_rollout_requires_complete_prospective_case_pairing(change):
    cases = [rollout_case(0), rollout_case(1)]
    rows = [rollout_case(0, success=True), rollout_case(1, success=False)]
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "missing":
        rows.pop()
    elif change == "unknown":
        rows[1]["task_id"] = 2
    elif change == "nonboolean":
        rows[0]["success"] = 1
    elif change == "case-duplicate":
        cases[1] = cases[0]
    else:
        rows[0]["task_id"] = False
    with pytest.raises(ValueError):
        quality.compare_rollout_checkpoints(rows, rows, cases, draws=10)


def test_rollout_does_not_manufacture_task_uncertainty_from_many_states():
    cases = [rollout_case(0, index) for index in range(100)]
    rows = [{**case, "success": True} for case in cases]
    with pytest.raises(ValueError, match="multiple task"):
        quality.compare_rollout_checkpoints(rows, rows, cases)


def test_read_json_hash_binds_exact_bytes(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"value": 1}))
    value, first = quality.read_json(path)
    path.write_text(json.dumps({"value": 2}))
    _, second = quality.read_json(path)
    assert value == {"value": 1} and first != second


@pytest.mark.parametrize("memory", [False, None, 1])
def test_missing_attributed_process_memory_blocks_combined_comparison(tmp_path, monkeypatch, memory):
    monkeypatch.setattr(quality, "load_groups", lambda *args: {"a": ("t", "e")})
    path = tmp_path / "development-result.json"
    path.write_text(
        json.dumps(
            {
                "mode": "nf4_lora",
                "seed": 17,
                "protocol_sha256": quality.PROTOCOL_SHA,
                "fixed_work": {"updates": 300},
                "endpoint": {"fresh_portable_exact_primary_parity": True},
                "attributed_process_memory_valid": memory,
            }
        )
    )
    with pytest.raises(ValueError, match="attributed process-memory"):
        quality.analyze([path], validation=tmp_path / "unused", manifest=tmp_path / "unused")
