# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native GR00T development ingestion and paired seed/task/episode uncertainty.

This consumes existing native evaluator rows, never launches or selects a run.
Windows are averaged within their episode. One episode draw for each selected
task occurrence is shared across every selected training seed and both arms.
The exposed development population cannot establish independent confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from tools.native_qlora_qualification.common import write_json_new
from tools.native_qlora_qualification.performance import quantile

PROTOCOL_SHA = "56c5419d3033fe71e9a9d364ff172e2fd35abd2cab43d92a92d5c3d7ebe3e0d7"
MANIFEST_SHA = "c726e37725c2d4be88efb17e8f96b956d018a07db9f72bde1f6825084392dd00"
VALIDATION_SHA = "0e74e90b82dfe4512c77734783bf0056bde222b49c10677f2ab32c3bb27317ac"
SEEDS = (17, 42, 123)
MODES = ("nf4_lora", "bf16_lora", "bf16_frozen")
METRIC = "normalized_action_mse"


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def episode_means(rows, groups, *, expected_count=None):
    """Equal windows within episodes, then equal episodes/tasks in later stages."""
    values = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        if not isinstance(row["ids"], list) or len(row["ids"]) != 1:
            raise ValueError("native action analysis requires one identified example per row")
        identifier = row["ids"][0]
        if identifier in seen or identifier not in groups:
            raise ValueError("evaluation row is duplicated or absent from the cohort")
        seen.add(identifier)
        numerator, count = (row["metrics"][METRIC][name] for name in ("sum", "count"))
        if (
            not all(type(value) in (int, float) and math.isfinite(value) for value in (numerator, count))
            or numerator < 0
            or count <= 0
            or (expected_count is not None and count != expected_count)
        ):
            raise ValueError("invalid native masked action-MSE numerator/denominator")
        task, episode = groups[identifier]
        values[task][episode].append(numerator / count)
    if seen != set(groups):
        raise ValueError("native evaluation omitted admitted cohort rows")
    return {
        task: {episode: sum(windows) / len(windows) for episode, windows in sorted(episodes.items())}
        for task, episodes in sorted(values.items())
    }


def paired_cubes(candidate, reference):
    """Retain trained-seed variation separately from shared task/episode variation."""
    seeds = sorted(reference)
    if seeds != sorted(candidate) or len(seeds) < 2:
        raise ValueError("paired recipe analysis requires at least two identical seed sets")
    tasks = sorted(reference[seeds[0]])
    if len(tasks) < 2:
        raise ValueError("paired recipe analysis requires at least two task clusters")
    episodes = {task: sorted(reference[seeds[0]][task]) for task in tasks}
    if any(len(items) < 2 for items in episodes.values()):
        raise ValueError("each task requires at least two episode clusters")
    cubes = []
    for arm in (candidate, reference):
        cube = []
        for seed in seeds:
            if sorted(arm[seed]) != tasks or any(sorted(arm[seed][task]) != episodes[task] for task in tasks):
                raise ValueError("tasks/episodes must be identical across every arm and training seed")
            cells = [[arm[seed][task][episode] for episode in episodes[task]] for task in tasks]
            if any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for cell in cells
                for value in cell
            ):
                raise ValueError("native episode MSE must be finite and nonnegative")
            cube.append(cells)
        cubes.append(cube)
    return cubes[0], cubes[1], seeds, tasks, episodes


def draw_mean(cube, seeds, tasks, episode_indices):
    """The same nested draw is applied to both precision arms and every seed.

    Repeated task indices are separate bootstrap task occurrences. Their
    episode draws may differ, but a draw never changes between seed indices.
    """
    if len(tasks) != len(episode_indices) or not seeds or not tasks:
        raise ValueError("empty or mismatched crossed sampling plan")
    return sum(
        sum(
            sum(cube[seed][task][episode] for episode in selected) / len(selected)
            for task, selected in zip(tasks, episode_indices, strict=True)
        )
        / len(tasks)
        for seed in seeds
    ) / len(seeds)


def compare_recipes(candidate, reference, *, draws=10000, seed=6809):
    left, right, seed_ids, task_ids, episode_ids = paired_cubes(candidate, reference)
    if type(draws) is not int or draws < 1:
        raise ValueError("bootstrap draws must be positive")
    s, t = len(seed_ids), len(task_ids)
    all_episodes = [list(range(len(episode_ids[task]))) for task in task_ids]
    point_left = draw_mean(left, range(s), range(t), all_episodes)
    point_right = draw_mean(right, range(s), range(t), all_episodes)
    if point_right <= 0:
        raise ValueError("relative MSE comparison requires a positive reference mean")
    rng, samples = random.Random(seed), []
    for _ in range(draws):
        selected_seeds = rng.choices(range(s), k=s)
        selected_tasks = rng.choices(range(t), k=t)
        selected_episodes = [rng.choices(all_episodes[task], k=len(all_episodes[task])) for task in selected_tasks]
        before = draw_mean(right, selected_seeds, selected_tasks, selected_episodes)
        if before <= 0:
            raise ValueError("zero reference in a bootstrap draw; relative bound is not estimable")
        samples.append(draw_mean(left, selected_seeds, selected_tasks, selected_episodes) / before - 1)
    upper = quantile(samples, 0.95)
    return {
        "candidate_equal_seed_task_episode_mse": point_left,
        "reference_equal_seed_task_episode_mse": point_right,
        "relative_increase": point_left / point_right - 1,
        "relative_increase_upper_one_sided95": upper,
        "draws": draws,
        "resampling_seed": seed,
        "training_seeds": seed_ids,
        "tasks": len(task_ids),
        "episodes_per_task": {task: len(episode_ids[task]) for task in task_ids},
        "method": "paired crossed seed/task resampling with one within-task episode draw per selected task occurrence, shared across all seed indices and both arms; fixed windows averaged first",
        "limitation": "Development uncertainty only; three training seeds and ten task clusters may not provide reliable tail coverage. Independent confirmation and closed-loop preservation remain separate requirements.",
    }


def load_groups(validation, manifest):
    raw = Path(validation).read_bytes()
    if hashlib.sha256(raw).hexdigest() != VALIDATION_SHA:
        raise ValueError("GR00T analysis requires the prospectively admitted validation JSONL")
    value, digest = read_json(manifest)
    if digest != MANIFEST_SHA:
        raise ValueError("GR00T grouping manifest differs from the admitted cohort")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    groups = {row["id"]: (row["observation"]["prompt"], row["episode_id"]) for row in rows}
    windows = defaultdict(lambda: defaultdict(int))
    for task, episode in groups.values():
        windows[task][episode] += 1
    if (
        len(rows) != len(groups)
        or len(groups) != 160
        or len(windows) != 10
        or any(len(episodes) != 4 or set(episodes.values()) != {4} for episodes in windows.values())
    ):
        raise ValueError("analysis requires exactly ten tasks/four episodes/four fixed windows")
    if {task: sorted(episodes) for task, episodes in windows.items()} != {
        task: sorted(episodes) for task, episodes in value["tasks"]["validation"].items()
    }:
        raise ValueError("JSONL task/episode identities differ from the admitted manifest")
    return groups


def analyze(paths, *, validation, manifest):
    groups = load_groups(validation, manifest)
    seen, tables, receipts, run_values = set(), {}, [], {}
    target_hashes = set()
    for path in paths:
        run, digest = read_json(path)
        key = run["mode"], run["seed"]
        if key in seen or key not in {(mode, seed) for mode in MODES for seed in SEEDS}:
            raise ValueError("development result has a duplicate or unplanned arm/seed")
        seen.add(key)
        if (
            run["protocol_sha256"] != PROTOCOL_SHA
            or run["fixed_work"]["updates"] != 300
            or not run["endpoint"]["fresh_portable_exact_primary_parity"]
        ):
            raise ValueError("development result does not match the frozen completed endpoint")
        if run.get("attributed_process_memory_valid") is not True:
            raise ValueError("development comparison requires valid attributed process-memory evidence")
        target_hashes.add(run["target_sha256"])
        score_path = run["endpoint"]["resident_score"]
        score, score_digest = read_json(score_path)
        if score["step"] != 300 or run["fixed_work"]["history"][-1] != {"step": 300, "metrics": score["metrics"]}:
            raise ValueError("selected endpoint differs from the native recorded history")
        tables[key] = episode_means(score["rows"], groups, expected_count=70)
        run_values[key] = run
        receipts.append(
            {"mode": key[0], "seed": key[1], "result_sha256": digest, "resident_score_sha256": score_digest}
        )
    if seen != {(mode, seed) for mode in MODES for seed in SEEDS} or len(target_hashes) != 1:
        raise ValueError("analysis requires all nine planned runs sharing one locked target")
    arms = {mode: {seed: tables[mode, seed] for seed in SEEDS} for mode in MODES}
    primary = compare_recipes(arms["nf4_lora"], arms["bf16_lora"])
    primary["predeclared_offline_preservation_margin"] = 0.05
    primary["development_offline_guard_met"] = primary["relative_increase_upper_one_sided95"] <= 0.05
    primary["development_quality_improvement_upper_bound_below_zero"] = (
        primary["relative_increase_upper_one_sided95"] < 0
    )
    point_rows = [
        {"mode": mode, "seed": seed, "task": task, "episode": episode, "mse": value}
        for (mode, seed), tasks in sorted(tables.items())
        for task, episodes in tasks.items()
        for episode, value in episodes.items()
    ]
    target_rows = [
        {
            "mode": mode,
            "seed": seed,
            **run_values[mode, seed]["target"],
            "calibration_cost": run_values[mode, seed]["calibration_cost"],
            "fixed_work": run_values[mode, seed]["fixed_work"],
            "sampled_memory": run_values[mode, seed]["sampled_memory"],
            "attributed_process_memory_valid": run_values[mode, seed]["attributed_process_memory_valid"],
        }
        for seed in SEEDS
        for mode in MODES
    ]
    return {
        "format": "cosmos-native-gr00t-development-analysis-v1",
        "scope": "exposed development only",
        "protocol_sha256": PROTOCOL_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "target_sha256": next(iter(target_hashes)),
        "input_receipts": receipts,
        "primary_nf4_vs_matched_bf16_lora": primary,
        "matched_head_policy_control": {
            **compare_recipes(arms["nf4_lora"], arms["bf16_frozen"]),
            "reference": "BF16 frozen backbone with shared FP32 head, deterministic transforms and fused AdamW; not the separate official FP32/HF-decay-group/train-augmentation recipe",
        },
        "episode_mse": point_rows,
        "individual_target_runtime_memory": target_rows,
        "time_to_target_rule": "All nine runs retained. Unreached endpoints remain censored; no attainer-only mean, no interpolated crossing, no claim fixed300 is equal compute.",
        "closed_loop_preservation": {
            "established": False,
            "reason": "requires actual paired rollout outcomes; offline action MSE cannot establish the2-percentage-point success guard",
        },
        "independent_confirmation": {
            "established": False,
            "reason": "exposed development episodes and unresolved original-base exposure",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, help="One native development-result.json; all nine required"
    )
    parser.add_argument("--validation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json_new(Path(args.output), analyze(args.run, validation=args.validation, manifest=args.manifest))


if __name__ == "__main__":
    main()
