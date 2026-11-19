# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Score LIBERO-10 closed-loop evaluations under the locked optimizer protocol.

Episodes are identified by (task_id, init_state). Arms are paired on the training
seed and on every episode identity. Uncertainty uses a two-way crossed bootstrap:
training seeds and episode identities are resampled independently (each episode
identity is shared by every seed and arm, so episodes are not independent draws
across seeds). Percentile intervals; the resampling seed is fixed by the protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_eval(directory: Path, field: str = "success") -> dict[tuple[int, int], float]:
    """Per-episode success (or control steps) from every client summary in one evaluation."""
    episodes: dict[tuple[int, int], float] = {}
    summaries = sorted(directory.glob("client-*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"no client summaries under {directory}")
    for path in summaries:
        for task in json.loads(path.read_text())["task_results"]:
            for index, episode in enumerate(task["episode_results"]):
                key = (int(task["task_id"]), int(episode.get("episode", index)))
                if key in episodes:
                    raise ValueError(f"duplicate episode {key} in {directory}")
                episodes[key] = float(bool(episode["success"]) if field == "success" else episode[field])
    return episodes


def matrix(evals: dict[int, dict[tuple[int, int], bool]], keys: list[tuple[int, int]]) -> np.ndarray:
    """seeds x episodes success matrix; every seed must cover exactly the same episodes."""
    seeds = sorted(evals)
    for seed in seeds:
        if set(evals[seed]) != set(keys):
            raise ValueError(f"seed {seed} does not cover the locked episode set")
    return np.array([[evals[seed][k] for k in keys] for seed in seeds], dtype=np.float64)


def crossed_bootstrap(diff: np.ndarray, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Mean of a seeds x episodes array under independent seed and episode resampling."""
    n_seed, n_episode = diff.shape
    out = np.empty(resamples)
    for b in range(resamples):
        s = rng.integers(0, n_seed, n_seed)
        e = rng.integers(0, n_episode, n_episode)
        out[b] = diff[np.ix_(s, e)].mean()
    return out


def interval(samples: np.ndarray) -> list[float]:
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="JSON: arm -> seed -> iteration -> eval dir")
    parser.add_argument("--clock", type=Path, required=True, help="JSON: arm -> seed -> iteration -> seconds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remap", nargs=2, metavar=("CLUSTER_PREFIX", "LOCAL_PREFIX"), help="rewrite index paths")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    index = json.loads(args.index.read_text())
    clock = json.loads(args.clock.read_text())
    rng = np.random.default_rng(protocol["statistics"]["bootstrap_seed"])
    resamples = protocol["statistics"]["bootstrap_resamples"]
    control, candidate = protocol["arms"]["control"], protocol["arms"]["candidate"]
    seeds = [str(s) for s in protocol["training_seeds"]]
    budget = str(protocol["equal_budget_iteration"])
    target = protocol["fixed_target"]["success_rate"]
    # Locked episode set: every (task_id 0..9, init_state 0..49) pair of LIBERO-10.
    keys = [(task, state) for task in range(10) for state in range(50)]
    if args.remap:
        index = {a: {s: {i: d.replace(args.remap[0], args.remap[1], 1) for i, d in its.items()} for s, its in ss.items()} for a, ss in index.items()}

    def evals(arm: str, iteration: str, field: str = "success") -> dict[int, dict]:
        return {int(s): load_eval(Path(index[arm][s][iteration]), field) for s in seeds}

    report: dict = {"protocol": str(args.protocol), "per_run": {}}
    ctrl = matrix(evals(control, budget), keys)
    cand = matrix(evals(candidate, budget), keys)
    diff = cand - ctrl
    boot = crossed_bootstrap(diff, resamples, rng)
    report["equal_budget"] = {
        "iteration": int(budget),
        "control_success": float(ctrl.mean()),
        "candidate_success": float(cand.mean()),
        "candidate_minus_control": float(diff.mean()),
        "interval95": interval(boot),
        "per_seed": {s: [float(ctrl[i].mean()), float(cand[i].mean())] for i, s in enumerate(seeds)},
    }
    ctrl_steps = matrix(evals(control, budget, "steps"), keys).mean()
    cand_steps = matrix(evals(candidate, budget, "steps"), keys).mean()
    report["equal_budget"]["mean_control_steps"] = {"control": float(ctrl_steps), "candidate": float(cand_steps)}
    report["protected_secondary_pass"] = bool(cand_steps <= 1.05 * ctrl_steps)
    margin = protocol["quality_preservation"]["margin"]
    report["quality_preservation_pass"] = report["equal_budget"]["interval95"][0] > -margin
    report["equal_budget_improvement_pass"] = bool(
        report["equal_budget"]["interval95"][0] > 0.0 and report["protected_secondary_pass"]
    )
    report["quality_preservation_pass"] = bool(report["quality_preservation_pass"])

    # Time to the fixed target: the first evaluated checkpoint at or above target;
    # censored runs never qualify. The clock is training wall time to that checkpoint.
    iterations = sorted({int(i) for arm in (control, candidate) for s in seeds for i in index[arm][s]})
    times: dict[str, dict[str, float | None]] = {}
    for arm in (control, candidate):
        times[arm] = {}
        for s in seeds:
            reached = None
            for it in iterations:
                directory = index[arm][s].get(str(it))
                if directory is None:
                    continue
                rate = float(np.mean([v for k, v in load_eval(Path(directory)).items() if k in set(keys)]))
                report["per_run"].setdefault(f"{arm}/{s}", {})[str(it)] = rate
                if reached is None and rate >= target:
                    reached = float(clock[arm][s][str(it)])
            times[arm][s] = reached
    report["time_to_target_seconds"] = times
    if all(times[arm][s] is not None for arm in (control, candidate) for s in seeds):
        ratios = np.array([times[control][s] / times[candidate][s] for s in seeds])
        log_boot = [np.exp(np.mean(np.log(ratios[rng.integers(0, len(ratios), len(ratios))]))) for _ in range(resamples)]
        report["time_ratio_control_over_candidate"] = {
            "geometric_mean": float(np.exp(np.mean(np.log(ratios)))),
            "per_seed": dict(zip(seeds, ratios.tolist())),
            "interval95": interval(np.array(log_boot)),
        }
        report["speed_pass"] = bool(
            report["time_ratio_control_over_candidate"]["interval95"][0] > protocol["speed"]["lower_bound"]
        )
    else:
        report["time_ratio_control_over_candidate"] = None
        report["speed_pass"] = False
        report["speed_censored"] = True
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_run"}, indent=2))


if __name__ == "__main__":
    main()
