"""Synthetic checks of the predeclared round-3 LIBERO-10 analysis."""

import json
import os
import random

from tools.quantized_optimizers.libero10_round3_analysis import analyse, bar, cluster_bootstrap


def _run(root, arm, seed, rates, *, seconds=(1000, 2000, 3000), steps=300.0, memory=None, independent=False):
    results = root / f"{arm}-{seed}"
    evaluation = results / "runs/cosmos3_action/joint_ppo/job/integration/evaluation"
    evaluation.mkdir(parents=True)
    (results / "allocation-start.txt").write_text("2026-11-18T00:00:00Z\n")
    rng = random.Random(seed * 7 + (len(arm) if independent else 0))
    for (update, rate), offset in zip(sorted(rates.items()), seconds, strict=True):
        episodes = [
            {"task_id": t, "init_state_id": s, "success": rng.random() < rate} for t in range(10) for s in range(50)
        ]
        record = {
            "metrics": {"val/success_rate": sum(e["success"] for e in episodes) / 500, "val/episode_steps": steps},
            "episodes": episodes,
            "optimizer_memory": memory,
        }
        path = evaluation / f"step_{update:06d}.json"
        path.write_text(json.dumps(record))
        stamp = 1794960000 + offset  # 2026-11-18T00:00:00Z + offset
        os.utime(path, (stamp, stamp))
    return {"arm": arm, "seed": seed, "results": str(results)}


def _memory(moments, masters):
    return [{"moment_bytes": moments, "metadata_bytes": 10, "master_bytes": masters} for _ in range(8)]


def test_clear_gain_noninferior_and_memory(tmp_path):
    runs = []
    for seed in (17, 29, 43):
        runs.append(_run(tmp_path, "fused", seed, {0: 0.85, 40: 0.92, 80: 0.97}, memory=_memory(4000, 2000)))
        runs.append(_run(tmp_path, "eightbit", seed, {0: 0.85, 40: 0.92, 80: 0.97}, memory=_memory(1000, 2000)))
    result = analyse(runs)
    assert result["fused_gain"]["pass"] and result["fused_gain"]["lower95"] > 0
    assert result["noninferiority"]["pass"]
    assert result["memory"]["moment_reduction"] > 0.74 and result["memory"]["pass"]
    # Same clocks in both arms: no speed claim.
    assert not result["time_to_target"]["pass"]
    assert abs(result["time_to_target"]["geometric_mean_ratio"] - 1) < 1e-9


def test_no_gain_and_bar_failure_are_reported(tmp_path):
    runs = []
    for seed in (17, 29, 43):
        final = 0.80 if seed == 29 else 0.86
        runs.append(_run(tmp_path, "fused", seed, {0: 0.86, 40: 0.86, 80: final}))
        runs.append(_run(tmp_path, "eightbit", seed, {0: 0.86, 40: 0.86, 80: 0.70}))
    result = analyse(runs)
    assert not result["bars"]["fused:29"]["pass"]
    assert not result["fused_gain"]["pass"] and not result["fused_gain"]["bar_all_seeds"]
    assert not result["noninferiority"]["pass"]
    assert result["time_to_target"]["eightbit_censored"] and not result["time_to_target"]["pass"]


def test_bar_uses_own_update_zero_and_step_ratio():
    base = {"success_rate": 0.9, "episode_steps": 300.0}
    assert bar({0: base, 80: {"success_rate": 0.88, "episode_steps": 330.0}})["pass"]
    assert not bar({0: base, 80: {"success_rate": 0.879, "episode_steps": 300.0}})["pass"]
    assert not bar({0: base, 80: {"success_rate": 0.95, "episode_steps": 331.0}})["pass"]
    assert not bar({0: base})["pass"]


def test_cluster_bootstrap_bounds_bracket_mean():
    values = {(t, s): float((t + s) % 3 == 0) for t in range(10) for s in range(50)}
    mean, low = cluster_bootstrap(values, lower=True)
    _, high = cluster_bootstrap(values, lower=False)
    assert low < mean < high


def test_independent_equal_arms_show_margin_resolution(tmp_path):
    runs = []
    for seed in (17, 29, 43):
        for arm in ("fused", "eightbit"):
            runs.append(_run(tmp_path, arm, seed, {0: 0.9, 40: 0.9, 80: 0.9}, independent=True))
    result = analyse(runs)
    # Equal true rates: the upper bound sits roughly 1-2 points above the observed difference.
    width = result["noninferiority"]["upper95"] - result["noninferiority"]["fused_minus_eightbit"]
    assert 0.01 < width < 0.025
    assert result["noninferiority"]["margin"] == 0.03
