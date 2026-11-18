"""Predeclared round-3 LIBERO-10 analysis: fused gain, 8-bit non-inferiority, time-to-target, memory.

Usage: python -m tools.quantized_optimizers.libero10_round3_analysis runs.json output.json
runs.json: [{"arm": "fused"|"eightbit", "seed": int, "results": "<RESULTS dir>"}, ...]
Each run's native output is <results>/runs/cosmos3_action/joint_ppo/*/integration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

B = 10_000
RNG_SEED = 0
BAR_DROP = 0.02
STEP_RATIO = 1.10
TARGET_MARGIN = 0.02
NONINFERIORITY_MARGIN = 0.03  # primary; the 0.02 result is reported as secondary
SECONDARY_MARGIN = 0.02
SPEED_BOUND = 1.05
STATE_SAVING = 0.60


def load_run(results: Path) -> dict:
    (integration,) = sorted(results.glob("runs/cosmos3_action/joint_ppo/*/integration"))
    start = datetime.fromisoformat((results / "allocation-start.txt").read_text().strip().replace("Z", "+00:00"))
    evaluations = {}
    for path in sorted((integration / "evaluation").glob("step_*.json")):
        record = json.loads(path.read_text())
        evaluations[int(path.stem.split("_")[1])] = {
            "success_rate": record["metrics"]["val/success_rate"],
            "episode_steps": record["metrics"]["val/episode_steps"],
            "cases": {(e["task_id"], e["init_state_id"]): float(bool(e["success"])) for e in record["episodes"]},
            "seconds": path.stat().st_mtime - start.timestamp(),
            "memory": record.get("optimizer_memory"),
        }
    return evaluations


def bar(evaluations: dict, update: int = 80) -> dict:
    zero, final = evaluations[0], evaluations.get(update)
    if final is None:
        return {"pass": False, "reason": f"update {update} evaluation missing"}
    ok = final["success_rate"] >= zero["success_rate"] - BAR_DROP - 1e-12 and (
        final["episode_steps"] <= STEP_RATIO * zero["episode_steps"] + 1e-9
    )
    return {"pass": ok, "success_0": zero["success_rate"], "success": final["success_rate"],
            "steps_0": zero["episode_steps"], "steps": final["episode_steps"]}


def cluster_bootstrap(per_case: dict, *, lower: bool) -> tuple[float, float]:
    """Mean of per-case values and its one-sided 95% bound; cases (task, init state) are the clusters."""
    keys = sorted(per_case)
    values = [per_case[k] for k in keys]
    rng = random.Random(RNG_SEED)
    draws = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(B))
    bound = draws[int(0.05 * B)] if lower else draws[int(0.95 * B) - 1]
    return sum(values) / len(values), bound


def per_case_mean(pairs: list[tuple[dict, dict]]) -> dict:
    """Mean over seeds of (left - right) per case; every seed must cover the same case set."""
    keys = set(pairs[0][0])
    for left, right in pairs:
        if set(left) != keys or set(right) != keys:
            raise ValueError("evaluation case sets differ across runs")
    return {k: sum(left[k] - right[k] for left, right in pairs) / len(pairs) for k in keys}


def first_target_time(evaluations: dict, target: float) -> tuple[float, bool]:
    for update in (40, 80):
        if update in evaluations and evaluations[update]["success_rate"] >= target:
            return evaluations[update]["seconds"], False
    last = max(evaluations)
    return evaluations[last]["seconds"], True


def state_bytes(memory: list[dict] | None) -> dict:
    if not memory:
        return {}
    moments = sum(r["moment_bytes"] + r["metadata_bytes"] for r in memory)
    return {"moments_metadata": moments, "with_masters": moments + sum(r["master_bytes"] for r in memory)}


def analyse(runs: list[dict]) -> dict:
    data = {(r["arm"], r["seed"]): load_run(Path(r["results"])) for r in runs}
    seeds = sorted({seed for _, seed in data})
    out: dict = {"seeds": seeds, "bars": {f"{a}:{s}": bar(e) for (a, s), e in data.items()}}

    fused = [data[("fused", s)] for s in seeds if ("fused", s) in data]
    if len(fused) == len(seeds) and all(80 in e for e in fused):
        mean, low = cluster_bootstrap(per_case_mean([(e[80]["cases"], e[0]["cases"]) for e in fused]), lower=True)
        bars = all(out["bars"][f"fused:{s}"]["pass"] for s in seeds)
        out["fused_gain"] = {"mean": mean, "lower95": low, "bar_all_seeds": bars, "pass": bars and low > 0}

    pairs = [s for s in seeds if all((a, s) in data and 80 in data[(a, s)] for a in ("fused", "eightbit"))]
    if pairs and len(pairs) == len(seeds):
        cases = [(data[("fused", s)][80]["cases"], data[("eightbit", s)][80]["cases"]) for s in pairs]
        mean, upper = cluster_bootstrap(per_case_mean(cases), lower=False)
        out["noninferiority"] = {
            "fused_minus_eightbit": mean,
            "upper95": upper,
            "margin": NONINFERIORITY_MARGIN,
            "pass": upper <= NONINFERIORITY_MARGIN,
            "secondary_margin": SECONDARY_MARGIN,
            "secondary_pass": upper <= SECONDARY_MARGIN,
        }
        mean, low = cluster_bootstrap(per_case_mean([(b, a) for a, b in cases]), lower=True)
        out["eightbit_equal_resource_gain"] = {"eightbit_minus_fused": mean, "lower95": low, "pass": low > 0}

        zeros = [data[k][0]["success_rate"] for k in data]
        target = sum(zeros) / len(zeros) + TARGET_MARGIN
        times = {}
        for s in pairs:
            tf, cf = first_target_time(data[("fused", s)], target)
            te, ce = first_target_time(data[("eightbit", s)], target)
            times[s] = {"fused_seconds": tf, "fused_censored": cf, "eightbit_seconds": te, "eightbit_censored": ce}
        ratios = [math.log(t["fused_seconds"] / t["eightbit_seconds"]) for t in times.values()]
        rng = random.Random(RNG_SEED)
        draws = sorted(
            math.exp(sum(ratios[rng.randrange(len(ratios))] for _ in ratios) / len(ratios)) for _ in range(B)
        )
        censored = any(t["eightbit_censored"] for t in times.values())
        out["time_to_target"] = {
            "target": target, "per_seed": times,
            "geometric_mean_ratio": math.exp(sum(ratios) / len(ratios)), "lower95": draws[int(0.05 * B)],
            "eightbit_censored": censored, "pass": (not censored) and draws[int(0.05 * B)] > SPEED_BOUND,
        }
        memory = {a: state_bytes(data[(a, pairs[0])][80]["memory"]) for a in ("fused", "eightbit")}
        if memory["fused"] and memory["eightbit"]:
            reduction = 1 - memory["eightbit"]["moments_metadata"] / memory["fused"]["moments_metadata"]
            total = 1 - memory["eightbit"]["with_masters"] / memory["fused"]["with_masters"]
            out["memory"] = {**memory, "moment_reduction": reduction, "with_masters_reduction": total,
                             "pass": reduction >= STATE_SAVING}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyse(json.loads(args.runs.read_text()))
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
