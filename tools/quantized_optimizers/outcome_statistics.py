"""The frozen paired-seed procedure, with recording families as bootstrap units."""

import numpy as np

from tools.quantized_optimizers.analyze_development import interval


def paired_quality(baseline, quantized, frozen, *, repetitions=10000):
    if len(baseline) != 3 or len(quantized) != 3:
        raise ValueError("quality requires three independent paired training seeds")
    identities = [(row["id"], row["group"]) for row in frozen]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate scored question")
    groups = sorted({group for _, group in identities})
    members = [[i for i, (_, group) in enumerate(identities) if group == name] for name in groups]
    if len({len(indices) for indices in members}) != 1:
        raise ValueError("protocol requires the same fixed question count in every recording group")
    for rows in [*baseline, *quantized]:
        if [(row["id"], row["group"]) for row in rows] != identities:
            raise ValueError("paired score identities/order differ")

    def values(runs, key):
        result = np.asarray([[np.mean([rows[i][key] for i in indices]) for indices in members] for rows in runs])
        if not np.isfinite(result).all():
            raise ValueError("nonfinite held-out metric")
        return result

    bn, qn = (values(runs, "answer_nll") for runs in (baseline, quantized))
    ba, qa = (values(runs, "restricted_choice_correct") for runs in (baseline, quantized))
    fa = values([frozen], "restricted_choice_correct")[0]
    if (bn <= 0).any() or (qn < 0).any():
        raise ValueError("invalid answer NLL")
    result = {
        "recording_groups": len(groups),
        "questions": len(frozen),
        "nll_ratio": interval(qn.mean(1) / bn.mean(1)),
        "accuracy_drop": interval((ba - qa).mean(1)),
        "nll_improvement": interval((bn - qn).mean(1)),
        "baseline_nll": interval(bn.mean(1)),
        "quantized_nll": interval(qn.mean(1)),
        "baseline_accuracy": interval(ba.mean(1)),
        "quantized_accuracy": interval(qa.mean(1)),
        "frozen_accuracy": float(fa.mean()),
    }
    rng = np.random.default_rng(2026)
    boot = {
        name: []
        for name in ("nll_ratio", "accuracy_drop", "nll_improvement", "baseline_adaptation", "quantized_adaptation")
    }
    for _ in range(repetitions):
        seeds = rng.integers(0, 3, 3)
        recordings = rng.integers(0, len(groups), len(groups))
        index = np.ix_(seeds, recordings)
        boot["nll_ratio"].append(float(qn[index].mean() / bn[index].mean()))
        boot["accuracy_drop"].append(float((ba - qa)[index].mean()))
        boot["nll_improvement"].append(float((bn - qn)[index].mean()))
        boot["baseline_adaptation"].append(float(ba[index].mean() - fa[recordings].mean()))
        boot["quantized_adaptation"].append(float(qa[index].mean() - fa[recordings].mean()))
    result["crossed_seed_recording_bootstrap"] = {
        name: {"ci95": np.quantile(draws, [0.025, 0.975]).tolist()} for name, draws in boot.items()
    }
    bootstrap = result["crossed_seed_recording_bootstrap"]
    result["guards"] = {
        "nll": result["nll_ratio"]["ci95"][1] <= 1.01 and bootstrap["nll_ratio"]["ci95"][1] <= 1.01,
        "accuracy": result["accuracy_drop"]["ci95"][1] <= 0.01 and bootstrap["accuracy_drop"]["ci95"][1] <= 0.01,
        "positive_baseline_adaptation": bootstrap["baseline_adaptation"]["ci95"][0] > 0,
        "positive_quantized_adaptation": bootstrap["quantized_adaptation"]["ci95"][0] > 0,
        "equal_budget_nll_gain": result["nll_improvement"]["ci95"][0] > 0
        and bootstrap["nll_improvement"]["ci95"][0] > 0,
    }
    return result
