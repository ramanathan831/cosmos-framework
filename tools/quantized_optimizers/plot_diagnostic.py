"""Plot diagnostic evidence, with across-seed uncertainty (not test-set quality)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

root = Path(__file__).resolve().parents[2] / "docs/posttrain/evidence/quantized_optimizers"
data = json.loads((root / "diagnostic_final.json").read_text())
summary = {}
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name in ("adamw", "adamw_bnb_8bit", "paged_adamw_8bit", "bnb32_diagnostic"):
    rows = [r for r in data["rows"] if r["optimizer"] == name]
    seeds = [[r for r in rows if r["seed"] == s] for s in (17, 29, 43)]
    seconds = np.array([np.mean([r["training_seconds"] for r in group]) for group in seeds])
    throughput = 64 / seconds
    peaks = np.array([np.mean([r["peak_allocated"] for r in group]) / 2**20 for group in seeds])
    err = t.ppf(0.975, 2) * throughput.std(ddof=1) / np.sqrt(3)
    label = "torch fused FP32 baseline" if name == "adamw" else name
    axes[0].errorbar(peaks.mean(), throughput.mean(), yerr=err, fmt="o", label=label, capsize=4)
    curves = np.array([np.mean([r["losses"] for r in group], axis=0) for group in seeds])
    times = np.mean([np.cumsum(r["training_seconds"]) for r in rows], axis=0)
    ci = t.ppf(0.975, 2) * curves.std(axis=0, ddof=1) / np.sqrt(3)
    axes[1].plot(times, curves.mean(axis=0), label=label)
    axes[1].fill_between(times, curves.mean(axis=0) - ci, curves.mean(axis=0) + ci, alpha=0.15)
    baseline = np.array(
        [
            np.mean([r["training_seconds"] for r in data["rows"] if r["optimizer"] == "adamw" and r["seed"] == s])
            for s in (17, 29, 43)
        ]
    )
    ratio = baseline / seconds
    margin = t.ppf(0.975, 2) * ratio.std(ddof=1) / np.sqrt(3)
    summary[name] = {
        "training_ms": float(seconds.mean() * 1000),
        "speed_ratio": float(ratio.mean()),
        "speed_ratio_ci95": [float(ratio.mean() - margin), float(ratio.mean() + margin)],
        "torch_peak_mib": float(peaks.mean()),
        "speed_gate": bool(ratio.mean() - margin > 1.05),
    }
axes[0].set(
    xlabel="Peak PyTorch allocated MiB (managed allocations excluded)",
    ylabel="Training samples/s",
    title="All-resident synthetic diagnostic, 95% seed CI",
)
axes[1].set(
    xlabel="Measured training seconds after warmup",
    ylabel="Synthetic training MSE (not held-out quality)",
    title="No task-quality or time-to-quality claim",
)
axes[0].legend(fontsize=7)
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(root / "diagnostic.png", dpi=160)
(root / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
