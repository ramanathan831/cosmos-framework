"""Plot the completed paired loading results without implying optimizer speed."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
r = json.loads((ROOT / "loading_summary.json").read_text())
rows = [
    r["paired_speed_ratios"]["overall"],
    r["paired_speed_ratios"]["fused"],
    r["paired_speed_ratios"]["eightbit"],
    r["nonoverlapping_verification_plus_checkpoint_phase_ratios"]["fused"],
    r["nonoverlapping_verification_plus_checkpoint_phase_ratios"]["eightbit"],
]
labels = [
    "Whole scoring process · paired sum",
    "Whole scoring process · fused",
    "Whole scoring process · eight-bit",
    "Verification + read/restore · fused",
    "Verification + read/restore · eight-bit",
]
fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
for i, row in enumerate(rows):
    center = row["mean"]
    low, high = row["ci95"]
    ax.errorbar(
        center, i, xerr=[[center - low], [high - center]], fmt="o", capsize=4, color="#1864ab" if i < 3 else "#087f5b"
    )
    ax.scatter(row["values"], [i] * 3, marker="|", color="0.5", alpha=0.7, s=70)
ax.axvline(1, color="0.35", label="Parity")
ax.axvline(1.05, color="#c92a2a", linestyle=":", label="Overall process gate: lower bound > 1.05")
ax.set_yticks(range(len(rows)), labels)
ax.invert_yaxis()
ax.set_xlabel("Serial / shared-loading elapsed time · mean and 95% paired t interval")
ax.set_title(
    "Exact development parity; whole-process speed gate fails\nThree counterbalanced restarts · same A100 node · two immutable checkpoints",
    fontsize=11,
)
ax.grid(axis="x", alpha=0.2)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
fig.savefig(ROOT / "shared_loading.png", dpi=180)
fig.savefig(ROOT / "shared_loading.svg")
