"""Render the collected native development observations; no statistical inference."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = Path(__file__).resolve().parent
d = json.loads((r / "native_exposure_development_summary.json").read_text())
arms = [d["fused"], d["nonpaged"]]
names = ["Fused AdamW\nFP32 masters", "Eight-bit AdamW\nFP32 masters"]
colors = ["#2474b7", "#df7727"]
fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.8))
for a, c, label in zip(arms, colors, ["Fused (final publication failed)", "Eight-bit (delivery completed)"]):
    x = sorted(map(int, a["evaluations"]))
    y = [100 * a["evaluations"][str(k)]["metrics"]["val/success_rate"] for k in x]
    ax[0].plot(x, y, "o-", color=c, label=label, lw=2)
ax[0].axhline(100, color="#666", ls=":", label="Fixed development target")
ax[0].set(
    xlabel="Completed optimizer updates",
    ylabel="LIBERO success (%)",
    ylim=(85, 101),
    xticks=[0, 40, 80],
    title="Task adaptation not demonstrated",
)
ax[0].legend(fontsize=7, loc="lower left")
master = [a["optimizer_storage_bytes_and_elements"]["master_bytes"] / 2**30 for a in arms]
moment = [
    sum(a["optimizer_storage_bytes_and_elements"][k] for k in ["moment_bytes", "metadata_bytes"]) / 2**30 for a in arms
]
ax[1].bar(names, master, color="#9ca3af", label="FP32 masters")
ax[1].bar(names, moment, bottom=master, color=colors, label="Moments + metadata")
ax[1].set(ylabel="Across all four ranks (GiB)", title="74.58% less moment state\n49.72% less including masters")
ax[1].legend(fontsize=8)
for i, a in enumerate(arms):
    peak = max(v["device_bytes"] for v in a["device_process_peaks"].values()) / 2**30
    host = a["full_launcher_peak_pss_bytes"] / 2**30
    ax[2].bar(i - 0.18, peak, 0.35, color=colors[i], label="Busiest whole GPU" if i == 0 else None)
    ax[2].bar(i + 0.18, host, 0.35, color=colors[i], alpha=0.4, label="Full-launcher host PSS" if i == 0 else None)
ax[2].set(xticks=[0, 1], xticklabels=names, ylabel="Sampled peak (GiB)", title="Observed total memory")
ax[2].legend(fontsize=8)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", alpha=0.2)
    a.set_axisbelow(True)
fig.suptitle("Native four-GPU GR00T / LIBERO Spatial — single-seed development, not final qualification", fontsize=12)
fig.text(
    0.5,
    0.015,
    "100 historically observed cases; no confidence intervals from one training seed. Publication implementations differ; no end-to-end speed comparison.",
    ha="center",
    fontsize=8,
)
fig.tight_layout(rect=[0, 0.055, 1, 0.93])
fig.savefig(r / "native_development.png", dpi=170)
print(r / "native_development.png")
