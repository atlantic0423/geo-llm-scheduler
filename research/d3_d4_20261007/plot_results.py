"""Rebuild the scientific overview from the published small statistical summaries."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
d3 = json.loads((root / "d3_report.json").read_text())
g = json.loads((root / "d4_guard_report.json").read_text())["summaries"]["all"]
w = json.loads((root / "d4_positions_report.json").read_text())["summaries"]["all"]
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), layout="constrained")
ax = axes[0, 0]
values = list(d3["primary"].values())
means = [v["mean"] * 1e6 for v in values]
ax.errorbar(
    means,
    range(3),
    xerr=[
        [m - v["ci"][0] * 1e6 for m, v in zip(means, values)],
        [v["ci"][1] * 1e6 - m for m, v in zip(means, values)],
    ],
    fmt="o",
    capsize=4,
    color="#4c6485",
)
ax.set_yticks(range(3), ["vs BASE", "vs RGAIN", "vs SHUFFLED"])
ax.set_ylim(-0.35, 2.65)
ax.axvline(0, color="gray", linewidth=1)
ax.set(title="D3: no frozen heldout gate passed", xlabel="RESPONSE gain difference (x 1e-6)")
ax.text(
    0.97, 0.94, "16 heldout bases; all Holm p = 1", transform=ax.transAxes, ha="right", fontsize=9
)
ax = axes[0, 1]
values = [g[a]["cpu_percent_saving"] for a in ("GUARD_ALL", "GUARD_SINGLE")]
means = [v["mean"] for v in values]
ax.bar(range(2), means, color=["#669e8e", "#217a66"])
ax.errorbar(
    range(2),
    means,
    yerr=[
        [m - v["ci"][0] for m, v in zip(means, values)],
        [v["ci"][1] - m for m, v in zip(means, values)],
    ],
    fmt="none",
    ecolor="black",
    capsize=4,
)
ax.set_xticks(range(2), ["ALL", "SINGLE"])
ax.set(title="D4-G: unchanged outputs, lower CPU", ylabel="Mean base CPU saving (%)", ylim=(0, 20))
for i, m in enumerate(means):
    ax.text(i, values[i]["ci"][1] + 0.4, f"{m:.2f}%", ha="center", fontsize=9)
ax.text(
    0.97, 0.05, "32 seen bases; construction only", transform=ax.transAxes, ha="right", fontsize=9
)
ax = axes[1, 0]
arms = ("REFERENCE", "RANDOM_SCORED", "PEAK_RANK")
rates = [100 * w[a]["means"]["nonempty"] for a in arms]
ax.bar(range(3), rates, color=["#8d9baa", "#c1cad4", "#427fb2"])
ax.set_xticks(range(3), ["Reference", "Scored control", "Peak rank"])
ax.set(title="D4-W: more calls return proposals", ylabel="Calls with a proposal (%)", ylim=(0, 70))
for i, m in enumerate(rates):
    ax.text(i, m + 1, f"{m:.2f}%", ha="center", fontsize=9)
ax.text(
    0.97, 0.05, "32 seen bases; 82,944 paired calls", transform=ax.transAxes, ha="right", fontsize=9
)
ax = axes[1, 1]
cpu = [w[a]["means"]["cpu"] * 1000 for a in arms]
gain = [w[a]["means"]["gain"] * 1000 for a in arms]
ax.bar([i - 0.18 for i in range(3)], cpu, width=0.36, color="#b77445", label="CPU (ms)")
ax.set_ylabel("Construction CPU / call (ms)", color="#b77445")
other = ax.twinx()
other.bar([i + 0.18 for i in range(3)], gain, width=0.36, color="#467fa5", label="Gain (x 1e-3)")
other.set_ylabel("Best scalar gain / call (x 1e-3)", color="#467fa5")
ax.set_xticks(range(3), ["Reference", "Scored control", "Peak rank"])
ax.set_title("D4-W: quality gains cost extra CPU")
fig.suptitle("Frozen offline studies: heldout D3 vs developmental D4 prototypes", fontsize=14)
fig.savefig(root / "overview.png", dpi=180)
fig.savefig(root / "overview.pdf")
plt.close(fig)
