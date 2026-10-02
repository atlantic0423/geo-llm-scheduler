"""Render standalone P1 research figures from committed clustered statistics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(folder: Path) -> None:
    """Plot base-level evidence with explicit units and exploratory captions."""
    data = json.loads((folder / "statistics.json").read_text())
    m = data["metrics"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for i, key in enumerate(["bill_loss", "continuation_scalar_delta"]):
        vals = m[f"h1/F6/timing/{key}"]["base_values"]
        ids = sorted(vals)
        factor = 100 if i == 0 else 1
        axes[i].bar(
            range(24), [vals[b] * factor for b in ids], color=["#2d7c9c"] * 12 + ["#b57c31"] * 12
        )
        axes[i].axhline(0, color="gray", linewidth=0.8)
        axes[i].set_xlabel("Independent base (12 x 50 jobs; 12 x 100 jobs)")
        axes[i].set_ylabel(
            "Bill increase on re-decoding (%)"
            if i == 0
            else "Scalar(re-decoded continuation) - Scalar(retained)"
        )
    fig.savefig(folder / "h1_bases.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    names = ["random", "direction_neighbors", "gain_order"]
    vals = [m[f"online_h2/F6/{name}_gain_ratio"] for name in names]
    means = [v["mean"] for v in vals]
    ax.errorbar(
        means,
        range(3),
        xerr=np.array(
            [[v["mean"] - v["ci"][0] for v in vals], [v["ci"][1] - v["mean"] for v in vals]]
        ),
        fmt="o",
        capsize=4,
    )
    ax.axvline(1, linestyle="--", color="gray")
    ax.set_yticks(
        range(3),
        [
            "Random birth-neighbor order",
            "Child-direction neighbors",
            "Birth-neighbor gain order (audit)",
        ],
    )
    ax.set_xlabel("Frozen scalar gains / original replacement\nBase mean and 95% cluster interval")
    fig.savefig(folder / "h2_online.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(8)
    for i, prefix in enumerate((3, 6, 10)):
        vals = [m[f"h4/F6/{a}/prefix{prefix}_success"]["mean"] * 100 for a in range(1, 9)]
        axes[0].bar(x + (i - 1) * 0.25, vals, width=0.25, label=f"Prefix {prefix}")
    axes[0].set_xticks(x, [f"A{a}" for a in range(1, 9)])
    axes[0].set_ylabel("Calls with scalar improvement (%)")
    axes[0].legend()
    for i, kind in enumerate(("late_6_10", "archive_only_late_6_10")):
        vals = [m[f"h4/F6/{a}/{kind}"]["mean"] * 100 for a in range(1, 9)]
        axes[1].bar(
            x + (i - 0.5) * 0.35,
            vals,
            width=0.35,
            label="Later scalar improvement" if i == 0 else "Archive-only later HV gain",
        )
    axes[1].set_xticks(x, [f"A{a}" for a in range(1, 9)])
    axes[1].set_ylabel("Calls benefiting from candidates 7-10 (%)")
    axes[1].legend(fontsize=8)
    fig.savefig(folder / "h4_budget.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    render(parser.parse_args().folder)
