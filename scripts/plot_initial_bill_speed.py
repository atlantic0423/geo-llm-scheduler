"""Fixed cost/quality figures and eight prespecified mixed-case Pareto views."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_d5_results import nondominated
from analyze_initial_bill_speed import points

ARMS = ("REF4", "FAST4", "FAST2", "FAST1")
COLORS = dict(zip(ARMS, ("#777777", "#2166ac", "#e08214", "#1b7837")))


def plot_costs(out: Path) -> list[Path]:
    """Plot all recorded cases, keeping audit clocks separate from production."""
    with (out / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (out / "paired.csv").open(encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, machines in zip(axes[0], (12, 24)):
        for arm in ARMS:
            values = [
                np.mean(
                    [
                        float(r["production_wall"])
                        for r in rows
                        if int(r["machines"]) == machines and int(r["n"]) == n and r["arm"] == arm
                    ]
                )
                for n in (20, 50, 100, 200)
            ]
            ax.plot((20, 50, 100, 200), values, marker="o", color=COLORS[arm], label=arm)
        ax.set(
            title=f"Three regions, {machines} instances",
            xlabel="Jobs",
            ylabel="Mean production wall time (s)",
        )
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.legend()
    ax = axes[1, 0]
    for arm in ("FAST2", "FAST1"):
        rr = [r for r in paired if r["arm"] == arm]
        x, y = [float(r["wall_ratio_to_fast"]) for r in rr], [float(r["hv_ratio"]) for r in rr]
        ax.scatter(x, y, color=COLORS[arm], alpha=0.25, s=18)
        ax.scatter(np.mean(x), np.mean(y), color=COLORS[arm], edgecolor="black", s=100, label=arm)
    ax.axhline(1.0, color="grey", linewidth=1)
    ax.set(
        xlabel="Production time / FAST4",
        ylabel="HV / FAST4",
        title="Paired observations; large markers are means",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    ax = axes[1, 1]
    labels = []
    layers = {k: [] for k in ("parent_wall", "bank_wall", "audit_wall")}
    for machines in (12, 24):
        for arm in ARMS:
            selected = [
                r
                for r in rows
                if int(r["n"]) == 200 and int(r["machines"]) == machines and r["arm"] == arm
            ]
            labels.append(f"{machines}\n{arm}")
            for key in layers:
                layers[key].append(float(np.mean([float(r[key]) for r in selected])))
    bottom = np.zeros(8)
    for key, label, color in (
        ("parent_wall", "Original P100", "#92c5de"),
        ("bank_wall", "BILL bank", "#2166ac"),
        ("audit_wall", "Independent audit", "#f4a582"),
    ):
        ax.bar(
            range(8),
            layers[key],
            bottom=bottom,
            color=color,
            label=label,
            hatch="//" if key == "audit_wall" else None,
        )
        bottom += layers[key]
    ax.set_xticks(range(8), labels, fontsize=7)
    ax.set(title="200 jobs: production phases plus separate audit", ylabel="Mean seconds")
    ax.legend(fontsize=7)
    files = [out / f"cost_quality.{ext}" for ext in ("png", "pdf")]
    for path in files:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    return files


def plot_fronts(raw: Path, out: Path) -> list[Path]:
    """Plot fixed mixed cases, seed15101; connecting lines are only visual guides."""
    specs = [json.loads(p.read_text()) for p in (raw / "specs").glob("*.json")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    for i, machines in enumerate((12, 24)):
        for j, n in enumerate((20, 50, 100, 200)):
            spec = next(
                s
                for s in specs
                if s["machines"] == machines
                and s["n"] == n
                and s["scenario"] == "mixed"
                and s["seed"] == 15101
            )
            ax = axes[i, j]
            for arm in ARMS:
                pp = points(
                    json.loads((raw / "runs" / spec["key"] / arm / "population.json").read_text())
                )
                front = nondominated(pp)
                ax.plot(
                    [p[0] for p in front],
                    [p[1] for p in front],
                    "o-",
                    markersize=3,
                    linewidth=2 if arm == "REF4" else 1,
                    alpha=0.8,
                    color=COLORS[arm],
                    label=arm,
                )
            ax.set(
                title=f"{n} jobs, {machines} instances",
                xlabel="Total flow time (s)",
                ylabel="Electricity bill (CNY)",
            )
            ax.grid(alpha=0.2)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0, 0].legend(fontsize=8)
    files = [out / f"fixed_mixed_fronts.{ext}" for ext in ("png", "pdf")]
    for path in files:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print([str(p) for p in plot_costs(args.out) + plot_fronts(args.raw, args.out)])
