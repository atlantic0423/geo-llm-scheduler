"""Render static P2 research figures from the verified derived data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"F6": "#486c9f", "CG": "#bd7542", "CT": "#2d8a73"}


def _save(figure: plt.Figure, output: Path, name: str) -> None:
    figure.savefig(output / (name + ".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output / (name + ".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Produce comparison, anytime, geometry and mechanism figures without sampling favorable cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with (args.analysis / "paired_contrasts.csv").open(newline="") as stream:
        contrasts = list(csv.DictReader(stream))
    summary = json.loads((args.analysis / "summary.json").read_text())
    diagnostics = json.loads((args.analysis / "supplementary_diagnostics.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, name in zip(axes, ("CG-F6", "CT-F6", "CT-CG"), strict=True):
        groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in contrasts:
            if row["contrast"] == name:
                groups[(row["jobs"], row["base_seed"])].append(float(row["hv_gain_pct"]))
        for jobs, color, position in (("50", "#638aac", 0), ("100", "#8f6699", 1)):
            values = [np.mean(v) for (j, _), v in sorted(groups.items()) if j == jobs]
            axis.scatter(
                np.linspace(position - 0.17, position + 0.17, len(values)),
                values,
                s=35,
                alpha=0.85,
                color=color,
            )
        s = summary["primary"][name]["hv_gain_pct"]
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axhline(s["mean"], color="gray", linestyle="--", linewidth=1)
        axis.axhspan(*s["ci95"], color="gray", alpha=0.13)
        axis.set(
            title=name,
            xticks=[0, 1],
            xticklabels=["50 jobs", "100 jobs"],
            ylabel="HV paired gain (%)",
        )
        axis.text(
            0.03,
            0.97,
            f"Mean {s['mean']:+.2f}%\n95% CI [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
    fig.suptitle("24 independent base structures; tariffs and seeds averaged within base")
    _save(fig, output, "01_paired_hv")
    with (args.analysis / "anytime_metrics.csv").open(newline="") as stream:
        snapshots = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, (jobs, tariff) in zip(
        axes.flat, (("50", "H"), ("50", "T"), ("100", "H"), ("100", "T")), strict=True
    ):
        for arm, color in COLORS.items():
            times, values = [], []
            for fraction in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
                chosen = [
                    r
                    for r in snapshots
                    if r["jobs"] == jobs
                    and r["tariff"] == tariff
                    and r["arm"] == arm
                    and float(r["fraction"]) == fraction
                ]
                times.append(fraction)
                values.append(float(np.mean([float(r["hv"]) for r in chosen])))
            axis.plot(times, values, "o-", label=arm, color=color, markersize=4)
        axis.set(
            title=f"{jobs} jobs / {tariff}",
            xlabel="Target fraction of engine budget",
            ylabel="Fixed-reference normalized HV",
        )
        axis.legend(frameon=False)
    fig.suptitle("Observed at first safe boundary after each target; descriptive means")
    _save(fig, output, "02_anytime_hv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for axis, field, title in zip(
        axes,
        ("population_unique_genotypes", "trigger_rate", "archive_unique_objectives"),
        (
            "Final population: distinct genotypes",
            "Trigger hit rate (%)",
            "Archive: distinct objective vectors",
        ),
        strict=True,
    ):
        values = [
            diagnostics["arm_descriptive_means"][arm][field]
            * (100 if field == "trigger_rate" else 1)
            for arm in COLORS
        ]
        axis.bar(list(COLORS), values, color=list(COLORS.values()), width=0.6)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.1f}", ha="center", va="bottom")
        axis.set(title=title, ylim=(0, max(values) * 1.18))
    fig.suptitle("Descriptive final-state evidence; counts are not quality or causal attribution")
    _save(fig, output, "03_mechanism_tradeoffs")
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    regions = ["igd_flow_edge", "igd_middle", "igd_bill_edge", "igd_arc_weighted"]
    for offset, name in enumerate(("CG-F6", "CT-F6", "CT-CG")):
        values = [diagnostics["contrasts"][name][field]["mean"] for field in regions]
        axis.bar(np.arange(4) + (offset - 1) * 0.25, values, width=0.25, label=name)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xticks=np.arange(4),
        xticklabels=["Flow edge", "Middle", "Bill edge", "Arc weighted"],
        ylabel="Normalized IGD+ decrease (positive better)",
    )
    axis.legend(frameon=False)
    fig.suptitle("Empirical front coverage: supplementary diagnostics")
    _save(fig, output, "04_front_coverage")


if __name__ == "__main__":
    main()
