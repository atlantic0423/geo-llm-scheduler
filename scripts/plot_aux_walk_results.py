"""Render scientific figures from the auxiliary campaign's complete analysis tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Render fixed comparison panels without filtering by observed significance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads((args.analysis / "analysis.json").read_text(encoding="utf-8"))
    stats, means = data["statistics"], data["arm_means"]
    colors = {"screen": "#7256a6", "confirm": "#176b91", "long": "#c57723"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig: plt.Figure, name: str) -> None:
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(args.out / (name + "." + ext), dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric in zip(axes, ("hv", "igd")):
        rows = [s for s in stats if s["stage"] != "screen" and s["metric"] == metric]
        for i, r in enumerate(rows):
            lo, hi = r["ci95"]
            ax.plot([lo, hi], [i, i], color=colors[r["stage"]], linewidth=2)
            ax.scatter(r["mean"], i, c=colors[r["stage"]], s=45)
            ax.annotate(
                f"Holm p={r['p_holm']:.4g}",
                (hi, i),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        ax.set_yticks(
            range(len(rows)), [f"{r['stage']}: {r['treatment']} - {r['control']}" for r in rows]
        )
        ax.axvline(0, color="0.5", linestyle="--")
        ax.set_title("HV gain" if metric == "hv" else "IGD+ reduction")
        ax.set_xlabel("Parent mean effect and unadjusted 95% CI (positive = better)")
        ax.invert_yaxis()
        ax.margins(x=0.3)
    save(fig, "confirmation")

    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    for ax, metric in zip(axes, ("hv", "igd")):
        rows = [r for r in stats if r["stage"] == "screen" and r["metric"] == metric]
        for i, r in enumerate(rows):
            lo, hi = r["ci95"]
            ax.plot([lo, hi], [i, i], c=colors["screen"])
            ax.scatter(r["mean"], i, c=colors["screen"], s=25)
        ax.set_yticks(range(len(rows)), [r["treatment"] for r in rows])
        ax.invert_yaxis()
        ax.axvline(0, c="0.5", linestyle="--")
        ax.set_title("HV gain vs W1" if metric == "hv" else "IGD+ reduction vs W1")
        ax.set_xlabel("80 screening parents; exploratory unadjusted 95% CI")
    save(fig, "screening")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    chosen = ["S0", "W0", "W1", "F1", "F2", "D2", "R0", "L0", "L1", "A0", "A1"]
    rows = [r for r in means if r["stage"] == "screen" and r["arm"] in chosen]
    for r in rows:
        axes[0].scatter(r["effective_per_step"], r["depth"], s=35)
        offset = {
            "S0": (-25, -19),
            "W0": (12, -19),
            "W1": (10, 10),
            "D2": (7, 11),
            "R0": (7, -15),
        }.get(r["arm"], (3, 4))
        axes[0].annotate(
            r["arm"],
            (r["effective_per_step"], r["depth"]),
            xytext=offset,
            textcoords="offset points",
        )
    axes[0].set(
        xlabel="Mean actual exact evaluations / step",
        ylabel="Mean actual steps / episode",
        title="Realized breadth and depth",
    )
    ms = json.loads((args.analysis / "mechanisms.json").read_text(encoding="utf-8"))
    lookup = {r["arm"]: r for r in ms if r["stage"] == "screen"}
    rr = [r for r in rows if r["arm"] != "S0"]
    rates = [
        100
        * sum(
            v for k, v in lookup[r["arm"]]["counts"].items() if k.startswith("breakthrough_after:")
        )
        / max(1, lookup[r["arm"]]["counts"].get("fixed_context_regressions", 0))
        for r in rr
    ]
    axes[1].barh([r["arm"] for r in rr], rates, color=colors["screen"])
    axes[1].set(
        xlabel="Regressions followed by a new historical best (%)",
        title="Within-episode recovery (descriptive)",
    )
    ratios = [
        100
        * lookup[r["arm"]]["timings"].get("polish", 0)
        / (r["production_seconds"] * r["parents"] * 4)
        for r in rows
    ]
    axes[2].barh([r["arm"] for r in rows], ratios, color="#176b91")
    axes[2].set(
        xlabel="Pooled Polish time / pooled production wall (%)",
        title="Measured cost; nested timers not added",
    )
    save(fig, "mechanisms")

    with (args.analysis / "parent_effects.csv").open(encoding="utf-8") as stream:
        effects = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, stage in zip(axes, ("confirm", "long")):
        for b, color in (("S0", "#7256a6"), ("W1", "#176b91"), ("R0", "#c57723")):
            rows = [
                r
                for r in effects
                if r["stage"] == stage and r["control"] == b and r["metric"] == "hv"
            ]
            xs, ys = [], []
            for size in (20, 50, 100, 200):
                values = [float(r["effect"]) for r in rows if int(r["jobs"]) == size]
                xs.append(size)
                ys.append(float(np.mean(values)))
            ax.plot(xs, ys, "o-", label=f"D2 - {b}", color=color)
        ax.axhline(0, c="0.5", linestyle="--")
        ax.set(
            xticks=[20, 50, 100, 200],
            xlabel="Jobs",
            ylabel="Mean HV gain (descriptive)",
            title=stage,
        )
        ax.legend()
    save(fig, "size_heterogeneity")

    with (args.analysis / "anytime_observations.csv").open(encoding="utf-8") as stream:
        curves = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cases = {}
    for ax, stage in zip(axes, ("screen", "confirm", "long")):
        ids = sorted({r["id"].rsplit("-", 1)[0] for r in curves if r["stage"] == stage})
        case = ids[0]
        cases[stage] = case
        for arm in ("S0", "W1", "D2", "R0"):
            rows = sorted(
                [r for r in curves if r["id"] == case + "-" + arm],
                key=lambda r: float(r["fraction"]),
            )
            ax.plot(
                [float(r["fraction"]) for r in rows],
                [float(r["hv"]) for r in rows],
                "o-",
                label=arm,
            )
        ax.set(
            xlabel="Recorded actual elapsed / nominal budget",
            ylabel="HV (separate anytime normalization)",
            title=stage,
        )
        ax.legend()
    save(fig, "anytime_fixed_cases")
    (args.out / "fixed_anytime_cases.json").write_text(
        json.dumps(cases, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
