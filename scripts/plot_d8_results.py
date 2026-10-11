"""Draw frozen effect intervals, anytime curves and descriptive budget summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_d8_results import ARMS, FACTORS


def main() -> None:
    """Read completed tables only; preserve all arms and all effect backgrounds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis
    report = json.loads((root / "report.json").read_text())
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    for scope, title, name in (
        ("statistics", "All fifteen arms versus BASE", "all_arm_effects"),
        (
            "interaction_statistics",
            "Pairwise interactions averaged over all backgrounds",
            "pair_interactions",
        ),
    ):
        contrasts = (
            [a + "-BASE" for a in ARMS[1:]] if scope == "statistics" else list(report[scope])
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 8 if scope == "statistics" else 4.5))
        for ax, metric, label in zip(
            axes,
            ("hv_diff", "igd_plus_gain"),
            ("Normalized HV difference", "Normalized IGD+ reduction"),
            strict=True,
        ):
            for i, contrast in enumerate(contrasts):
                value = report[scope][contrast][metric]
                mean = value["mean"]
                low, high = value["ci95"]
                ax.plot([low, high], [i, i], color="#4279a9")
                ax.plot(mean, i, "o", color="#4279a9" if mean >= 0 else "#d55d3e")
            ax.set_yticks(range(len(contrasts)), [c.removesuffix("-BASE") for c in contrasts])
            ax.invert_yaxis()
            ax.axvline(0, color="#555", linewidth=0.8)
            ax.grid(axis="x", alpha=0.2)
            ax.set_xlabel(label + " (positive = better)")
        fig.suptitle(title + "; unadjusted 95% cluster intervals")
        fig.tight_layout()
        fig.savefig(figures / (name + ".png"), dpi=160)
        plt.close(fig)
    selected = [ARMS[m] for m in (0, 1, 2, 4, 8, 15)]
    snapshots = list(csv.DictReader((root / "anytime_metrics.csv").open(encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(8, 5))
    anytime = {}
    for arm in ARMS:
        grouped = defaultdict(list)
        for row in snapshots:
            if row["arm"] == arm:
                grouped[float(row["fraction"])].append(float(row["hv"]))
        means = {f: float(np.mean(v)) for f, v in sorted(grouped.items())}
        anytime[arm] = means
        if arm in selected:
            ax.plot(list(means), list(means.values()), "o-", label=arm)
    ax.set(
        xlabel="Fraction of equal wall budget",
        ylabel="Mean normalized HV",
        title="Descriptive anytime quality",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "anytime_selected.png", dpi=160)
    plt.close(fig)
    budget = list(csv.DictReader((root / "budget_metrics.csv").open(encoding="utf-8")))
    q = list(csv.DictReader((root / "q_state_metrics.csv").open(encoding="utf-8")))
    actions = list(csv.DictReader((root / "state_action_metrics.csv").open(encoding="utf-8")))
    diagnostics = {"anytime": anytime, "max_snapshot_delay_seconds": report["max_snapshot_delay"]}
    for arm in ARMS:
        hist = defaultdict(int)
        for row in budget:
            if row["arm"] == arm:
                hist[(int(row["requested"]), int(row["effective"]))] += int(row["calls"])
        qr = [row for row in q if row["arm"] == arm]
        ar = [row for row in actions if row["arm"] == arm]
        counts = defaultdict(int)
        for row in ar:
            calls = float(row["calls"])
            if not calls.is_integer():
                raise ValueError("Observed call counts must be whole numbers")
            counts[int(row["action"])] += int(calls)
        total = sum(counts.values())
        diagnostics[arm] = {
            "requested_effective_budget_calls": {str(k): v for k, v in sorted(hist.items())},
            "q_rows": len(qr),
            "unvisited_state_fraction": sum(int(r["visits"]) == 0 for r in qr) / len(qr),
            "action_call_share": {str(k): v / total for k, v in sorted(counts.items())},
        }
    (root / "descriptive_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(json.dumps({"figures": 3, "arms": len(ARMS), "factors": FACTORS}))


if __name__ == "__main__":
    main()
