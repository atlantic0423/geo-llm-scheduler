"""Export every instance, learning diagnostics and comparable anytime curves."""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(sys.argv[1]) / "outputs/lsv2_budget12_20261028"
A = ROOT / "analysis"
F = ROOT / "figures"
Q6 = "Q1_NET_G9_B6L5"
Q3 = "Q1_NET_G9_B3L3"
Q1 = "Q1_NET_G0_B1L1"
U6 = "UNIFORM_B6L5"
NAMES = {
    Q6: "Q gamma=.9 B6L5",
    Q3: "Q gamma=.9 B3L3",
    Q1: "Q gamma=0 B1L1",
    "Q1_NET_G0_B6L5": "Q gamma=0 B6L5",
    "Q1_NET_G0_B3L3": "Q gamma=0 B3L3",
    U6: "Uniform B6L5",
    "UNIFORM_B3L3": "Uniform B3L3",
    "UNIFORM_B1L1": "Uniform B1L1",
    Q6 + "_NOA8": "Q gamma=.9 B6L5 no A8",
    "UNIFORM_B6L5_NOA8": "Uniform B6L5 no A8",
    "BILL_RANDOM": "BILL4 + MOEA/D",
    "EXTH_SPEA2": "Enhanced SPEA2",
    "EXTH_MOEAD": "Enhanced MOEA/D",
    "EXTH_NSGAII": "Enhanced NSGA-II",
    "EXTH_EPS_IBEA": "Enhanced epsilon-IBEA",
    "EXTH_RK_MOPSO": "Enhanced RK-MOPSO",
}
COLORS = {
    Q6: "#d7354a",
    Q3: "#1c6dc1",
    Q1: "#9c52a7",
    U6: "#222222",
    "UNIFORM_B3L3": "#1c9c77",
    "UNIFORM_B1L1": "#b28b19",
    Q6 + "_NOA8": "#e79025",
    "UNIFORM_B6L5_NOA8": "#7a7a7a",
    "Q1_NET_G0_B6L5": "#f299a7",
    "Q1_NET_G0_B3L3": "#78abde",
    "BILL_RANDOM": "#333333",
    "EXTH_SPEA2": "#009688",
    "EXTH_MOEAD": "#8957a1",
    "EXTH_NSGAII": "#c58a2b",
    "EXTH_EPS_IBEA": "#6f889a",
    "EXTH_RK_MOPSO": "#ac8e79",
}


def _csv(name: str) -> list[dict]:
    return list(csv.DictReader((A / name).open(encoding="utf-8-sig")))


def main() -> None:
    """Render all predefined panels, with identical units and unpooled single-run fronts."""
    F.mkdir(exist_ok=True)
    fronts = json.loads((A / "fronts.json").read_text())
    result = json.loads((A / "analysis.json").read_text())
    parents = sorted(fronts, key=lambda p: (fronts[p]["n"], p))
    groups = [
        ("Q budget comparison", [Q6, "Q1_NET_G0_B6L5", Q3, Q1]),
        ("Uniform budget comparison", [U6, "UNIFORM_B3L3", "UNIFORM_B1L1"]),
        ("A8 ablation", [Q6, Q6 + "_NOA8", U6, "UNIFORM_B6L5_NOA8"]),
        (
            "External methods (historical curves noted in report)",
            [
                Q6,
                Q3,
                "BILL_RANDOM",
                "EXTH_SPEA2",
                "EXTH_MOEAD",
                "EXTH_NSGAII",
                "EXTH_EPS_IBEA",
                "EXTH_RK_MOPSO",
            ],
        ),
    ]

    def draw(ax, par, arms):
        for arm in arms:
            p = np.array(fronts[par]["runs"][arm])
            ax.plot(
                p[:, 0],
                p[:, 1],
                label=NAMES[arm],
                color=COLORS[arm],
                lw=1.7 if arm in (Q6, Q3, U6) else 1.15,
                alpha=0.95,
            )
        ax.set_xlabel("Total flow time")
        ax.set_ylabel("Electricity bill")
        ax.grid(alpha=0.2)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.legend(fontsize=7, ncol=2)

    for par in parents:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        for ax, (title, arms) in zip(axs.flat, groups):
            draw(ax, par, arms)
            ax.set_title(title, fontsize=11)
        fig.suptitle(
            f"{par} | seed 16901 | lower-left is better | each curve = one run", fontsize=14
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(F / f"{par}.png", dpi=170)
        plt.close(fig)
    internal = [a for a in result["arms"] if not a.startswith("EXTH")]
    metrics = _csv("run_metrics.csv")
    by = {(r["parent"], r["arm"]): float(r["hv"]) for r in metrics}
    matrix = np.array([[by[p, a] - by[p, U6] for a in internal] for p in parents])
    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-0.12, vmax=0.12, aspect="auto")
    ax.set_xticks(
        range(len(internal)), [NAMES[a] for a in internal], rotation=35, ha="right", fontsize=9
    )
    ax.set_yticks(range(16), parents, fontsize=9)
    for i in range(16):
        for j in range(len(internal)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(matrix[i, j]) > 0.075 else "black",
            )
    ax.set_title(
        "HV difference vs matched Uniform B6L5 | blue = better\nColor scale saturates at +/-0.12; numbers are exact differences",
        fontsize=12,
    )
    fig.colorbar(im, ax=ax, label="Normalized HV difference", shrink=0.75)
    fig.tight_layout()
    fig.savefig(F / "hv_differences.png", dpi=180)
    plt.close(fig)
    anyrows = _csv("anytime.csv")
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    for col, n in enumerate((20, 50, 100, 200)):
        for row, arms in enumerate(([Q6, Q3, Q1], [U6, "UNIFORM_B3L3", "UNIFORM_B1L1"])):
            ax = axs[row, col]
            for a in arms:
                zz = [r for r in anyrows if int(r["n"]) == n and r["arm"] == a]
                xs = sorted({float(r["fraction"]) for r in zz})
                ys = [
                    np.mean([float(r["hv"]) for r in zz if float(r["fraction"]) == x]) for x in xs
                ]
                ax.plot(xs, ys, "o-", color=COLORS[a], label=NAMES[a], ms=3)
            ax.set_title(f"n={n} | 4 instances")
            ax.set_xlabel("Fraction of wall-clock budget")
            ax.set_ylabel("Mean normalized HV")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
    fig.suptitle(
        "Budget trajectories | fixed post-hoc reference; initialization is not backdated",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(F / "anytime.png", dpi=180)
    plt.close(fig)
    ph = _csv("phase_diagnostics.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for a in (Q6, Q3, Q1):
        for ax, field, title in zip(
            axs, ("p3_share", "empty_share"), ("P3 selection share", "Empty-proposal step share")
        ):
            yy = [
                100
                * np.mean([float(r[field]) for r in ph if r["arm"] == a and int(r["phase"]) == i])
                for i in range(4)
            ]
            ax.plot(range(1, 5), yy, "o-", label=NAMES[a], color=COLORS[a])
            ax.set_title(title)
            ax.set_xticks(range(1, 5))
            ax.set_xlabel("Cumulative LS-time quarter")
            ax.set_ylabel("Per-run mean (%)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Descriptive learning behavior, not a causal action-value estimate")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(F / "learning_behavior.png", dpi=180)
    plt.close(fig)
    for n in (100, 200):
        ps = [p for p in parents if fronts[p]["n"] == n]
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        for ax, par in zip(axs.flat, ps):
            draw(ax, par, [Q6, Q3, Q1, U6, Q6 + "_NOA8"])
            ax.set_title(par)
        fig.suptitle(f"All four n={n} development instances | seed 16901 | no pooled runs")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(F / f"overview_n{n}.png", dpi=180)
        plt.close(fig)
    print(json.dumps(dict(png_files=len(list(F.glob("*.png"))))))


if __name__ == "__main__":
    main()
