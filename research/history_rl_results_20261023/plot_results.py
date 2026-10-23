"""Scientific summary figures and one fixed-case four-scale front sheet per arm."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from geo_llm_scheduler.experiments.history_rl import ARMS

LABELS = tuple(a.replace("_", " ") for a in ARMS)


def table(path: Path) -> list[dict]:
    """Read scalar observations without changing their original precision."""
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, folder: Path, name: str) -> None:
    """Export paired publication figures at a fixed reproducible resolution."""
    fig.savefig(folder / (name + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot(output: Path, figures: Path) -> None:
    """Generate scale, cost, behavior and per-arm front comparisons."""
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    rows = table(output / "arm_summary.csv")
    lookup = {(int(r["n"]), r["arm"]): r for r in rows}
    values = np.array(
        [[float(lookup[(n, a)]["hv_vs_bill_pct"]) for n in (20, 50, 100, 200)] for a in ARMS]
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    bound = max(1, float(abs(values).max()))
    im = ax.imshow(
        values, cmap="RdBu", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto"
    )
    ax.set_xticks(range(4), ["20 jobs", "50 jobs", "100 jobs", "200 jobs"])
    ax.set_yticks(range(12), LABELS)
    for i in range(12):
        for j in range(4):
            ax.text(j, i, f"{values[i, j]:+.2f}%", ha="center", va="center", fontsize=8)
    ax.set_title(
        "Paired final-Archive HV relative to plain permuted MOEA/D\nDescriptive by scale; four development parents per scale"
    )
    fig.colorbar(im, ax=ax, label="Paired relative HV (%)", shrink=0.7)
    fig.tight_layout()
    save(fig, figures, "01_quality_by_scale")

    fig, ax = plt.subplots(figsize=(10, 7))
    cost_fields = (
        "initialization_share",
        "ls_measured_share",
        "polish_share",
        "outer_evaluation_share",
        "unclassified_share",
    )
    names = (
        "Initialization",
        "LS measured components",
        "Polish",
        "Outer evaluation",
        "Remaining support/outer work",
    )
    left = np.zeros(12)
    for field, name in zip(cost_fields, names):
        v = np.array([float(lookup[(200, a)][field]) * 100 for a in ARMS])
        ax.barh(range(12), v, left=left, label=name)
        left += v
    ax.set_yticks(range(12), LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean fraction of production wall clock (%)")
    ax.set_title(
        "200-job mutually exclusive wall-clock decomposition\nRemaining work is not Q-learning time"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, figures, "02_cost_200")

    parents = table(output / "parent_metrics.csv")
    by = {(int(r["base_seed"]), r["arm"]): r for r in parents}
    bases = sorted({b for b, _ in by})
    sizes = np.array([int(by[(b, ARMS[0])]["n"]) for b in bases])
    fig, ax = plt.subplots(figsize=(9, 7))
    rng = np.random.default_rng(2026102302)
    for i, arm in enumerate(ARMS):
        x = np.array([float(by[(b, arm)]["hv"]) - float(by[(b, ARMS[1])]["hv"]) for b in bases])
        boot = np.zeros(20000)
        for n in (20, 50, 100, 200):
            z = x[sizes == n]
            boot += z[rng.integers(0, 4, (20000, 4))].mean(1) / 4
        lo, hi = np.quantile(boot, [0.025, 0.975])
        ax.errorbar(
            x.mean(), i, xerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o", color="C0", capsize=2
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(12), LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized HV difference vs Q42 two-focus (larger is better)")
    ax.set_title(
        "Descriptive comparisons to Q42: sixteen paired parents, marginal 95% intervals\nIntervals do not establish multiplicity-corrected significance"
    )
    fig.tight_layout()
    save(fig, figures, "03_paired_effects")

    actions = table(output / "action_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    for col, field in enumerate(("steps", "action_seconds", "accepted")):
        matrix = np.zeros((12, 8))
        for i, arm in enumerate(ARMS):
            sub = [r for r in actions if r["arm"] == arm and int(r["n"]) == 200]
            for j in range(8):
                chosen = [r for r in sub if int(r["action"]) == j + 1]
                numerator = sum(float(r[field]) for r in chosen)
                denominator = (
                    sum(float(r["steps"]) for r in chosen)
                    if field == "accepted"
                    else sum(float(r[field]) for r in sub)
                )
                matrix[i, j] = 100 * numerator / max(1e-12, denominator)
        im = axes[col].imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
        axes[col].set_xticks(range(8), [f"A{i}" for i in range(1, 9)])
        axes[col].set_title(
            ("Selected-step share", "Action-time share", "Conditional acceptance")[col]
        )
        axes[col].set_yticks(range(12), LABELS)
    fig.suptitle(
        "200-job operator exposure and conditional returns (%)\nPooled calls describe visited contexts; they are not randomized operator effects"
    )
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, pad=0.02)
    save(fig, figures, "04_action_exposure_200")

    fronts = json.loads((output / "normalized_fronts.json").read_text())
    fixed = {
        n: min(b for b in bases if int(by[(b, ARMS[0])]["n"]) == n) for n in (20, 50, 100, 200)
    }
    selected = {}
    for n, base in fixed.items():
        keys = sorted(
            k.split("|")[0]
            for k in fronts
            if k.endswith("|" + ARMS[0]) and f"b{base}_v0" == fronts[k]["reference"]
        )
        selected[n] = keys[0]
    for i, arm in enumerate(ARMS):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (n, key) in zip(axes.flat, selected.items()):
            for other, color, label in [
                (ARMS[0], "C1", "Plain permuted"),
                (ARMS[1], "C2", "Q42 two-focus"),
                (arm, "C0", LABELS[i]),
            ]:
                a = np.asarray(fronts[key + "|" + other]["archive"])
                ax.scatter(a[:, 0], a[:, 1], s=6, color=color, label=label, alpha=0.7)
            initial = np.asarray(fronts[key + "|" + arm]["initial_population"])
            ax.scatter(
                initial[:, 0], initial[:, 1], marker="x", s=18, color="0.4", label="Initial P100"
            )
            ax.set_xlim(-0.03, 1.12)
            ax.set_ylim(-0.03, 1.12)
            ax.set_title(f"{n} jobs | fixed parent {fixed[n]} | variant0/seed16301")
            ax.set_xlabel("Normalized Flow (lower is better)")
            ax.set_ylabel("Normalized Bill")
            ax.legend(fontsize=6)
            ax.grid(alpha=0.2)
        fig.suptitle(
            LABELS[i] + ": fixed cases, no best-case selection; empirical union normalization"
        )
        fig.tight_layout()
        save(fig, figures, f"arm_{i + 1:02d}_fixed_fronts")
    (figures / "FIXED_CASES.json").write_text(
        json.dumps({str(n): dict(parent=fixed[n], key=k) for n, k in selected.items()}, indent=2)
    )
    print(json.dumps(dict(figure_pairs=16, fixed_cases=selected)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    plot(args.output, args.figures)
