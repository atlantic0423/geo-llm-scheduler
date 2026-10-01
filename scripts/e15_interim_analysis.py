"""Analyze only complete E15 stages, preserving protocol and base-instance pairing."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPLETE = {"A_GEN": 288, "A_TIME": 288, "B_GEN": 480}


def cluster_differences(
    rows: list[dict], stage: str, left: str, right: str, metric: str
) -> list[float]:
    """Average three seeds and both tariffs within each independent base instance."""
    groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["stage"] == stage and row["arm"] in (left, right):
            groups[(row["base"], row["arm"])].append(float(row[metric]))
    bases = sorted({base for base, _ in groups})
    if not bases or any(len(groups[(b, a)]) != 6 for b in bases for a in (left, right)):
        raise ValueError("Pairing requires all three seeds and both tariffs for each arm")
    return [mean(groups[(b, right)]) - mean(groups[(b, left)]) for b in bases]


def summarize_differences(values: list[float], seed: int = 20261001) -> dict:
    """Compute cluster bootstrap mean CI and a two-sided exact nonzero sign test."""
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Finite nonempty differences required")
    rng = np.random.default_rng(seed)
    array = np.asarray(values)
    boot = array[rng.integers(0, len(array), size=(20000, len(array)))].mean(axis=1)
    positive = sum(v > 0 for v in values)
    negative = sum(v < 0 for v in values)
    n = positive + negative
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1))
    return {
        "n_bases": len(values),
        "mean_difference": mean(values),
        "median_difference": median(values),
        "positive_bases": positive,
        "negative_bases": negative,
        "ties": len(values) - n,
        "bootstrap_mean_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "sign_p": min(1.0, 2 * tail / (2**n)) if n else 1.0,
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Adjust the declared family of sign tests with Holm's step-down method."""
    output = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        output[index] = running
    return output


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path, output: Path) -> dict:
    """Create derived tables and figures; refuse incomplete or duplicate stage cells."""
    rows = json.loads((root / "ops/completed_stage_metrics_20261001.json").read_text())
    for stage, count in COMPLETE.items():
        cells = [r for r in rows if r["stage"] == stage]
        identities = {(r["instance"], r["seed"], r["arm"]) for r in cells}
        if len(cells) != count or len(identities) != count:
            raise ValueError(f"Incomplete or duplicated stage: {stage}")
    if {r["stage"] for r in rows} != set(COMPLETE):
        raise ValueError("Do not mix unfinished protocols or calibration with complete stages")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    summaries = {}
    for path in (root / "runs").glob("*/summary.json"):
        value = json.loads(path.read_text())
        summaries[value["job_key"]] = value
    aggregate = []
    for stage in COMPLETE:
        for arm in sorted({r["arm"] for r in rows if r["stage"] == stage}):
            subset = [r for r in rows if r["stage"] == stage and r["arm"] == arm]
            raw = [s for k, s in summaries.items() if k.startswith(f"{stage}__{arm}__")]
            budgets: Counter = Counter()
            actions: dict[str, Counter] = defaultdict(Counter)
            for key, summary in summaries.items():
                if not key.startswith(f"{stage}__{arm}__"):
                    continue
                budgets.update(summary["budget_counts"])
                funnel = json.loads((root / "runs" / key / "budget_funnel.json").read_text())
                for action, values in funnel.items():
                    actions[action].update(values)
            for action, values in actions.items():
                values["positive_rate"] = values["positive"] / max(values["calls"], 1)
                values["no_candidate_rate"] = values["no_candidate"] / max(values["calls"], 1)
                values["fulfillment"] = values["effective"] / max(values["requested"], 1)
            aggregate.append(
                {
                    "stage": stage,
                    "arm": arm,
                    "runs": len(subset),
                    **{
                        f"mean_{metric}": mean(r[metric] for r in subset)
                        for metric in (
                            "hv",
                            "igd_plus",
                            "elapsed",
                            "exact",
                            "archive_size",
                            "flow_extreme",
                            "electricity_extreme",
                        )
                    },
                    "median_elapsed": median(r["elapsed"] for r in subset),
                    "mean_archive_fraction": mean(
                        s["timings"].get("archive", 0) / s["elapsed"] for s in raw
                    ),
                    "mean_ssgs_fraction": mean(
                        s["timings"].get("ssgs", 0) / s["elapsed"] for s in raw
                    ),
                    "mean_exact_fraction": mean(
                        s["timings"].get("exact", 0) / s["elapsed"] for s in raw
                    ),
                    "mean_generations": mean(s["generations_completed"] for s in raw),
                    "budget_counts": dict(budgets),
                    "action_funnel": dict(actions),
                    "mean_adaptive_cpu": mean(s["adaptive_cpu_seconds"] for s in raw),
                    "adaptive_counts": dict(
                        sum((Counter(s["adaptive_counts"]) for s in raw), Counter())
                    ),
                }
            )
    contrasts = []
    for stage in COMPLETE:
        pairs = (
            [("F6", a) for a in ("C2", "S2", "SEQ")]
            if stage[0] == "A"
            else [("M0", "N0"), ("M0", "M1"), ("N0", "N1"), ("M1", "N1")]
        )
        for left, right in pairs:
            for metric in ("hv", "igd_plus"):
                values = cluster_differences(rows, stage, left, right, metric)
                contrasts.append(
                    {
                        "stage": stage,
                        "left": left,
                        "right": right,
                        "metric": metric,
                        **summarize_differences(values),
                        "base_differences": values,
                    }
                )
    for record, adjusted in zip(contrasts, holm_adjust([r["sign_p"] for r in contrasts])):
        record["holm_p_20_tests"] = adjusted
    result = {
        "aggregate": aggregate,
        "contrasts": contrasts,
        "source_commit": next(iter(summaries.values()))["source_commit"],
        "selector": json.loads((root / "selected_adaptive_macrosearch.json").read_text()),
        "progress": json.loads((root / "ops/progress_validation_20261001.json").read_text())[
            "stages"
        ],
    }
    (output / "statistics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _csv(output / "complete_stage_metrics.csv", rows)
    _csv(
        output / "arm_summary.csv",
        [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in aggregate],
    )
    _csv(
        output / "paired_statistics.csv",
        [{k: v for k, v in r.items() if k != "base_differences"} for r in contrasts],
    )
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})

    def save(fig: plt.Figure, name: str) -> None:
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"{name}.{suffix}", dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, stage in enumerate(("A_GEN", "A_TIME")):
        for j, metric in enumerate(("hv", "igd_plus")):
            subset = [r for r in contrasts if r["stage"] == stage and r["metric"] == metric]
            ax = axes[i, j]
            for x, row in enumerate(subset):
                ax.scatter([x] * row["n_bases"], row["base_differences"], s=16, alpha=0.6)
                ax.scatter(x, row["mean_difference"], marker="D", c="black", s=35)
            ax.set_xticks(range(3), [r["right"] for r in subset])
            ax.axhline(0, c="gray", lw=1)
            ax.set_title(f"{'200代' if i == 0 else '等时间'}：相对固定6的{metric.upper()}差")
            ax.set_ylabel("按基础实例平均，点=实例；菱形=均值")
    save(fig, "01_stageA_paired")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ("hv", "igd_plus")):
        subset = [r for r in aggregate if r["stage"] == "B_GEN"]
        ax.bar([r["arm"] for r in subset], [r[f"mean_{metric}"] for r in subset])
        ax.set_title(f"B：200代，{metric.upper()} {'越高越好' if metric == 'hv' else '越低越好'}")
    save(fig, "02_stageB_generation_quality")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ("elapsed", "exact")):
        subset = [r for r in aggregate if r["stage"] == "B_GEN"]
        ax.bar([r["arm"] for r in subset], [r[f"mean_{metric}"] for r in subset])
        ax.set_title("B：200代，" + ("实际运行秒数" if metric == "elapsed" else "精确评价次数"))
    save(fig, "03_stageB_resource_cost")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, stage in zip(axes, ("A_GEN", "A_TIME")):
        subset = [r for r in aggregate if r["stage"] == stage]
        for i, budget in enumerate(("3", "6", "10")):
            vals = [
                r["budget_counts"].get(budget, 0) / max(sum(r["budget_counts"].values()), 1)
                for r in subset
            ]
            ax.bar(np.arange(4) + (i - 1) * 0.23, vals, width=0.23, label=f"预算{budget}")
        ax.set_xticks(range(4), [r["arm"] for r in subset])
        ax.set_title(f"{stage}：累计调用预算比例")
        ax.legend()
    save(fig, "04_budget_activation")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, stage in zip(axes, ("A_GEN", "B_GEN")):
        subset = [r for r in aggregate if r["stage"] == stage]
        ax.bar([r["arm"] for r in subset], [r["mean_archive_fraction"] for r in subset])
        ax.set_title(f"{stage}：档案维护占总时间比例")
    save(fig, "05_archive_cost")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, rate in zip(axes, ("positive_rate", "no_candidate_rate")):
        for offset, arm in enumerate(("M1", "N1")):
            entry = next(r for r in aggregate if r["stage"] == "B_GEN" and r["arm"] == arm)
            vals = [entry["action_funnel"][str(a)][rate] for a in range(1, 9)]
            ax.bar(np.arange(8) + (offset - 0.5) * 0.35, vals, width=0.35, label=arm)
        ax.set_xticks(range(8), [f"A{a}" for a in range(1, 9)])
        ax.set_title(
            "B：200代，" + ("正奖励调用比例" if rate == "positive_rate" else "无有效候选调用比例")
        )
        ax.legend()
    save(fig, "06_action_outcomes")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    subset = [r for r in contrasts if r["stage"] == "B_GEN" and r["metric"] == "hv"]
    for x, row in enumerate(subset):
        ax.scatter([x] * row["n_bases"], row["base_differences"], s=18, alpha=0.6)
        ax.scatter(x, row["mean_difference"], marker="D", c="black", s=40)
    ax.set_xticks(range(4), [r["right"] + "−" + r["left"] for r in subset])
    ax.axhline(0, c="gray", lw=1)
    ax.set_title("B：200代，20个独立基础实例的配对HV差")
    save(fig, "07_stageB_paired_hv")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.root, args.output)
