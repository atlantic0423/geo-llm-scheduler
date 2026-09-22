"""D03 calibration scoring, severity-scale statistics, transitions, and figures."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from geo_llm_scheduler.experiments.diagnostics import SEVERITY_NAMES
from geo_llm_scheduler.rl.state import CONDITION_LABELS, PREFERENCE_LABELS

ACTIONS = tuple(f"A{i}" for i in range(1, 9))
GENERATION_BINS = (
    (1, 1, "Generation 1"),
    (1, 10, "1-10"),
    (11, 20, "11-20"),
    (21, 30, "21-30"),
    (31, 40, "31-40"),
    (41, 50, "41-50"),
)
LOW_SAMPLE = 30


def read_steps(run_directories: list[Path]) -> list[dict[str, str]]:
    """Read exact run directories only, never glob unrelated calibration data."""
    rows: list[dict[str, str]] = []
    for directory in run_directories:
        with (directory / "rl_steps.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _ratio(row: dict[str, str], name: str) -> float:
    key = f"ratio_{name}"
    return (
        float(row[key]) if key in row and row[key] != "" else float(row[f"severity_{name}"]) / 0.2
    )


def _generation(row: dict[str, str]) -> int:
    return int(row["generation"]) + 1


def _bin_labels(generation: int) -> list[str]:
    labels = []
    for low, high, label in GENERATION_BINS:
        if low <= generation <= high:
            labels.append(label)
    return labels


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a stable union schema for diagnostic tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else math.nan


def severity_tables(
    rows: list[dict[str, str]], suite: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return required pooled/run/generation severity and argmax competition tables."""
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        labels = ["all", *_bin_labels(_generation(row))]
        for label in labels:
            grouped[(row["run_id"], label, "run")].append(row)
            grouped[(suite, label, "suite")].append(row)
    summary: list[dict] = []
    margins: list[dict] = []
    for (group, generation_bin, level), values in grouped.items():
        ratios_by_row = [[_ratio(row, name) for name in SEVERITY_NAMES] for row in values]
        for index, name in enumerate(SEVERITY_NAMES):
            raw = [float(row[f"severity_{name}"]) for row in values]
            ratio = [item[index] for item in ratios_by_row]
            exceeded = [value > 1 for value in ratio]
            wins = [item[index] == max(item) for item in ratios_by_row]
            summary.append(
                {
                    "suite": suite,
                    "level": level,
                    "group": group,
                    "generation_bin": generation_bin,
                    "severity": name,
                    "count": len(ratio),
                    "raw_mean": statistics.fmean(raw),
                    "raw_median": statistics.median(raw),
                    "raw_q1": _quantile(raw, 0.25),
                    "raw_q3": _quantile(raw, 0.75),
                    "raw_p90": _quantile(raw, 0.90),
                    "raw_p95": _quantile(raw, 0.95),
                    "raw_min": min(raw),
                    "raw_max": max(raw),
                    "ratio_mean": statistics.fmean(ratio),
                    "ratio_median": statistics.median(ratio),
                    "ratio_q1": _quantile(ratio, 0.25),
                    "ratio_q3": _quantile(ratio, 0.75),
                    "ratio_iqr": _quantile(ratio, 0.75) - _quantile(ratio, 0.25),
                    "ratio_p90": _quantile(ratio, 0.90),
                    "ratio_p95": _quantile(ratio, 0.95),
                    "ratio_min": min(ratio),
                    "ratio_max": max(ratio),
                    "p_above_threshold": sum(exceeded) / len(values),
                    "p_argmax": sum(wins) / len(values),
                    "p_argmax_given_above": sum(w and e for w, e in zip(wins, exceeded))
                    / max(1, sum(exceeded)),
                }
            )
        top = [sorted(item, reverse=True) for item in ratios_by_row]
        margins.append(
            {
                "suite": suite,
                "level": level,
                "group": group,
                "generation_bin": generation_bin,
                "count": len(top),
                "margin_mean": statistics.fmean(x[0] - x[1] for x in top),
                "margin_median": statistics.median(x[0] - x[1] for x in top),
                "margin_p90": _quantile([x[0] - x[1] for x in top], 0.90),
            }
        )
    pairwise: list[dict] = []
    for left, lname in enumerate(SEVERITY_NAMES):
        for right, rname in enumerate(SEVERITY_NAMES):
            pairwise.append(
                {
                    "suite": suite,
                    "left": lname,
                    "right": rname,
                    "count": len(rows),
                    "win_rate": sum(_ratio(row, lname) > _ratio(row, rname) for row in rows)
                    / max(1, len(rows)),
                }
            )
    return summary, pairwise, margins


def calibration_summary(rows: list[dict[str, str]], target: str | None = None) -> dict[str, object]:
    """Score a candidate profile from observed labels without changing state semantics."""
    counts = Counter(row["dominant_condition"] for row in rows)
    per_run: dict[str, Counter] = defaultdict(Counter)
    transitions = 0
    for row in rows:
        per_run[row["run_id"]][row["dominant_condition"]] += 1
    ordered = sorted(
        rows,
        key=lambda row: (
            row["run_id"],
            _generation(row),
            int(row["subproblem"]),
            int(row.get("trajectory_step", 0)),
        ),
    )
    for previous, current in zip(ordered, ordered[1:]):
        transitions += int(
            previous["run_id"] == current["run_id"]
            and previous["dominant_condition"] != current["dominant_condition"]
        )
    effective = [sum(value >= 10 for value in counter.values()) for counter in per_run.values()]
    return {
        "steps": len(rows),
        "counts": dict(counts),
        "target_share": counts[target] / max(1, len(rows)) if target else None,
        "distinct_mean": statistics.fmean(len(c) for c in per_run.values()),
        "effective_distinct_mean": statistics.fmean(effective),
        "transition_count": transitions,
    }


def transition_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Compute per-run coverage, transition, dwell, and first-generation diagnostics."""
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
    output = []
    for run_id, values in by_run.items():
        values.sort(
            key=lambda row: (
                _generation(row),
                int(row["subproblem"]),
                int(row.get("trajectory_step", 0)),
            )
        )
        labels = [row["dominant_condition"] for row in values]
        counts = Counter(labels)
        dwell = best = 1
        for previous, current in zip(labels, labels[1:]):
            dwell = dwell + 1 if current == previous else 1
            best = max(best, dwell)
        first = {
            label: min(_generation(row) for row in values if row["dominant_condition"] == label)
            for label in counts
        }
        output.append(
            {
                "run_id": run_id,
                "instance_seed": values[0]["instance_seed"],
                "algorithm_seed": values[0]["algorithm_seed"],
                "distinct": len(counts),
                **{
                    f"effective_{n}": sum(value >= n for value in counts.values())
                    for n in (1, 5, 10, 20)
                },
                "transitions": sum(a != b for a, b in zip(labels, labels[1:])),
                "longest_dwell": best,
                "counts_json": json.dumps(counts, sort_keys=True),
                "first_generation_json": json.dumps(first, sort_keys=True),
            }
        )
    return output


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_figures(rows: list[dict[str, str]], suite: str, directory: Path) -> None:
    """Create the D03 v2 severity, state, transition, and action figure set."""
    condition_counts = Counter(row["dominant_condition"] for row in rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(CONDITION_LABELS, [condition_counts[x] for x in CONDITION_LABELS])
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"{suite}: DominantCondition counts")
    _save(fig, directory, "condition_distribution")
    data = [[_ratio(row, name) for row in rows] for name in SEVERITY_NAMES]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=SEVERITY_NAMES, showfliers=False)
    ax.axhline(1, color="red", ls="--")
    ax.set_ylabel("Severity ratio R=d/theta")
    ax.set_title(f"{suite}: pooled severity-ratio distribution")
    _save(fig, directory, "severity_ratio_distribution")
    means = np.zeros((6, 6))
    counts = np.zeros((6, 6), dtype=int)
    labels = ("Gen 1", "1-10", "11-20", "21-30", "31-40", "41-50")
    for j, (low, high, _) in enumerate(GENERATION_BINS):
        generation_rows = [row for row in rows if low <= _generation(row) <= high]
        for i, name in enumerate(SEVERITY_NAMES):
            counts[i, j] = len(generation_rows)
            means[i, j] = (
                statistics.fmean([_ratio(row, name) for row in generation_rows])
                if generation_rows
                else math.nan
            )
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(means, aspect="auto", cmap="viridis")
    ax.set_xticks(range(6), labels)
    ax.set_yticks(range(6), SEVERITY_NAMES)
    fig.colorbar(image, ax=ax, label="Mean R")
    ax.set_title(f"{suite}: severity ratio by generation")
    _save(fig, directory, "severity_ratio_by_generation")
    wins = np.array(
        [
            [
                sum(_ratio(row, a) > _ratio(row, b) for row in rows) / max(1, len(rows))
                for b in SEVERITY_NAMES
            ]
            for a in SEVERITY_NAMES
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(wins, vmin=0, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(6), SEVERITY_NAMES, rotation=30)
    ax.set_yticks(range(6), SEVERITY_NAMES)
    fig.colorbar(image, ax=ax, label="P(row > column)")
    ax.set_title(f"{suite}: pairwise ratio win rate")
    _save(fig, directory, "severity_pairwise_win_rate")
    margins = []
    for row in rows:
        ordered = sorted((_ratio(row, name) for name in SEVERITY_NAMES), reverse=True)
        margins.append(ordered[0] - ordered[1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(margins, bins=40)
    ax.set_xlabel("Top1 - Top2 severity ratio")
    ax.set_title(f"{suite}: argmax competition margin")
    _save(fig, directory, "severity_top_margin")
    visits = np.zeros((2, 3, 7), dtype=int)
    for row in rows:
        visits[
            int(row["search_progress"] == "Stagnating"),
            PREFERENCE_LABELS.index(row["preference"]),
            CONDITION_LABELS.index(row["dominant_condition"]),
        ] += 1
    for progress, label in enumerate(("Improving", "Stagnating")):
        fig, ax = plt.subplots(figsize=(11, 4))
        image = ax.imshow(np.log1p(visits[progress]), aspect="auto", cmap="Blues")
        ax.set_xticks(range(7), CONDITION_LABELS, rotation=30)
        ax.set_yticks(range(3), PREFERENCE_LABELS)
        for i in range(3):
            for j in range(7):
                ax.text(j, i, str(visits[progress, i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="log(1+visits)")
        ax.set_title(f"{suite}: {label} state coverage")
        _save(fig, directory, f"state_coverage_{label.lower()}")
    cgen = np.zeros((7, 5), dtype=int)
    for row in rows:
        g = _generation(row)
        generation_bin = min(4, (g - 1) // 10)
        cgen[CONDITION_LABELS.index(row["dominant_condition"]), generation_bin] += 1
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(cgen, aspect="auto", cmap="magma")
    ax.set_xticks(range(5), ("1-10", "11-20", "21-30", "31-40", "41-50"))
    ax.set_yticks(range(7), CONDITION_LABELS)
    fig.colorbar(image, ax=ax, label="Visits")
    ax.set_title(f"{suite}: condition by generation")
    _save(fig, directory, "condition_by_generation")
    trans = np.zeros((7, 7), dtype=int)
    byrun: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        byrun[row["run_id"]].append(row)
    for values in byrun.values():
        values.sort(
            key=lambda row: (
                _generation(row),
                int(row["subproblem"]),
                int(row.get("trajectory_step", 0)),
            )
        )
        for prior_row, next_row in zip(values, values[1:]):
            trans[
                CONDITION_LABELS.index(prior_row["dominant_condition"]),
                CONDITION_LABELS.index(next_row["dominant_condition"]),
            ] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(trans, cmap="magma")
    ax.set_xticks(range(7), CONDITION_LABELS, rotation=30)
    ax.set_yticks(range(7), CONDITION_LABELS)
    fig.colorbar(image, ax=ax, label="Transitions")
    ax.set_title(f"{suite}: adjacent condition transitions")
    _save(fig, directory, "condition_transition_matrix")
    selection_counts = np.zeros((7, 8), dtype=int)
    success = np.zeros((7, 8))
    reward = np.zeros((7, 8))
    for row in rows:
        condition_index = CONDITION_LABELS.index(row["dominant_condition"])
        action_index = int(row["selected_action"][1:]) - 1
        selection_counts[condition_index, action_index] += 1
        success[condition_index, action_index] += float(row["reward"]) > 0
        reward[condition_index, action_index] += float(row["reward"])
    for matrix, stem, title in (
        (
            selection_counts / np.maximum(selection_counts.sum(axis=1, keepdims=True), 1),
            "condition_action_selection",
            "Selection probability",
        ),
        (
            success / np.maximum(selection_counts, 1),
            "condition_action_success",
            "Positive improvement rate",
        ),
        (reward / np.maximum(selection_counts, 1), "condition_action_reward", "Mean reward"),
    ):
        fig, ax = plt.subplots(figsize=(11, 7))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(8), ACTIONS)
        ax.set_yticks(range(7), CONDITION_LABELS)
        for i in range(7):
            for j in range(8):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}\nn={selection_counts[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if selection_counts[i, j] >= LOW_SAMPLE else "gray",
                )
        fig.colorbar(image, ax=ax)
        ax.set_title(f"{suite}: {title}")
        _save(fig, directory, stem)
    entropy = []
    for b in range(5):
        subset = [row for row in rows if min(4, (_generation(row) - 1) // 10) == b]
        action_counts = Counter(row["selected_action"] for row in subset)
        total = sum(action_counts.values())
        p = [v / total for v in action_counts.values()] if total else []
        entropy.append(-sum(x * math.log(x) for x in p))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(("1-10", "11-20", "21-30", "31-40", "41-50"), entropy, marker="o")
    ax.set_ylabel("Selection entropy")
    ax.set_title(f"{suite}: policy entropy over search")
    _save(fig, directory, "policy_entropy")


def aggregate_suite(
    rows: list[dict[str, str]], suite: str, output: Path, figures: Path
) -> dict[str, object]:
    """Write all required D03 v2 tables and figures for one isolated suite."""
    summary, pairwise, margins = severity_tables(rows, suite)
    write_csv(output / "severity_scale_summary.csv", summary)
    write_csv(output / "severity_argmax_competition.csv", pairwise)
    write_csv(output / "severity_margin_summary.csv", margins)
    transitions = transition_rows(rows)
    write_csv(output / "run_transition_summary.csv", transitions)
    by_generation = []
    for generation in sorted({_generation(row) for row in rows}):
        subset = [row for row in rows if _generation(row) == generation]
        for name in SEVERITY_NAMES:
            values = [_ratio(row, name) for row in subset]
            by_generation.append(
                {
                    "suite": suite,
                    "generation": generation,
                    "severity": name,
                    "count": len(values),
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "p_above_threshold": sum(v > 1 for v in values) / len(values),
                    "p_argmax": sum(
                        _ratio(row, name) == max(_ratio(row, n) for n in SEVERITY_NAMES)
                        for row in subset
                    )
                    / len(subset),
                }
            )
    write_csv(output / "severity_by_generation.csv", by_generation)
    create_figures(rows, suite, figures)
    covered = len({int(row["state_id"]) for row in rows})
    counts = Counter(row["dominant_condition"] for row in rows)
    result = {
        "suite": suite,
        "steps": len(rows),
        "state_coverage": covered,
        "condition_counts": dict(counts),
        "runs": len({row["run_id"] for row in rows}),
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def write_initial_severity_summary(rows: list[dict[str, str]], suite: str, output: Path) -> None:
    """Summarize generation-zero population severities outside RL visit counts."""
    summary = []
    for name in SEVERITY_NAMES:
        values = [_ratio(row, name) for row in rows]
        summary.append(
            {
                "suite": suite,
                "generation": 0,
                "severity": name,
                "count": len(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "q1": _quantile(values, 0.25),
                "q3": _quantile(values, 0.75),
                "p_above_threshold": sum(value > 1 for value in values) / len(values),
                "p_argmax": sum(
                    _ratio(row, name) == max(_ratio(row, other) for other in SEVERITY_NAMES)
                    for row in rows
                )
                / len(rows),
            }
        )
    write_csv(output / "initial_population_severity.csv", summary)


def write_policy_consensus(run_directories: list[Path], output: Path) -> None:
    """Write final greedy argmax consensus with run and visit support per state."""
    rows = []
    qtables = [
        json.loads((directory / "qtable.json").read_text(encoding="utf-8"))
        for directory in run_directories
    ]
    for state in range(42):
        actions = []
        visits = 0
        for table in qtables:
            count = int(table["visits"][state])
            visits += count
            if count:
                row = table["q"][state]
                actions.append(int(np.argmax(row)) + 1)
        mode = Counter(actions).most_common(1)[0] if actions else (0, 0)
        rows.append(
            {
                "state_id": state,
                "argmax_action": f"A{mode[0]}" if mode[0] else "",
                "consensus": mode[1] / len(actions) if actions else math.nan,
                "effective_runs": len(actions),
                "visits": visits,
            }
        )
    write_csv(output / "greedy_policy_consensus.csv", rows)


def create_representative_timelines(
    rows: list[dict[str, str]], output: Path, figures: Path
) -> None:
    """Select min/median/max distinct runs by a deterministic rule and plot timelines."""
    summaries = transition_rows(rows)
    ordered = sorted(summaries, key=lambda row: (int(str(row["distinct"])), str(row["run_id"])))
    chosen = [ordered[0], ordered[(len(ordered) - 1) // 2], ordered[-1]]
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
    figure, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    selection_rows = []
    for axis, label, selected in zip(axes, ("minimum", "median", "maximum"), chosen):
        values = by_run[str(selected["run_id"])]
        matrix = np.zeros((7, 50), dtype=int)
        for row in values:
            matrix[CONDITION_LABELS.index(row["dominant_condition"]), _generation(row) - 1] += 1
        axis.stackplot(np.arange(1, 51), matrix, labels=CONDITION_LABELS)
        axis.set_title(f"{label}: {selected['run_id']} (distinct={selected['distinct']})")
        axis.set_ylabel("RL visits")
        selection_rows.append({"selection_rule": label, **selected})
    axes[-1].set_xlabel("Generation")
    axes[0].legend(ncol=4, fontsize=7, loc="upper right")
    _save(figure, figures, "representative_condition_timelines")
    write_csv(output / "representative_timeline_selection.csv", selection_rows)
