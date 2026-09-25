"""E14 offline relabeling of frozen D03 raw severities; no Q-table reuse."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from geo_llm_scheduler.rl.state import CONDITION_LABELS, PREFERENCE_LABELS, PROGRESS_LABELS

SEVERITY_NAMES = ("resource", "kv", "region", "tou", "demand", "compressible")


def relabel(values: tuple[float, ...], thresholds: tuple[float, ...]) -> int:
    """Apply the current six-threshold ratio rule, with Normal as residual state."""
    ratios = tuple(value / threshold for value, threshold in zip(values, thresholds))
    return 0 if max(ratios) <= 1 else 1 + max(range(6), key=lambda i: ratios[i])


def _quantile(values: list[float], portion: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(portion * (len(ordered) - 1)))]


def offline_shortlist(raw_root: Path) -> dict[str, Any]:
    """Summarize all D03 raw steps and produce baseline plus three reproducible candidates."""
    paths = sorted(
        path for suite in ("type_i", "type_ii") for path in (raw_root / suite).rglob("rl_steps.csv")
    )
    if not paths:
        raise FileNotFoundError("Frozen D03 raw severity files unavailable")
    rows: list[tuple[tuple[float, ...], str, int, int, str, str]] = []
    samples: list[list[float]] = [[] for _ in range(6)]
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                values = tuple(float(row[f"severity_{name}"]) for name in SEVERITY_NAMES)
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                    raise ValueError(f"Invalid raw severity: {path}")
                for i, value in enumerate(values):
                    samples[i].append(value)
                rows.append(
                    (
                        values,
                        str(row["run_id"]),
                        int(row["generation"]),
                        int(row["subproblem"]),
                        str(row["preference"]),
                        str(row["search_progress"]),
                    )
                )
    if not rows:
        raise ValueError("D03 raw files contain no steps")
    candidates: dict[str, tuple[float, ...]] = {"baseline": (0.2,) * 6}
    for percentile in (0.50, 0.65, 0.80):
        candidates[f"q{int(percentile * 100)}"] = tuple(
            round(max(0.05, min(0.9, _quantile(column, percentile))), 6) for column in samples
        )
    analysis: dict[str, Any] = {}
    for name, thresholds in candidates.items():
        exceed = [0] * 6
        labels: Counter[int] = Counter()
        states: set[int] = set()
        support: dict[int, set[str]] = defaultdict(set)
        margins: list[float] = []
        transitions = 0
        opportunities = 0
        previous: dict[tuple[str, int, int], int] = {}
        for values, run_id, generation, subproblem, preference, progress in rows:
            ratios = [v / t for v, t in zip(values, thresholds)]
            exceed = [n + int(v > t) for n, v, t in zip(exceed, values, thresholds)]
            label = relabel(values, thresholds)
            labels[label] += 1
            support[label].add(run_id.rsplit("_a", 1)[0])
            states.add(
                (PREFERENCE_LABELS.index(preference) * 7 + label) * 2
                + PROGRESS_LABELS.index(progress)
            )
            ordered = sorted(ratios, reverse=True)
            margins.append(ordered[0] - ordered[1])
            key = (run_id, generation, subproblem)
            if key in previous:
                opportunities += 1
                transitions += int(previous[key] != label)
            previous[key] = label
        total = len(rows)
        probabilities = [labels[i] / total for i in range(len(CONDITION_LABELS))]
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        analysis[name] = {
            "thresholds": thresholds,
            "exceedance_probability": dict(zip(SEVERITY_NAMES, (n / total for n in exceed))),
            "dominant_probability": dict(zip(CONDITION_LABELS, probabilities)),
            "state_coverage": len(states),
            "condition_entropy": entropy,
            "top1_top2_margin_median": median(margins),
            "within_trajectory_transition_rate": transitions / max(1, opportunities),
            "independent_instance_support": {
                CONDITION_LABELS[i]: len(support[i]) for i in range(len(CONDITION_LABELS))
            },
        }
    return {
        "status": "offline_relabel_only",
        "raw_files": len(paths),
        "raw_rows": len(rows),
        "candidate_thresholds": candidates,
        "analysis": analysis,
        "selection_policy": "baseline and q50/q65/q80 vectors; online retraining required",
        "statistically_concluded": False,
    }
