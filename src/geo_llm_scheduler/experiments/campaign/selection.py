"""Common-reference metrics and deterministic, explicitly provisional selectors."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus


def nondominated(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the unique nondominated minimizing points in stable sorted order."""
    unique = sorted(set(points))
    return [
        p for p in unique if not any(q != p and all(a <= b for a, b in zip(q, p)) for q in unique)
    ]


def read_objectives(path: Path) -> list[tuple[float, float]]:
    """Read exact Flow and CNY bill pairs from a validated run artifact."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = [(float(row["flow_seconds"]), float(row["bill_cny"])) for row in rows]
    if not points or not all(math.isfinite(v) for point in points for v in point):
        raise ValueError("Missing or nonfinite Pareto objectives")
    return points


def common_metrics(
    arms: dict[str, list[tuple[float, float]]],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Evaluate every arm against one pooled, per-instance normalized reference."""
    pooled = [point for values in arms.values() for point in values]
    if not pooled:
        raise ValueError("Common metrics require results")
    lo = tuple(min(p[k] for p in pooled) for k in range(2))
    hi = tuple(max(p[k] for p in pooled) for k in range(2))
    scale = tuple(max(hi[k] - lo[k], 1e-12) for k in range(2))

    def normalize(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [((p[0] - lo[0]) / scale[0], (p[1] - lo[1]) / scale[1]) for p in points]

    reference_front = nondominated(normalize(pooled))
    reference_point = (1.1, 1.1)
    metrics = {
        arm: {
            "hv": hypervolume(normalize(points), reference_point),
            "igd_plus": igd_plus(normalize(points), reference_front),
        }
        for arm, points in arms.items()
    }
    provenance = {
        "pooled_points": len(pooled),
        "front_points": len(reference_front),
        "ideal": lo,
        "maximum": hi,
        "scale": scale,
        "reference_point_normalized": reference_point,
        "reference_front_normalized": reference_front,
    }
    return metrics, provenance


def select_stage_winner(
    results: list[dict[str, Any]],
    arms: tuple[str, ...],
    fallback: str,
) -> dict[str, Any]:
    """Rank only complete paired cells, resolving near ties by cost or explicit fallback."""
    if fallback not in arms:
        raise ValueError("Fallback must be one of the tested arms")
    cells: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        if row["arm"] in arms:
            cells[str(row["cell"])][str(row["arm"])] = row
    complete = {cell: by_arm for cell, by_arm in cells.items() if set(arms) <= set(by_arm)}
    missing = sorted(set(cells) - set(complete))
    ranks: dict[str, dict[str, list[float]]] = {
        arm: {key: [] for key in ("hv", "igd_plus", "elapsed", "exact")} for arm in arms
    }
    wins = {arm: {other: 0 for other in arms if other != arm} for arm in arms}
    for by_arm in complete.values():
        for field, reverse in (
            ("hv", True),
            ("igd_plus", False),
            ("elapsed", False),
            ("exact", False),
        ):
            for arm in arms:
                value = by_arm[arm][field]
                better = sum(
                    (by_arm[other][field] > value if reverse else by_arm[other][field] < value)
                    for other in arms
                    if other != arm
                )
                ranks[arm][field].append(float(1 + better))
        for arm in arms:
            for other in arms:
                if arm != other:
                    a, b = by_arm[arm], by_arm[other]
                    if (
                        a["hv"] >= b["hv"]
                        and a["igd_plus"] <= b["igd_plus"]
                        and (a["hv"] > b["hv"] or a["igd_plus"] < b["igd_plus"])
                    ):
                        wins[arm][other] += 1
    score = {
        arm: (sum(ranks[arm]["hv"]) + sum(ranks[arm]["igd_plus"])) / max(1, 2 * len(complete))
        for arm in arms
    }
    ordered = sorted(arms, key=lambda arm: (score[arm], arm))
    selected = fallback
    ambiguous = True
    reason = "no complete paired cells"
    if complete:
        best = ordered[0]
        gap = score[ordered[1]] - score[best] if len(ordered) > 1 else math.inf
        if gap > 0.10:
            selected, ambiguous, reason = best, False, "paired objective rank advantage"
        else:
            near = [arm for arm in arms if score[arm] - score[best] <= 0.10]
            runtime = {
                arm: sum(complete[cell][arm]["elapsed"] for cell in complete) / len(complete)
                for arm in near
            }
            cheapest = min(near, key=lambda arm: (runtime[arm], arm))
            other_costs = sorted(runtime.values())
            if len(other_costs) > 1 and other_costs[1] > 1.05 * other_costs[0]:
                selected, ambiguous, reason = cheapest, False, "near objective tie; lower runtime"
            else:
                reason = "paired objective and runtime differences ambiguous"
    return {
        "selected_arm": selected,
        "fallback_arm": fallback,
        "fallback_used": selected == fallback and ambiguous,
        "ambiguity": ambiguous,
        "reason": reason,
        "pair_count": len(complete),
        "missing_pairs": missing,
        "mean_ranks": {
            arm: {
                key: sum(values) / len(values) if values else None for key, values in data.items()
            }
            for arm, data in ranks.items()
        },
        "score": score,
        "win_matrix": wins,
        "selector_convention": "cell-wise HV/IGD+ ranks; 0.10 near-tie; 5% runtime tie-break",
        "statistically_concluded": False,
    }
