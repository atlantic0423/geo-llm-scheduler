"""Preregistered screening rule; independent confirmation never retunes it."""

from __future__ import annotations

from collections import defaultdict

from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.macrosearch.trajectory_budget import ARM_SPECS

ELIGIBLE = ("D1", "D2", "D3", "D4", "D5", "D6", "A1")


def nondominated(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the exact two-objective minimizing union front with stable ties."""
    result = []
    best = float("inf")
    for x, y in sorted(set(points)):
        if y < best:
            result.append((x, y))
            best = y
    return result


def choose(rows: list[dict]) -> dict:
    """Select one C30 distance arm by mean within-parent HV/IGD+ rank, ID tie break."""
    cases: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        cases[row["spec"].get("input", row["spec"]["case"])].append(row)
    parent_metrics: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for group in cases.values():
        arms = {row["spec"]["arm"] for row in group}
        if arms != set(ARM_SPECS):
            raise ValueError("Screening case lacks all 18 arms")
        points = [tuple(p) for row in group for p in row["archive_points"]]
        if not points:
            raise ValueError("Empty screening archive union")
        lower = tuple(min(p[k] for p in points) for k in range(2))
        spans = tuple(max(max(p[k] for p in points) - lower[k], 1e-9) for k in range(2))

        def norm(items: list) -> list[tuple[float, float]]:
            return [tuple((p[k] - lower[k]) / spans[k] for k in range(2)) for p in items]

        front = nondominated(norm(points))
        for row in group:
            spec = row["spec"]
            normalized = norm(row["archive_points"])
            parent_metrics[spec["parent"]][spec["arm"]].append(
                (hypervolume(normalized, (1.1, 1.1)), igd_plus(normalized, front))
            )
    rank_totals = dict.fromkeys(ELIGIBLE, 0.0)
    for arms in parent_metrics.values():
        means = {
            a: tuple(sum(v[k] for v in values) / len(values) for k in range(2))
            for a, values in arms.items()
        }
        for a in ELIGIBLE:
            hv, igd = means[a]
            rank_totals[a] += sum(means[b][0] > hv for b in ELIGIBLE) + 0.5 * sum(
                b != a and means[b][0] == hv for b in ELIGIBLE
            )
            rank_totals[a] += sum(means[b][1] < igd for b in ELIGIBLE) + 0.5 * sum(
                b != a and means[b][1] == igd for b in ELIGIBLE
            )
    selected = min(ELIGIBLE, key=lambda a: (rank_totals[a], a))
    counts = [
        sum(
            row["diagnostic_counts"].get(f"full_request:{b}", 0)
            for row in rows
            if row["spec"]["arm"] == selected
        )
        for b in (3, 6, 10)
    ]
    probabilities = [v / sum(counts) for v in counts] if sum(counts) else [0.4, 0.3, 0.3]
    return {
        "selected": selected,
        "rank_totals": rank_totals,
        "parents": len(parent_metrics),
        "probabilities": probabilities,
        "request_counts": counts,
        "random_aid": ARM_SPECS[selected].aid,
        "rule": "equal parent/size/topology/pressure weights; mean HV and IGD+ ordinal ranks over D1-D6/A1; ID ties; all 18 arms in empirical per-case reference",
    }
