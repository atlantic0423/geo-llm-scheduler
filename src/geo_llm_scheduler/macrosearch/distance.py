"""Representation-aware assignment, operation-order and timing distances."""

from __future__ import annotations

from dataclasses import dataclass

from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.utils.numeric import EPS_RATIO, TOL


@dataclass(frozen=True)
class Distances:
    """Three separately normalized components; no objective normalization is used."""

    assignment: float
    order: float
    timing: float

    def combined(self, metric: str, previous_action: int) -> float:
        """Combine components using a predeclared metric and an already executed action."""
        if metric == "mean":
            return (self.assignment + self.order + self.timing) / 3
        if metric == "max":
            return max(self.assignment, self.order, self.timing)
        if metric == "action":
            return self.timing if previous_action in (7, 8) else max(self.assignment, self.order)
        raise ValueError("Unknown trajectory distance metric")


def operation_order(jobs: tuple[int, ...]) -> tuple[int, ...]:
    """Label repeated job occurrences as distinct Prefill/Decode operation indices."""
    n = len(jobs) // 2
    count = [0] * n
    result = []
    if len(jobs) % 2:
        raise ValueError("OS must contain two occurrences per job")
    for job in jobs:
        if not 0 <= job < n or count[job] >= 2:
            raise ValueError("Invalid OS job/occurrence")
        result.append(2 * job + count[job])
        count[job] += 1
    if any(value != 2 for value in count):
        raise ValueError("OS must contain every job twice")
    return tuple(result)


def _inversions(values: list[int]) -> int:
    tree = [0] * (len(values) + 1)
    total = 0
    for seen, value in enumerate(values):
        i = value + 1
        prefix = 0
        while i:
            prefix += tree[i]
            i -= i & -i
        total += seen - prefix
        i = value + 1
        while i < len(tree):
            tree[i] += 1
            i += i & -i
    return total


def components(problem: ProblemInstance, left: Candidate, right: Candidate) -> Distances:
    """Compute normalized distances in O(N log N), including timing-only movement."""
    count = problem.operation_count
    if not count:
        return Distances(0.0, 0.0, 0.0)
    if any(
        len(values) != count
        for candidate in (left, right)
        for values in (candidate.genotype.ms, candidate.genotype.os, candidate.schedule.starts)
    ):
        raise ValueError("Candidate dimensions do not match the problem")
    a, b = (operation_order(candidate.genotype.os) for candidate in (left, right))
    positions = {operation: index for index, operation in enumerate(a)}
    denominator = count * (count - 1) // 2 - len(problem.jobs)
    order = (
        _inversions([positions[operation] for operation in b]) / denominator if denominator else 0.0
    )
    assignment = sum(x != y for x, y in zip(left.genotype.ms, right.genotype.ms)) / count
    timing = (
        sum(
            min(1.0, abs(x - y) / max(problem.profile(o).duration, TOL.time))
            for o, (x, y) in enumerate(zip(left.schedule.starts, right.schedule.starts))
        )
        / count
    )
    return Distances(assignment, order, timing)


def trajectory_signal(
    problem: ProblemInstance,
    positions: list[Candidate],
    metric: str,
    window: int,
    kind: str,
    previous_action: int,
) -> tuple[float, Distances]:
    """Measure the completed trajectory suffix only; never join different episodes."""
    if not positions or window < 1:
        raise ValueError("Nonempty history and a positive window are required")
    suffix = positions[-(window + 1) :]
    parts = components(problem, suffix[0], suffix[-1])
    endpoint = parts.combined(metric, previous_action)
    if kind == "endpoint":
        return endpoint, parts
    if kind != "efficiency" or metric == "action":
        raise ValueError("Efficiency requires a fixed metric")
    path = sum(
        components(problem, a, b).combined(metric, previous_action)
        for a, b in zip(suffix, suffix[1:])
    )
    return endpoint / (path + EPS_RATIO), parts
