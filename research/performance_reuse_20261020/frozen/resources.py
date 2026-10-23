"""Exact half-open resource intervals shared by decoding and timing moves."""

from bisect import insort

from geo_llm_scheduler.domain.models import Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.utils.numeric import TOL, less


def intervals(
    problem: ProblemInstance,
    genotype: Genotype,
    schedule: Schedule,
    machine: int,
    excluded: frozenset[int] = frozenset(),
) -> list[tuple[int, float, float]]:
    """Read assignment from MS and return scheduled intervals on one server."""
    return [
        (o, s, s + problem.profile(o).duration)
        for o, s in enumerate(schedule.starts)
        if o not in excluded and genotype.ms[o] == machine and s >= 0
    ]


def fits(
    problem: ProblemInstance,
    genotype: Genotype,
    schedule: Schedule,
    operation: int,
    start: float,
    excluded: frozenset[int] = frozenset(),
) -> bool:
    """Check a proposed interval against frozen operations, excluding itself."""
    machine = genotype.ms[operation]
    others = intervals(problem, genotype, schedule, machine, excluded | {operation})
    return _fits_intervals(problem, machine, operation, start, others)


def _fits_intervals(
    problem: ProblemInstance,
    machine: int,
    operation: int,
    start: float,
    others: list[tuple[int, float, float]],
) -> bool:
    """Check a frozen interval list in the reference operation summation order."""
    p = problem.profile(operation)
    end = start + p.duration
    points = sorted({start, end} | {t for _, a, b in others for t in (a, b) if start < t < end})
    for a, b in zip(points, points[1:]):
        if b <= a:
            continue
        active = [problem.profile(o) for o, s, e in others if s <= a < e]
        if less(1.0, p.compute + sum(q.compute for q in active), TOL.compute):
            return False
        if less(problem.instances[machine].vram, p.vram + sum(q.vram for q in active), TOL.vram):
            return False
    return True


def earliest(
    problem: ProblemInstance, genotype: Genotype, schedule: Schedule, operation: int, est: float
) -> float:
    """Find earliest feasible interval using completion events, never time steps."""
    machine = genotype.ms[operation]
    others = intervals(problem, genotype, schedule, machine, frozenset({operation}))
    return _earliest_intervals(problem, machine, operation, est, others)


def _earliest_intervals(
    problem: ProblemInstance,
    machine: int,
    operation: int,
    est: float,
    others: list[tuple[int, float, float]],
) -> float:
    """Reuse the same frozen intervals across the unchanged completion events."""
    events = {est} | {b for _, _, b in others if b >= est}
    for t in sorted(events):
        if _fits_intervals(problem, machine, operation, t, others):
            return t
    raise ValueError("Operation cannot fit on assigned instance")


class ResourceCalendar:
    """Index scheduled prefix operations by machine in operation-index order."""

    def __init__(self, problem: ProblemInstance) -> None:
        self.problem = problem
        self.items: list[list[tuple[int, float, float]]] = [[] for _ in problem.instances]

    def commit(self, machine: int, operation: int, start: float) -> None:
        """Insert one newly scheduled operation without regrouping resource sums."""
        insort(
            self.items[machine],
            (operation, start, start + self.problem.profile(operation).duration),
        )

    def earliest(
        self,
        machine: int,
        operation: int,
        est: float,
        temporary: tuple[int, float, float] | None = None,
    ) -> float:
        """Return the reference event, including an optional uncommitted Prefill."""
        others = [item for item in self.items[machine] if item[0] != operation]
        if temporary is not None and temporary[0] != operation:
            insort(others, temporary)
        return _earliest_intervals(self.problem, machine, operation, est, others)


def est_for(
    problem: ProblemInstance, genotype: Genotype, schedule: Schedule, operation: int
) -> float:
    """Release or preceding Prefill completion plus assignment-dependent KV."""
    job = problem.jobs[operation // 2]
    if operation % 2 == 0:
        return job.release
    return schedule.end(problem, operation - 1) + job.kv_delay * (
        genotype.ms[operation] != genotype.ms[operation - 1]
    )


def union_length(items: list[tuple[float, float]]) -> float:
    """Compute union length without double-counting concurrent operations."""
    total = 0.0
    right = float("-inf")
    for a, b in sorted(items):
        total += max(0.0, b - max(a, right))
        right = max(right, b)
    return total
