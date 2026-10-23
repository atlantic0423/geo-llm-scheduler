"""Exact half-open resource intervals shared by decoding and timing moves."""

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
    p = problem.profile(operation)
    end = start + p.duration
    others = intervals(problem, genotype, schedule, machine, excluded | {operation})
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
    events = {est} | {
        b
        for _, _, b in intervals(problem, genotype, schedule, machine, frozenset({operation}))
        if b >= est
    }
    for t in sorted(events):
        if fits(problem, genotype, schedule, operation, t):
            return t
    raise ValueError("Operation cannot fit on assigned instance")


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
