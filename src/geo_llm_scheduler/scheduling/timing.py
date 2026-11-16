"""Finite event-derived single-operation timing moves, without global horizon."""

from dataclasses import replace

from geo_llm_scheduler.domain.models import Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.scheduling.resources import (
    ResourceView,
    est_for,
    fits,
    intervals,
    union_length,
)
from geo_llm_scheduler.utils.numeric import TOL, close, less


def legal_bounds(
    problem: ProblemInstance, genotype: Genotype, schedule: Schedule, operation: int
) -> tuple[float, float]:
    """P is bounded by frozen D; D has no artificial planning upper bound."""
    lower = est_for(problem, genotype, schedule, operation)
    upper = float("inf")
    if operation % 2 == 0:
        job = problem.jobs[operation // 2]
        upper = (
            schedule.starts[operation + 1]
            - problem.profile(operation).duration
            - job.kv_delay * (genotype.ms[operation] != genotype.ms[operation + 1])
        )
    return lower, upper


def critical_times(
    problem: ProblemInstance,
    genotype: Genotype,
    schedule: Schedule,
    operation: int,
    *,
    resources: ResourceView | None = None,
) -> tuple[float, ...]:
    """Shared A7/Compressible finite activity/resource boundary starts."""
    if resources is not None:
        resources.check_binding(problem, genotype, schedule)
    lower, upper = legal_bounds(problem, genotype, schedule, operation)
    p = problem.profile(operation).duration
    events = (
        {
            t
            for _, a, b in intervals(problem, genotype, schedule, genotype.ms[operation])
            for t in (a, b)
        }
        if resources is None
        else set(resources.events(genotype.ms[operation]))
    )
    values = {lower, schedule.starts[operation]} | events | {e - p for e in events}
    if upper != float("inf"):
        values.add(upper)
    return tuple(
        sorted(
            t
            for t in values
            if t >= 0 and not less(t, lower, TOL.time) and not less(upper, t, TOL.time)
        )
    )


def move(
    problem: ProblemInstance,
    genotype: Genotype,
    schedule: Schedule,
    operation: int,
    start: float,
    *,
    resources: ResourceView | None = None,
) -> Schedule | None:
    """Clone one timing field if legal; frozen operations never propagate."""
    if resources is not None:
        resources.check_binding(problem, genotype, schedule)
    lower, upper = legal_bounds(problem, genotype, schedule, operation)
    if start < 0 or less(start, lower, TOL.time) or less(upper, start, TOL.time):
        return None
    feasible = (
        fits(problem, genotype, schedule, operation, start)
        if resources is None
        else resources.fits(operation, start)
    )
    if not feasible:
        return None
    starts = list(schedule.starts)
    starts[operation] = start
    return replace(schedule, starts=tuple(starts), resource_wait=(), intentional_wait=())


def pack_gain(
    problem: ProblemInstance, genotype: Genotype, before: Schedule, after: Schedule, operation: int
) -> float:
    """Return exact reduction of the assigned instance's active union."""
    m = genotype.ms[operation]
    return union_length(
        [(a, b) for _, a, b in intervals(problem, genotype, before, m)]
    ) - union_length([(a, b) for _, a, b in intervals(problem, genotype, after, m)])


def packing_moves(
    problem: ProblemInstance,
    genotype: Genotype,
    schedule: Schedule,
    *,
    resources: ResourceView | None = None,
) -> list[tuple[int, float, float, float]]:
    """Return (operation, start, union gain, delta Flow) for positive packing."""
    result = []
    if resources is None:
        resources = ResourceView(problem, genotype, schedule)
    else:
        resources.check_binding(problem, genotype, schedule)
    lengths = [union_length([(a, b) for _, a, b in items]) for items in resources.items]
    for o, old in enumerate(schedule.starts):
        machine = genotype.ms[o]
        for t in critical_times(problem, genotype, schedule, o, resources=resources):
            if close(old, t, TOL.time):
                continue
            candidate = move(problem, genotype, schedule, o, t, resources=resources)
            if candidate is None:
                continue
            # Preserve the exact sorted-union arithmetic, replacing only one
            # interval instead of rescanning every other server twice.
            if old >= 0:
                after = [
                    (t, t + problem.profile(o).duration) if operation == o else (a, b)
                    for operation, a, b in resources.items[machine]
                ]
                gain = lengths[machine] - union_length(after)
            else:
                gain = pack_gain(problem, genotype, schedule, candidate, o)
            if less(0, gain, TOL.time):
                result.append((o, t, gain, t - old if o % 2 else 0.0))
    return result
