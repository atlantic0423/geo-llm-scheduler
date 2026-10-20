"""Exact event integrals with invocation-local indexes, no time-slot sampling."""

import math
from bisect import bisect_left, bisect_right

from geo_llm_scheduler.domain.models import EvaluationResult, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.io.validation import validate_genotype
from geo_llm_scheduler.scheduling.resources import ResourceView, est_for
from geo_llm_scheduler.utils.numeric import TOL, less


def violations(problem: ProblemInstance, genotype: Genotype, schedule: Schedule) -> tuple[str, ...]:
    """Return every discoverable schedule feasibility violation."""
    return _checked_resources(problem, genotype, schedule)[0]


def _checked_resources(
    problem: ProblemInstance, genotype: Genotype, schedule: Schedule
) -> tuple[tuple[str, ...], ResourceView | None]:
    """Validate once and retain the same invocation-local occupancy for billing."""
    try:
        validate_genotype(problem, genotype)
    except ValueError as exc:
        return (str(exc),), None
    if len(schedule.starts) != problem.operation_count:
        return ("Timing dimension mismatch",), None
    if any(not math.isfinite(t) for t in schedule.starts):
        return ("Non-finite start",), None
    resources = ResourceView(problem, genotype, schedule)
    errors = []
    for o, start in enumerate(schedule.starts):
        if start < 0 or less(start, est_for(problem, genotype, schedule, o), TOL.time):
            errors.append(f"release/precedence:{o}")
        if not resources.fits(o, start):
            errors.append(f"resource:{o}")
    return tuple(errors), resources


def power_segments(
    problem: ProblemInstance, genotype: Genotype, schedule: Schedule, region: int
) -> tuple[tuple[float, float, float], ...]:
    """Return piecewise constant regional power on the actual batch interval."""
    return _power_segments(problem, ResourceView(problem, genotype, schedule), region)


def _power_segments(
    problem: ProblemInstance, resources: ResourceView, region: int
) -> tuple[tuple[float, float, float], ...]:
    """Keep every reference breakpoint and machine-sum order; index occupancy only."""
    end = resources.schedule.horizon(problem)
    machines = [m for m, server in enumerate(problem.instances) if server.region == region]
    occupied = {m: resources.items[m] for m in machines}
    points = sorted(
        {0.0, end}
        | {t for ops in occupied.values() for _, a, b in ops for t in (a, b) if 0 <= t <= end}
    )
    result = []
    # Counts use only comparisons, with the original half-open convention.
    # Keeping *all* points (including hidden overlapping-operation boundaries)
    # preserves the original integral's floating-point addition order.
    bounds = {m: resources.activity_bounds(m) for m in machines}
    for a, b in zip(points, points[1:]):
        power = sum(
            problem.instances[m].active_kw
            if bisect_right(bounds[m][0], a) > bisect_right(bounds[m][1], a)
            else problem.instances[m].idle_kw
            for m in machines
        )
        result.append((a, b, power))
    return tuple(result)


def regional_windows(
    segments: tuple[tuple[float, float, float], ...], end: float
) -> tuple[float, ...]:
    """Integrate all fixed 900-second windows, retaining the fixed tail divisor."""
    result = [0.0] * math.ceil(end / 900)
    for a, b, power in segments:
        for window in range(int(a // 900), min(len(result), math.ceil(b / 900))):
            overlap = max(0.0, min(b, (window + 1) * 900) - max(a, window * 900))
            result[window] += power * overlap / 900
    return tuple(result)


def evaluate(problem: ProblemInstance, genotype: Genotype, schedule: Schedule) -> EvaluationResult:
    """Compute feasibility, Flow and the exact electricity-bill decomposition.

    Infeasible schedules have infinite objectives. Missing tariff coverage is an
    input error, never an implicit tariff extension or artificially cheap tail.
    """
    errors, resources = _checked_resources(problem, genotype, schedule)
    if errors:
        return EvaluationResult(False, math.inf, math.inf, (), (), errors)
    end = schedule.horizon(problem)
    flow = sum(schedule.end(problem, 2 * i + 1) - j.release for i, j in enumerate(problem.jobs))
    assert resources is not None
    tou = 0.0
    windows = []
    demand = []
    for r, region in enumerate(problem.regions):
        if region.tariffs[-1].end < end:
            raise ValueError(f"Tariff coverage ends before batch: {region.name}")
        segments = _power_segments(problem, resources, r)
        starts = tuple(t.start for t in region.tariffs)
        ends = tuple(t.end for t in region.tariffs)
        for a, b, power in segments:
            # Validated tariffs are contiguous and sorted. Omitted terms are
            # exactly zero; nonzero additions retain the reference order.
            # Finite individual powers can still overflow during regional sum.
            # In that exceptional case retain zero terms too (inf*0 is NaN),
            # exactly matching the original evaluator's arithmetic behavior.
            tariffs = (
                region.tariffs
                if not math.isfinite(power)
                else region.tariffs[bisect_right(ends, a) : bisect_left(starts, b)]
            )
            for tariff in tariffs:
                overlap = max(0.0, min(b, tariff.end) - max(a, tariff.start))
                tou += power * overlap * tariff.price / 3600
        averages = regional_windows(segments, end)
        windows.append(averages)
        demand.append(region.demand_rate * max(averages, default=0.0))
    return EvaluationResult(True, flow, tou, tuple(demand), tuple(windows))


def tied_peaks(averages: tuple[float, ...]) -> tuple[int, ...]:
    """Return all windows tied at the maximum using power tolerance."""
    peak = max(averages, default=0.0)
    return tuple(b for b, p in enumerate(averages) if not less(p, peak, TOL.power))
