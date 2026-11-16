"""Pure reference evaluator: exact event integrals, no time-slot sampling."""

import math

from geo_llm_scheduler.domain.models import EvaluationResult, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.io.validation import validate_genotype
from geo_llm_scheduler.scheduling.resources import est_for, fits, intervals
from geo_llm_scheduler.utils.numeric import TOL, less


def violations(problem: ProblemInstance, genotype: Genotype, schedule: Schedule) -> tuple[str, ...]:
    """Return every discoverable schedule feasibility violation."""
    try:
        validate_genotype(problem, genotype)
    except ValueError as exc:
        return (str(exc),)
    if len(schedule.starts) != problem.operation_count:
        return ("Timing dimension mismatch",)
    if any(not math.isfinite(t) for t in schedule.starts):
        return ("Non-finite start",)
    errors = []
    for o, start in enumerate(schedule.starts):
        if start < 0 or less(start, est_for(problem, genotype, schedule, o), TOL.time):
            errors.append(f"release/precedence:{o}")
        if not fits(problem, genotype, schedule, o, start):
            errors.append(f"resource:{o}")
    return tuple(errors)


def power_segments(
    problem: ProblemInstance, genotype: Genotype, schedule: Schedule, region: int
) -> tuple[tuple[float, float, float], ...]:
    """Return piecewise constant regional power on the actual batch interval."""
    end = schedule.horizon(problem)
    machines = [m for m, server in enumerate(problem.instances) if server.region == region]
    occupied = {m: intervals(problem, genotype, schedule, m) for m in machines}
    points = sorted(
        {0.0, end}
        | {t for ops in occupied.values() for _, a, b in ops for t in (a, b) if 0 <= t <= end}
    )
    result = []
    for a, b in zip(points, points[1:]):
        power = sum(
            problem.instances[m].active_kw
            if any(s <= a < e for _, s, e in occupied[m])
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
    errors = violations(problem, genotype, schedule)
    if errors:
        return EvaluationResult(False, math.inf, math.inf, (), (), errors)
    end = schedule.horizon(problem)
    flow = sum(schedule.end(problem, 2 * i + 1) - j.release for i, j in enumerate(problem.jobs))
    tou = 0.0
    windows = []
    demand = []
    for r, region in enumerate(problem.regions):
        if region.tariffs[-1].end < end:
            raise ValueError(f"Tariff coverage ends before batch: {region.name}")
        segments = power_segments(problem, genotype, schedule, r)
        for a, b, power in segments:
            for tariff in region.tariffs:
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
