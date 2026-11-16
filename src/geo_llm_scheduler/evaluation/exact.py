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
    # Scheduled operations are checked in one sweep per machine; an unscheduled
    # (negative) start is not in the view and keeps the per-operation check.
    fitted = resources.self_fits()
    errors = []
    for o, start in enumerate(schedule.starts):
        if start < 0 or less(start, est_for(problem, genotype, schedule, o), TOL.time):
            errors.append(f"release/precedence:{o}")
        if not (fitted[o] if o in fitted else resources.fits(o, start)):
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
    #
    # A machine's power changes only at its own activity boundaries, so each
    # machine's value is evaluated once per own elementary interval, with the
    # reference expression at that interval's start, and then read by a
    # forward sweep. Region power still sums machines in the reference order.
    levels = [_machine_levels(problem, resources, m) for m in machines]
    cursor = [-1] * len(machines)
    idle = [problem.instances[m].idle_kw for m in machines]
    for a, b in zip(points, points[1:]):
        values = []
        for k, (times, level) in enumerate(levels):
            i = cursor[k]
            while i + 1 < len(times) and times[i + 1] <= a:
                i += 1
            cursor[k] = i
            values.append(level[i] if i >= 0 else idle[k])
        result.append((a, b, sum(values)))
    return tuple(result)


def _machine_levels(
    problem: ProblemInstance, resources: ResourceView, machine: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """One machine's reference power at each of its own activity boundaries.

    Between consecutive boundaries the busy test and the running load are
    constant, so the value at a boundary holds until the next one; before the
    first boundary the machine is idle.
    """
    # The reference busy test counts started minus ended positive-length
    # operations, i.e. whether any runs at t; its load steps sit on the same
    # boundaries. One sweep yields both, with each load a fresh sum in
    # operation order exactly as ``_load_steps`` / ``_machine_power`` compute it.
    server = problem.instances[machine]
    legacy = problem.power_model.legacy
    begins: dict[float, list[int]] = {}
    finish: dict[float, list[int]] = {}
    for o, a, b in resources.items[machine]:
        if a < b:
            begins.setdefault(a, []).append(o)
            finish.setdefault(b, []).append(o)
    times = tuple(sorted(begins.keys() | finish.keys()))
    if not legacy:
        kw = {o: problem.load_kw(o, machine) for ops in begins.values() for o in ops}
        busy_kw = server.idle_kw + problem.floor_kw(machine)
    active: set[int] = set()
    values = []
    for t in times:
        active.difference_update(finish.get(t, ()))
        active.update(begins.get(t, ()))
        if not active:
            values.append(server.idle_kw)
        elif legacy:
            values.append(server.active_kw)
        else:
            values.append(busy_kw + sum(kw[o] for o in sorted(active)))
    return times, tuple(values)


def _load_steps(
    problem: ProblemInstance, machine: int, items: tuple[tuple[int, float, float], ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-machine load (kW above the busy floor) on each own elementary interval.

    Each value is a fresh sum over the operations covering that interval, in
    operation order, so no running total accumulates rounding drift.
    """
    begins: dict[float, list[int]] = {}
    ends: dict[float, list[int]] = {}
    for o, a, b in items:
        if a < b:
            begins.setdefault(a, []).append(o)
            ends.setdefault(b, []).append(o)
    times = sorted(begins.keys() | ends.keys())
    kw = {o: problem.load_kw(o, machine) for o, a, b in items if a < b}
    active: set[int] = set()
    loads = []
    for t in times:
        # Half-open [a, b): an operation ending at t is no longer running at t.
        active.difference_update(ends.get(t, ()))
        active.update(begins.get(t, ()))
        loads.append(sum(kw[o] for o in sorted(active)))
    return tuple(times), tuple(loads)


def _machine_power(
    problem: ProblemInstance,
    machine: int,
    steps: tuple[tuple[float, ...], tuple[float, ...]],
    at: float,
) -> float:
    """Busy-machine power at a segment start: idle + floor + running load."""
    times, loads = steps
    server = problem.instances[machine]
    load = loads[bisect_right(times, at) - 1]
    return server.idle_kw + problem.floor_kw(machine) + load


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
