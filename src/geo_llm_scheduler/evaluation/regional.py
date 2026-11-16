"""Regional objective ledgers for R1 block screening (research path, not default).

Under the current model every Job's Prefill and Decode share one region, machines
belong to exactly one region, and the batch horizon ``H`` is the latest Decode
completion. For a fixed feasible schedule the objectives are therefore
conditionally separable by region:

* Flow is a plain sum of regional Job contributions.
* Region ``r`` has no activity after its own end ``E_r <= H``; on ``[E_r, H)`` its
  power is the constant sum of its machines' idle power. Its TOU cost and all
  fixed 900-second window averages are a horizon-independent core on
  ``[0, E_r)`` plus an idle tail that depends only on ``H``.

A :class:`RegionLedger` stores that core once per solution. Combining one ledger
per region (possibly from different parents with equal regional Job sets)
predicts the objectives of the spliced solution without SSGS decoding. The
prediction is a screen only: the exact evaluator stays the ground truth for
feasibility, scalar acceptance and Archive updates.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from geo_llm_scheduler.domain.models import EvaluationResult, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.evaluation.exact import (
    _checked_resources,
    _power_segments,
    regional_windows,
)
from geo_llm_scheduler.scheduling.resources import ResourceView

WINDOW = 900.0


@dataclass(frozen=True)
class RegionLedger:
    """Horizon-independent objective core of one region of one solution.

    ``core_windows[k]`` is the window-``k`` power integral divided by 900 seconds,
    counting only ``[0, end)``; it has ``ceil(end / 900)`` entries. ``prefix_peak``
    is the maximum of every core window strictly before the window containing
    ``end`` (``-inf`` when there is none).
    """

    region: int
    jobs: frozenset[int]
    flow: float
    end: float
    idle_kw: float
    core_tou: float
    core_windows: tuple[float, ...]
    prefix_peak: float

    def same_core(self, other: RegionLedger) -> bool:
        """Return True when both ledgers contribute identical objective terms."""
        return (
            self.jobs == other.jobs
            and self.flow == other.flow
            and self.end == other.end
            and self.core_tou == other.core_tou
            and self.core_windows == other.core_windows
        )


@dataclass(frozen=True)
class TariffIndex:
    """Cumulative price integral of one region's contiguous tariff schedule."""

    starts: tuple[float, ...]
    ends: tuple[float, ...]
    prices: tuple[float, ...]
    cumulative: tuple[float, ...]

    def integral(self, a: float, b: float) -> float:
        """Return the integral of price over ``[a, b)`` in CNY·s/kWh.

        Raises:
            ValueError: if ``b`` exceeds tariff coverage, matching the exact evaluator.
        """
        if b <= a:
            return 0.0
        if b > self.ends[-1]:
            raise ValueError("Tariff coverage ends before batch")
        return self._at(b) - self._at(a)

    def _at(self, t: float) -> float:
        k = max(0, bisect_right(self.starts, t) - 1)
        return self.cumulative[k] + (min(t, self.ends[k]) - self.starts[k]) * self.prices[k]


def tariff_indexes(problem: ProblemInstance) -> tuple[TariffIndex, ...]:
    """Build one cumulative tariff index per region (validated contiguous tariffs)."""
    result = []
    for region in problem.regions:
        cumulative = [0.0]
        for tariff in region.tariffs[:-1]:
            cumulative.append(cumulative[-1] + (tariff.end - tariff.start) * tariff.price)
        result.append(
            TariffIndex(
                tuple(t.start for t in region.tariffs),
                tuple(t.end for t in region.tariffs),
                tuple(t.price for t in region.tariffs),
                tuple(cumulative),
            )
        )
    return tuple(result)


def region_ledgers(
    problem: ProblemInstance, genotype: Genotype, schedule: Schedule
) -> tuple[RegionLedger, ...]:
    """Return every region's ledger for one feasible schedule.

    The schedule must be feasible (callers use exact-evaluated candidates). Power
    segments come from the exact evaluator's own segment builder, so every core
    term uses the reference power values and breakpoints.

    Raises:
        ValueError: if a Job's Prefill and Decode are assigned to different regions.
    """
    resources = ResourceView(problem, genotype, schedule)
    horizon = schedule.horizon(problem)
    return tuple(
        _ledger(
            problem,
            schedule,
            r,
            jobs,
            _power_segments(problem, resources, r) if horizon > 0 else (),
        )
        for r, jobs in enumerate(_regional_jobs(problem, genotype))
    )


def evaluate_with_ledgers(
    problem: ProblemInstance, genotype: Genotype, schedule: Schedule
) -> tuple[EvaluationResult, tuple[RegionLedger, ...]]:
    """Exact evaluation and regional ledgers from one shared resource/segment pass.

    The returned ``EvaluationResult`` is computed with the reference evaluator's
    own validation, segment builder and arithmetic order; tests assert equality
    with :func:`geo_llm_scheduler.evaluation.exact.evaluate`. Infeasible
    schedules return the infeasible result and no ledgers.
    """
    errors, resources = _checked_resources(problem, genotype, schedule)
    if errors:
        return EvaluationResult(False, math.inf, math.inf, (), (), errors), ()
    assert resources is not None
    end = schedule.horizon(problem)
    flow = sum(schedule.end(problem, 2 * i + 1) - j.release for i, j in enumerate(problem.jobs))
    tou = 0.0
    windows = []
    demand = []
    ledgers = []
    jobs_by_region = _regional_jobs(problem, genotype)
    for r, region in enumerate(problem.regions):
        if region.tariffs[-1].end < end:
            raise ValueError(f"Tariff coverage ends before batch: {region.name}")
        segments = _power_segments(problem, resources, r)
        starts = tuple(t.start for t in region.tariffs)
        ends = tuple(t.end for t in region.tariffs)
        for a, b, power in segments:
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
        ledgers.append(_ledger(problem, schedule, r, jobs_by_region[r], segments))
    return EvaluationResult(True, flow, tou, tuple(demand), tuple(windows)), tuple(ledgers)


def _regional_jobs(problem: ProblemInstance, genotype: Genotype) -> list[frozenset[int]]:
    jobs: list[set[int]] = [set() for _ in problem.regions]
    for j in range(len(problem.jobs)):
        rp = problem.instances[genotype.ms[2 * j]].region
        rd = problem.instances[genotype.ms[2 * j + 1]].region
        if rp != rd:
            raise ValueError("P/D region mismatch")
        jobs[rp].add(j)
    return [frozenset(s) for s in jobs]


def _ledger(
    problem: ProblemInstance,
    schedule: Schedule,
    r: int,
    jobs: frozenset[int],
    segments: tuple[tuple[float, float, float], ...],
) -> RegionLedger:
    """Build one ledger from the region's reference power segments."""
    region = problem.regions[r]
    flow = 0.0
    end = 0.0
    for j in sorted(jobs):
        finish = schedule.end(problem, 2 * j + 1)
        flow += finish - problem.jobs[j].release
        end = max(end, finish, schedule.end(problem, 2 * j))
    idle = sum(m.idle_kw for m in problem.instances if m.region == r)
    starts = tuple(t.start for t in region.tariffs)
    ends = tuple(t.end for t in region.tariffs)
    tou = 0.0
    windows = [0.0] * math.ceil(end / WINDOW)
    for a, b, power in segments:
        if b > end:
            break
        for tariff in region.tariffs[bisect_right(ends, a) : bisect_left(starts, b)]:
            overlap = max(0.0, min(b, tariff.end) - max(a, tariff.start))
            tou += power * overlap * tariff.price / 3600
        for k in range(int(a // WINDOW), min(len(windows), math.ceil(b / WINDOW))):
            overlap = max(0.0, min(b, (k + 1) * WINDOW) - max(a, k * WINDOW))
            windows[k] += power * overlap / WINDOW
    boundary = int(end // WINDOW)
    return RegionLedger(
        r, jobs, flow, end, idle, tou, tuple(windows), max(windows[:boundary], default=-math.inf)
    )


def tail_windows(ledger: RegionLedger, horizon: float) -> tuple[float, ...]:
    """Return all fixed-window averages of the region for a batch ending at ``horizon``.

    Diagnostic/test helper, O(number of windows); ``regional_bill`` uses O(1) peaks.
    """
    if horizon < ledger.end:
        raise ValueError("Horizon precedes the region's own end")
    windows = list(ledger.core_windows) + [0.0] * (
        math.ceil(horizon / WINDOW) - len(ledger.core_windows)
    )
    if horizon > ledger.end:
        for k in range(int(ledger.end // WINDOW), len(windows)):
            overlap = max(0.0, min(horizon, (k + 1) * WINDOW) - max(ledger.end, k * WINDOW))
            windows[k] += ledger.idle_kw * overlap / WINDOW
    return tuple(windows)


def regional_bill(
    problem: ProblemInstance, index: TariffIndex, ledger: RegionLedger, horizon: float
) -> tuple[float, float]:
    """Return ``(TOU, Demand)`` of one region for a batch ending at ``horizon``.

    The tail ``[end, horizon)`` runs at the region's idle power; window peaks use
    the cached prefix maximum, so the cost does not grow with the tail length.
    """
    end = ledger.end
    if horizon < end:
        raise ValueError("Horizon precedes the region's own end")
    tou = ledger.core_tou + ledger.idle_kw * index.integral(end, horizon) / 3600
    count = math.ceil(horizon / WINDOW)
    peak = ledger.prefix_peak
    boundary = int(end // WINDOW)
    if boundary < count:
        own = ledger.core_windows[boundary] if boundary < len(ledger.core_windows) else 0.0
        overlap = max(0.0, min(horizon, (boundary + 1) * WINDOW) - end)
        peak = max(peak, own + ledger.idle_kw * overlap / WINDOW)
        if boundary + 1 < count:
            # Later windows are idle only: full windows equal idle power, the last
            # one keeps the fixed 900-second divisor and so cannot exceed it.
            last = (horizon - (count - 1) * WINDOW) / WINDOW
            peak = max(peak, ledger.idle_kw * (1.0 if boundary + 2 < count else last))
    peak = max(peak, 0.0)
    return tou, problem.regions[ledger.region].demand_rate * peak


@dataclass(frozen=True)
class RegionalPrediction:
    """Objectives predicted from regional ledgers; not an exact evaluation."""

    flow: float
    tou: float
    demand: tuple[float, ...]
    horizon: float

    @property
    def bill(self) -> float:
        """Return predicted TOU plus regional Demand Charge."""
        return self.tou + sum(self.demand)

    @property
    def objectives(self) -> tuple[float, float]:
        """Return the predicted minimizing objective vector."""
        return self.flow, self.bill


def combine(
    problem: ProblemInstance,
    indexes: tuple[TariffIndex, ...],
    ledgers: tuple[RegionLedger, ...],
) -> RegionalPrediction:
    """Predict the objectives of a solution assembled from one ledger per region.

    Requires the ledgers to be indexed by region and to partition the Jobs. The
    horizon is the latest regional end, which equals the latest Decode end.
    """
    if [g.region for g in ledgers] != list(range(len(problem.regions))):
        raise ValueError("Ledgers must be ordered by region")
    if sum(len(g.jobs) for g in ledgers) != len(problem.jobs) or len(
        frozenset().union(*(g.jobs for g in ledgers))
    ) != len(problem.jobs):
        raise ValueError("Regional Job sets must partition all Jobs")
    horizon = max(g.end for g in ledgers)
    tou = 0.0
    demand = []
    for g in ledgers:
        t, d = regional_bill(problem, indexes[g.region], g, horizon)
        tou += t
        demand.append(d)
    return RegionalPrediction(sum(g.flow for g in ledgers), tou, tuple(demand), horizon)


def splice(
    recipient: tuple[RegionLedger, ...], donor: tuple[RegionLedger, ...], region: int
) -> tuple[RegionLedger, ...]:
    """Replace one region's ledger by the donor's; requires equal nonempty Job sets."""
    if not donor[region].jobs or donor[region].jobs != recipient[region].jobs:
        raise ValueError("Requires equal nonempty regional job sets")
    return tuple(donor[region] if r == region else g for r, g in enumerate(recipient))
