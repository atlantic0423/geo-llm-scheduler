"""Bounded exact reuse for a single Prefill shift with frozen Decode timing.

Reference arithmetic order is retained: changed-region segments are rebuilt,
all other regional terms are reused, and TOU additions remain region/segment/
tariff ordered. No delta subtraction or tariff antiderivative is introduced.
"""

from bisect import bisect_left, bisect_right
from math import isfinite

from geo_llm_scheduler.domain.models import (
    Candidate,
    EvaluationResult,
    Genotype,
    ProblemInstance,
    Schedule,
)
from geo_llm_scheduler.evaluation.exact import _power_segments, evaluate, regional_windows
from geo_llm_scheduler.scheduling.resources import ResourceView, est_for
from geo_llm_scheduler.scheduling.timing import legal_bounds
from geo_llm_scheduler.utils.numeric import TOL, less


class PrefillEvaluationContext:
    """Reuse only an independently exact-verified immutable parent and its region terms."""

    def __init__(self, problem: ProblemInstance, parent: Candidate) -> None:
        reference = evaluate(problem, parent.genotype, parent.schedule)
        if not reference.feasible or reference != parent.evaluation:
            raise ValueError("Prefill context requires a verified feasible exact parent")
        self.problem = problem
        self.parent = parent
        self.resources = ResourceView(problem, parent.genotype, parent.schedule)
        self.end = parent.schedule.horizon(problem)
        self.tariff_starts = tuple(tuple(t.start for t in r.tariffs) for r in problem.regions)
        self.tariff_ends = tuple(tuple(t.end for t in r.tariffs) for r in problem.regions)
        self.terms = tuple(
            self._region_terms(self.resources, r)[0] for r in range(len(problem.regions))
        )
        self.evaluations = 0

    def _region_terms(
        self, resources: ResourceView, region: int
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        problem = self.problem
        segments = _power_segments(problem, resources, region)
        tariffs = problem.regions[region].tariffs
        terms = []
        for a, b, power in segments:
            selected = (
                tariffs
                if not isfinite(power)
                else tariffs[
                    bisect_right(self.tariff_ends[region], a) : bisect_left(
                        self.tariff_starts[region], b
                    )
                ]
            )
            for tariff in selected:
                overlap = max(0.0, min(b, tariff.end) - max(a, tariff.start))
                terms.append(power * overlap * tariff.price / 3600)
        return tuple(terms), regional_windows(segments, self.end)

    def evaluate(self, genotype: Genotype, schedule: Schedule) -> EvaluationResult:
        """Return reference-identical results for exactly one legal Prefill timing change.

        The parent is fully verified once. Only the affected machine's capacity
        and the moved P / frozen D precedence need rechecking. Capacity is checked
        in each affected operation's reference sum order, including tolerance
        boundaries. Every Decode, and hence Flow and the batch horizon, is fixed.
        Misbound/multi-operation inputs are rejected instead of silently cached.
        """
        parent = self.parent
        if genotype is not parent.genotype or len(schedule.starts) != len(parent.schedule.starts):
            raise ValueError("Prefill context belongs to another genotype or timing dimension")
        changed = [
            o
            for o, (old, new) in enumerate(zip(parent.schedule.starts, schedule.starts))
            if old != new
        ]
        if len(changed) != 1 or changed[0] % 2:
            raise ValueError("Prefill reuse requires exactly one moved Prefill, frozen Decode")
        operation = changed[0]
        start = schedule.starts[operation]
        if not isfinite(start):
            raise ValueError("Prefill reuse requires a finite start")
        lower, upper = legal_bounds(self.problem, genotype, parent.schedule, operation)
        if (
            start < 0
            or less(start, lower, TOL.time)
            or less(upper, start, TOL.time)
            or not self.resources.fits(operation, start)
        ):
            # The normal move gate omits infeasible proposals. If this context
            # is used directly, retain the full reference violation diagnostics.
            return evaluate(self.problem, genotype, schedule)
        resources = ResourceView(self.problem, genotype, schedule)
        machine = genotype.ms[operation]
        if less(
            schedule.starts[operation + 1],
            est_for(self.problem, genotype, schedule, operation + 1),
            TOL.time,
        ) or any(not resources.fits(o, a) for o, a, _ in resources.items[machine]):
            return evaluate(self.problem, genotype, schedule)
        region = self.problem.instances[machine].region
        changed_terms, averages = self._region_terms(resources, region)
        tou = 0.0
        for r, terms in enumerate(self.terms):
            for term in changed_terms if r == region else terms:
                tou += term
        windows = list(parent.evaluation.windows)
        demand = list(parent.evaluation.demand)
        windows[region] = averages
        demand[region] = self.problem.regions[region].demand_rate * max(averages, default=0.0)
        self.evaluations += 1
        return EvaluationResult(True, parent.evaluation.flow, tou, tuple(demand), tuple(windows))
