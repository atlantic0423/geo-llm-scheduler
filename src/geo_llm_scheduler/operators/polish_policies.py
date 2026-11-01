"""Research-only two-focus and sparse sequential Prefill right-shift policies."""

from dataclasses import replace
from math import ceil
from time import perf_counter

from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import tied_peaks
from geo_llm_scheduler.evaluation.prefill import PrefillEvaluationContext
from geo_llm_scheduler.experiments.initial_bill_index import TariffIndex
from geo_llm_scheduler.operators.right_shift import PolishStats, polish, right_shift_times
from geo_llm_scheduler.scheduling.resources import ResourceView
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.scheduling.timing import legal_bounds, move
from geo_llm_scheduler.utils.numeric import TOL, close, identity, less


class ReusedPrefillEvaluator:
    """Forward every feasible candidate to the unchanged Archive dispatch, reusing exact terms."""

    def __init__(self, gateway: EvaluationGateway, parent: Candidate, reuse: bool = True) -> None:
        self.gateway = gateway
        self.parent = parent
        self.reuse = reuse
        self.context: PrefillEvaluationContext | None = None

    @property
    def ideal(self) -> tuple[float, float]:
        """Expose the gateway's live ideal, including every nonselected candidate."""
        return self.gateway.ideal

    @ideal.setter
    def ideal(self, value: tuple[float, float]) -> None:
        self.gateway.ideal = value

    def evaluate(
        self, genotype: Genotype, schedule: Schedule | None = None, origin: str = "polish"
    ) -> Candidate:
        """Lazily verify the parent once and charge all reuse setup inside production time."""
        if not self.reuse:
            return self.gateway.evaluate(genotype, schedule, origin)
        if self.context is None:
            started = perf_counter()
            self.context = PrefillEvaluationContext(self.gateway.problem, self.parent)
            self.gateway.counts["polish_parent_audits"] += 1
            self.gateway.seconds["polish_context"] += perf_counter() - started
        return self.gateway.evaluate(genotype, schedule, origin, prefill_context=self.context)


def focus_prefills(
    problem: ProblemInstance, parent: Candidate, resources: ResourceView | None = None
) -> tuple[int, ...]:
    """Pick TOU exposure then a distinct peak-window exposure Prefill, at most two.

    Only activity uniquely attributable to an operation contributes to its score;
    overlapping work does not double-count machine power. These are inexpensive
    priority scores, never substitute objective values or acceptance decisions.
    All tied regional peak windows contribute to the demand exposure score.
    """
    g, schedule = parent.genotype, parent.schedule
    if resources is None:
        resources = ResourceView(problem, g, schedule)
    else:
        resources.check_binding(problem, g, schedule)
    eligible = {
        o
        for o in range(0, problem.operation_count, 2)
        if less(schedule.starts[o], legal_bounds(problem, g, schedule, o)[1], TOL.time)
    }
    if not eligible:
        return ()
    tou = dict.fromkeys(eligible, 0.0)
    demand = dict.fromkeys(eligible, 0.0)
    tariffs = TariffIndex(problem)
    peaks = [set(tied_peaks(w)) for w in parent.evaluation.windows]
    for m, items in enumerate(resources.items):
        if not any(o in eligible for o, _, _ in items):
            continue
        server = problem.instances[m]
        region = server.region
        power = server.active_kw - server.idle_kw
        points = resources.events(m)
        for a, b in zip(points, points[1:]):
            active = [o for o, s, e in items if s <= a < e]
            if len(active) != 1 or active[0] not in eligible:
                continue
            o = active[0]
            tou[o] += power * tariffs.integral(problem, region, a, b) / 3600
            for window in range(int(a // 900), ceil(b / 900)):
                if window in peaks[region]:
                    overlap = max(0.0, min(b, (window + 1) * 900) - max(a, window * 900))
                    demand[o] += power * overlap / 900 * problem.regions[region].demand_rate
    first = min(eligible, key=lambda o: (-tou[o], o))
    others = eligible - {first}
    return (first, min(others, key=lambda o: (-demand[o], o))) if others else (first,)


def _best_for_operation(
    problem: ProblemInstance,
    current: Candidate,
    operation: int,
    gateway: EvaluationGateway,
    stats: PolishStats,
    evaluator: ReusedPrefillEvaluator,
    resources: ResourceView,
) -> Candidate:
    options: list[tuple[Candidate, float]] = []
    old = current.schedule.starts[operation]
    for start in right_shift_times(
        problem, current.genotype, current.schedule, operation, resources
    ):
        stats.candidates += 1
        schedule = move(
            problem, current.genotype, current.schedule, operation, start, resources=resources
        )
        if schedule is None:
            continue
        candidate = evaluator.evaluate(current.genotype, schedule, "polish")
        stats.exact += 1
        if candidate.evaluation.feasible:
            if candidate.evaluation.flow != current.evaluation.flow:
                raise AssertionError("Prefill-only timing changed Flow")
            options.append((candidate, start - old))
    if not options:
        return current
    minimum = min(c.evaluation.bill for c, _ in options)
    tied = [(c, d) for c, d in options if close(c.evaluation.bill, minimum, TOL.cost)]
    best, _ = min(tied, key=lambda pair: (pair[1], identity(pair[0])))
    if less(best.evaluation.bill, current.evaluation.bill, TOL.cost):
        stats.accepted += 1
        return best
    return current


def policy_polish(
    problem: ProblemInstance,
    parent: Candidate,
    gateway: EvaluationGateway,
    policy: str,
    stats: PolishStats | None = None,
    *,
    reuse: bool = True,
) -> Candidate:
    """Apply a declared policy with all candidate costs/Archive attempts retained.

    ``reference_reuse`` has exactly the original enumeration and one-best-move
    rule. ``two_focus`` freezes TOU/demand picks, then accepts up to two moves.
    ``sequential_scan`` visits all Prefills in reverse frozen-D readiness order,
    rechecking the current schedule after each accepted strict Bill decrease.
    Both new policies freeze Decode, MS/OS, and preserve Flow exactly. There is
    no new demand-balance candidate set in this study: critical starts match the
    reference, so invocation/operation selection remain separately interpretable.
    """
    if policy not in ("reference_reuse", "two_focus", "sequential_scan"):
        raise ValueError("Unknown experimental Polish policy")
    if gateway.problem is not problem:
        raise ValueError("Polish gateway belongs to another input")
    stats = stats if stats is not None else PolishStats()
    if policy == "reference_reuse":
        return polish(problem, parent, ReusedPrefillEvaluator(gateway, parent, reuse), stats)
    current = parent
    resources = ResourceView(problem, parent.genotype, parent.schedule)
    if policy == "two_focus":
        selected = focus_prefills(problem, parent, resources)
    else:
        selected = tuple(
            sorted(
                range(0, problem.operation_count, 2),
                key=lambda o: (parent.schedule.starts[o + 1], parent.schedule.starts[o], o),
                reverse=True,
            )
        )
    gateway.counts["polish_selected_prefills"] += len(selected)
    evaluator = ReusedPrefillEvaluator(gateway, current, reuse)
    for operation in selected:
        before = current
        current = _best_for_operation(
            problem, current, operation, gateway, stats, evaluator, resources
        )
        if current is not before:
            resources = ResourceView(problem, current.genotype, current.schedule)
            evaluator = ReusedPrefillEvaluator(gateway, current, reuse)
    if current is not parent:
        current = replace(
            current,
            schedule=refresh_diagnostics(problem, current.genotype, current.schedule),
        )
    return current
