"""Deterministic single Prefill polish, separate from Q credit and MacroSearch budget."""

import math
from dataclasses import dataclass, replace

from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.macrosearch.search import Evaluator
from geo_llm_scheduler.scheduling.resources import ResourceView
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.scheduling.timing import legal_bounds, move
from geo_llm_scheduler.utils.numeric import TOL, close, identity, less


@dataclass
class PolishStats:
    """Separate enumeration, exact evaluation and accepted-move accounting."""

    candidates: int = 0
    exact: int = 0
    accepted: int = 0


def right_shift_times(
    problem: ProblemInstance,
    genotype: Genotype,
    schedule: Schedule,
    operation: int,
    resources: ResourceView,
) -> tuple[float, ...]:
    """Return the original finite critical Prefill starts in unchanged sorted order."""
    resources.check_binding(problem, genotype, schedule)
    if operation % 2:
        raise ValueError("Right-shift Polish moves Prefill only")
    old = schedule.starts[operation]
    _, upper = legal_bounds(problem, genotype, schedule, operation)
    if not less(old, upper, TOL.time):
        return ()
    duration = problem.profile(operation).duration
    region = problem.instances[genotype.ms[operation]].region
    events = set(resources.events(genotype.ms[operation]))
    events |= {
        t
        for tariff in problem.regions[region].tariffs
        for t in (tariff.start, tariff.end)
        if old - duration <= t <= upper + duration
    }
    events |= {float(b * 900) for b in range(math.ceil((upper + duration) / 900) + 1)}
    return tuple(
        t
        for t in sorted({upper} | events | {e - duration for e in events})
        if less(old, t, TOL.time) and not less(upper, t, TOL.time)
    )


def polish(
    problem: ProblemInstance,
    incumbent: Candidate,
    evaluator: Evaluator,
    stats: PolishStats | None = None,
) -> Candidate:
    """Evaluate every feasible critical right-shift and execute at most one best move."""
    g = incumbent.genotype
    stats = stats if stats is not None else PolishStats()
    options = []
    resources = ResourceView(problem, g, incumbent.schedule)
    for o in range(0, problem.operation_count, 2):
        old = incumbent.schedule.starts[o]
        for t in right_shift_times(problem, g, incumbent.schedule, o, resources):
            stats.candidates += 1
            schedule = move(problem, g, incumbent.schedule, o, t, resources=resources)
            if schedule is None:
                continue
            candidate = evaluator.evaluate(g, schedule, "polish")
            stats.exact += 1
            if candidate.evaluation.feasible:
                if candidate.evaluation.flow != incumbent.evaluation.flow:
                    raise AssertionError("Prefill-only polish changed Flow")
                options.append((candidate, t - old))
    if not options:
        return incumbent
    minimum = min(c.evaluation.bill for c, _ in options)
    tied = [(c, d) for c, d in options if close(c.evaluation.bill, minimum, TOL.cost)]
    best, _ = min(tied, key=lambda pair: (pair[1], identity(pair[0])))
    if not less(best.evaluation.bill, incumbent.evaluation.bill, TOL.cost):
        return incumbent
    stats.accepted += 1
    return replace(best, schedule=refresh_diagnostics(problem, g, best.schedule))
