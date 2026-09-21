"""Deterministic serial schedule generation on continuous resource events."""

from geo_llm_scheduler.domain.models import Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.io.validation import validate_genotype
from geo_llm_scheduler.scheduling.resources import earliest, est_for


def decode(problem: ProblemInstance, genotype: Genotype) -> Schedule:
    """Interpret first/second OS occurrence as P/D and place earliest feasible."""
    validate_genotype(problem, genotype)
    starts = [-1.0] * problem.operation_count
    waits = [0.0] * problem.operation_count
    seen = [0] * len(problem.jobs)
    for job in genotype.os:
        o = 2 * job + seen[job]
        seen[job] += 1
        partial = Schedule(tuple(starts))
        est = est_for(problem, genotype, partial, o)
        starts[o] = earliest(problem, genotype, partial, o, est)
        waits[o] = starts[o] - est
    return Schedule(tuple(starts), tuple(waits), (0.0,) * problem.operation_count)


def refresh_diagnostics(
    problem: ProblemInstance, genotype: Genotype, schedule: Schedule
) -> Schedule:
    """Fully recompute frozen-profile resource/intentional waits without decoding."""
    resource, intentional = [], []
    for o, start in enumerate(schedule.starts):
        est = est_for(problem, genotype, schedule, o)
        minimum = earliest(problem, genotype, schedule, o, est)
        resource.append(max(0.0, minimum - est))
        intentional.append(max(0.0, start - minimum))
    return Schedule(schedule.starts, tuple(resource), tuple(intentional))
