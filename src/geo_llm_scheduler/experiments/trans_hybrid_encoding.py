"""Round-trip helpers for research hybrids using the continuous-key adapter."""

from __future__ import annotations

import numpy as np

from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.experiments.trans_common import Array
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.scheduling.timing import legal_bounds, move
from geo_llm_scheduler.utils.numeric import TOL, close


def encode_schedule(problem: ProblemInstance, candidate: Candidate) -> Array | None:
    """Encode an exact schedule when the adapter can reproduce every start time.

    Key ordering matches ``SearchContext.evaluate``.  The wait keys are rebuilt
    in job order with all other operations frozen; schedules outside this decoder's
    image return None rather than pairing a phenotype with inconsistent genes.
    This helper neither evaluates nor archives any candidate.
    """
    genotype: Genotype = candidate.genotype
    n = len(problem.jobs)
    keys: Array = np.zeros(4 * n, dtype=np.float64)
    for job in range(n):
        legal = paths(problem, job)
        pair = (genotype.ms[2 * job], genotype.ms[2 * job + 1])
        try:
            index = legal.index(pair)
        except ValueError:
            return None
        keys[job] = (index + 0.5) / len(legal)
    seen = [0] * n
    for rank, job in enumerate(genotype.os):
        if job < 0 or job >= n or seen[job] >= 2:
            return None
        operation = 2 * job + seen[job]
        seen[job] += 1
        keys[n + operation] = (rank + 0.5) / (2 * n)
    if seen != [2] * n:
        return None
    current = decode(problem, genotype)
    target = candidate.schedule
    for job in range(n):
        op = 2 * job
        start = current.starts[op]
        destination = target.starts[op]
        _, upper = legal_bounds(problem, genotype, current, op)
        span = max(0.0, upper - start)
        if destination < start - TOL.time or destination > upper + TOL.time:
            return None
        fraction = (destination - start) / span if span else 0.0
        keys[3 * n + job] = np.clip(fraction, 0.0, 1.0)
        moved = move(problem, genotype, current, op, start + keys[3 * n + job] * span)
        if moved is None:
            return None
        current = moved
    if any(not close(a, b, TOL.time) for a, b in zip(current.starts, target.starts, strict=True)):
        return None
    return keys
