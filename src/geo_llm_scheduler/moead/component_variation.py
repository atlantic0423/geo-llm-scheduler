"""Explicit, opt-in structural operators for the v4 component study.

These functions return genotypes only. Callers must rebuild and evaluate them
through EvaluationGateway; they do not inherit timing from either parent.
"""

import random
from collections import Counter

from geo_llm_scheduler.domain.models import Genotype, ProblemInstance
from geo_llm_scheduler.moead.variation import _DifferentJobPairs, crossover, paths

CROSSOVERS = ("job_linked", "independent_masks", "segment_fill", "precedence_merge")
ORDER_MUTATIONS = ("swap", "insert", "block2_insert", "reverse4")
WITHIN_MUTATIONS = ("single_phase", "paired_phases", "swap_job_paths")


def cross(a: Genotype, b: Genotype, rule: str, rng: random.Random) -> Genotype:
    """Cross legal repeated-job encodings, retaining same-region MS pairs."""
    if rule not in CROSSOVERS:
        raise ValueError("Unknown crossover rule")
    if rule == "job_linked":
        return crossover(a, b, rng)
    n = len(a.ms) // 2
    if len(b.ms) != len(a.ms) or Counter(a.os) != Counter(b.os):
        raise ValueError("Parents have incompatible jobs")
    if rule == "independent_masks":
        order = crossover(a, b, rng).os
    elif rule == "segment_fill":
        if len(a.os) < 2:
            order = a.os
        else:
            lo, hi = sorted(rng.sample(range(len(a.os) + 1), 2))
            remaining = Counter(a.os)
            remaining.subtract(a.os[lo:hi])
            fill = []
            for job in b.os:
                if remaining[job] > 0:
                    fill.append(job)
                    remaining[job] -= 1
            order = tuple(fill[:lo]) + a.os[lo:hi] + tuple(fill[lo:])
    else:

        def operations(order: tuple[int, ...]) -> list[int]:
            seen: Counter[int] = Counter()
            result = []
            for job in order:
                result.append(2 * job + seen[job])
                seen[job] += 1
            return result

        left, right = operations(a.os), operations(b.os)
        merged = []
        while left:
            operation = (left if rng.random() < 0.5 else right)[0]
            merged.append(operation // 2)
            left.remove(operation)
            right.remove(operation)
        order = tuple(merged)
    selected = {j for j in range(n) if rng.random() < 0.5}
    ms = tuple((a if o // 2 in selected else b).ms[o] for o in range(2 * n))
    return Genotype(ms, order)


def mutate_order(g: Genotype, rule: str, rng: random.Random) -> Genotype:
    """Apply one bounded move; repeated labels may yield a recorded no-op."""
    if rule not in ORDER_MUTATIONS:
        raise ValueError("Unknown order mutation")
    order = list(g.os)
    if rule == "swap":
        pairs = _DifferentJobPairs(order)
        if pairs:
            i, j = rng.choice(pairs)
            order[i], order[j] = order[j], order[i]
    elif rule == "reverse4":
        if len(order) >= 4:
            i = rng.randrange(len(order) - 3)
            order[i : i + 4] = reversed(order[i : i + 4])
    else:
        width = 1 if rule == "insert" else 2
        if len(order) > width:
            i = rng.randrange(len(order) - width + 1)
            block = order[i : i + width]
            del order[i : i + width]
            slots = [j for j in range(len(order) + 1) if j != i]
            j = rng.choice(slots)
            order[j:j] = block
    return Genotype(g.ms, tuple(order))


def mutate_within(problem: ProblemInstance, g: Genotype, rule: str, rng: random.Random) -> Genotype:
    """Change instances without changing regions; return unchanged if no legal move."""
    if rule not in WITHIN_MUTATIONS:
        raise ValueError("Unknown within-region mutation")
    if not problem.jobs:
        return g
    job = rng.randrange(len(problem.jobs))
    ms = list(g.ms)
    region = problem.instances[ms[2 * job]].region
    if rule == "single_phase":
        op = 2 * job + rng.randrange(2)
        choices = [m for m in problem.eligible(op, region) if m != ms[op]]
        if choices:
            ms[op] = rng.choice(choices)
    elif rule == "paired_phases":
        pair_choices = [
            (p, d)
            for p, d in paths(problem, job, region)
            if p != ms[2 * job] and d != ms[2 * job + 1]
        ]
        if pair_choices:
            ms[2 * job : 2 * job + 2] = rng.choice(pair_choices)
    else:
        partners = [
            j
            for j in range(len(problem.jobs))
            if j != job and problem.instances[ms[2 * j]].region == region
        ]
        for other in rng.sample(partners, min(4, len(partners))):
            a = tuple(ms[2 * job : 2 * job + 2])
            b = tuple(ms[2 * other : 2 * other + 2])
            if a != b and b in paths(problem, job, region) and a in paths(problem, other, region):
                ms[2 * job : 2 * job + 2] = b
                ms[2 * other : 2 * other + 2] = a
                break
    return Genotype(tuple(ms), g.os)
