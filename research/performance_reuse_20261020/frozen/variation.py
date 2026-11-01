"""Job-linked MS/OS crossover and three elementary structural mutations."""

import random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype, ProblemInstance


def paths(problem: ProblemInstance, job: int, region: int | None = None) -> list[tuple[int, int]]:
    """Enumerate legal same-region Prefill/Decode assignments."""
    return [
        (p, d)
        for p in problem.eligible(2 * job, region)
        for d in problem.eligible(2 * job + 1, region)
        if problem.instances[p].region == problem.instances[d].region
    ]


def crossover(a: Genotype, b: Genotype, rng: random.Random) -> Genotype:
    """Inherit both occurrences and both MS genes together for each selected job."""
    n = len(a.ms) // 2
    selected = {i for i in range(n) if rng.random() < 0.5}
    fill = iter(j for j in b.os if j not in selected)
    order = tuple(j if j in selected else next(fill) for j in a.os)
    ms = tuple((a if o // 2 in selected else b).ms[o] for o in range(2 * n))
    return Genotype(ms, order)


def reproduce(
    problem: ProblemInstance, a: Genotype, b: Genotype, config: Config, rng: random.Random
) -> Genotype:
    """Generate one structural child; mutation is applied after crossover decision."""
    g = crossover(a, b, rng) if rng.random() < config.crossover_probability else rng.choice((a, b))
    if not problem.jobs or rng.random() >= config.mutation_probability:
        return g
    kind = rng.choices(range(3), weights=config.mutation_weights)[0]
    ms, os = list(g.ms), list(g.os)
    job = rng.randrange(len(problem.jobs))
    region = problem.instances[ms[2 * job]].region
    if kind == 0:
        pairs = [(i, j) for i in range(len(os)) for j in range(i + 1, len(os)) if os[i] != os[j]]
        if pairs:
            i, j = rng.choice(pairs)
            os[i], os[j] = os[j], os[i]
    elif kind == 1:
        o = 2 * job + rng.randrange(2)
        choices = [m for m in problem.eligible(o, region) if m != ms[o]]
        if choices:
            ms[o] = rng.choice(choices)
    else:
        path_choices = [
            (p, d) for p, d in paths(problem, job) if problem.instances[p].region != region
        ]
        if path_choices:
            ms[2 * job : 2 * job + 2] = rng.choice(path_choices)
    return Genotype(tuple(ms), tuple(os))
