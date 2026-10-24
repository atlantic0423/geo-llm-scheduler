"""Job-linked MS/OS crossover and three elementary structural mutations."""

import random
from bisect import bisect_right
from collections.abc import Sequence
from typing import overload

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype, ProblemInstance


class _DifferentJobPairs(Sequence[tuple[int, int]]):
    """The original lexicographic swap-pair sequence, represented in O(N) space.

    random.choice still receives the exact original length, calls the same
    _randbelow once, and selects the same pair at that index. No rejection
    sampling, new draws, or changes to the mutation distribution are involved.
    """

    def __init__(self, order: list[int]) -> None:
        self.order = order
        counts: dict[int, int] = {}
        sizes = [0] * len(order)
        for i in range(len(order) - 1, -1, -1):
            job = order[i]
            sizes[i] = len(order) - i - 1 - counts.get(job, 0)
            counts[job] = counts.get(job, 0) + 1
        self.prefix = []
        total = 0
        for size in sizes:
            total += size
            self.prefix.append(total)

    def __len__(self) -> int:
        return self.prefix[-1] if self.prefix else 0

    @overload
    def __getitem__(self, index: int) -> tuple[int, int]: ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[int, int]]: ...

    def __getitem__(self, index: int | slice) -> tuple[int, int] | list[tuple[int, int]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("swap pair index out of range")
        i = bisect_right(self.prefix, index)
        local = index - (self.prefix[i - 1] if i else 0)
        for j in range(i + 1, len(self.order)):
            if self.order[i] != self.order[j]:
                if local == 0:
                    return i, j
                local -= 1
        raise AssertionError("swap-pair prefix is inconsistent")


def paths(problem: ProblemInstance, job: int, region: int | None = None) -> list[tuple[int, int]]:
    """Enumerate legal same-region Prefill/Decode assignments."""
    prefill = problem.eligible(2 * job, region)
    decode = problem.eligible(2 * job + 1, region)
    return [
        (p, d)
        for p in prefill
        for d in decode
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
        pairs = _DifferentJobPairs(os)
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
