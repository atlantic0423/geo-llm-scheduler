"""Shared scheduling adapter and numerical kernels for attributed PlatEMO ports.

Algorithm-specific rules remain in their own modules. This adapter supplies
legal P/D assignments, repeated-job priorities and objective-blind waiting genes.
See docs/experiments/trans_python.md for upstream attribution and deviations.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.scheduling.timing import legal_bounds, move
from geo_llm_scheduler.utils.rng import RNGManager

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class Individual:
    """Continuous decision vector bound to its exact feasible scheduling phenotype."""

    genes: Array
    candidate: Candidate


class SearchContext:
    """Own explicit RNG and one inclusive wall clock, including adapter construction."""

    def __init__(
        self,
        problem: ProblemInstance,
        seed: int,
        seconds: float,
        *,
        wait_genes: bool = True,
        max_generations: int | None = None,
    ) -> None:
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Positive wall budget required")
        self.started = perf_counter()
        self.seconds = seconds
        self.problem = problem
        self.wait_genes = wait_genes
        self.dimensions = len(problem.jobs) * (4 if wait_genes else 3)
        self.legal = [paths(problem, i) for i in range(len(problem.jobs))]
        if not problem.jobs or any(not p for p in self.legal):
            raise ValueError("Every job needs a legal P/D path")
        self.rng = np.random.default_rng(RNGManager(seed).stream("trans_native").getrandbits(64))
        self.gateway = EvaluationGateway(problem, Archive(), decode_cache_size=10000)
        self.evaluations = 0
        self.generations = 0
        if max_generations is not None and max_generations < 0:
            raise ValueError("Negative generation limit")
        self.max_generations = max_generations
        self.trace: list[dict] = []

    def running(self) -> bool:
        """Wall stop, with an optional deterministic engineering-test generation cap."""
        return self.progress() < 1 and (
            self.max_generations is None or self.generations < self.max_generations
        )

    def progress(self) -> float:
        """Charged wall fraction replacing FE/maxFE in wall-budget adaptations."""
        return min(1.0, (perf_counter() - self.started) / self.seconds)

    def evaluate(self, genes: Array) -> Individual:
        """Map bounded keys, repair waiting by resource feasibility, then exact evaluate.

        Waiting repair never tests the objective. The earliest decoded candidate
        is also a complete exact candidate and therefore reaches the passive Archive.
        """
        x = np.asarray(genes, dtype=float)
        if x.shape != (self.dimensions,) or not np.all(np.isfinite(x)):
            raise ValueError("Invalid continuous genotype")
        x = np.clip(x, 0, 1)
        n = len(self.legal)
        assignment = tuple(
            m
            for choices, key in zip(self.legal, x[:n], strict=True)
            for m in choices[min(int(float(key) * len(choices)), len(choices) - 1)]
        )
        order = tuple(int(i) // 2 for i in np.argsort(x[n : 3 * n], kind="stable"))
        genotype = Genotype(assignment, order)
        current = self.gateway.evaluate(genotype, origin="trans_keys")
        schedule = current.schedule
        if self.wait_genes:
            for j, fraction in enumerate(x[3 * n :]):
                operation = 2 * j
                start = schedule.starts[operation]
                _, latest = legal_bounds(self.problem, genotype, schedule, operation)
                target = start + float(fraction) * max(0.0, latest - start)
                shifted = move(self.problem, genotype, schedule, operation, target)
                if shifted is not None:
                    schedule = shifted
            if schedule.starts != current.schedule.starts:
                current = self.gateway.evaluate(genotype, schedule, origin="trans_wait_genes")
        if not current.evaluation.feasible:
            raise AssertionError("Feasible scheduling adapter returned an infeasible candidate")
        self.evaluations += 1
        return Individual(x.copy(), current)

    def initialize(self, size: int) -> list[Individual]:
        """Draw a native continuous population, charging every decoded evaluation."""
        if size < 4:
            raise ValueError("At least four individuals required")
        return [self.evaluate(x) for x in self.rng.random((size, self.dimensions))]

    def offspring(self, genes: Array) -> list[Individual]:
        """Stop evaluation at complete-candidate boundaries when wall time expires."""
        result = []
        for x in genes:
            if self.progress() >= 1:
                break
            result.append(self.evaluate(x))
        return result


def objectives(population: list[Individual]) -> Array:
    """Read raw Flow and total bill without changing upstream objective scaling."""
    return np.asarray([x.candidate.evaluation.objectives for x in population], dtype=float)


def decisions(population: list[Individual]) -> Array:
    """Stack decision keys for native continuous GA and DE operators."""
    return np.asarray([x.genes for x in population], dtype=float)


def dominance_mask(values: Array, cv: Array | None = None, *, round10: bool = False) -> IntArray:
    """CMOEA-CD's Pareto/constraint-Pareto mask with first equal point retained."""
    f = values.copy()
    if round10:
        f = np.sign(f) * np.floor(np.abs(f) * 1e10 + 0.5) / 1e10
    n = len(f)
    violation = np.zeros(n) if cv is None else cv
    dominated = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if np.array_equal(f[i], f[j]):
                dominated[j if violation[i] <= violation[j] else i] = True
            elif np.all(f[i] >= f[j]) and violation[j] <= violation[i]:
                dominated[i] = True
            elif np.all(f[j] >= f[i]) and violation[i] <= violation[j]:
                dominated[j] = True
    return np.flatnonzero(~dominated)


def truncate(values: Array, size: int) -> list[int]:
    """SPEA2 nearest-distance lexicographic deletion, stable source-order ties."""
    if not 1 <= size <= len(values):
        raise ValueError("Invalid truncation size")
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    remaining = list(range(len(values)))
    while len(remaining) > size:
        removed = min(remaining, key=lambda i: tuple(np.sort(distances[i, remaining])))
        remaining.remove(removed)
    return remaining


def spea_fitness(values: Array, cv: Array | None = None) -> Array:
    """Upstream unnormalized strength plus kth-neighbor density, with constraint first."""
    violation = np.zeros(len(values)) if cv is None else cv
    pareto = np.all(values[:, None, :] <= values[None, :, :], axis=2) & np.any(
        values[:, None, :] < values[None, :, :], axis=2
    )
    domination = (violation[:, None] < violation[None, :]) | (
        (violation[:, None] == violation[None, :]) & pareto
    )
    raw = domination.sum(axis=1) @ domination
    distance = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    k = max(0, int(np.sqrt(len(values))) - 1)
    return raw + 1 / (np.sort(distance, axis=1)[:, k] + 2)


def select_spea(
    population: list[Individual], size: int
) -> tuple[list[Individual], Array, list[int]]:
    """Port AOP/FLCS selection, preserving unsorted membership indices for rewards."""
    values = objectives(population)
    fit = spea_fitness(values)
    selected: list[int] = np.flatnonzero(fit < 1).tolist()
    if len(selected) < size:
        selected = sorted(np.argsort(fit, kind="stable")[:size].tolist())
    elif len(selected) > size:
        selected = [selected[i] for i in truncate(values[selected], size)]
    order = sorted(selected, key=lambda i: fit[i])
    return [population[i] for i in order], fit[order], selected


def tournament(fitness: Array, count: int, rng: np.random.Generator) -> IntArray:
    """Binary tournament with replacement and first-draw ties."""
    draws = rng.integers(len(fitness), size=(2, count))
    return np.where(fitness[draws[0]] <= fitness[draws[1]], draws[0], draws[1])


def polynomial_mutation(x: Array, rng: np.random.Generator, eta: float = 20) -> Array:
    """PlatEMO unit-box polynomial mutation with expected one changed variable."""
    y = np.clip(x, 0, 1).copy()
    site = rng.random(y.shape) < 1 / y.shape[1]
    u = rng.random(y.shape)
    low = site & (u <= 0.5)
    high = site & (u > 0.5)
    y[low] += (2 * u[low] + (1 - 2 * u[low]) * (1 - y[low]) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    y[high] += 1 - (2 * (1 - u[high]) + 2 * (u[high] - 0.5) * y[high] ** (eta + 1)) ** (
        1 / (eta + 1)
    )
    return y


def ga(parents: Array, rng: np.random.Generator, *, half: bool = True, eta_m: float = 20) -> Array:
    """PlatEMO real SBX and polynomial mutation, default crossover probability one."""
    count = len(parents) // 2
    a, b = parents[:count], parents[count : 2 * count]
    u = rng.random(a.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / 21), (2 - 2 * u) ** (-1 / 21))
    beta *= 1 - 2 * rng.integers(2, size=a.shape)
    beta[rng.random(a.shape) < 0.5] = 1
    children = (a + b) / 2 + beta * (a - b) / 2
    if not half:
        children = np.vstack((children, (a + b) / 2 - beta * (a - b) / 2))
    return polynomial_mutation(children, rng, eta_m)


def de(a: Array, b: Array, c: Array, rng: np.random.Generator, *, eta_m: float = 20) -> Array:
    """PlatEMO DE with CR=1, F=0.5 and expected-one polynomial mutation."""
    return polynomial_mutation(a + 0.5 * (b - c), rng, eta_m)


def hv2(values: Array, reference: Array) -> float:
    """Exact minimization HV for two objectives and a supplied common reference."""
    kept = values[np.all(values <= reference, axis=1)]
    if not len(kept):
        return 0.0
    previous = float(reference[1])
    area = 0.0
    for x, y in kept[np.argsort(kept[:, 0], kind="stable")]:
        if y < previous:
            area += float(reference[0] - x) * (previous - float(y))
            previous = float(y)
    return area
