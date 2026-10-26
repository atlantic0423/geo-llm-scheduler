"""SPEA2, additive-epsilon IBEA and explicitly adapted random-key MOPSO.

Their bounded native mating repositories are independent of the passive,
unbounded measurement Archive used by every study arm.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from time import perf_counter
from typing import Callable

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import RunResult
from geo_llm_scheduler.experiments.nsga2 import dominates
from geo_llm_scheduler.moead.variation import paths, reproduce
from geo_llm_scheduler.rl.controller import Controller
from geo_llm_scheduler.utils.numeric import identity
from geo_llm_scheduler.utils.rng import RNGManager


def normalized_objectives(population: list[Candidate]) -> list[tuple[float, float]]:
    """Scale each current selection pool to unit ranges, including degenerate axes."""
    values = [c.evaluation.objectives for c in population]
    low = [min(v[k] for v in values) for k in (0, 1)]
    span = [max(v[k] for v in values) - low[k] or 1.0 for k in (0, 1)]
    return [((v[0] - low[0]) / span[0], (v[1] - low[1]) / span[1]) for v in values]


def spea_fitness(population: list[Candidate]) -> tuple[list[float], list[list[float]]]:
    """Strength-sum raw fitness plus kth-neighbor density, k=floor(sqrt(pool))."""
    count = len(population)
    if not count:
        raise ValueError("Empty SPEA2 pool")
    beaten = [
        {j for j in range(count) if dominates(population[i], population[j])} for i in range(count)
    ]
    strength = [len(items) for items in beaten]
    raw = [sum(strength[j] for j in range(count) if i in beaten[j]) for i in range(count)]
    points = normalized_objectives(population)
    distances = [[math.dist(a, b) for b in points] for a in points]
    k = min(max(1, int(math.sqrt(count))), count - 1)
    fitness = [raw[i] + 1 / (sorted(distances[i])[k] + 2) for i in range(count)]
    return fitness, distances


def spea_select(population: list[Candidate], size: int) -> tuple[list[Candidate], list[float]]:
    """Select nondominated solutions, lexicographically truncate nearest distances."""
    if not 1 <= size <= len(population):
        raise ValueError("Invalid SPEA2 size")
    fitness, distances = spea_fitness(population)
    chosen = [i for i, f in enumerate(fitness) if f < 1]
    if len(chosen) < size:
        chosen += sorted(
            (i for i, f in enumerate(fitness) if f >= 1), key=lambda i: (fitness[i], i)
        )[: size - len(chosen)]
    while len(chosen) > size:
        removed = min(
            chosen, key=lambda i: (tuple(sorted(distances[i][j] for j in chosen if j != i)), -i)
        )
        chosen.remove(removed)
    return [population[i] for i in chosen], [fitness[i] for i in chosen]


def epsilon_contributions(population: list[Candidate], kappa: float = 0.05) -> list[list[float]]:
    """Return positive epsilon indicator losses from row donor to column recipient."""
    if not population or kappa <= 0:
        raise ValueError("Invalid indicator pool/kappa")
    values = normalized_objectives(population)
    indicators = [[max(a[k] - b[k] for k in (0, 1)) for b in values] for a in values]
    scale = max((abs(v) for row in indicators for v in row), default=0.0) or 1.0
    return [
        [0.0 if i == j else math.exp(-value / (scale * kappa)) for j, value in enumerate(row)]
        for i, row in enumerate(indicators)
    ]


def ibea_select(population: list[Candidate], size: int) -> tuple[list[Candidate], list[float]]:
    """Iteratively delete maximal positive loss and update remaining contributions."""
    if not 1 <= size <= len(population):
        raise ValueError("Invalid IBEA size")
    matrix = epsilon_contributions(population)
    loss = [sum(row[j] for row in matrix) for j in range(len(population))]
    chosen = list(range(len(population)))
    while len(chosen) > size:
        worst = max(chosen, key=lambda i: (loss[i], i))
        chosen.remove(worst)
        for i in chosen:
            loss[i] -= matrix[worst][i]
    return [population[i] for i in chosen], [loss[i] for i in chosen]


def encode_keys(
    problem: ProblemInstance,
    genotype: Genotype,
    legal_paths: list[list[tuple[int, int]]] | None = None,
) -> list[float]:
    """Embed a legal MS/OS exactly into N legal-path keys and 2N priority keys."""
    n = len(problem.jobs)
    result = []
    for j in range(n):
        legal = legal_paths[j] if legal_paths is not None else paths(problem, j)
        result.append(
            (legal.index((genotype.ms[2 * j], genotype.ms[2 * j + 1])) + 0.5) / len(legal)
        )
    counts = [0] * n
    order = [0.0] * (2 * n)
    for rank, job in enumerate(genotype.os):
        operation = 2 * job + counts[job]
        counts[job] += 1
        order[operation] = (rank + 0.5) / (2 * n)
    return result + order


def decode_keys(
    problem: ProblemInstance, keys: list[float], legal: list[list[tuple[int, int]]] | None = None
) -> Genotype:
    """Map clipped random keys to same-region paths and repeated-job priority OS."""
    n = len(problem.jobs)
    if len(keys) != 3 * n or any(not math.isfinite(x) or not 0 <= x <= 1 for x in keys):
        raise ValueError("Invalid random keys")
    legal = legal if legal is not None else [paths(problem, j) for j in range(n)]
    ms = tuple(
        m
        for j, x in enumerate(keys[:n])
        for m in legal[j][min(int(x * len(legal[j])), len(legal[j]) - 1)]
    )
    os = tuple(o // 2 for o in sorted(range(2 * n), key=lambda o: (keys[n + o], o)))
    return Genotype(ms, os)


def grid_repository(candidates: list[Candidate], size: int, rng: random.Random) -> list[Candidate]:
    """Maintain a bounded native nondominated repository, crowding-biased deletion."""
    unique = list({identity(c): c for c in candidates}.values())
    result = [
        a
        for i, a in enumerate(unique)
        if not any(dominates(b, a) for j, b in enumerate(unique) if i != j)
    ]
    while len(result) > size:
        cells = _grid_cells(result)
        key = rng.choices(list(cells), weights=[len(v) ** 2 for v in cells.values()])[0]
        del result[rng.choice(cells[key])]
    return result


def _grid_cells(repository: list[Candidate]) -> dict[tuple[int, int], list[int]]:
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, point in enumerate(normalized_objectives(repository)):
        cell = tuple(min(9, max(0, int((x + 0.1) / 1.2 * 10))) for x in point)
        cells[cell].append(i)  # type: ignore[index]
    return dict(cells)


def select_leader(repository: list[Candidate], rng: random.Random) -> Candidate:
    """Favor sparsely occupied native grid cells, then sample a member uniformly."""
    cells = _grid_cells(repository)
    key = rng.choices(list(cells), weights=[1 / len(v) for v in cells.values()])[0]
    return repository[rng.choice(cells[key])]


def update_repository(
    repository: list[Candidate],
    candidate: Candidate,
    size: int,
    rng: random.Random,
) -> list[Candidate]:
    """Insert one candidate into a previously nondominated native repository."""
    key = identity(candidate)
    if any(identity(c) == key or dominates(c, candidate) for c in repository):
        return repository
    result = [c for c in repository if not dominates(candidate, c)] + [candidate]
    while len(result) > size:
        cells = _grid_cells(result)
        cell = rng.choices(list(cells), weights=[len(v) ** 2 for v in cells.values()])[0]
        del result[rng.choice(cells[cell])]
    return result


def run_classic(
    problem: ProblemInstance,
    config: Config,
    method: str,
    initializer: Callable[[EvaluationGateway, RNGManager], list[Candidate]],
    *,
    observer: Callable[[str, dict], None] | None = None,
) -> RunResult:
    """Run native SPEA2/epsilon-IBEA or random-key MOPSO under one total wall clock."""
    if method not in ("SPEA2", "EPS_IBEA", "RK_MOPSO"):
        raise ValueError("Unknown classic baseline")
    begin = perf_counter()
    streams = RNGManager(config.seed)
    gateway = EvaluationGateway(
        problem, Archive(), config.exact_evaluation_cap, config.decode_cache_size
    )
    if observer:
        gateway.candidate_sink = lambda c: observer(
            "candidate", dict(candidate=c, gateway=gateway, elapsed=perf_counter() - begin)
        )
    population = initializer(gateway, streams)
    gateway.seconds["initialization"] = perf_counter() - begin
    rng = streams.stream("variation")
    processed = 0
    reason = "generation_limit"
    native: list[Candidate] = []
    vectors: list[list[float]] = []
    velocities: list[list[float]] = []
    best_vectors: list[list[float]] = []
    personal: list[Candidate] = []
    legal = [paths(problem, j) for j in range(len(problem.jobs))] if method == "RK_MOPSO" else []
    if method == "RK_MOPSO":
        vectors = [encode_keys(problem, c.genotype, legal) for c in population]
        velocities = [[0.0] * len(v) for v in vectors]
        best_vectors = [v.copy() for v in vectors]
        personal = population.copy()
        native = grid_repository(population, config.population, rng)
    for generation in range(config.generations):
        if config.seconds is not None and perf_counter() - begin >= config.seconds:
            reason = "time_budget"
            break
        if method == "SPEA2":
            native, fitness = spea_select(population + native, config.population)
        elif method == "EPS_IBEA":
            native, fitness = ibea_select(population, config.population)
        offspring = []
        for i in range(config.population):
            if config.seconds is not None and perf_counter() - begin >= config.seconds:
                reason = "time_budget"
                break
            if (
                config.exact_evaluation_cap is not None
                and gateway.counts["exact"] >= config.exact_evaluation_cap
            ):
                reason = "exact_evaluation_cap"
                break
            if method == "RK_MOPSO":
                leader = encode_keys(problem, select_leader(native, rng).genotype, legal)
                for j, x in enumerate(vectors[i]):
                    v = (
                        0.4 * velocities[i][j]
                        + 1.5 * rng.random() * (best_vectors[i][j] - x)
                        + 1.5 * rng.random() * (leader[j] - x)
                    )
                    v = max(-0.2, min(0.2, v))
                    y = x + v
                    if not 0 <= y <= 1:
                        y, v = min(1.0, max(0.0, y)), -v
                    vectors[i][j], velocities[i][j] = y, v
                if rng.random() < config.mutation_probability:
                    vectors[i][rng.randrange(len(vectors[i]))] = rng.random()
                child = gateway.evaluate(decode_keys(problem, vectors[i], legal), origin="mopso")
                if dominates(child, personal[i]) or (
                    not dominates(personal[i], child) and rng.random() < 0.5
                ):
                    personal[i], best_vectors[i] = child, vectors[i].copy()
                population[i] = child
                native = update_repository(native, child, config.population, rng)
            else:

                def parent() -> Candidate:
                    a, b = rng.sample(range(len(native)), 2)
                    return native[min((a, b), key=lambda j: (fitness[j], j))]

                child = gateway.evaluate(
                    reproduce(problem, parent().genotype, parent().genotype, config, rng)
                )
                offspring.append(child)
            processed += 1
            if observer:
                observer(
                    "offspring",
                    dict(
                        row=dict(
                            generation=generation, exact_count=gateway.counts["exact"], steps=[]
                        ),
                        processed=processed,
                        gateway=gateway,
                        elapsed=perf_counter() - begin,
                    ),
                )
        if method == "EPS_IBEA" and offspring:
            population, _ = ibea_select(native + offspring, config.population)
        elif method == "SPEA2" and offspring:
            if len(offspring) == config.population:
                population = offspring
            else:
                native, _ = spea_select(native + offspring, config.population)
                population = native.copy()
        if reason in ("time_budget", "exact_evaluation_cap"):
            break
    if method == "SPEA2" and native:
        population, _ = spea_select(population + native, config.population)
    return RunResult(
        population,
        gateway.archive,
        gateway,
        [],
        perf_counter() - begin,
        Controller(config),
        reason,
        offspring_count=processed,
    )
