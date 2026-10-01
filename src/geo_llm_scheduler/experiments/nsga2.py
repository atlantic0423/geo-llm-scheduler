"""Auditable two-objective NSGA-II sharing the MOEA/D representation and evaluator."""

from __future__ import annotations

import random
from time import perf_counter
from typing import Callable

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationCapReached, EvaluationGateway
from geo_llm_scheduler.engine.run import RunResult
from geo_llm_scheduler.engine.trajectory import improve
from geo_llm_scheduler.engine.trigger import triggered
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.validation import validate_problem
from geo_llm_scheduler.macrosearch.adaptive import AdaptiveBudget
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, weights
from geo_llm_scheduler.moead.variation import reproduce
from geo_llm_scheduler.rl.controller import Controller
from geo_llm_scheduler.rl.state import associate
from geo_llm_scheduler.utils.numeric import TOL, identity, less
from geo_llm_scheduler.utils.rng import RNGManager


def dominates(a: Candidate, b: Candidate) -> bool:
    """Return strict Pareto dominance for the two minimized exact objectives."""
    x, y = a.evaluation.objectives, b.evaluation.objectives
    return all(p <= q for p, q in zip(x, y)) and any(p < q for p, q in zip(x, y))


def nondominated_fronts(population: list[Candidate]) -> list[list[int]]:
    """Perform standard fast non-dominated sorting with stable input-index ties."""
    count = [0] * len(population)
    beaten: list[list[int]] = [[] for _ in population]
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            if dominates(population[i], population[j]):
                beaten[i].append(j)
                count[j] += 1
            elif dominates(population[j], population[i]):
                beaten[j].append(i)
                count[i] += 1
    front = [i for i, n in enumerate(count) if n == 0]
    fronts: list[list[int]] = []
    while front:
        fronts.append(front)
        following: list[int] = []
        for i in front:
            for j in beaten[i]:
                count[j] -= 1
                if count[j] == 0:
                    following.append(j)
        front = sorted(following)
    return fronts


def crowding_distance(population: list[Candidate], front: list[int]) -> dict[int, float]:
    """Compute normalized NSGA-II crowding, preserving objective endpoints."""
    distance = {i: 0.0 for i in front}
    if not front:
        return distance
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    for objective in range(2):
        ordered = sorted(front, key=lambda i: (population[i].evaluation.objectives[objective], i))
        lo = population[ordered[0]].evaluation.objectives[objective]
        hi = population[ordered[-1]].evaluation.objectives[objective]
        distance[ordered[0]] = distance[ordered[-1]] = float("inf")
        if hi > lo:
            for k in range(1, len(ordered) - 1):
                i = ordered[k]
                if distance[i] != float("inf"):
                    left = population[ordered[k - 1]].evaluation.objectives[objective]
                    right = population[ordered[k + 1]].evaluation.objectives[objective]
                    distance[i] += (right - left) / (hi - lo)
    return distance


def rank_and_crowding(population: list[Candidate]) -> tuple[list[int], list[float]]:
    """Return per-candidate Pareto rank and crowding for tournament selection."""
    rank = [0] * len(population)
    crowding = [0.0] * len(population)
    for r, front in enumerate(nondominated_fronts(population)):
        distances = crowding_distance(population, front)
        for i in front:
            rank[i] = r
            crowding[i] = distances[i]
    return rank, crowding


def tournament(rank: list[int], crowding: list[float], rng: random.Random) -> int:
    """Select a parent by rank then crowding; stable index breaks exact ties."""
    a, b = rng.sample(range(len(rank)), 2)
    return min((a, b), key=lambda i: (rank[i], -crowding[i], i))


def environmental_selection(population: list[Candidate], size: int) -> list[Candidate]:
    """Choose the best fronts and crowding-trim the last front, without Archive feedback."""
    selected: list[Candidate] = []
    for front in nondominated_fronts(population):
        if len(selected) + len(front) <= size:
            selected.extend(population[i] for i in front)
            continue
        distance = crowding_distance(population, front)
        chosen = sorted(front, key=lambda i: (-distance[i], i))[: size - len(selected)]
        selected.extend(population[i] for i in chosen)
        break
    return selected


def run_nsga2(
    problem: ProblemInstance,
    config: Config,
    initial: list[Genotype] | None = None,
    *,
    retain_trace: bool = True,
    observer: Callable[[str, dict], None] | None = None,
) -> RunResult:
    """Run a passive-Archive NSGA-II baseline under a soft wall-clock limit."""
    validate_problem(problem)
    begin = perf_counter()
    streams = RNGManager(config.seed)
    gateway = EvaluationGateway(problem, Archive(), config.exact_evaluation_cap)
    if observer is not None:
        gateway.candidate_sink = lambda candidate: observer(
            "candidate",
            {"candidate": candidate, "gateway": gateway, "elapsed": perf_counter() - begin},
        )
    initialization_start = perf_counter()
    population = [
        gateway.evaluate(g, origin="initialization")
        for g in (
            initial
            if initial is not None
            else initial_genotypes(problem, config, streams.stream("initialization"))
        )
    ]
    gateway.seconds["initialization"] += perf_counter() - initialization_start
    if len(population) != config.population:
        raise ValueError("Frozen initialization size does not match population")
    trace: list[dict] = []
    processed = 0
    reason = "generation_limit"
    rng = streams.stream("variation")
    controller = Controller(config)
    adaptive = (
        AdaptiveBudget()
        if config.budget_policy in ("coverage_v2", "severity_v2", "sequential")
        else None
    )
    lambdas = weights(config.population)
    stagnant: dict[tuple, int] = {}
    for generation in range(config.generations):
        rank, crowding = rank_and_crowding(population)
        offspring: list[Candidate] = []
        while len(offspring) < config.population:
            if config.seconds is not None and perf_counter() - begin >= config.seconds:
                reason = "time_budget"
                break
            if (
                config.exact_evaluation_cap is not None
                and gateway.counts["exact"] >= config.exact_evaluation_cap
            ):
                reason = "exact_evaluation_cap"
                break
            a = tournament(rank, crowding, rng)
            b = tournament(rank, crowding, rng)
            parent_context = (
                NormalizationContext(gateway.ideal, maximum(population))
                if config.method == "nsga2_memetic"
                else None
            )
            frozen_parent_direction = (
                associate(parent_context.normalize(population[a]), lambdas)
                if parent_context is not None
                else None
            )
            child = gateway.evaluate(
                reproduce(problem, population[a].genotype, population[b].genotype, config, rng)
            )
            hit = False
            steps: list[dict] = []
            parent_direction: int | None = None
            child_direction: int | None = None
            if config.method == "nsga2_memetic" and (
                config.exact_evaluation_cap is None
                or gateway.counts["exact"] < config.exact_evaluation_cap
            ):
                assert parent_context is not None and frozen_parent_direction is not None
                context = parent_context
                parent_direction = frozen_parent_direction
                child_direction = associate(context.normalize(child), lambdas)
                gateway.counts["trigger_attempts"] += 1
                hit = triggered(
                    child,
                    population[a],
                    parent_direction,
                    lambdas,
                    context,
                    config,
                    streams.stream("trigger"),
                )
                if hit:
                    gateway.counts["trigger_hits"] += 1
                    before_search = child
                    progress = (generation * config.population + len(offspring)) / (
                        config.generations * config.population
                    )
                    if config.seconds is not None:
                        progress = min(1.0, (perf_counter() - begin) / config.seconds)
                    elif config.exact_evaluation_cap is not None:
                        progress = min(1.0, gateway.counts["exact"] / config.exact_evaluation_cap)
                    try:
                        child, steps = improve(
                            child,
                            parent_direction,
                            stagnant.get(identity(population[a]), 0),
                            progress,
                            lambdas,
                            maximum(population),
                            gateway,
                            controller,
                            config,
                            streams,
                            population,
                            adaptive,
                        )
                    except EvaluationCapReached:
                        reason = "exact_evaluation_cap"
                    final_context = NormalizationContext(gateway.ideal, maximum(population))
                    if less(
                        final_context.scalar(child, lambdas[parent_direction]),
                        final_context.scalar(before_search, lambdas[parent_direction]),
                        TOL.scalar,
                    ):
                        gateway.counts["trigger_successes"] += 1
                        stagnant[identity(child)] = 0
                    else:
                        stagnant[identity(child)] = stagnant.get(identity(population[a]), 0) + 1
            offspring.append(child)
            processed += 1
            row = {
                "generation": generation,
                "subproblem": len(offspring) - 1,
                "objectives": child.evaluation.objectives,
                "replaced": 0,
                "elapsed": perf_counter() - begin,
                "exact_count": gateway.counts["exact"],
                "trigger": hit,
                "steps": steps,
                "parent_direction": parent_direction,
                "offspring_direction": child_direction,
            }
            if retain_trace:
                row["archive_objectives"] = [
                    c.evaluation.objectives for c in gateway.archive.members
                ]
                trace.append(row)
            if observer is not None:
                observer("offspring", {"row": row})
        if offspring:
            population = environmental_selection(population + offspring, config.population)
            if config.method == "nsga2_memetic":
                stagnant = {identity(c): stagnant.get(identity(c), 0) for c in population}
        if reason in ("time_budget", "exact_evaluation_cap"):
            break
    return RunResult(
        population,
        gateway.archive,
        gateway,
        trace,
        perf_counter() - begin,
        controller,
        reason,
        adaptive,
        processed,
    )
