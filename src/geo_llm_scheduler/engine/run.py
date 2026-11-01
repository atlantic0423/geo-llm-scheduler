"""Run orchestration for the exact-evaluated plain MOEA/D reference baseline."""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Callable

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationCapReached, EvaluationGateway
from geo_llm_scheduler.engine.offspring import create_offspring
from geo_llm_scheduler.engine.trajectory import improve
from geo_llm_scheduler.engine.trigger import triggered
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.validation import validate_problem
from geo_llm_scheduler.macrosearch.adaptive import AdaptiveBudget
from geo_llm_scheduler.moead.core import (
    NormalizationContext,
    maximum,
    neighborhoods,
    replace_neighbors,
    weights,
)
from geo_llm_scheduler.moead.replacement import replacement_neighborhood
from geo_llm_scheduler.operators.base import Operator
from geo_llm_scheduler.rl.controller import Controller
from geo_llm_scheduler.utils.numeric import TOL, less
from geo_llm_scheduler.utils.rng import RNGManager


@dataclass
class RunResult:
    """Final population, archive and reproducibility trace for a single run."""

    population: list[Candidate]
    archive: Archive
    gateway: EvaluationGateway
    trace: list[dict]
    elapsed: float
    controller: Controller
    termination_reason: str
    adaptive: AdaptiveBudget | None = None
    offspring_count: int = 0
    observer_seconds: float = 0.0


def run(
    problem: ProblemInstance,
    config: Config,
    initial: list[Genotype] | None = None,
    *,
    retain_trace: bool = True,
    observer: Callable[[str, dict], None] | None = None,
    operator_overrides: dict[int, Operator] | None = None,
    initializer: Callable[[EvaluationGateway, RNGManager], list[Candidate]] | None = None,
    trajectory_override: Callable[..., tuple[Candidate, list[dict]]] | None = None,
    offspring_transform: Callable[[Candidate, bool, EvaluationGateway, RNGManager], Candidate]
    | None = None,
    safe_boundary_deadline: bool = False,
    budget_excluded_seconds: Callable[[], float] | None = None,
    generation_transform: Callable[
        [int, list[Candidate], EvaluationGateway, RNGManager, list[int]], None
    ]
    | None = None,
    interleave: Callable[
        [list[Candidate], EvaluationGateway, RNGManager, Callable[[], float]], None
    ]
    | None = None,
) -> RunResult:
    """Execute generations; optional synchronous observers never own algorithm RNGs.

    ``retain_trace=False`` omits historical snapshots without trimming the archive.
    Observer work is included in the engine wall-clock budget.
    A custom initializer runs inside that same clock and owns gateway dispatch;
    it is mutually exclusive with precomputed genotype input.
    ``trajectory_override`` is an explicit research seam for Polish-only studies;
    omitted callbacks preserve the original trajectory and RNG behavior.
    ``offspring_transform`` is an optional charged post-trajectory research seam
    for both selected and unselected offspring; it runs before replacement.
    ``safe_boundary_deadline`` rechecks after a potentially blocking observer;
    only explicit online-policy callers opt in, preserving historical controls.
    ``budget_excluded_seconds`` optionally reports cumulative synchronous API
    blocking time. Only deadline/progress calculations subtract it; result and
    observer elapsed values remain actual wall time. Initialization is charged.
    ``generation_transform`` is an opt-in charged round-boundary research seam.
    It runs only after a complete traversal and may refine the current population.
    Omitting it preserves the original child trigger, replacement and RNG path.
    ``interleave`` is an opt-in charged research seam called before every offspring
    slot with the population, gateway, streams and the budget clock. It may run
    its own search on separately named streams and replace population members;
    the deadline is rechecked after it returns. Omitting it preserves every path.
    """
    if initial is not None and initializer is not None:
        raise ValueError("Choose initial genotypes or a charged initializer")
    validate_problem(problem)
    begin = perf_counter()

    def budget_elapsed() -> float:
        wall = perf_counter() - begin
        excluded = 0.0 if budget_excluded_seconds is None else budget_excluded_seconds()
        if not isfinite(excluded) or excluded < 0 or excluded > wall:
            raise ValueError("Invalid excluded API time")
        return wall - excluded

    observer_seconds = 0.0
    streams = RNGManager(config.seed)
    gateway = EvaluationGateway(
        problem, Archive(), config.exact_evaluation_cap, config.decode_cache_size
    )
    if observer is not None:

        def observe_candidate(candidate: Candidate) -> None:
            nonlocal observer_seconds
            tick = perf_counter()
            assert observer is not None
            observer(
                "candidate",
                {"candidate": candidate, "gateway": gateway, "elapsed": perf_counter() - begin},
            )
            observer_seconds += perf_counter() - tick

        gateway.candidate_sink = observe_candidate
    adaptive = (
        AdaptiveBudget()
        if config.budget_policy in ("coverage_v2", "severity_v2", "sequential")
        else None
    )
    initialization_start = perf_counter()
    population = (
        initializer(gateway, streams)
        if initializer is not None
        else [
            gateway.evaluate(g, origin="initialization")
            for g in (
                initial
                if initial is not None
                else initial_genotypes(problem, config, streams.stream("initialization"))
            )
        ]
    )
    if len(population) != config.population:
        raise ValueError("Frozen initialization size does not match population")
    gateway.seconds["initialization"] += perf_counter() - initialization_start
    lambdas = weights(config.population)
    neighbors = neighborhoods(lambdas, config.neighborhood)
    trace: list[dict] = []
    controller = Controller(config)
    stagnation = [0] * config.population
    processed = 0

    def observe(event: str, extra: dict) -> None:
        nonlocal observer_seconds
        if observer is not None:
            tick = perf_counter()
            observer(
                event,
                {
                    "population": population,
                    "gateway": gateway,
                    "controller": controller,
                    "streams": streams,
                    "stagnation": stagnation,
                    "elapsed": perf_counter() - begin,
                    **extra,
                },
            )
            observer_seconds += perf_counter() - tick

    observe("initialization", {})

    def finish(reason: str) -> RunResult:
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
            observer_seconds,
        )

    for generation in range(config.generations):
        order = list(range(config.population))
        streams.stream("traversal").shuffle(order)
        for i in order:
            if config.seconds is not None and budget_elapsed() >= config.seconds:
                return finish("time_budget")
            if (
                config.exact_evaluation_cap is not None
                and gateway.counts["exact"] >= config.exact_evaluation_cap
            ):
                return finish("exact_evaluation_cap")
            if interleave is not None:
                try:
                    interleave(population, gateway, streams, budget_elapsed)
                except EvaluationCapReached:
                    return finish("exact_evaluation_cap")
                if config.seconds is not None and budget_elapsed() >= config.seconds:
                    return finish("time_budget")
            streams.stream("variation")
            observe("before_offspring", {"generation": generation, "subproblem": i})
            if (
                safe_boundary_deadline
                and config.seconds is not None
                and budget_elapsed() >= config.seconds
            ):
                return finish("time_budget")

            def observe_structure(event: str, values: dict) -> None:
                nonlocal observer_seconds
                tick = perf_counter()
                payload = {
                    "generation": generation,
                    "subproblem": i,
                    "weight": lambdas[i],
                    "context": values.get("context")
                    or NormalizationContext(gateway.ideal, maximum(population)),
                    **values,
                }
                observer_seconds += perf_counter() - tick
                observe(event, payload)

            child = create_offspring(
                population,
                i,
                neighbors[i],
                config,
                gateway,
                streams,
                observer=observe_structure if observer is not None else None,
            )
            offspring_route = child.origin
            context = NormalizationContext(gateway.ideal, maximum(population))
            previous = population[i]
            steps: list[dict] = []
            hit = False
            before_search = child
            if config.method == "full":
                gateway.counts["trigger_attempts"] += 1
                hit = triggered(
                    child, previous, i, lambdas, context, config, streams.stream("trigger")
                )
                if hit:
                    gateway.counts["trigger_hits"] += 1
                    progress = (generation * config.population + processed % config.population) / (
                        config.generations * config.population
                    )
                    if config.seconds is not None:
                        progress = min(1.0, budget_elapsed() / config.seconds)
                    elif config.exact_evaluation_cap is not None:
                        progress = min(1.0, gateway.counts["exact"] / config.exact_evaluation_cap)
                    try:
                        child, steps = (trajectory_override or improve)(
                            child,
                            i,
                            stagnation[i],
                            progress,
                            lambdas,
                            maximum(population),
                            gateway,
                            controller,
                            config,
                            streams,
                            population,
                            adaptive,
                            observer=observe_structure if observer is not None else None,
                            operator_overrides=operator_overrides,
                        )
                    except EvaluationCapReached:
                        return finish("exact_evaluation_cap")
            if offspring_transform is not None:
                try:
                    child = offspring_transform(child, hit, gateway, streams)
                except EvaluationCapReached:
                    return finish("exact_evaluation_cap")
            if hit:
                final_context = NormalizationContext(gateway.ideal, maximum(population))
                if less(
                    final_context.scalar(child, lambdas[i]),
                    final_context.scalar(before_search, lambdas[i]),
                    TOL.scalar,
                ):
                    gateway.counts["trigger_successes"] += 1
            context = NormalizationContext(gateway.ideal, maximum(population))
            replacement_order = replacement_neighborhood(
                config.replacement_policy, i, child, neighbors, lambdas, context, streams
            )
            observe(
                "before_replacement",
                {
                    "candidate": child,
                    "context": context,
                    "neighbors": replacement_order,
                    "generation": generation,
                    "subproblem": i,
                },
            )
            replaced = replace_neighbors(
                population, child, replacement_order, lambdas, context, config.replacement_cap
            )
            improved = less(
                context.scalar(population[i], lambdas[i]),
                context.scalar(previous, lambdas[i]),
                TOL.scalar,
            )
            stagnation[i] = 0 if improved else stagnation[i] + 1
            processed += 1
            row = {
                "generation": generation,
                "subproblem": i,
                "objectives": child.evaluation.objectives,
                "replaced": replaced,
                "elapsed": perf_counter() - begin,
                "exact_count": gateway.counts["exact"],
                "trigger": hit,
                "steps": steps,
            }
            if config.offspring_policy != "variation":
                row["offspring_route"] = offspring_route
            if config.replacement_policy != "birth":
                row["replacement_neighbors"] = replacement_order
            if retain_trace:
                row["archive_objectives"] = [
                    c.evaluation.objectives for c in gateway.archive.members
                ]
                trace.append(row)
            observe("offspring", {"row": row, "processed": processed})
        if generation_transform is not None:
            if config.seconds is not None and budget_elapsed() >= config.seconds:
                return finish("time_budget")
            try:
                generation_transform(generation, population, gateway, streams, stagnation)
            except EvaluationCapReached:
                return finish("exact_evaluation_cap")
    return finish("generation_limit")
