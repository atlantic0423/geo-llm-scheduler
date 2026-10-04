"""Run orchestration for the exact-evaluated plain MOEA/D reference baseline."""

from dataclasses import dataclass
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
) -> RunResult:
    """Execute generations; optional synchronous observers never own algorithm RNGs.

    ``retain_trace=False`` omits historical snapshots without trimming the archive.
    Observer work is included in the engine wall-clock budget.
    """
    validate_problem(problem)
    begin = perf_counter()
    observer_seconds = 0.0
    streams = RNGManager(config.seed)
    gateway = EvaluationGateway(problem, Archive(), config.exact_evaluation_cap)
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
    population = [
        gateway.evaluate(g, origin="initialization")
        for g in (
            initial
            if initial is not None
            else initial_genotypes(problem, config, streams.stream("initialization"))
        )
    ]
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
            if config.seconds is not None and perf_counter() - begin >= config.seconds:
                return finish("time_budget")
            if (
                config.exact_evaluation_cap is not None
                and gateway.counts["exact"] >= config.exact_evaluation_cap
            ):
                return finish("exact_evaluation_cap")
            streams.stream("variation")
            observe("before_offspring", {"generation": generation, "subproblem": i})

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
                        progress = min(1.0, (perf_counter() - begin) / config.seconds)
                    elif config.exact_evaluation_cap is not None:
                        progress = min(1.0, gateway.counts["exact"] / config.exact_evaluation_cap)
                    before_search = child
                    try:
                        child, steps = improve(
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
                        )
                    except EvaluationCapReached:
                        return finish("exact_evaluation_cap")
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
    return finish("generation_limit")
