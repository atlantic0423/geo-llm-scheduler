"""Run orchestration for the exact-evaluated plain MOEA/D reference baseline."""

from dataclasses import dataclass
from time import perf_counter

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationCapReached, EvaluationGateway
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
from geo_llm_scheduler.moead.variation import reproduce
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


def run(
    problem: ProblemInstance, config: Config, initial: list[Genotype] | None = None
) -> RunResult:
    """Execute baseline generations, preserving every complete evaluated candidate."""
    validate_problem(problem)
    begin = perf_counter()
    streams = RNGManager(config.seed)
    gateway = EvaluationGateway(problem, Archive(), config.exact_evaluation_cap)
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
            rng = streams.stream("variation")
            pool = (
                neighbors[i]
                if rng.random() < config.neighbor_probability
                else tuple(range(len(population)))
            )
            a, b = rng.sample(pool, 2)
            child = gateway.evaluate(
                reproduce(problem, population[a].genotype, population[b].genotype, config, rng)
            )
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
                    progress = (generation * config.population + len(trace) % config.population) / (
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
            replaced = replace_neighbors(
                population, child, neighbors[i], lambdas, context, config.replacement_cap
            )
            improved = less(
                context.scalar(population[i], lambdas[i]),
                context.scalar(previous, lambdas[i]),
                TOL.scalar,
            )
            stagnation[i] = 0 if improved else stagnation[i] + 1
            trace.append(
                {
                    "generation": generation,
                    "subproblem": i,
                    "objectives": child.evaluation.objectives,
                    "replaced": replaced,
                    "elapsed": perf_counter() - begin,
                    "exact_count": gateway.counts["exact"],
                    "archive_objectives": [
                        c.evaluation.objectives for c in gateway.archive.members
                    ],
                    "trigger": hit,
                    "steps": steps,
                }
            )
    return finish("generation_limit")
