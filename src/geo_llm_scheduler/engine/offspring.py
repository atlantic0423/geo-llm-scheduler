"""Opt-in CG/CT controls around the unchanged MOEA/D variation pathway."""

from dataclasses import replace
from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.moead.variation import reproduce
from geo_llm_scheduler.utils.rng import RNGManager


def create_offspring(
    population: list[Candidate],
    subproblem: int,
    neighbors: tuple[int, ...],
    config: Config,
    gateway: EvaluationGateway,
    streams: RNGManager,
    *,
    observer: Callable[[str, dict], None] | None = None,
) -> Candidate:
    """Evaluate one child; CT retains the whole incumbent schedule, CG rebuilds it.

    The paired controls clone the current birth subproblem's incumbent. Both use
    the same independent routing stream and still consume an exact evaluation.
    Disabled routing creates no extra stream and preserves legacy variation draws.
    """
    if config.offspring_policy != "variation":
        if streams.stream("P2:clone").random() < config.clone_probability:
            incumbent = population[subproblem]
            if config.offspring_policy == "phenotype_clone":
                return gateway.evaluate(
                    incumbent.genotype, replace(incumbent.schedule), origin="clone_phenotype"
                )
            return gateway.evaluate(incumbent.genotype, origin="clone_genotype")
    rng = streams.stream("variation")
    pool = (
        neighbors if rng.random() < config.neighbor_probability else tuple(range(len(population)))
    )
    a, b = rng.sample(pool, 2)
    target = reproduce(gateway.problem, population[a].genotype, population[b].genotype, config, rng)
    if observer is not None:
        observer(
            "structure",
            {"source": population[a], "target": target, "path": "variation", "parents": (a, b)},
        )
    return gateway.evaluate(target)
