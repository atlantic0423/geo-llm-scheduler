"""Single-factor BILL4 retention; candidate generation and Archive stay intact."""

from collections import Counter
from collections.abc import Callable
from time import perf_counter

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.runner import digest, solution
from geo_llm_scheduler.initialization.retention import retain_pool
from geo_llm_scheduler.utils.rng import RNGManager

POLICIES = ("legacy", "global_pool")


def pool_initializer(
    problem: ProblemInstance, config: Config, evidence: dict
) -> Callable[[EvaluationGateway, RNGManager], list[Candidate]]:
    """Wrap unchanged BILL4 and charge collection/retention to its engine clock.

    Preserve any existing candidate observer, including on failure. Record the
    legacy population hash separately; the standard hash describes the actual
    retained population. Existing run RNG streams receive no additional draws.
    """
    if config.population != 100:
        raise ValueError("The frozen BILL4 source pool requires population 100")
    original = charged_initializer(problem, config, "BILL_MOEAD", evidence)

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        start = perf_counter()
        pool: list[Candidate] = []
        prior = gateway.candidate_sink

        def collect(c: Candidate) -> None:
            pool.append(c)
            if prior is not None:
                prior(c)

        gateway.candidate_sink = collect
        try:
            legacy = original(gateway, streams)
        finally:
            gateway.candidate_sink = prior
        tick = perf_counter()
        members = retain_pool(pool, config.population)
        selection_seconds = perf_counter() - tick
        legacy_ids = {id(c) for c in legacy}
        evidence.update(
            initial_retention="global_pool",
            retention_protocol="pareto-maxmin-physical/v1",
            legacy_initial_population_sha256=evidence["initial_population_sha256"],
            legacy_initialization_seconds=evidence["initialization_seconds"],
            initial_population_sha256=digest([solution(c) for c in members]),
            retention=dict(
                pool_count=len(pool),
                selected_count=len(members),
                selected_origins=dict(Counter(c.origin for c in members)),
                physical_unique=len({(c.genotype.ms, c.schedule.starts) for c in members}),
                legacy_overlap=sum(id(c) in legacy_ids for c in members),
                selection_seconds=selection_seconds,
            ),
        )
        evidence["initialization_seconds"] = perf_counter() - start
        return members

    return initialize
