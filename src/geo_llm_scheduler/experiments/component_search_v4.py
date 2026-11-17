"""Local v4 component configuration entry; the frozen RL23 arm guard is untouched."""

from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import RunResult, run
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.right_shift_policies import PolishPostprocess
from geo_llm_scheduler.experiments.rl23_support import StudyPolicy
from geo_llm_scheduler.experiments.s4_moead import arm_config, guided_options, make_search
from geo_llm_scheduler.lsv2.guided_variation import GuidedVariation
from geo_llm_scheduler.lsv2.moead_search import MoeadMixin
from geo_llm_scheduler.lsv2.search import LSV2Search
from geo_llm_scheduler.moead.component_selection import (
    CROSS_REGION,
    PARENTS,
    REPLACEMENT,
    TERMINAL,
    TerminalSelector,
    mutate_cross_region,
    order_replacements,
    select_parents,
    select_pool,
)
from geo_llm_scheduler.moead.component_variation import (
    CROSSOVERS,
    ORDER_MUTATIONS,
    WITHIN_MUTATIONS,
    cross,
    mutate_order,
    mutate_within,
)
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, weights
from geo_llm_scheduler.moead.variation import reproduce
from geo_llm_scheduler.utils.rng import RNGManager

BASE = "GS4_GUIDED_RAND_SWITCH_D4"
SLOTS = {
    "initialization": (
        "current_bill4",
        "historical_mixed",
        "legal_random",
        "quality_diversity_pool",
    ),
    "parents": PARENTS,
    "crossover": CROSSOVERS,
    "order_mutation": ORDER_MUTATIONS,
    "within_region_mutation": WITHIN_MUTATIONS,
    "cross_region_mutation": CROSS_REGION,
    "replacement_order": REPLACEMENT,
    "terminal_target_order": TERMINAL,
}
NUMERIC = {
    "mutation_weights": ((0.4, 0.4, 0.2), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)),
    "crossover_probability": (0.7, 0.9, 1.0),
    "mutation_probability": (0.1, 0.2, 0.4),
    "neighborhood": (10, 20, 30),
    "neighbor_probability": (0.7, 0.9, 1.0),
    "replacement_cap": (1, 2, 3),
    "ls_width": (1, 3, 6),
    "ls_depth": (1, 3, 5),
}


@dataclass(frozen=True)
class Components:
    """Strictly declared 8 slots/30 candidates plus 8 finite numeric dimensions."""

    initialization: str = "current_bill4"
    parents: str = "random_pair"
    crossover: str = "job_linked"
    order_mutation: str = "swap"
    within_region_mutation: str = "single_phase"
    cross_region_mutation: str = "random_legal_path"
    replacement_order: str = "random"
    terminal_target_order: str = "flow_ascending"
    mutation_weights: tuple[float, ...] = (0.4, 0.4, 0.2)
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2
    neighborhood: int = 20
    neighbor_probability: float = 0.9
    replacement_cap: int = 2
    ls_width: int = 3
    ls_depth: int = 3

    def __post_init__(self) -> None:
        for name, choices in {**SLOTS, **NUMERIC}.items():
            value = getattr(self, name)
            if isinstance(value, bool) or value not in choices:
                raise ValueError(f"Undeclared v4 value: {name}={value!r}")
        for name in ("neighborhood", "replacement_cap", "ls_width", "ls_depth"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"Integer required: {name}")

    def engine_config(self, seed: int, seconds: float) -> Config:
        """Keep the adopted engine fixed except for the six declared numeric fields."""
        names = tuple(name for name in NUMERIC if not name.startswith("ls_"))
        return replace(arm_config(BASE, seed, seconds), **{n: getattr(self, n) for n in names})


def initializer(
    problem: ProblemInstance,
    config: Config,
    spec: Components,
    evidence: dict,
) -> Callable[[EvaluationGateway, RNGManager], list[Candidate]]:
    """Return a charged initializer; pool selection retains all exact archive attempts."""
    arms = {
        "current_bill4": "BILL_MOEAD",
        "historical_mixed": "MIXED_MOEAD",
        "legal_random": "EXT_MOEAD",
    }
    if spec.initialization in arms:
        return charged_initializer(problem, config, arms[spec.initialization], evidence)

    def pooled(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        parts: list[dict] = [{}, {}]
        bill = charged_initializer(problem, config, "BILL_MOEAD", parts[0])(gateway, streams)
        random = charged_initializer(problem, config, "EXT_MOEAD", parts[1])(gateway, streams)
        selected = select_pool(bill + random, config.population)
        evidence.update(
            policy="quality_diversity_pool",
            pools=parts,
            pool_size=len(bill) + len(random),
            accepted_count=len(selected),
        )
        return selected

    return pooled


def offspring_factory(spec: Components) -> Callable[..., Candidate]:
    """Create a bounded ordinary offspring callback; guided routing remains external."""
    legacy = all(
        getattr(spec, name) == SLOTS[name][0]
        for name in (
            "crossover",
            "order_mutation",
            "within_region_mutation",
            "cross_region_mutation",
        )
    )

    def offspring(
        population: list[Candidate],
        birth: int,
        neighbors: tuple[int, ...],
        config: Config,
        gateway: EvaluationGateway,
        streams: RNGManager,
        *,
        observer: Callable[[str, dict], None] | None = None,
        global_pool: tuple[int, ...] | None = None,
    ) -> Candidate:
        rng = streams.stream("variation")
        pool = (
            neighbors
            if rng.random() < config.neighbor_probability
            else (tuple(range(len(population))) if global_pool is None else global_pool)
        )
        context = NormalizationContext(gateway.ideal, maximum(population))
        a, b = select_parents(
            population, birth, pool, spec.parents, context, weights(config.population)[birth], rng
        )
        left, right = population[a].genotype, population[b].genotype
        if legacy:
            target = reproduce(gateway.problem, left, right, config, rng)
        else:
            target = (
                cross(left, right, spec.crossover, rng)
                if (rng.random() < config.crossover_probability)
                else rng.choice((left, right))
            )
            if gateway.problem.jobs and rng.random() < config.mutation_probability:
                kind = rng.choices(range(3), weights=config.mutation_weights)[0]
                if kind == 0:
                    target = mutate_order(target, spec.order_mutation, rng)
                elif kind == 1:
                    target = mutate_within(
                        gateway.problem, target, spec.within_region_mutation, rng
                    )
                else:
                    target = mutate_cross_region(
                        gateway.problem, target, spec.cross_region_mutation, rng
                    )
        if observer is not None:
            observer(
                "structure",
                {"source": population[a], "target": target, "path": "variation", "parents": (a, b)},
            )
        return gateway.evaluate(target)

    return offspring


def run_components(
    problem: ProblemInstance,
    spec: Components,
    seed: int,
    seconds: float,
    calibration: dict,
    evidence: dict,
    *,
    observer: Callable[[str, dict], None] | None = None,
    search_sink: Callable[[LSV2Search], None] | None = None,
    generations: int | None = None,
    guided_timer: Callable[[], float] | None = None,
) -> RunResult:
    """Run opt-in v4 locally under the inclusive clock; generations optionally caps smoke tests.

    This is not an irace search campaign or a new adopted default. No network,
    deployment or automatic tuning is performed by this entry point.
    """
    config = spec.engine_config(seed, seconds)
    if generations is not None:
        config = replace(config, generations=generations)
    search = make_search(BASE, config, calibration)
    assert isinstance(search, MoeadMixin)
    search.config = replace(search.config, budget=spec.ls_width, trajectory_steps=spec.ls_depth)
    search.terminal_proposal_budget = 3
    search.terminal_selector = TerminalSelector(spec.terminal_target_order)
    if search_sink is not None:
        search_sink(search)
    options = guided_options(BASE)
    assert options is not None
    guided = GuidedVariation(
        config,
        options,
        gain_probe=search.last_gain,
        terminal_probe=search.in_terminal,
        timer=guided_timer or perf_counter,
    )
    search.guided = guided  # type: ignore[attr-defined]
    study = StudyPolicy("BILL_PERMUTED", 1.0)
    factory = initializer(problem, config, spec, evidence)

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        members = factory(gateway, streams)
        study.initial_context = NormalizationContext(gateway.ideal, maximum(members))
        if observer is not None:
            observer("initial_ready", {"population": members, "gateway": gateway})
        return members

    def observe(event: str, data: dict) -> None:
        if event == "initialization":
            study.learner(data["controller"], config)
        if observer is not None:
            observer(event, data)

    def sort_replacements(
        pop: list[Candidate],
        child: Candidate,
        neighbors: tuple[int, ...],
        lambdas: tuple[tuple[float, float], ...],
        context: NormalizationContext,
    ) -> tuple[int, ...]:
        return order_replacements(pop, child, neighbors, lambdas, context, spec.replacement_order)

    result = run(
        problem,
        config,
        initializer=initialize,
        observer=observe,
        retain_trace=False,
        trajectory_override=study,
        offspring_transform=PolishPostprocess("BILL_MOEAD"),
        interleave=search,
        guided_offspring=guided,
        ordinary_path=guided.ordinary,
        offspring_factory=offspring_factory(spec),
        replacement_sorter=None if spec.replacement_order == "random" else sort_replacements,
    )
    evidence.update(
        components=asdict(spec),
        engine=asdict(config),
        terminal_budget=3,
        terminal_share=search.terminal_share,
        status="local_execution",
    )
    return result
