"""Explicit initialization policies and component controls for a fresh study.

Classic baselines are discrete scheduling adaptations, not literal reproductions
of continuous benchmark papers. No parameters are selected to impair a baseline.
Historical campaigns and default initialization remain unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from heapq import heappop, heappush
from math import isfinite
from time import perf_counter
from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import RunResult, run
from geo_llm_scheduler.experiments.classic_baselines import run_classic
from geo_llm_scheduler.experiments.comparison13 import initialization_factory
from geo_llm_scheduler.experiments.initial_bill_benchmark import traced_original
from geo_llm_scheduler.experiments.nsga2 import run_nsga2
from geo_llm_scheduler.experiments.runner import digest, solution
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.operators.right_shift import PolishStats, polish
from geo_llm_scheduler.utils.rng import RNGManager

EXTERNAL_ARMS = ("EXT_MOEAD", "EXT_NSGAII", "EXT_RK_MOPSO", "EXT_SPEA2", "EXT_EPS_IBEA")
ARMS = (
    *EXTERNAL_ARMS,
    "MIXED_MOEAD",
    "BILL_MOEAD",
    *(f"ONLY_A{i}" for i in range(1, 9)),
    "POLISH_ONLY",
    "RPERM_ONLY",
    "POOL_RANDOM_B1_L1",
    "POOL_RANDOM_B6_L1",
    "POOL_Q_B6_L1",
    "POOL_Q_B6_L5",
    "POOL_Q_STRICT_B6_L5",
    "POOL_Q_SEVERITY_L5",
    "POOL_Q_COVERAGE_L5",
    "FULL_CURRENT_I4",
    "FULL_STRICT_NO_POLISH_I4",
)
BUDGETS = {20: 300, 50: 600, 100: 1200, 200: 2400}
STUDY_SEED = 16101


def initializer_policy(arm: str) -> str:
    """Separate pure-random external baselines from the two designed initializers."""
    if arm not in ARMS:
        raise ValueError("Unknown component-study arm")
    if arm in EXTERNAL_ARMS:
        return "random"
    return (
        "bill4" if arm in ("BILL_MOEAD", "FULL_CURRENT_I4", "FULL_STRICT_NO_POLISH_I4") else "mixed"
    )


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Resolve predeclared controls without weakening external variation settings.

    Existing discrete crossover/mutation probabilities .9/.2 are retained for
    all genetic methods. They are scheduling-adapter choices, not an assertion
    that continuous per-variable polynomial mutation is identical to .2.
    """
    initializer_policy(arm)
    plain = arm in (*EXTERNAL_ARMS, "MIXED_MOEAD", "BILL_MOEAD")
    cfg = Config(
        seed=seed,
        seconds=seconds,
        generations=10**9,
        method="nsga2" if arm == "EXT_NSGAII" else "plain" if plain else "full",
        polish=False,
        controller="random",
        trigger_mode="always",
        fixed_budget=1,
        rl_steps=1,
    )
    if arm.startswith("ONLY_A"):
        action = int(arm.removeprefix("ONLY_A"))
        cfg = replace(
            cfg, enabled_operators=(action,), a3_region_policy="tariff" if action == 3 else "load"
        )
    if arm == "RPERM_ONLY":
        cfg = replace(
            cfg, replacement_policy="permuted", trigger_mode="fixed", fixed_ls_probability=0.0
        )
    if arm.startswith(("POOL_", "FULL_")):
        cfg = replace(cfg, a3_region_policy="tariff")
    if arm.startswith("POOL_"):
        cfg = replace(
            cfg,
            controller="qlearning" if arm.startswith("POOL_Q") else "random",
            fixed_budget=1 if "_B1_" in arm else 6,
            rl_steps=5 if arm.endswith("_L5") else 1,
            trigger_mode="strict" if "_STRICT_" in arm else "always",
            trigger_delta=0.0 if "_STRICT_" in arm else 0.10,
        )
        if arm in ("POOL_Q_SEVERITY_L5", "POOL_Q_COVERAGE_L5"):
            cfg = replace(
                cfg,
                budget_policy="severity_v2" if "SEVERITY" in arm else "coverage_v2",
            )
    if arm.startswith("FULL_"):
        cfg = replace(
            cfg,
            controller="qlearning",
            fixed_budget=6,
            rl_steps=5,
            replacement_policy="permuted",
            trigger_mode="preference" if arm == "FULL_CURRENT_I4" else "strict",
            trigger_delta=0.10 if arm == "FULL_CURRENT_I4" else 0.0,
            polish=arm == "FULL_CURRENT_I4",
        )
    return cfg


def charged_initializer(
    problem: ProblemInstance, config: Config, arm: str, evidence: dict
) -> Callable[[EvaluationGateway, RNGManager], list[Candidate]]:
    """Build and charge random, original mixed, or BILL4 populations in the engine.

    Random assignments sample legal same-region P/D paths uniformly; shuffled
    repeated-job OS has no release/load/price ranking. Duplicate rejection uses
    genotype equality only. Feasibility or objective results never rank samples.
    """
    policy = initializer_policy(arm)
    if policy == "bill4":
        original = initialization_factory(problem, config, 4, config.seed + 100000, evidence)

        def bill(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
            members = original(gateway, streams)
            evidence["policy"] = policy
            return members

        return bill

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        started = perf_counter()
        attempts: int | None
        if policy == "mixed":
            members = traced_original(problem, config, evaluator=gateway.evaluate)
            attempts = None  # The unchanged mixed initializer does not expose draw attempts.
        else:
            rng = streams.stream("initialization")
            legal = [paths(problem, job) for job in range(len(problem.jobs))]
            if any(not choices for choices in legal):
                raise ValueError("Random initializer requires a legal P/D path for every job")
            genes: list[Genotype] = []
            seen: set[Genotype] = set()
            attempts = 0
            while len(genes) < config.population and attempts < config.initialization_attempts:
                attempts += 1
                assignment = tuple(machine for choices in legal for machine in rng.choice(choices))
                order = [job for job in range(len(problem.jobs)) for _ in (0, 1)]
                rng.shuffle(order)
                genotype = Genotype(assignment, tuple(order))
                if genotype not in seen:
                    seen.add(genotype)
                    genes.append(genotype)
            if len(genes) != config.population:
                raise ValueError(
                    "Random genotype space/attempt limit cannot supply unique population"
                )
            members = [gateway.evaluate(g, origin="pure_random_initialization") for g in genes]
        if not all(c.evaluation.feasible for c in members):
            raise AssertionError("Initializer produced an infeasible exact candidate")
        evidence.update(
            policy=policy,
            random_draw_attempts=attempts,
            accepted_count=len(members),
            initial_population_sha256=digest([solution(c) for c in members]),
            initial_archive_sha256=digest([solution(c) for c in gateway.archive.members]),
            initialization_seconds=perf_counter() - started,
            config=asdict(config),
        )
        return members

    return initialize


def polish_only(
    child: Candidate,
    index: int,
    stagnant_count: int,
    progress: float,
    weights: tuple[tuple[float, float], ...],
    maximum: tuple[float, float],
    gateway: EvaluationGateway,
    controller: object,
    config: Config,
    streams: RNGManager,
    population: list[Candidate] | None = None,
    adaptive: object | None = None,
    **kwargs: object,
) -> tuple[Candidate, list[dict]]:
    """Execute one charged Polish call and no RL action, Q update or RNG draw."""
    started = perf_counter()
    stats = PolishStats()
    gateway.counts["polish_calls"] += 1
    candidate = polish(gateway.problem, child, gateway, stats)
    for field in ("candidates", "exact", "accepted"):
        gateway.counts["polish_" + field] += getattr(stats, field)
    gateway.seconds["polish"] += perf_counter() - started
    return candidate, []


def run_arm(
    problem: ProblemInstance,
    config: Config,
    arm: str,
    evidence: dict,
    *,
    observer: Callable[[str, dict], None] | None = None,
) -> RunResult:
    """Run the declared arm with charged initialization and no historical reuse."""
    factory = charged_initializer(problem, config, arm, evidence)

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        members = factory(gateway, streams)
        if observer is not None:
            observer("initial_ready", {"population": members, "gateway": gateway})
        return members

    if arm in ("EXT_RK_MOPSO", "EXT_SPEA2", "EXT_EPS_IBEA"):
        return run_classic(problem, config, arm.removeprefix("EXT_"), initialize, observer=observer)
    if arm == "EXT_NSGAII":
        return run_nsga2(
            problem, config, initializer=initialize, observer=observer, retain_trace=False
        )
    return run(
        problem,
        config,
        initializer=initialize,
        observer=observer,
        retain_trace=False,
        trajectory_override=polish_only if arm == "POLISH_ONLY" else None,
    )


def study_specs(seed: int = STUDY_SEED, *, reserve: bool = False) -> list[dict]:
    """Reuse twenty old parents, or four separate reserve parents, by identity.

    This is a development panel, not a fresh holdout. Every four consecutive
    inputs rotate 20/50/100/200 jobs. Variants are paired within parent; they
    are not counted as independent replicates.
    """
    topologies = (
        ((0, 3, 12, "arrival"),)
        if reserve
        else (
            (20, 5, 12, "arrival"),
            (30, 3, 6, "arrival"),
            (40, 3, 24, "arrival"),
            (100, 3, 12, "kv"),
            (110, 2, 12, "kv"),
        )
    )
    return [
        dict(
            old_key=f"r0_n{n}_R{r}M{m}_{pressure}_v{variant}_s{seed}",
            n=n,
            regions=r,
            machines=m,
            pressure=pressure,
            variant=variant,
            base_seed=202610160000 + offset + index,
            seed=seed,
            seconds=BUDGETS[n],
        )
        for variant in (0, 1)
        for offset, r, m, pressure in topologies
        for index, n in enumerate(BUDGETS)
    ]


def balanced_order(index: int) -> list[str]:
    """Return an even-treatment Williams row; a partial prefix is not full balance."""
    count = len(ARMS)
    row = [0] + [j // 2 + 1 if j % 2 == 0 else count - 1 - j // 2 for j in range(count - 1)]
    return [ARMS[(j + index) % count] for j in row]


def planning_summary(workers: int = 48, overhead_seconds: float = 60.0) -> dict:
    """Calculate a deterministic slot model, not a measured runtime guarantee.

    This plans run-level dispatch on one host. Sequentially executing an entire
    26-arm 200-job block on one slot would exceed the requested overnight window.
    No experiment is dispatched and no historic result is counted as a new run.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("Workers must be a positive integer")
    if not isfinite(overhead_seconds) or overhead_seconds < 0:
        raise ValueError("Overhead must be finite and nonnegative")
    specs = study_specs()
    slots = [0.0] * workers
    for _arm_round in ARMS:
        for spec in specs:
            heappush(slots, heappop(slots) + spec["seconds"] + overhead_seconds)
    solver_seconds = sum(spec["seconds"] for spec in specs) * len(ARMS)
    return dict(
        status="planning_only_not_dispatched",
        arms=list(ARMS),
        parents=len({spec["base_seed"] for spec in specs}),
        input_versions=len(specs),
        algorithm_seeds=[STUDY_SEED],
        runs=len(specs) * len(ARMS),
        workers=workers,
        memory_cap_bytes=500_000_000_000,
        production_solver_hours=solver_seconds / 3600,
        ideal_lower_bound_hours=solver_seconds / workers / 3600,
        modeled_overhead_per_run_seconds=overhead_seconds,
        modeled_completion_hours=max(slots) / 3600,
        reserve_parents=4,
        reserve_runs=len(study_specs(reserve=True)) * len(ARMS),
        specs=specs,
        configs={arm: asdict(arm_config(arm, STUDY_SEED, 300)) for arm in ARMS},
    )
