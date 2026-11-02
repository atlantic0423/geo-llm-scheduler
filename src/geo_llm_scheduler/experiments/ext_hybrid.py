"""External baselines with a weakened hybrid initializer and per-generation old Polish.

Research-only study arms. Each external method keeps its component33 search
settings (``component_random.arm_config`` of the matching ``EXT_`` arm); only two
things change, identically for all five methods:

* a charged hybrid initializer that is deliberately weaker than BILL4
  (90 original heuristic parents + 10 BILL seeds with four tournament trials):
  20 original heuristic parents (two per source mode), one single-trial BILL
  seed (Flow weight 0, i.e. the first BILL4 bank slot built with one trial) and
  79 uniformly random legal genotypes. The level was fixed by comparing initial
  populations only: on all 32 component33 inputs its initial front is behind
  BILL4 in HV and at least 95% weakly dominated by BILL4 (median per size);
* after every complete generation, the unchanged old right-shift Polish is tried
  on a uniformly sampled 20% of the current population. An accepted Polish keeps
  Flow, lowers Bill and replaces that member; every Polish candidate reaches the
  measurement Archive through the gateway.

Reference arms ``FULL_CURRENT_I4`` and ``BILL_MOEAD`` replay component33 unchanged.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from time import perf_counter
from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import RunResult, run
from geo_llm_scheduler.experiments import component_random
from geo_llm_scheduler.experiments.classic_baselines import run_classic
from geo_llm_scheduler.experiments.initial_bill_benchmark import traced_original
from geo_llm_scheduler.experiments.initial_bill_mixture import guided_genotype
from geo_llm_scheduler.experiments.nsga2 import run_nsga2
from geo_llm_scheduler.experiments.runner import digest, solution
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.operators.right_shift import PolishStats, polish
from geo_llm_scheduler.utils.rng import RNGManager

ARMS = ("EXTH_MOEAD", "EXTH_NSGAII", "EXTH_RK_MOPSO", "EXTH_SPEA2", "EXTH_EPS_IBEA")
REFERENCE_ARMS = ("FULL_CURRENT_I4", "BILL_MOEAD")
STUDY_ARMS = (*ARMS, *REFERENCE_ARMS)
HEURISTIC_PER_MODE = 2
POLISH_SHARE = 0.2
POLICY = "hybrid_h20_b1t1_r79"


def base_arm(arm: str) -> str:
    """Map a hybrid arm to its component33 external arm; reference arms map to themselves."""
    if arm in ARMS:
        return "EXT_" + arm.removeprefix("EXTH_")
    if arm in REFERENCE_ARMS:
        return arm
    raise ValueError("Unknown hybrid-baseline study arm")


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Reuse the unchanged component33 configuration of the mapped arm."""
    return component_random.arm_config(base_arm(arm), seed, seconds)


def retain_balanced(parents: list[Candidate], per_mode: int) -> list[Candidate]:
    """Keep the first ``per_mode`` parents of each of the ten original source modes.

    Selection follows construction order only; no objective value ranks parents.
    """
    if not 1 <= per_mode <= 10 or len(parents) != 100:
        raise ValueError("Require P100 original parents and 1..10 per mode")
    counts = [0] * 10
    kept = []
    for candidate in parents:
        mode = int(candidate.origin.removeprefix("initial_mode_"))
        if counts[mode] < per_mode:
            kept.append(candidate)
            counts[mode] += 1
    if counts != [per_mode] * 10:
        raise ValueError("Original source modes are not balanced")
    return kept


def random_genotypes(
    problem: ProblemInstance,
    count: int,
    rng: random.Random,
    seen: set[Genotype],
    attempts: int,
) -> tuple[list[Genotype], int]:
    """Draw unique uniform legal P/D paths with shuffled repeated-job OS.

    Same sampling as the component33 pure-random initializer; genotypes already in
    ``seen`` are rejected. Returns the genotypes and the number of draws used.
    """
    legal = [paths(problem, job) for job in range(len(problem.jobs))]
    if any(not choices for choices in legal):
        raise ValueError("Random initializer requires a legal P/D path for every job")
    genes: list[Genotype] = []
    used = 0
    while len(genes) < count and used < attempts:
        used += 1
        assignment = tuple(machine for choices in legal for machine in rng.choice(choices))
        order = [job for job in range(len(problem.jobs)) for _ in (0, 1)]
        rng.shuffle(order)
        genotype = Genotype(assignment, tuple(order))
        if genotype not in seen:
            seen.add(genotype)
            genes.append(genotype)
    if len(genes) != count:
        raise ValueError("Random genotype space/attempt limit cannot supply unique members")
    return genes, used


def bill_seed(
    problem: ProblemInstance, rng: random.Random, seen: set[Genotype], attempts: int = 100
) -> Genotype:
    """Return the first unseen single-trial BILL seed at Flow weight 0.

    Identical to slot 0 of the BILL4 seed bank built with one trial instead of four.
    """
    for _ in range(attempts):
        g, _ = guided_genotype(problem, "BILL", rng, 0.0, accelerated=True)
        if g not in seen:
            seen.add(g)
            return g
    raise ValueError("Unique BILL seed unavailable within frozen attempt limit")


def hybrid_initializer(
    problem: ProblemInstance, config: Config, evidence: dict
) -> Callable[[EvaluationGateway, RNGManager], list[Candidate]]:
    """Build the charged 20 heuristic + 1 BILL(1 trial) + 79 random population.

    The original heuristic parents are constructed and evaluated inside the clock,
    but only the 20 retained parents pass through the gateway, so discarded
    parents never reach the measurement Archive.
    """
    if config.population != 100:
        raise ValueError("Hybrid initializer requires P100")

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        started = perf_counter()
        parents = traced_original(problem, config)
        heuristic = [
            gateway.evaluate(c.genotype, origin=c.origin)
            for c in retain_balanced(parents, HEURISTIC_PER_MODE)
        ]
        seen = {c.genotype for c in parents}
        seed = gateway.evaluate(
            bill_seed(problem, random.Random(config.seed + 100000), seen),
            origin="bill_seed_BILL",
        )
        genes, draws = random_genotypes(
            problem,
            config.population - len(heuristic) - 1,
            streams.stream("hybrid_random_initialization"),
            seen,
            config.initialization_attempts,
        )
        randoms = [gateway.evaluate(g, origin="hybrid_random_initialization") for g in genes]
        members = heuristic + [seed] + randoms
        if len({c.genotype for c in members}) != config.population:
            raise AssertionError("Hybrid initializer produced duplicate genotypes")
        if not all(c.evaluation.feasible for c in members):
            raise AssertionError("Hybrid initializer produced an infeasible exact candidate")
        evidence.update(
            policy=POLICY,
            heuristic=len(heuristic),
            bill=1,
            bill_trials=1,
            random=len(randoms),
            random_draw_attempts=draws,
            initial_population_sha256=digest([solution(c) for c in members]),
            initial_archive_sha256=digest([solution(c) for c in gateway.archive.members]),
            initialization_seconds=perf_counter() - started,
            config=asdict(config),
        )
        return members

    return initialize


def polish_generation(
    problem: ProblemInstance,
    population: list[Candidate],
    gateway: EvaluationGateway,
    rng: random.Random,
    expired: Callable[[], bool],
    share: float = POLISH_SHARE,
) -> list[int]:
    """Try the old Polish on ``ceil(share * P)`` uniformly sampled members, in place.

    Members are sampled without replacement and processed in index order; the pass
    stops before the next call once ``expired()`` is true. Returns replaced indices.
    """
    if not 0 < share <= 1:
        raise ValueError("Polish share must be in (0, 1]")
    chosen = sorted(rng.sample(range(len(population)), math.ceil(share * len(population))))
    changed = []
    for i in chosen:
        if expired():
            break
        started = perf_counter()
        stats = PolishStats()
        gateway.counts["polish_calls"] += 1
        polished = polish(problem, population[i], gateway, stats)
        for field in ("candidates", "exact", "accepted"):
            gateway.counts["polish_" + field] += getattr(stats, field)
        gateway.seconds["polish"] += perf_counter() - started
        if polished is not population[i]:
            population[i] = polished
            changed.append(i)
    return changed


def run_arm(
    problem: ProblemInstance,
    config: Config,
    arm: str,
    evidence: dict,
    *,
    observer: Callable[[str, dict], None] | None = None,
) -> RunResult:
    """Run a hybrid external arm, or replay a component33 reference arm unchanged."""
    if arm in REFERENCE_ARMS:
        return component_random.run_arm(problem, config, arm, evidence, observer=observer)
    method = base_arm(arm)
    factory = hybrid_initializer(problem, config, evidence)
    begin = perf_counter()

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        members = factory(gateway, streams)
        if observer is not None:
            observer("initial_ready", {"population": members, "gateway": gateway})
        return members

    def expired() -> bool:
        return config.seconds is not None and perf_counter() - begin >= config.seconds

    def transform(
        generation: int,
        population: list[Candidate],
        gateway: EvaluationGateway,
        streams: RNGManager,
        *_: object,
    ) -> list[int]:
        return polish_generation(
            problem, population, gateway, streams.stream("ext_polish"), expired
        )

    def moead_transform(
        generation: int,
        population: list[Candidate],
        gateway: EvaluationGateway,
        streams: RNGManager,
        stagnation: list[int],
    ) -> None:
        transform(generation, population, gateway, streams)

    if method in ("EXT_RK_MOPSO", "EXT_SPEA2", "EXT_EPS_IBEA"):
        return run_classic(
            problem,
            config,
            method.removeprefix("EXT_"),
            initialize,
            observer=observer,
            generation_transform=transform,
        )
    if method == "EXT_NSGAII":
        return run_nsga2(
            problem,
            config,
            initializer=initialize,
            observer=observer,
            retain_trace=False,
            generation_transform=transform,
        )
    return run(
        problem,
        config,
        initializer=initialize,
        observer=observer,
        retain_trace=False,
        generation_transform=moead_transform,
    )
