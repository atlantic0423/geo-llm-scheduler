"""Observe the unchanged initializer with method provenance and exact verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.e15_instances import generate_e15_pair
from geo_llm_scheduler.initialization.generators import generate, initial_genotypes
from geo_llm_scheduler.io.validation import validate_problem
from geo_llm_scheduler.utils.rng import RNGManager

SCENARIOS = ("mixed", "burst", "kv_delay", "compute_tight", "vram_tight", "demand_high")
METHODS = (
    "MIXED",
    "FCFS_LOAD",
    "FCFS_RANDOM",
    "SPT_LOAD",
    "SPT_RANDOM",
    "LPT_LOAD",
    "LPT_RANDOM",
    "RANDOM_LOAD",
    "RANDOM",
    "KV_AWARE",
    "ENERGY_AWARE",
)


@dataclass
class PopulationTrace:
    """Keep accepted method labels aligned after the original population shuffle."""

    genotypes: list[Genotype]
    modes: list[int]
    attempts: int
    duplicates: int
    generation_seconds: float
    replay_seconds: float
    original_replay_verified: bool


def diagnostic_problem(jobs: int, seed: int, tariff: str, scenario: str) -> ProblemInstance:
    """Apply one declared stress intervention to an H/T-paired immutable instance.

    Original durations, server power/capacity and all unrelated fields are retained.
    The final tariff segment is extended to cover a conservative serial schedule.
    """
    if scenario not in SCENARIOS or tariff not in ("H", "T"):
        raise ValueError("Unknown initialization diagnostic condition")
    problem = generate_e15_pair(jobs, seed)[tariff == "T"]
    if scenario == "burst":
        problem = replace(problem, jobs=tuple(replace(j, release=0.0) for j in problem.jobs))
    elif scenario == "kv_delay":
        problem = replace(
            problem, jobs=tuple(replace(j, kv_delay=4 * j.kv_delay) for j in problem.jobs)
        )
    elif scenario in ("compute_tight", "vram_tight"):
        rows = []
        for job in problem.jobs:
            profiles = []
            for profile in (job.prefill, job.decode):
                if scenario == "compute_tight":
                    profiles.append(replace(profile, compute=min(0.95, 1.5 * profile.compute)))
                else:
                    profiles.append(replace(profile, vram=min(15.5, 2.0 * profile.vram)))
            rows.append(replace(job, prefill=profiles[0], decode=profiles[1]))
        problem = replace(problem, jobs=tuple(rows))
    elif scenario == "demand_high":
        problem = replace(
            problem,
            regions=tuple(replace(r, demand_rate=5 * r.demand_rate) for r in problem.regions),
        )
    horizon = (
        max(j.release for j in problem.jobs)
        + sum(j.prefill.duration + j.decode.duration + j.kv_delay for j in problem.jobs)
        + 86400.0
    )
    problem = replace(
        problem,
        regions=tuple(
            replace(
                r,
                tariffs=r.tariffs[:-1]
                + (replace(r.tariffs[-1], end=max(horizon, r.tariffs[-1].end)),),
            )
            for r in problem.regions
        ),
    )
    validate_problem(problem)
    return problem


def sample_population(problem: ProblemInstance, config: Config, method: str) -> PopulationTrace:
    """Sample P unique genotypes from an explicit initialization RNG stream.

    MIXED reproduces the frozen initializer, records accepted construction modes,
    and checks its full output and RNG post-state against the original function.
    Each single-mode control uses that mode exclusively, without random fallback.
    An insufficient unique population is a recorded diagnostic limitation, not a
    reason to silently change the method or to retry a crashed experiment.
    """
    if method not in METHODS:
        raise ValueError("Unknown initialization diagnostic method")
    rng = RNGManager(config.seed).stream("initialization")
    quotas = [config.population // 10 + (i < config.population % 10) for i in range(10)]
    accepted: dict[Genotype, int] = {}
    result: list[Genotype] = []
    tick = perf_counter()
    attempt = -1
    for attempt in range(config.initialization_attempts):
        if method == "MIXED":
            pending = [i for i, q in enumerate(quotas) if q > 0]
            if not pending:
                break
            mode = pending[attempt % len(pending)] if attempt < config.population * 20 else 7
        else:
            mode = METHODS.index(method) - 1
            pending = []
        g = generate(problem, mode, rng, config.initialization_perturbation)
        if g not in accepted:
            accepted[g] = mode
            result.append(g)
            if method == "MIXED":
                target = mode if quotas[mode] > 0 else pending[0]
                quotas[target] -= 1
        if len(result) == config.population:
            rng.shuffle(result)
            break
    duration = perf_counter() - tick
    replay_seconds = 0.0
    verified = False
    if method == "MIXED" and len(result) == config.population:
        tick = perf_counter()
        replay_rng = RNGManager(config.seed).stream("initialization")
        reference = initial_genotypes(problem, config, replay_rng)
        if result != reference or rng.getstate() != replay_rng.getstate():
            raise AssertionError("Diagnostic tracing changed original initialization or RNG")
        verified = True
        replay_seconds = perf_counter() - tick
    return PopulationTrace(
        result,
        [accepted[g] for g in result],
        attempt + 1,
        attempt + 1 - len(result),
        duration,
        replay_seconds,
        verified,
    )


def evaluate_population(
    problem: ProblemInstance, trace: PopulationTrace
) -> tuple[list[Candidate], Archive, dict]:
    """Decode every initial individual and independently repeat its exact evaluation.

    No evolutionary iterations, memetic action, timing polish or extra selection
    is applied. Gateway measurements and independent verification cost are separate.
    """
    gateway = EvaluationGateway(problem, Archive())
    population = [
        gateway.evaluate(g, origin=f"initial_mode_{mode}")
        for g, mode in zip(trace.genotypes, trace.modes, strict=True)
    ]
    tick = perf_counter()
    for c in population:
        if evaluate(problem, c.genotype, c.schedule) != c.evaluation:
            raise AssertionError("Independent exact evaluation mismatch")
    independent_seconds = perf_counter() - tick
    if any(not c.evaluation.feasible for c in population):
        raise AssertionError("Initializer produced an infeasible decoded candidate")
    return (
        population,
        gateway.archive,
        {
            "counts": dict(gateway.counts),
            "seconds": dict(gateway.seconds),
            "independent_exact_seconds": independent_seconds,
            "independent_exact_verified": len(population),
            "archive_size": len(gateway.archive.members),
        },
    )
