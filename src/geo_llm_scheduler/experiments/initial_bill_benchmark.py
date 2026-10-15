"""Independent P100 initialization recipes with production and audit clocks."""

from __future__ import annotations

import random
from time import perf_counter, process_time
from typing import Callable

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.initial_bill_mixture import retain_original, seed_bank
from geo_llm_scheduler.initialization.generators import generate, initial_genotypes
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.rng import RNGManager

METHODS = ("BASE", "BILL10", "BILL20", "DEMAND20")
ORDERS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))


def traced_original(
    problem: ProblemInstance,
    config: Config,
    *,
    evaluator: Callable[..., Candidate] | None = None,
) -> list[Candidate]:
    """Reproduce the unchanged initializer, retaining exact construction labels.

    This is a production path: no repeated evaluation or generation is hidden
    inside its clock. Reference replay is performed separately by the caller.
    """
    if config.population != 100:
        raise ValueError("Benchmark requires P100")
    rng = RNGManager(config.seed).stream("initialization")
    quotas = [10] * 10
    accepted: dict[Genotype, int] = {}
    genes: list[Genotype] = []
    for attempt in range(config.initialization_attempts):
        pending = [i for i, q in enumerate(quotas) if q > 0]
        if not pending:
            break
        mode = pending[attempt % len(pending)] if attempt < 2000 else 7
        g = generate(problem, mode, rng, config.initialization_perturbation)
        if g in accepted:
            continue
        accepted[g] = mode
        genes.append(g)
        quotas[mode if quotas[mode] else pending[0]] -= 1
        if len(genes) == 100:
            rng.shuffle(genes)
            break
    if len(genes) != 100:
        raise ValueError("Original unique population unavailable")
    result = []
    for g in genes:
        if evaluator is not None:
            candidate = evaluator(g, origin=f"initial_mode_{accepted[g]}")
            if not candidate.evaluation.feasible:
                raise AssertionError("Original initializer infeasible")
            result.append(candidate)
            continue
        schedule = decode(problem, g)
        ev = evaluate(problem, g, schedule)
        if not ev.feasible:
            raise AssertionError("Original initializer infeasible")
        result.append(Candidate(g, schedule, ev, f"initial_mode_{accepted[g]}"))
    return result


def benchmark_population(
    problem: ProblemInstance,
    config: Config,
    method: str,
    bank_seed: int,
    progress: Callable[[dict], None] | None = None,
    *,
    accelerated: bool = False,
    trials: int = 4,
) -> tuple[list[Candidate], list[Candidate], dict]:
    """Build one arm from scratch and independently verify all its final members.

    Each invocation regenerates its original P100. BILL10 builds only ten seeds;
    BILL20 builds twenty with the identical RNG prefix. There are no shared banks
    or shared global duplicate sets across separately timed invocations.
    """
    if method not in METHODS:
        raise ValueError("Unknown benchmark method")
    started, cpu_started = perf_counter(), process_time()
    parents = traced_original(problem, config)
    parent_wall, parent_cpu = perf_counter() - started, process_time() - cpu_started
    stats: dict = {
        "exact": 0,
        "verification_seconds": 0.0,
        "verification_cpu_seconds": 0.0,
        "observations": [],
    }
    bank_wall = bank_cpu = 0.0
    if method == "BASE":
        new = []
        share = 0
    else:
        family, share = method[:-2], int(method[-2:])
        bank, stats = seed_bank(
            problem,
            parents,
            random.Random(bank_seed),
            trials,
            progress,
            families=(family,),
            size=share,
            accelerated=accelerated,
        )
        bank_wall, bank_cpu = stats["production_seconds"], stats["production_cpu_seconds"]
        new = bank[family]
    wall, cpu = perf_counter(), process_time()
    members = parents if method == "BASE" else retain_original(parents, share) + new
    if len(members) != 100 or len({c.genotype for c in members}) != 100:
        raise AssertionError("Final population size/uniqueness mismatch")
    archive = Archive()
    for candidate in members:
        archive.consider(candidate)
    selection_wall, selection_cpu = perf_counter() - wall, process_time() - cpu
    production_wall = parent_wall + bank_wall + selection_wall
    production_cpu = parent_cpu + bank_cpu + selection_cpu
    wall, cpu = perf_counter(), process_time()
    reference = initial_genotypes(problem, config, RNGManager(config.seed).stream("initialization"))
    if reference != [p.genotype for p in parents]:
        raise AssertionError("Original generation reference mismatch")
    for candidate in members:
        replay = decode(problem, candidate.genotype)
        if (
            replay != candidate.schedule
            or evaluate(problem, candidate.genotype, replay) != candidate.evaluation
        ):
            raise AssertionError("Final independent SSGS/exact mismatch")
    audit_wall = perf_counter() - wall + stats["verification_seconds"]
    audit_cpu = process_time() - cpu + stats["verification_cpu_seconds"]
    return (
        members,
        archive.members,
        {
            "method": method,
            "production_wall_seconds": production_wall,
            "production_cpu_seconds": production_cpu,
            "parent_wall_seconds": parent_wall,
            "parent_cpu_seconds": parent_cpu,
            "bank_wall_seconds": bank_wall,
            "bank_cpu_seconds": bank_cpu,
            "selection_wall_seconds": selection_wall,
            "audit_wall_seconds": audit_wall,
            "audit_cpu_seconds": audit_cpu,
            "total_elapsed_seconds": perf_counter() - started,
            "parent_exact": 100,
            "new_candidate_exact": stats["exact"],
            "candidate_independent_exact_verified": stats["exact"],
            "final_independent_exact_verified": 100,
            "parents": [(p.genotype.ms, p.genotype.os) for p in parents],
            "bank_stats": stats,
        },
    )
