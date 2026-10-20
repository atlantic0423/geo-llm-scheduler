"""Isolated thirteen-arm end-to-end study; no change to default algorithms."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, replace
from time import perf_counter
from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments.initial_bill_benchmark import traced_original
from geo_llm_scheduler.experiments.initial_bill_mixture import retain_original, seed_bank
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.utils.rng import RNGManager

ARMS = (
    "M11_I4",
    "M11_I2",
    "M11_I1",
    "M00_I4",
    "M10_I4",
    "M01_I4",
    "DS_I4",
    "DC_I4",
    "MOEAD",
    "NSGAII",
    "RK_MOPSO",
    "SPEA2",
    "EPS_IBEA",
)
SIZES = (20, 50, 100, 200)
TOPOLOGIES = ((3, 12), (2, 12), (5, 12), (3, 6), (3, 24))
PRESSURES = ("arrival", "kv", "cheap_capacity", "demand")
BUDGETS = {20: 300, 50: 600, 100: 1200, 200: 2400}


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Resolve the thirteen frozen mechanisms, keeping baseline Q/Polish disabled."""
    if arm not in ARMS:
        raise ValueError("Unknown comparison arm")
    internal = arm.startswith(("M", "DS_", "DC_")) and arm != "MOEAD"
    return Config(
        seed=seed,
        seconds=seconds,
        generations=1000000000,
        method="full" if internal else "nsga2" if arm == "NSGAII" else "plain",
        polish=internal,
        a3_region_policy="tariff" if internal and arm not in ("M00_I4", "M01_I4") else "load",
        replacement_policy="permuted"
        if arm.startswith(("M11_", "M01_", "DS_", "DC_"))
        else "birth",
        budget_policy="severity_v2"
        if arm == "DS_I4"
        else "coverage_v2"
        if arm == "DC_I4"
        else "fixed",
    )


def initial_trials(arm: str) -> int:
    """Return initializer strength independently of the local-search width."""
    if arm not in ARMS:
        raise ValueError("Unknown comparison arm")
    return 2 if arm == "M11_I2" else 1 if arm == "M11_I1" else 4


def balanced_order(index: int) -> list[str]:
    """Return one of 26 Williams sequences for thirteen odd-numbered treatments."""
    row = [0] + [j // 2 + 1 if j % 2 == 0 else 12 - j // 2 for j in range(12)]
    # [0,1,12,2,11,...,6,7] has each directed difference exactly once.
    result = [ARMS[(j + index % 13) % 13] for j in row]
    return result if index % 26 < 13 else result[::-1]


def initialization_factory(
    problem: ProblemInstance,
    config: Config,
    trials: int,
    bank_seed: int,
    evidence: dict,
    observer: Callable[[str, dict], None] | None = None,
) -> Callable[[EvaluationGateway, RNGManager], list[Candidate]]:
    """Construct and charge all parents/trials inside the engine's original clock.

    Every complete feasible parent and trial reaches the run measurement Archive.
    Extra reference evaluations are deferred to the final audit, not performed
    invisibly outside production. Default standalone benchmark behavior is intact.
    """

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        tick = perf_counter()
        parents = traced_original(problem, config, evaluator=gateway.evaluate)
        parent_seconds = perf_counter() - tick
        bank, stats = seed_bank(
            problem,
            parents,
            random.Random(bank_seed),
            trials,
            families=("BILL",),
            size=10,
            accelerated=True,
            evaluator=gateway.evaluate,
            verify_trials=False,
        )
        members = retain_original(parents, 10) + bank["BILL"]
        assert len(members) == len({c.genotype for c in members}) == 100
        evidence.update(
            trials=trials,
            bank_seed=bank_seed,
            parent_seconds=parent_seconds,
            bank_stats=stats,
            original_parent_count=len(parents),
            initial_population_sha256=hashlib.sha256(
                json.dumps([solution(c) for c in members], sort_keys=True).encode()
            ).hexdigest(),
            initial_archive_sha256=hashlib.sha256(
                json.dumps([solution(c) for c in gateway.archive.members], sort_keys=True).encode()
            ).hexdigest(),
            initialization_seconds=perf_counter() - tick,
            config=asdict(config),
        )
        if observer is not None:
            observer(
                "initial_ready", {"population": members, "gateway": gateway, "evidence": evidence}
            )
        return members

    return initialize


def case_problem(
    n: int, regions: int, machines: int, seed: int, pressure: str, variant: int
) -> ProblemInstance:
    """Generate a paired single-factor workload with native heterogeneous prices."""
    from geo_llm_scheduler.domain.models import Job, Profile, Region, ServingInstance, Tariff
    from geo_llm_scheduler.io.validation import validate_problem

    if (
        n < 6
        or regions not in (2, 3, 5)
        or machines < regions
        or pressure not in PRESSURES
        or variant not in (0, 1)
    ):
        raise ValueError("Invalid comparison case")
    rng = random.Random(seed)
    jobs = []
    for i in range(n):
        release = float(rng.randrange(61) * 60)
        p = Profile(
            rng.choice((60, 120, 240, 360)),
            rng.choice((0.15, 0.3, 0.45, 0.6)),
            rng.choice((2, 4, 6)),
        )
        d = Profile(
            rng.choice((180, 300, 600, 900)),
            rng.choice((0.15, 0.3, 0.45, 0.6)),
            rng.choice((2, 4, 6)),
        )
        kv = float(rng.choice((15, 30, 60)))
        jobs.append(Job(f"j{i}", release, p, d, kv))
    if pressure == "arrival" and variant:
        jobs = [replace(j, release=float(int(j.release // 1200) * 1200)) for j in jobs]
    if pressure == "kv":
        jobs = [replace(j, kv_delay=j.kv_delay * (4 if variant else 0.25)) for j in jobs]
    # Identical coverage in both versions, despite release/KV changes.
    upper = 3600 + sum(j.prefill.duration + j.decode.duration + 240 for j in jobs) + 86400
    locations = tuple(
        Region(
            f"r{r}",
            tuple(
                Tariff(
                    float(t),
                    min(float(t + 4800), float(upper)),
                    (0.35, 0.8, 0.5)[(t // 4800 + r) % 3] * (1 + 0.15 * r),
                )
                for t in range(0, int(upper), 4800)
            ),
            (20.0 if pressure == "demand" and variant else 2.0 if pressure == "demand" else 5.0)
            * (1 + 0.05 * r),
        )
        for r in range(regions)
    )
    counts = [machines // regions + (r < machines % regions) for r in range(regions)]
    if pressure == "cheap_capacity":
        # Same multiset of instance counts and same M: reverse the capacity/price association.
        extra = machines - regions
        heavy = (2 * extra + 2) // 3
        counts = [1 + heavy] + [
            1 + (extra - heavy) // (regions - 1) + (r < (extra - heavy) % (regions - 1))
            for r in range(regions - 1)
        ]
        if variant:
            counts = counts[::-1]
    servers = tuple(
        ServingInstance(f"m{r}_{k}", r, 16.0, 0.06, 0.4)
        for r, c in enumerate(counts)
        for k in range(c)
    )
    problem = ProblemInstance(tuple(jobs), locations, servers)
    validate_problem(problem)
    return problem


def campaign_specs(replicates: int = 5, *, pilot: bool = False) -> list[dict]:
    """Round-robin sizes within every successive topology/pressure/seed cycle."""
    result: list[dict] = []
    for rep in range(replicates):
        for pressure_index, pressure in enumerate(PRESSURES[:1] if pilot else PRESSURES):
            for topology_index, (regions, machines) in enumerate(
                TOPOLOGIES[:1] if pilot else TOPOLOGIES
            ):
                for variant in (0, 1):
                    for seed in (16001,) if pilot else (16101, 16201):
                        for size_index, n in enumerate(SIZES):
                            base_seed = (
                                (202610169000 if pilot else 202610160000)
                                + rep * 1000
                                + pressure_index * 100
                                + topology_index * 10
                                + size_index
                            )
                            key = f"r{rep}_n{n}_R{regions}M{machines}_{pressure}_v{variant}_s{seed}"
                            result.append(
                                dict(
                                    key=key,
                                    n=n,
                                    regions=regions,
                                    machines=machines,
                                    pressure=pressure,
                                    variant=variant,
                                    base_seed=base_seed,
                                    seed=seed,
                                    bank_seed=seed + 100000,
                                    seconds=BUDGETS[n],
                                    order=balanced_order(len(result)),
                                )
                            )
    return result
