"""Frozen initialization-only speed/strength comparison, with native prices."""

import random

from geo_llm_scheduler.domain.models import (
    Job,
    ProblemInstance,
    Profile,
    Region,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.io.validation import validate_problem

ARMS = ("REF4", "FAST4", "FAST2", "FAST1")
SCENARIOS = ("mixed", "burst", "vram_tight", "demand_high")
SEEDS = (15101, 15201)
ORDERS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))


def speed_problem(n: int, seed: int, machines: int, scenario: str) -> ProblemInstance:
    """Generate a reproducible three-region case with periodic heterogeneous TOU.

    The fixed scenario changes one declared pressure. No H/T duplication,
    quality-based filtering, or historical diagnostic instances are used.
    """
    if n < 6 or machines not in (12, 24) or scenario not in SCENARIOS:
        raise ValueError("Invalid speed diagnostic dimensions")
    rng = random.Random(seed)
    jobs = []
    for i in range(n):
        release = float(rng.randrange(0, 3601, 60))
        pp = Profile(
            rng.choice((60, 120, 240, 360)),
            rng.choice((0.15, 0.3, 0.45, 0.6)),
            rng.choice((2, 4, 6)),
        )
        dp = Profile(
            rng.choice((180, 300, 600, 900)),
            rng.choice((0.15, 0.3, 0.45, 0.6)),
            rng.choice((2, 4, 6)),
        )
        kv = float(rng.choice((0, 30, 60, 120)))
        jobs.append(Job(f"j{i}", 0.0 if scenario == "burst" else release, pp, dp, kv))
    upper = (
        max(j.release for j in jobs)
        + sum(j.prefill.duration + j.decode.duration + j.kv_delay for j in jobs)
        + 86400
    )
    prices = (0.35, 0.8, 0.5)
    regions = tuple(
        Region(
            f"r{r}",
            tuple(
                Tariff(
                    float(t),
                    min(float(t + 4800), upper),
                    prices[(t // 4800 + r) % 3] * (1 + 0.12 * r),
                )
                for t in range(0, int(upper), 4800)
            ),
            (20.0 if scenario == "demand_high" else 5.0) * (1 + 0.05 * r),
        )
        for r in range(3)
    )
    instances = tuple(
        ServingInstance(f"m{i}", i % 3, 8.0 if scenario == "vram_tight" else 16.0, 0.06, 0.4)
        for i in range(machines)
    )
    problem = ProblemInstance(tuple(jobs), regions, instances)
    validate_problem(problem)
    return problem


def speed_specs(*, pilot: bool = False) -> list[dict]:
    """Return disjoint pilot/formal case seeds and balanced four-arm block order."""
    specs: list[dict] = []
    base = 202610151000 if pilot else 202610152000
    sizes = (20, 200) if pilot else (20, 50, 100, 200)
    scenarios = ("mixed",) if pilot else SCENARIOS
    for scenario in scenarios:
        for machines in (12, 24):
            for n in sizes:
                base_seed = base + len(specs) * 10
                for seed in (15001,) if pilot else SEEDS:
                    key = f"n{n}_m{machines}_{scenario}_b{base_seed}_s{seed}"
                    specs.append(
                        {
                            "key": key,
                            "n": n,
                            "machines": machines,
                            "scenario": scenario,
                            "base_seed": base_seed,
                            "seed": seed,
                            "bank_seed": seed + 100000,
                            "order": [ARMS[i] for i in ORDERS[len(specs) % 4]],
                        }
                    )
    return specs
