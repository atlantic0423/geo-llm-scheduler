"""New parent instances with single-factor paired pressures for auxiliary walks."""

from __future__ import annotations

import random
from dataclasses import replace

from geo_llm_scheduler.domain.models import ProblemInstance, Region, ServingInstance, Tariff
from geo_llm_scheduler.experiments.mixed_instances import generate_mixed_instance
from geo_llm_scheduler.io.validation import validate_problem

TOPOLOGIES = ((3, 12), (2, 12), (5, 12), (3, 6), (3, 24))
PRESSURES = ("arrival", "kv", "cheap_capacity", "demand")


def paired_instance(
    jobs: int,
    regions: int,
    servers: int,
    pressure: str,
    version: int,
    rng: random.Random,
) -> ProblemInstance:
    """Use identical base draws for a pair, changing only the declared pressure factor."""
    if regions < 2 or servers < regions or pressure not in PRESSURES or version not in (0, 1):
        raise ValueError("Invalid topology or paired pressure")
    base = generate_mixed_instance(jobs, rng.getrandbits(63))
    requests = base.jobs
    if pressure == "arrival" and version == 1:
        requests = tuple(replace(job, release=job.release / 18) for job in requests)
    if pressure == "kv":
        requests = tuple(
            replace(job, kv_delay=job.kv_delay / 4 if version == 0 else 180 + 8 * job.kv_delay)
            for job in requests
        )
    counts = [servers // regions + int(r < servers % regions) for r in range(regions)]
    rng.shuffle(counts)
    if pressure == "cheap_capacity":
        high = min(servers - regions + 1, servers // regions + max(1, servers // (2 * regions)))
        remaining = servers - high
        counts = [high] + [
            remaining // (regions - 1) + int(r < remaining % (regions - 1))
            for r in range(regions - 1)
        ]
        if version == 1:
            lowest = min(range(1, regions), key=lambda r: counts[r])
            counts[0], counts[lowest] = counts[lowest], counts[0]
    coverage = (
        max(job.release for job in base.jobs)
        + sum(
            job.prefill.duration + job.decode.duration + 180 + 8 * job.kv_delay for job in base.jobs
        )
        + 86400
    )
    boundaries = (0.0, 900.0, 1800.0, 3600.0, 7200.0, 14400.0, coverage)
    prices = (0.42, 0.72, 0.35, 0.62, 0.48, 0.40)
    rate = (0.2 if version == 0 else 20.0) if pressure == "demand" else 5.0
    territories = tuple(
        Region(
            f"r{r}",
            tuple(
                Tariff(a, b, prices[k] * (0.6 + 0.25 * r))
                for k, (a, b) in enumerate(zip(boundaries, boundaries[1:]))
            ),
            rate,
        )
        for r in range(regions)
    )
    machines = tuple(
        ServingInstance(f"r{r}m{m}", r, 16.0, 10.0, 100.0)
        for r, count in enumerate(counts)
        for m in range(count)
    )
    result = ProblemInstance(requests, territories, machines)
    validate_problem(result)
    return result
