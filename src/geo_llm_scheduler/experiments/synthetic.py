"""Seeded synthetic instances; parameter choices are explicit research inputs."""

import random

from geo_llm_scheduler.domain.models import (
    Job,
    ProblemInstance,
    Profile,
    Region,
    ServingInstance,
    Tariff,
)


def synthetic(jobs: int, regions: int, servers_per_region: int, seed: int) -> ProblemInstance:
    """Create feasible positive-capacity cases without claiming real profiling fidelity."""
    if min(jobs, regions, servers_per_region) < 1:
        raise ValueError("Positive synthetic dimensions required")
    rng = random.Random(seed)
    requests = tuple(
        Job(
            f"j{i}",
            rng.choice((0, 30, 60)),
            Profile(rng.choice((60, 120, 300)), rng.choice((0.25, 0.5)), rng.choice((2, 4))),
            Profile(rng.choice((120, 300, 600)), rng.choice((0.25, 0.5)), rng.choice((2, 4))),
            rng.choice((0, 15, 30)),
        )
        for i in range(jobs)
    )
    # Tariff coverage is input data, not a schedule constraint or search horizon.
    coverage = (
        max(j.release for j in requests)
        + sum(j.prefill.duration + j.decode.duration + j.kv_delay for j in requests)
        + 86400
    )
    locations = tuple(
        Region(
            f"r{r}",
            (
                Tariff(0, 900, 0.1 + 0.05 * r),
                Tariff(900, 1800, max(0.01, 0.3 - 0.03 * r)),
                Tariff(1800, coverage, 0.1 + 0.02 * r),
            ),
            2 + r,
        )
        for r in range(regions)
    )
    machines = tuple(
        ServingInstance(f"r{r}m{k}", r, 16, 10, 100)
        for r in range(regions)
        for k in range(servers_per_region)
    )
    return ProblemInstance(requests, locations, machines)
