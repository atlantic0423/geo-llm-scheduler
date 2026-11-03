"""Stress instance generator v2: stronger Flow/Bill conflict and larger search space.

The component33 panel is easy in specific ways (2026-10-26 diagnosis): utilization
is low at small sizes (0.12 at 20 jobs, 0.27 at 50), every release falls inside one
80-minute tariff period with a 2.3:1 price ratio, regions differ only by a price
factor, and the Bill end is 60-86% Demand Charge. A Demand-dominated Bill end is
reached by serializing work on few machines, which the MS/OS encoding already
expresses; a TOU-dominated Bill end needs deliberate delays that SSGS never makes.
Six two-level factors turn these pressures up independently; level 0 reproduces the
component33 setting of that factor and level 1 is the stressed version:

- ``load``: target compute utilization 0.55 / 0.85 over a one-hour window
  (machine count from the work, at least two per region);
- ``tariff``: 4800 s periods at 0.35/0.8/0.5 / 1800 s periods at 0.6/1.65/3.0
  (5:1, three times the price level); regions are phase-shifted (time zones), so
  shifting work in time or across regions pays;
- ``arrival``: uniform over one hour / four hours with bursts at peak-period starts;
- ``demand``: Demand Charge rate 5 / 0.5 per kW (level 1 makes the Bill
  TOU-dominated; a 2026-10-26 probe showed a stronger Demand rate shrinks the space);
- ``region``: balanced machine counts with 15-60 s KV / cheap regions scarce and
  4x KV, so the cheap region is contested and cross-instance P/D is costly;
- ``jobs``: component33 duration mix / heavy-tailed Decode (up to one hour).

The model boundary is unchanged: same-stage instances are homogeneous, power is
idle/active, every tariff covers the whole horizon. Candidates are screened later
only with baseline runs; this module never looks at search results.
"""

from __future__ import annotations

import math
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

FACTORS = ("load", "tariff", "arrival", "demand", "region", "jobs")


def design_points() -> list[dict[str, int]]:
    """Resolution-IV 2^(6-2) fractional factorial: E = ABC, F = BCD (16 points)."""
    points = []
    for i in range(16):
        a, b, c, d = (i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1
        e, f = a ^ b ^ c, b ^ c ^ d
        points.append(dict(zip(FACTORS, (a, b, c, d, e, f))))
    return points


def _job_profiles(rng: random.Random, heavy: bool) -> tuple[Profile, Profile, float]:
    if heavy:
        prefill = rng.choices((60, 120, 240, 360, 600), weights=(3, 3, 2, 1, 1))[0]
        decode = rng.choices((120, 300, 600, 900, 1800, 3600), weights=(5, 5, 4, 3, 2, 1))[0]
    else:
        prefill = rng.choice((60, 120, 240, 360))
        decode = rng.choice((180, 300, 600, 900))
    p = Profile(float(prefill), rng.choice((0.15, 0.3, 0.45, 0.6)), float(rng.choice((2, 4, 6))))
    d = Profile(float(decode), rng.choice((0.15, 0.3, 0.45, 0.6)), float(rng.choice((2, 4, 6))))
    return p, d, float(rng.choice((15, 30, 60)))


def stress_problem(
    n: int,
    regions: int,
    levels: dict[str, int],
    seed: int,
    *,
    demand_rate: float | None = None,
    price_scale: float = 1.0,
) -> ProblemInstance:
    """Generate one instance; ``levels`` maps every factor in ``FACTORS`` to 0 or 1.

    ``demand_rate`` and ``price_scale`` override the factor-implied Demand rate and
    scale all prices; they exist for generator probing only.
    """
    if n < 6 or regions not in (2, 3, 5) or set(levels) != set(FACTORS):
        raise ValueError("Invalid stress instance request")
    if any(v not in (0, 1) for v in levels.values()):
        raise ValueError("Factor levels must be 0 or 1")
    rng = random.Random(seed)
    period = 1800 if levels["tariff"] else 4800
    prices = (0.6, 1.65, 3.0) if levels["tariff"] else (0.35, 0.8, 0.5)
    span = 14400 if levels["arrival"] else 3600
    profiles = [_job_profiles(rng, bool(levels["jobs"])) for _ in range(n)]
    if levels["arrival"]:
        # Bursts open at region-0 peak periods; 70% of jobs arrive within 5 min of one.
        top = prices.index(max(prices))
        peaks = [t for t in range(0, span, period) if (t // period) % 3 == top] or [0]
        releases = []
        for _ in range(n):
            if rng.random() < 0.7:
                t = rng.choice(peaks) + rng.randrange(6) * 60
            else:
                t = rng.randrange(span // 60 + 1) * 60
            releases.append(float(min(t, span)))
    else:
        releases = [float(rng.randrange(61) * 60) for _ in range(n)]
    kv_scale = 4.0 if levels["region"] else 1.0
    jobs = tuple(
        Job(f"j{i}", releases[i], p, d, kv * kv_scale) for i, (p, d, kv) in enumerate(profiles)
    )
    work = sum(
        j.prefill.duration * j.prefill.compute + j.decode.duration * j.decode.compute for j in jobs
    )
    mean_len = sum(j.prefill.duration + j.decode.duration for j in jobs) / n
    target = 0.85 if levels["load"] else 0.55
    # Utilization is targeted over a one-hour window for both arrival levels, so the
    # four-hour bursty level congests at bursts and idles between them (room for
    # time shifting). Every region keeps at least two machines.
    machines = max(2 * regions, round(work / (target * (3600 + 2 * mean_len))))
    if levels["region"]:
        # Cheapest region (r = 0) gets the fewest machines: weights 1, 2, ..., R.
        weights = [r + 1 for r in range(regions)]
        raw = [machines * w / sum(weights) for w in weights]
        counts = [max(2, math.floor(x)) for x in raw]
        while sum(counts) < machines:
            counts[max(range(regions), key=lambda r: raw[r] - counts[r])] += 1
    else:
        counts = [machines // regions + (r < machines % regions) for r in range(regions)]
    upper = span + sum(j.prefill.duration + j.decode.duration + j.kv_delay + 240 for j in jobs)
    upper += 86400
    rate = demand_rate if demand_rate is not None else 0.5 if levels["demand"] else 5.0
    prices = (prices[0] * price_scale, prices[1] * price_scale, prices[2] * price_scale)
    locations = tuple(
        Region(
            f"r{r}",
            tuple(
                Tariff(
                    float(t),
                    min(float(t + period), float(upper)),
                    prices[(t // period + r) % 3] * (1 + 0.15 * r),
                )
                for t in range(0, int(upper), period)
            ),
            rate * (1 + 0.05 * r),
        )
        for r in range(regions)
    )
    servers = tuple(
        ServingInstance(f"m{r}_{k}", r, 16.0, 0.06, 0.4)
        for r, c in enumerate(counts)
        for k in range(c)
    )
    problem = ProblemInstance(jobs, locations, servers)
    validate_problem(problem)
    return problem


def candidate_specs(base_seed: int = 202610260000) -> list[dict]:
    """Screening candidates: 16 design points x 4 sizes, regions cycling 2/3/5/3."""
    specs = []
    for size_index, n in enumerate((20, 50, 100, 200)):
        for i, levels in enumerate(design_points()):
            specs.append(
                dict(
                    key=f"st_n{n}_d{i:02d}",
                    n=n,
                    regions=(2, 3, 5, 3)[i % 4],
                    levels=levels,
                    seed=base_seed + 100 * size_index + i,
                )
            )
    return specs


__all__ = ["FACTORS", "candidate_specs", "design_points", "stress_problem"]
