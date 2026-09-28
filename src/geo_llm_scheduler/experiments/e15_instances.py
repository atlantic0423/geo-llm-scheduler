"""Tariff-paired E15 instances with frozen, non-overlapping base seeds."""

from __future__ import annotations

from dataclasses import replace

from geo_llm_scheduler.domain.models import ProblemInstance, Region, Tariff
from geo_llm_scheduler.experiments.mixed_instances import generate_mixed_instance, mixed_parameters
from geo_llm_scheduler.io.validation import validate_problem


def e15_generator_manifest() -> dict[str, object]:
    """Record all sampling ranges and the sole tariff intervention."""
    return {
        "base": mixed_parameters(),
        "base_seed_A": list(range(1001, 1013)),
        "base_seed_B": list(range(2001, 2021)),
        "base_seed_D": list(range(3001, 3013)),
        "pilot_seeds": [9001, 9002],
        "tariff_H": "unchanged homogeneous D02 mixed tariffs and demand rates",
        "tariff_T": {
            "price_multipliers": [0.8, 1.0, 1.2],
            "four_segment_rotations": [0, 1, 2],
            "demand_cny_per_kw": [4.0, 5.0, 6.0],
            "all_other_fields": "bitwise identical to H base",
        },
    }


def generate_e15_pair(jobs: int, seed: int) -> tuple[ProblemInstance, ProblemInstance]:
    """Return H/T variants that differ only in Region tariffs and demand rate."""
    homogeneous = generate_mixed_instance(jobs, seed)
    scales = (0.8, 1.0, 1.2)
    bounds = (0.0, 900.0, 1800.0, 2700.0, 3600.0)
    prices = (0.42, 0.72, 0.35, 0.62)
    horizon = homogeneous.regions[0].tariffs[-1].end
    regions = []
    for index, region in enumerate(homogeneous.regions):
        shifted = prices[index:] + prices[:index]
        tariffs = tuple(
            Tariff(bounds[j], bounds[j + 1], shifted[j] * scales[index]) for j in range(4)
        ) + (Tariff(3600.0, horizon, 0.48 * scales[index]),)
        regions.append(Region(region.name, tariffs, (4.0, 5.0, 6.0)[index]))
    heterogeneous = replace(homogeneous, regions=tuple(regions))
    validate_problem(homogeneous)
    validate_problem(heterogeneous)
    return homogeneous, heterogeneous
