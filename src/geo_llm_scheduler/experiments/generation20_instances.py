"""Frozen 20-hour campaign instances within the current homogeneous hardware model."""

from dataclasses import replace

from geo_llm_scheduler.domain.models import ProblemInstance, Region, Tariff
from geo_llm_scheduler.experiments.mixed_instances import generate_mixed_instance
from geo_llm_scheduler.io.validation import validate_problem


def generate_generation20_instance(seed: int, heterogeneous_prices: bool) -> ProblemInstance:
    """Generate a new independent instance; vary prices/rates, never hardware, if requested."""
    base = generate_mixed_instance(50, seed)
    if not heterogeneous_prices:
        return base
    # Fixed before observing campaign outcomes: two price scales and phase offsets.
    scales = (0.8, 1.0, 1.2)
    bounds = (0, 900, 1800, 2700, 3600)
    original = (0.42, 0.72, 0.35, 0.62)
    horizon = base.regions[0].tariffs[-1].end
    regions = []
    for index, region in enumerate(base.regions):
        prices = original[index:] + original[:index] if index else original
        tariffs = tuple(
            Tariff(float(bounds[j]), float(bounds[j + 1]), prices[j] * scales[index])
            for j in range(4)
        ) + (Tariff(3600.0, horizon, 0.48 * scales[index]),)
        regions.append(Region(region.name, tariffs, (4.0, 5.0, 6.0)[index]))
    problem = replace(base, regions=tuple(regions))
    validate_problem(problem)
    return problem
