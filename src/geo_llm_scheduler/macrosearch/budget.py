"""Separate experimental severity/coverage policies and three budget controls."""

import random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.rl.state import associate, preference
from geo_llm_scheduler.utils.numeric import TOL, close


def objective_coverage(archive: list[Candidate]) -> tuple[Candidate, ...]:
    """Return deterministic objective-space representatives without changing Archive semantics."""
    representatives: list[Candidate] = []
    ordered = sorted(archive, key=lambda candidate: candidate.evaluation.objectives)
    for candidate in ordered:
        objectives = candidate.evaluation.objectives
        duplicate = False
        for existing in reversed(representatives):
            flow = existing.evaluation.flow
            if objectives[0] - flow > TOL.time + TOL.relative * max(abs(objectives[0]), abs(flow)):
                break
            if close(objectives[0], flow, TOL.time) and close(
                objectives[1], existing.evaluation.bill, TOL.cost
            ):
                duplicate = True
                break
        if duplicate:
            continue
        representatives.append(candidate)
    return tuple(representatives)


def choose_budget(
    action: int,
    values: tuple[float, ...],
    stagnant_count: int,
    index: int,
    archive: list[Candidate],
    weights: tuple[tuple[float, float], ...],
    context: NormalizationContext,
    config: Config,
    rng: random.Random,
) -> int:
    """Choose one per-invocation exact budget; policies never compose implicitly."""
    low, medium, high = config.budgets
    policy = config.budget_policy
    if policy == "fixed":
        return config.fixed_budget
    if policy == "static":
        return config.static_budgets[action - 1]
    if policy == "random":
        return rng.choice(config.budgets)
    if policy == "severity":
        ratios = [v / t for v, t in zip(values, config.severity_thresholds)]
        low_cutoff, high_cutoff = config.severity_budget_cutoffs
        if action in (4, 5):
            return (high, medium, low)[preference(index, config.population)]
        if action == 6:
            return high if stagnant_count >= config.stagnation_threshold else low
        value = {1: ratios[0], 2: max(ratios[:2]), 3: ratios[2], 7: ratios[5], 8: ratios[4]}[action]
        return low if value <= low_cutoff else medium if value <= high_cutoff else high
    if policy == "coverage":
        coverage = objective_coverage(archive)
        if len(coverage) < config.population:
            return medium
        counts = [0] * len(weights)
        for candidate in coverage:
            counts[associate(context.normalize(candidate), weights)] += 1
        ratio = counts[index] / (len(coverage) / len(weights))
        return high if ratio < 0.5 else medium if ratio <= 1.5 else low
    raise ValueError(f"Unknown budget policy: {policy}")
