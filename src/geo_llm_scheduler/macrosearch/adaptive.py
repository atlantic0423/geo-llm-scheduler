"""E15 experimental stateful budget policies; legacy policies stay unchanged."""

from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass, field

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.macrosearch.budget import objective_coverage
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.rl.state import direction, preference
from geo_llm_scheduler.utils.numeric import TOL


def action_severity(
    action: int, values: tuple[float, ...], stagnant: int, index: int, config: Config
) -> float:
    """Use the current action-specific source before E15 percentile binning."""
    ratios = tuple(v / t for v, t in zip(values, config.severity_thresholds))
    if action in (4, 5):
        return float(2 - preference(index, config.population))
    if action == 6:
        return stagnant / config.stagnation_threshold
    return {1: ratios[0], 2: max(ratios[:2]), 3: ratios[2], 7: ratios[5], 8: ratios[4]}[action]


def empirical_percentile(history: list[float], value: float) -> float:
    """Midrank empirical percentile; exact ties do not depend on call order."""
    if not history:
        return 0.5
    ordered = sorted(history)
    lo = bisect.bisect_left(ordered, value)
    hi = bisect.bisect_right(ordered, value)
    return (lo + hi) / (2 * len(ordered))


def gap_budgets(
    gaps: tuple[float, ...], levels: tuple[int, ...]
) -> tuple[tuple[int, ...], float, float, float]:
    """Quantile rank gaps with an absolute and relative uniformity gate."""
    if not gaps:
        return (), 0.0, 0.0, 0.0
    ordered = sorted(gaps)
    q25 = ordered[int(0.25 * (len(ordered) - 1))]
    q75 = ordered[int(0.75 * (len(ordered) - 1))]
    spread = q75 - q25
    if spread <= max(1e-3, 0.05 * max(q75, 1e-9)):
        return (levels[1],) * len(gaps), q25, q75, spread
    result = tuple(levels[0] if g < q25 else levels[2] if g > q75 else levels[1] for g in gaps)
    return result, q25, q75, spread


def direction_gaps(
    population: list[Candidate],
    archive: list[Candidate],
    weights: tuple[tuple[float, float], ...],
    context: NormalizationContext,
) -> tuple[float, ...]:
    """Measure nearest representative angular distance for every auxiliary ray."""
    representatives = objective_coverage(population + archive)
    rays = []
    for candidate in representatives:
        x, y = context.normalize(candidate)
        if math.hypot(x, y) > 1e-12:
            rays.append(math.atan2(max(0.0, y), max(0.0, x)))
    if not rays:
        return (0.0,) * len(weights)
    rays.sort()
    gaps = []
    for weight in weights:
        x, y = direction(weight)
        angle = math.atan2(y, x)
        at = bisect.bisect_left(rays, angle)
        gaps.append(min(abs(angle - rays[j]) for j in (max(0, at - 1), min(at, len(rays) - 1))))
    return tuple(gaps)


@dataclass
class AdaptiveBudget:
    """Per-run deterministic policy history, coverage cache and audit counters."""

    histories: dict[int, list[float]] = field(default_factory=lambda: {a: [] for a in range(1, 9)})
    coverage_key: tuple[object, ...] | None = None
    coverage_values: tuple[int, ...] = ()
    coverage_gaps: tuple[float, ...] = ()
    coverage_quantiles: tuple[float, float, float] = (0.0, 0.0, 0.0)
    counts: Counter[str] = field(default_factory=Counter)
    cpu_seconds: float = 0.0

    def choose(
        self,
        action: int,
        values: tuple[float, ...],
        stagnant: int,
        index: int,
        population: list[Candidate],
        archive: list[Candidate],
        weights: tuple[tuple[float, float], ...],
        context: NormalizationContext,
        config: Config,
    ) -> tuple[int, dict[str, float | int]]:
        """Return requested budget and a per-invocation audit record."""
        if config.budget_policy == "sequential":
            return config.budgets[2], {"requested_budget": config.budgets[2]}
        if config.budget_policy == "severity_v2":
            raw = action_severity(action, values, stagnant, index, config)
            history = self.histories[action]
            percentile = empirical_percentile(history, raw)
            count = len(history)
            budget = (
                config.budgets[1]
                if count < 5
                else (
                    config.budgets[0]
                    if percentile < 0.2
                    else config.budgets[2]
                    if percentile >= 0.8
                    else config.budgets[1]
                )
            )
            history.append(raw)
            self.counts[f"severity_budget:{budget}"] += 1
            return budget, {
                "raw_severity": raw,
                "action_percentile": percentile,
                "history_count": count,
                "requested_budget": budget,
            }
        if config.budget_policy != "coverage_v2":
            raise ValueError("Adaptive controller needs an E15 policy")
        # Objective-tolerance quantization is the cache identity, never phenotype identity.
        key = (
            frozenset(
                (round(c.evaluation.flow / TOL.time), round(c.evaluation.bill / TOL.cost))
                for c in population + archive
            ),
            context.ideal,
            context.maximum,
        )
        if key != self.coverage_key:
            self.coverage_gaps = direction_gaps(population, archive, weights, context)
            budgets, q25, q75, spread = gap_budgets(self.coverage_gaps, config.budgets)
            self.coverage_values = budgets
            self.coverage_quantiles = q25, q75, spread
            self.coverage_key = key
            self.counts["coverage_recompute_count"] += 1
        else:
            self.counts["coverage_cache_hit"] += 1
        q25, q75, spread = self.coverage_quantiles
        budget = self.coverage_values[index]
        self.counts[f"coverage_budget:{budget}"] += 1
        return budget, {
            "gap": self.coverage_gaps[index],
            "gap_q25": q25,
            "gap_q75": q75,
            "gap_spread": spread,
            "requested_budget": budget,
        }
