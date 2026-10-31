"""Threshold-free, input-scoped alternatives to repeatedly polishing empty inputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from random import Random

from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.lsv2.polish_selection import (
    PhysicalKey,
    PolishSelector,
    Selection,
    candidate_pool,
    physical_key,
    segment_of,
)
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.rl.state import associate
from geo_llm_scheduler.utils.numeric import TOL, close, identity, less

OPPORTUNITY_POLICIES = (
    "eligible_random",
    "eligible_region",
    "fresh_region",
    "frontier_region",
    "fresh_direction",
)


@dataclass
class OpportunitySelector:
    """Retain visit counts only for inputs still in the current physical pool.

    Empty certificates come from the generator, never from lack of Archive gain.
    Fresh means least completed searches in the chosen region/direction, not a
    cooldown or age cutoff. Dominated inputs remain eligible in every policy.
    """

    visits: dict[PhysicalKey, int] = field(default_factory=dict)
    accounting: PolishSelector = field(default_factory=PolishSelector)
    diagnostics: dict = field(default_factory=dict)

    def choose(
        self,
        policy: str,
        population: Sequence[Candidate],
        endpoint: Candidate,
        frame: FrozenFrame,
        context: NormalizationContext,
        lambdas: tuple[tuple[float, float], ...],
        rng: Random,
        known_empty: Callable[[Candidate], bool],
    ) -> Selection | None:
        """Choose one non-certified input, or None when no opportunity remains."""
        if policy not in OPPORTUNITY_POLICIES:
            raise ValueError("Unknown opportunity selection policy")
        pool = candidate_pool(population, endpoint)
        keys = {physical_key(c) for c in pool}
        self.visits = {k: v for k, v in self.visits.items() if k in keys}
        eligible = [c for c in pool if not known_empty(c)]
        self.diagnostics = dict(
            raw_pool_size=len(pool),
            eligible_pool_size=len(eligible),
            certified_empty_excluded=len(pool) - len(eligible),
            direction_diagnostics=policy == "fresh_direction",
        )
        if not eligible:
            return None
        regions: list[list[int]] = [[], [], []]
        directions: list[list[int]] = [[] for _ in lambdas] if policy == "fresh_direction" else []
        labels = []
        for i, c in enumerate(eligible):
            region = segment_of(frame.normalize(c.evaluation.objectives))
            direction = associate(context.normalize(c), lambdas) if directions else -1
            regions[region].append(i)
            if directions:
                directions[direction].append(i)
            labels.append((region, direction))
        if policy == "eligible_random":
            candidates = list(range(len(eligible)))
        elif policy == "fresh_direction":
            group = next(
                (self.accounting.direction_cursor + k) % len(lambdas)
                for k in range(len(lambdas))
                if directions[(self.accounting.direction_cursor + k) % len(lambdas)]
            )
            candidates = directions[group]
            self.accounting.direction_cursor = (group + 1) % len(lambdas)
        else:
            seconds = self.accounting.region_seconds
            minimum = min(seconds[r] for r in range(3) if regions[r])
            group = next(
                (self.accounting.region_cursor + k) % 3
                for k in range(3)
                if regions[(self.accounting.region_cursor + k) % 3]
                and seconds[(self.accounting.region_cursor + k) % 3] == minimum
            )
            candidates = regions[group]
            self.accounting.region_cursor = (group + 1) % 3
        if policy in ("fresh_region", "frontier_region", "fresh_direction"):
            least = min(self.visits.get(physical_key(eligible[i]), 0) for i in candidates)
            candidates = [
                i for i in candidates if self.visits.get(physical_key(eligible[i]), 0) == least
            ]
        if policy == "frontier_region":
            # Pareto preference only within the least-visited tier. This is not a
            # permanent dominance filter: every finite tier can be visited.
            values = {i: frame.normalize(eligible[i].evaluation.objectives) for i in candidates}
            candidates = [
                i
                for i in candidates
                if not any(
                    # Exact non-worsening makes the preference acyclic; use
                    # the contract tolerance only for the strict improvement.
                    all(x <= y for x, y in zip(values[j], values[i]))
                    and any(less(x, y, TOL.scalar) for x, y in zip(values[j], values[i]))
                    for j in candidates
                    if j != i
                )
            ]
        if policy == "fresh_direction":
            scores = {i: context.scalar(eligible[i], lambdas[group]) for i in candidates}
            minimum_score = min(scores.values())
            chosen = min(
                (i for i in candidates if close(scores[i], minimum_score, TOL.scalar)),
                key=lambda i: (identity(eligible[i]), physical_key(eligible[i])),
            )
        else:
            chosen = rng.choice(candidates)
        c = eligible[chosen]
        key = physical_key(c)
        self.diagnostics.update(
            selection_tier_size=len(candidates), prior_visits=self.visits.get(key, 0)
        )
        in_pop = any(physical_key(p) == key for p in population)
        is_endpoint = physical_key(endpoint) == key
        return Selection(
            c,
            *labels[chosen],
            len(eligible),
            tuple(map(len, regions)),
            tuple(map(len, directions)),
            "both" if in_pop and is_endpoint else "population" if in_pop else "endpoint",
        )

    def completed(self, candidate: Candidate) -> None:
        """Count a fully evaluated search; incomplete/deadline attempts do not count."""
        key = physical_key(candidate)
        self.visits[key] = self.visits.get(key, 0) + 1

    def charge(self, region: int, seconds: float) -> None:
        """Charge the entire external call, including choosing and diagnosing input."""
        self.accounting.charge(region, seconds)
