"""Opt-in solution selection for episode-external right shift; no failure memory."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from random import Random

from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.rl.state import associate
from geo_llm_scheduler.utils.numeric import TOL, close, identity

SEGMENTS = ("flow_end", "middle", "bill_end")
POOL_POLICIES = ("pool_random", "direction", "region_time")
PhysicalKey = tuple[tuple[int, ...], tuple[float, ...]]


def segment_of(norm: tuple[float, float]) -> int:
    """Use the existing frozen normalized-Flow thirds, including boundary ties."""
    return 0 if norm[0] < 1 / 3 else 1 if norm[0] < 2 / 3 else 2


def physical_key(candidate: Candidate) -> PhysicalKey:
    """Exact physical schedule within one immutable problem: assignments and starts.

    OS and derived diagnostics do not change an already scheduled phenotype.
    Do not round starts: different physical inputs must remain distinct even when
    their objectives or the numeric-comparison identity happen to coincide.
    """
    return candidate.genotype.ms, candidate.schedule.starts


def physical_fingerprint(candidate: Candidate) -> str:
    """Stable binary SHA256 within one instance; pair with the run's input SHA.

    Zero is canonicalized so numeric equality and pool deduplication agree.
    This is diagnostic provenance, never an online eligibility test.
    """
    ms, starts = physical_key(candidate)
    digest = hashlib.sha256(b"rightshift-physical-v1\0")
    digest.update(struct.pack("<Q", len(ms)))
    digest.update(struct.pack(f"<{len(ms)}q", *ms))
    digest.update(struct.pack(f"<{len(starts)}d", *(t if t else 0.0 for t in starts)))
    return digest.hexdigest()


def candidate_pool(population: Sequence[Candidate], endpoint: Candidate) -> list[Candidate]:
    """Deduplicate the current population plus endpoint, without dominance pruning.

    Stable physical order makes slot order irrelevant. Identical schedules with
    different OS retain the smallest canonical identity. No state survives a call.
    """
    unique: dict[PhysicalKey, Candidate] = {}
    for candidate in (*population, endpoint):
        key = physical_key(candidate)
        previous = unique.get(key)
        if previous is None or candidate.genotype.os < previous.genotype.os:
            unique[key] = candidate
    return [unique[key] for key in sorted(unique)]


@dataclass(frozen=True)
class Selection:
    """One chosen input and the current candidate-set composition."""

    candidate: Candidate
    region: int
    direction: int
    pool_size: int
    region_sizes: tuple[int, ...]
    direction_sizes: tuple[int, ...]
    source: str


@dataclass
class PolishSelector:
    """Only rotation pointers and measured service seconds persist across calls."""

    direction_cursor: int = 0
    region_cursor: int = 0
    region_seconds: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def choose(
        self,
        policy: str,
        population: Sequence[Candidate],
        endpoint: Candidate,
        frame: FrozenFrame,
        context: NormalizationContext,
        lambdas: tuple[tuple[float, float], ...],
        rng: Random,
    ) -> Selection:
        """Select from the shared pool using an explicit, selection-only RNG.

        Direction association uses MOEA/D's current context and inverse-weight
        rays; region association uses the independent frozen LS frame. Empty
        groups are skipped. Direction score ties are anchored at the minimum.
        """
        if policy not in POOL_POLICIES:
            raise ValueError("Unknown external right-shift selection policy")
        pool = candidate_pool(population, endpoint)
        regions: list[list[int]] = [[] for _ in SEGMENTS]
        directions: list[list[int]] = [[] for _ in lambdas]
        labels = []
        for i, candidate in enumerate(pool):
            region = segment_of(frame.normalize(candidate.evaluation.objectives))
            direction = associate(context.normalize(candidate), lambdas)
            regions[region].append(i)
            directions[direction].append(i)
            labels.append((region, direction))
        if policy == "pool_random":
            chosen = rng.randrange(len(pool))
        elif policy == "direction":
            group = next(
                (self.direction_cursor + k) % len(lambdas)
                for k in range(len(lambdas))
                if directions[(self.direction_cursor + k) % len(lambdas)]
            )
            scores = [(context.scalar(pool[i], lambdas[group]), i) for i in directions[group]]
            best = min(score for score, _ in scores)
            tied = [i for score, i in scores if close(score, best, TOL.scalar)]
            chosen = min(tied, key=lambda i: (identity(pool[i]), physical_key(pool[i])))
            self.direction_cursor = (group + 1) % len(lambdas)
        else:
            best_seconds = min(self.region_seconds[r] for r in range(3) if regions[r])
            # Measured seconds are ordered directly: no new tunable threshold.
            group = next(
                (self.region_cursor + k) % 3
                for k in range(3)
                if regions[(self.region_cursor + k) % 3]
                and self.region_seconds[(self.region_cursor + k) % 3] == best_seconds
            )
            chosen = rng.choice(regions[group])
            self.region_cursor = (group + 1) % 3
        candidate = pool[chosen]
        key = physical_key(candidate)
        in_pop = any(physical_key(c) == key for c in population)
        is_endpoint = physical_key(endpoint) == key
        return Selection(
            candidate,
            *labels[chosen],
            len(pool),
            tuple(map(len, regions)),
            tuple(map(len, directions)),
            "both" if in_pop and is_endpoint else "population" if in_pop else "endpoint",
        )

    def charge(self, region: int, seconds: float) -> None:
        """Charge actual service cost, excluding shared pool and diagnostic overhead."""
        if region not in range(3) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid measured right-shift service cost")
        self.region_seconds[region] += seconds
