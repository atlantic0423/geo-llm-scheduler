"""Opt-in enhancement for continuous-key Transactions baselines.

This uses the existing EXTH mixed initializer and old single-Prefill Polish;
it does not alter the unenhanced baseline implementation or search budget.
"""

from __future__ import annotations

import math
from random import Random
from time import perf_counter

from geo_llm_scheduler.domain.models import ProblemInstance
from geo_llm_scheduler.experiments import ext_hybrid
from geo_llm_scheduler.experiments.trans_common import Individual, SearchContext
from geo_llm_scheduler.experiments.trans_hybrid_encoding import encode_schedule
from geo_llm_scheduler.operators.right_shift import PolishStats, polish
from geo_llm_scheduler.utils.numeric import TOL, close
from geo_llm_scheduler.utils.rng import RNGManager


class HybridSearchContext(SearchContext):
    """Same initialization and right-shift recipe as the five enhanced baselines."""

    def __init__(
        self,
        problem: ProblemInstance,
        *,
        seed: int,
        seconds: float,
        wait_genes: bool = True,
        max_generations: int | None = None,
    ) -> None:
        if not wait_genes:
            raise ValueError("Hybrid right-shift requires wait genes")
        super().__init__(problem, seed, seconds, wait_genes=True, max_generations=max_generations)
        self.seed = seed
        self.initialization_count = 0
        self.hybrid_evidence: list[dict] = []
        self.polish_rng: Random = RNGManager(seed).stream("trans_hybrid:polish")
        self.polish_counts = {"calls": 0, "accepted": 0, "reencoded": 0, "unrepresentable": 0}

    def initialize(self, size: int) -> list[Individual]:
        """Create full mixed population and preserve exact genotype round trips.

        Native dual-population methods get distinct deterministic initialization
        streams.  Small-size returns are for engineering tests only; full work
        stays charged to the same wall clock.
        """
        if not 4 <= size <= 100:
            raise ValueError("Enhanced continuous populations must have 4..100 members")
        seed = (
            self.seed
            if self.initialization_count == 0
            else self.seed + 1000003 * self.initialization_count
        )
        self.initialization_count += 1
        config = ext_hybrid.arm_config("EXTH_MOEAD", seed, self.seconds)
        evidence: dict = {}
        members = ext_hybrid.hybrid_initializer(self.problem, config, evidence)(
            self.gateway, RNGManager(seed)
        )
        self.hybrid_evidence.append(evidence)
        result = []
        for member in members[:size]:
            genes = encode_schedule(self.problem, member)
            if genes is None:
                raise AssertionError("Initial schedule cannot be represented by native keys")
            result.append(Individual(genes, member))
        self.evaluations += len(members)
        return result

    def polish_population(self, population: list[Individual]) -> list[Individual]:
        """Apply old Polish to 20% and replace only round-trip-consistent members."""
        selected = sorted(
            self.polish_rng.sample(range(len(population)), math.ceil(0.2 * len(population)))
        )
        result = population.copy()
        for index in selected:
            if self.progress() >= 1:
                break
            current = result[index]
            started = perf_counter()
            stats = PolishStats()
            candidate = polish(self.problem, current.candidate, self.gateway, stats)
            self.gateway.seconds["polish"] += perf_counter() - started
            self.polish_counts["calls"] += 1
            self.polish_counts["accepted"] += stats.accepted
            if not stats.accepted:
                continue
            keys = encode_schedule(self.problem, candidate)
            if keys is None:
                self.polish_counts["unrepresentable"] += 1
                continue
            replay = self.evaluate(keys)
            if replay.candidate.genotype != candidate.genotype or any(
                not close(a, b, TOL.time)
                for a, b in zip(
                    replay.candidate.schedule.starts, candidate.schedule.starts, strict=True
                )
            ):
                raise AssertionError("Right-shift re-encoding failed exact schedule replay")
            result[index] = replay
            self.polish_counts["reencoded"] += 1
        return result
