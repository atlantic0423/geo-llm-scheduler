"""Prototype 3: Flow-invariant Bill move groups (generalizes the single right-shift Polish).

Only Prefill starts move, inside ``[EST, S_D - KV - p_P]`` with every Decode frozen,
so all Decode completions, Total Flow and ``T_end`` are unchanged by construction.
Unlike RightShiftPolish, moves may go left or right and several improving moves are
committed greedily in one proposal. Moves are ranked with incremental Bill deltas;
the gateway still exact-evaluates every complete proposal.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance, Schedule
from geo_llm_scheduler.lsv2.billing import WINDOW, PowerProfile
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.scheduling.resources import _fits_intervals
from geo_llm_scheduler.scheduling.timing import legal_bounds
from geo_llm_scheduler.utils.numeric import TOL, close, less


@dataclass
class _PreparedFlow:
    base: PowerProfile
    upper: dict[int, float]
    bounds: dict[int, tuple[float, float]]
    saving: dict[int, float]
    targets: list[int]
    max_moves: int
    empty_attempts: int | None = None


def prefill_times(
    problem: ProblemInstance,
    machine: int,
    operation: int,
    lower: float,
    upper: float,
    others: list[tuple[int, float, float]],
) -> list[float]:
    """Finite critical Prefill starts: resource events, TOU and 900 s boundaries, minus duration."""
    p = problem.profile(operation).duration
    region = problem.instances[machine].region
    events = {t for _, a, b in others for t in (a, b)}
    events |= {t for tariff in problem.regions[region].tariffs for t in (tariff.start, tariff.end)}
    first, last = int(max(0.0, lower - p) // WINDOW), math.ceil((upper + p) / WINDOW)
    events |= {w * WINDOW for w in range(first, last + 1)}
    values = {lower, upper} | events | {e - p for e in events}
    return sorted(
        t for t in values if not less(t, lower, TOL.time) and not less(upper, t, TOL.time)
    )


@dataclass
class FlowSafeBillMoves:
    """Greedy multi-move Prefill retiming that never changes Total Flow."""

    action: int = 103
    max_moves_ratio: float = 0.1
    min_moves: int = 5
    target_ratio: float = 0.2
    min_targets: int = 20
    stochastic: float = 0.3
    right_only: bool = False
    all_targets: bool = False

    def propose(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        config: Config,
        rng: random.Random,
    ) -> ProposalBatch:
        """Return up to ``budget`` distinct timing proposals from one frozen incumbent."""
        prepared = self._prepare(problem, incumbent)
        return self._propose_prepared(problem, incumbent, budget, rng, prepared)

    def _prepare(self, problem: ProblemInstance, incumbent: Candidate) -> _PreparedFlow:
        g, s = incumbent.genotype, incumbent.schedule
        n = len(problem.jobs)
        base = PowerProfile(problem, g.ms, s.starts)
        bounds = {}
        upper = {}
        for o in range(0, problem.operation_count, 2):
            lo, hi = legal_bounds(problem, g, s, o)
            upper[o] = hi
            if self.right_only:
                lo = max(lo, s.starts[o])
            if less(lo, hi, TOL.time):
                bounds[o] = (lo, hi)
        # Rank targets by their exclusive Bill contribution (cost if their interval vanished).
        saving = {o: -base.delta_for({g.ms[o]: {o: None}}) for o in bounds}
        k = max(self.min_targets, math.ceil(self.target_ratio * n))
        targets = sorted(bounds, key=lambda o: -saving[o])
        if not self.all_targets:
            targets = targets[:k]
        max_moves = max(self.min_moves, math.ceil(self.max_moves_ratio * n))
        return _PreparedFlow(base, upper, bounds, saving, targets, max_moves)

    def _propose_prepared(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        rng: random.Random,
        prepared: _PreparedFlow,
        *,
        reuse_empty: bool = False,
    ) -> ProposalBatch:
        g, s = incumbent.genotype, incumbent.schedule
        base, bounds, saving = prepared.base, prepared.bounds, prepared.saving
        targets, max_moves = prepared.targets, prepared.max_moves
        batch = ProposalBatch()
        seen = set()
        scanned, moves, capped = [], [], []
        for b in range(max(1, budget)):
            order = list(targets)
            if b:
                rng.shuffle(order)
                order.sort(key=lambda o: -saving[o] * (1 - self.stochastic * rng.random()))
            # A complete zero-move deterministic pass leaves every competitor
            # unchanged. Reordering these same targets cannot enable a first move.
            # Still consume the original shuffle and key randomness on every call.
            if reuse_empty and prepared.empty_attempts is not None:
                scanned.append(len(order))
                moves.append(0)
                capped.append(False)
                continue
            starts = list(s.starts)
            profile = base.copy()
            machines: dict[int, list[tuple[int, float, float]]] = {}
            moved: list[int] = []
            checked = 0
            for o in order:
                if len(moved) >= max_moves:
                    break
                checked += 1
                m = g.ms[o]
                if m not in machines:
                    machines[m] = [
                        (q, starts[q], starts[q] + problem.profile(q).duration)
                        for q in range(problem.operation_count)
                        if g.ms[q] == m
                    ]
                others = [item for item in machines[m] if item[0] != o]
                lo, hi = bounds[o]
                p = problem.profile(o).duration
                best_t, best_d = None, -TOL.cost
                for t in prefill_times(problem, m, o, lo, hi, others):
                    if close(t, starts[o], TOL.time):
                        continue
                    if not _fits_intervals(problem, m, o, t, others):
                        continue
                    batch.attempts += 1
                    d = profile.delta_for({m: {o: (t, t + p)}})
                    if d < best_d:
                        best_t, best_d = t, d
                if best_t is None:
                    continue
                profile.apply({m: {o: (best_t, best_t + p)}})
                starts[o] = best_t
                machines[m] = others + [(o, best_t, best_t + p)]
                moved.append(o)
            scanned.append(checked)
            moves.append(len(moved))
            capped.append(len(moved) >= max_moves)
            key = tuple(round(t / TOL.time) for t in starts)
            if moved and key not in seen:
                seen.add(key)
                batch.proposals.append(
                    Proposal(g, Schedule(tuple(starts)), selected=tuple(sorted(moved)))
                )
        batch.diagnostics = dict(
            targets=len(targets), eligible=len(bounds), proposals=len(batch.proposals)
        )
        if self.right_only:
            batch.diagnostics.update(
                scanned=scanned, moves=moves, move_cap_reached=capped, max_moves=max_moves
            )
        if reuse_empty:
            hit = prepared.empty_attempts is not None
            batch.diagnostics["empty_cache_hit"] = hit
            batch.diagnostics["saved_attempts"] = prepared.empty_attempts if hit else 0
            if not hit and not any(moves):
                prepared.empty_attempts = batch.attempts
        return batch
