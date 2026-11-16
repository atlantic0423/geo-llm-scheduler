"""Deterministic decoder-embedded timing refinement (research arms P2a/P2b).

The engine's SSGS decoder places every operation at its earliest feasible start,
so a timing improvement found by a separate refinement operator is lost when a
child is decoded again. These refiners run inside the decoder instead: the
refined timetable is a deterministic function of ``(genotype, refine_class)``
and is therefore inherited like any decoded schedule.

``flow_safe_pass``
    One deterministic greedy pass of :class:`FlowSafeBillMoves` restricted to
    right moves: Prefills move later inside ``[S_P, S_D - KV - p_P]`` with every
    Decode frozen, so Total Flow is unchanged and only Bill-improving moves are
    committed (incremental Bill deltas; the gateway exact-evaluates the result).
``tradeoff_pass``
    For a Bill-leaning class only: Decodes may move later, never beyond the
    current batch end, when the class weight values the Bill decrease more than
    the Flow increase, ``lambda_B * dBill / s_B > lambda_F * dFlow / s_F`` with
    frozen scales ``(s_F, s_B)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.lsv2.billing import PowerProfile
from geo_llm_scheduler.lsv2.flowsafe import FlowSafeBillMoves, prefill_times
from geo_llm_scheduler.scheduling.resources import _fits_intervals
from geo_llm_scheduler.utils.numeric import TOL, close

# Representative Bill weight per class; class 0 (Flow-leaning) never trades Flow.
CLASS_BILL_WEIGHT = {0: None, 1: 0.5, 2: 5.0 / 6.0}


def weight_class(bill_weight: float) -> int:
    """Map a Tchebycheff Bill weight in ``[0, 1]`` to class 0 (<1/3), 1 (<2/3) or 2."""
    if not 0.0 <= bill_weight <= 1.0:
        raise ValueError("Bill weight must lie in [0, 1]")
    return 0 if bill_weight < 1.0 / 3.0 else 1 if bill_weight < 2.0 / 3.0 else 2


def _incumbent(genotype: Genotype, schedule: Schedule) -> Candidate:
    # FlowSafeBillMoves reads only the genotype and the schedule of its incumbent.
    return cast(Candidate, SimpleNamespace(genotype=genotype, schedule=schedule))


def flow_safe_pass(
    problem: ProblemInstance,
    genotype: Genotype,
    schedule: Schedule,
    *,
    target_ratio: float = 0.1,
    min_targets: int = 5,
    max_moves_ratio: float = 0.05,
    min_moves: int = 3,
) -> Schedule:
    """Return the first deterministic right-only Flow-safe greedy pass (or the input)."""
    mover = FlowSafeBillMoves(
        right_only=True,
        target_ratio=target_ratio,
        min_targets=min_targets,
        max_moves_ratio=max_moves_ratio,
        min_moves=min_moves,
    )
    incumbent = _incumbent(genotype, schedule)
    prepared = mover._prepare(problem, incumbent)
    if not prepared.targets:
        return schedule
    # budget=1: only the deterministic first pass runs; the RNG is never drawn.
    batch = mover._propose_prepared(problem, incumbent, 1, None, prepared)  # type: ignore[arg-type]
    refined = batch.proposals[0].schedule if batch.proposals else None
    return refined if refined is not None else schedule


def tradeoff_pass(
    problem: ProblemInstance,
    genotype: Genotype,
    schedule: Schedule,
    bill_weight: float,
    scales: tuple[float, float],
    *,
    target_ratio: float = 0.1,
    min_targets: int = 5,
    max_moves_ratio: float = 0.05,
    min_moves: int = 3,
) -> Schedule:
    """Move Decodes later when the weighted Bill gain exceeds the weighted Flow cost."""
    if not 0.0 < bill_weight < 1.0 or min(scales) <= 0:
        raise ValueError("Trade-off needs an interior weight and positive scales")
    g, starts = genotype, list(schedule.starts)
    n = len(problem.jobs)
    duration = [problem.profile(o).duration for o in range(problem.operation_count)]
    t_end = max(starts[o] + duration[o] for o in range(1, problem.operation_count, 2))
    profile = PowerProfile(problem, g.ms, starts)
    decodes = [
        o
        for o in range(1, problem.operation_count, 2)
        if t_end - duration[o] - starts[o] > TOL.time
    ]
    if not decodes:
        return schedule
    saving = {o: -profile.delta_for({g.ms[o]: {o: None}}) for o in decodes}
    k = max(min_targets, math.ceil(target_ratio * n))
    max_moves = max(min_moves, math.ceil(max_moves_ratio * n))
    lam_b, lam_f = bill_weight, 1.0 - bill_weight
    s_f, s_b = scales
    machines: dict[int, list[tuple[int, float, float]]] = {}
    moved = 0
    for o in sorted(decodes, key=lambda q: (-saving[q], q))[:k]:
        if moved >= max_moves:
            break
        m = g.ms[o]
        if m not in machines:
            machines[m] = [
                (q, starts[q], starts[q] + duration[q])
                for q in range(problem.operation_count)
                if g.ms[q] == m
            ]
        others = [item for item in machines[m] if item[0] != o]
        lo, hi = starts[o], t_end - duration[o]
        best_t, best_score = None, 0.0
        for t in prefill_times(problem, m, o, lo, hi, others):
            if close(t, starts[o], TOL.time) or not _fits_intervals(problem, m, o, t, others):
                continue
            d = profile.delta_for({m: {o: (t, t + duration[o])}})
            score = lam_b * (-d) / s_b - lam_f * (t - starts[o]) / s_f
            if score > best_score + 1e-12:
                best_t, best_score = t, score
        if best_t is None:
            continue
        profile.apply({m: {o: (best_t, best_t + duration[o])}})
        starts[o] = best_t
        machines[m] = others + [(o, best_t, best_t + duration[o])]
        moved += 1
    return Schedule(tuple(starts)) if moved else schedule


@dataclass
class DecoderRefiner:
    """Gateway ``refine`` callable: ``flow_safe`` (P2a) or ``weighted`` (P2b).

    ``scales`` stays ``None`` until frozen by the caller; a ``weighted`` refiner
    behaves like ``flow_safe`` before that and for class 0.
    """

    problem: ProblemInstance
    mode: str = "flow_safe"
    scales: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("flow_safe", "weighted"):
            raise ValueError("Unknown refine mode")

    def __call__(self, genotype: Genotype, schedule: Schedule, refine_class: int) -> Schedule:
        """Refine one decoded schedule deterministically for the given weight class."""
        refined = flow_safe_pass(self.problem, genotype, schedule)
        weight = CLASS_BILL_WEIGHT.get(refine_class)
        if self.mode == "weighted" and weight is not None and self.scales is not None:
            traded = tradeoff_pass(self.problem, genotype, refined, weight, self.scales)
            if traded is not refined:
                refined = flow_safe_pass(self.problem, genotype, traded)
        return refined


__all__ = [
    "CLASS_BILL_WEIGHT",
    "DecoderRefiner",
    "flow_safe_pass",
    "tradeoff_pass",
    "weight_class",
]
