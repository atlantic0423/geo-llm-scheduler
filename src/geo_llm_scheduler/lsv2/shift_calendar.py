"""Conservative forbidden displacement intervals for a rigid moving group.

Only certified infeasible interiors are rejected. Endpoints, numeric uncertainty
and every remaining displacement use the unchanged reference feasibility test.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence

from geo_llm_scheduler.domain.models import ProblemInstance
from geo_llm_scheduler.utils.numeric import TOL, less

Interval = tuple[int, float, float]
Segment = tuple[float, float, tuple[int, ...]]


def _segments(items: Sequence[Interval]) -> list[Segment]:
    events: dict[float, tuple[list[int], list[int]]] = {}
    for operation, start, end in items:
        if start < end:
            events.setdefault(start, ([], []))[1].append(operation)
            events.setdefault(end, ([], []))[0].append(operation)
    active: set[int] = set()
    result = []
    points = sorted(events)
    for start, end in zip(points, points[1:]):
        ends, begins = events[start]
        active.difference_update(ends)
        active.update(begins)
        if active:
            result.append((start, end, tuple(sorted(active))))
    return result


class ShiftCalendar:
    """Build a frozen load profile and rigid-group forbidden shift interiors.

    Pairing a moving segment [a,b) with a fixed segment [c,d) yields the open
    overlap interval (c-b,d-a). A conflicting resource sum certifies rejection
    there. Shrink by a floating-point roundoff guard before taking the union;
    boundaries always fall through to the legacy test, preserving half-open ties.
    """

    def __init__(
        self,
        problem: ProblemInstance,
        machine: int,
        moving: Sequence[Interval],
        fixed: Sequence[Interval],
        upper: float,
    ) -> None:
        self.pairs = 0
        forbidden = []
        scale = max([abs(upper), 1.0, *(abs(t) for _, a, b in (*moving, *fixed) for t in (a, b))])
        # Representation guard, not a new physical feasibility tolerance.
        self.guard = 32 * math.ulp(4 * scale)

        def loads(items: Sequence[Interval]) -> list[tuple[float, float, float, float, int]]:
            return [
                (
                    a,
                    b,
                    math.fsum(problem.profile(o).compute for o in active),
                    math.fsum(problem.profile(o).vram for o in active),
                    len(active),
                )
                for a, b, active in _segments(items)
            ]

        fixed_segments = loads(fixed)
        capacity = problem.instances[machine].vram
        for a, b, gc, gv, gn in loads(moving):
            if b - a <= 2 * self.guard:
                continue
            for c, d, fc, fv, fn in fixed_segments:
                low, high = max(0.0, c - b), min(upper, d - a)
                if high - low <= 2 * self.guard or d - c <= 2 * self.guard:
                    continue
                self.pairs += 1
                # Nonnegative demands: a conservative lower bound covers any
                # reference p + sum(other operations) rounding order. If close
                # to the physical tolerance, leave the decision to reference.
                count = gn + fn + 1
                compute, vram = gc + fc, gv + fv
                compute_lower = compute - 8 * count * math.ulp(max(1.0, compute))
                vram_lower = vram - 8 * count * math.ulp(max(1.0, capacity, vram))
                conflict = less(1.0, compute_lower, TOL.compute) or less(
                    capacity, vram_lower, TOL.vram
                )
                if conflict:
                    forbidden.append((low + self.guard, high - self.guard))
        merged: list[tuple[float, float]] = []
        for low, high in sorted(forbidden):
            # Do not close a feasible point at exactly touching open intervals.
            if merged and low < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
            else:
                merged.append((low, high))
        self.forbidden = tuple(merged)
        self._left = tuple(low for low, _ in merged)

    def rejects(self, delta: float) -> bool:
        """Return true only for a certified capacity-conflicting interior."""
        i = bisect_right(self._left, delta) - 1
        return i >= 0 and self.forbidden[i][0] < delta < self.forbidden[i][1]
