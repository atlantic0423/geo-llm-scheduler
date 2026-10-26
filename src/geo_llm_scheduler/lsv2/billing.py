"""Incremental Bill bookkeeping for scoring candidate moves; exact evaluation stays authoritative.

The profile mirrors the exact model: every instance draws idle power on
``[0, T_end)`` and active power on the union of its operation intervals; TOU is
the price integral of regional power divided by 3600, and Demand is each Region's
maximum fixed 900-second window mean times its rate. Scores computed here only
rank moves. Final candidates are always re-evaluated by ``evaluation.exact``.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping

from geo_llm_scheduler.domain.models import ProblemInstance

WINDOW = 900.0
Changes = Mapping[int, Mapping[int, "tuple[float, float] | None"]]


def merge(items: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the sorted disjoint union of half-open intervals."""
    out: list[list[float]] = []
    for a, b in sorted(items):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def difference(
    old: list[tuple[float, float]], new: list[tuple[float, float]]
) -> list[tuple[float, float, int]]:
    """Signed segments where membership changes: +1 newly active, -1 newly idle."""
    points = sorted({t for a, b in old for t in (a, b)} | {t for a, b in new for t in (a, b)})
    result: list[tuple[float, float, int]] = []
    i = j = 0
    for a, b in zip(points, points[1:]):
        mid = (a + b) / 2
        while i < len(old) and old[i][1] <= mid:
            i += 1
        while j < len(new) and new[j][1] <= mid:
            j += 1
        inside_old = i < len(old) and old[i][0] <= mid
        inside_new = j < len(new) and new[j][0] <= mid
        if inside_old != inside_new:
            sign = 1 if inside_new else -1
            if result and result[-1][2] == sign and math.isclose(result[-1][1], a):
                result[-1] = (result[-1][0], b, sign)
            else:
                result.append((a, b, sign))
    return result


class PowerProfile:
    """Mutable per-call bookkeeping of active unions, TOU and regional window means."""

    def __init__(
        self,
        problem: ProblemInstance,
        machines: tuple[int, ...] | list[int],
        starts: tuple[float, ...] | list[float],
        horizon: float | None = None,
    ) -> None:
        self.problem = problem
        self.items: list[dict[int, tuple[float, float]]] = [{} for _ in problem.instances]
        for o, s in enumerate(starts):
            if s >= 0:
                self.items[machines[o]][o] = (s, s + problem.profile(o).duration)
        if horizon is None:
            horizon = max(
                (s + problem.profile(o).duration for o, s in enumerate(starts) if s >= 0 and o % 2),
                default=0.0,
            )
        self.horizon = horizon
        self._tariffs = [
            (tuple(t.start for t in r.tariffs), tuple(t.end for t in r.tariffs), r.tariffs)
            for r in problem.regions
        ]
        self.unions = [merge(list(b.values())) for b in self.items]
        self._rebuild()

    # -------- exact-shaped totals --------
    def _price_integral(self, region: int, a: float, b: float) -> float:
        starts, ends, tariffs = self._tariffs[region]
        total = 0.0
        i = max(0, bisect_right(starts, a) - 1)
        while i < len(tariffs) and tariffs[i].start < b:
            t = tariffs[i]
            total += max(0.0, min(b, t.end) - max(a, t.start)) * t.price
            i += 1
        return total

    def _add_segment(
        self, region: int, a: float, b: float, kw: float, windows: list[float]
    ) -> float:
        """Add constant power kw on [a,b) clipped to the horizon; return its TOU cost."""
        a, b = max(0.0, a), min(b, self.horizon)
        if b <= a or kw == 0:
            return 0.0
        first, last = int(a // WINDOW), min(len(windows), math.ceil(b / WINDOW))
        for w in range(first, last):
            overlap = min(b, (w + 1) * WINDOW) - max(a, w * WINDOW)
            if overlap > 0:
                windows[w] += kw * overlap / WINDOW
        return kw * self._price_integral(region, a, b) / 3600

    def _rebuild(self) -> None:
        count = math.ceil(self.horizon / WINDOW)
        self.windows = [[0.0] * count for _ in self.problem.regions]
        self.tou = 0.0
        for m, server in enumerate(self.problem.instances):
            r = server.region
            self.tou += self._add_segment(r, 0.0, self.horizon, server.idle_kw, self.windows[r])
            for a, b in self.unions[m]:
                self.tou += self._add_segment(
                    r, a, b, server.active_kw - server.idle_kw, self.windows[r]
                )

    def demand(self) -> float:
        return sum(
            region.demand_rate * max(w, default=0.0)
            for region, w in zip(self.problem.regions, self.windows)
        )

    def bill(self) -> float:
        return self.tou + self.demand()

    # -------- moves --------
    def delta_for(self, changes: Changes) -> float:
        """Bill change for {machine: {operation: new interval or None}}, horizon fixed."""
        tou = 0.0
        touched = {self.problem.instances[m].region for m in changes}
        windows = {r: (list(w) if r in touched else w) for r, w in enumerate(self.windows)}
        for m, edits in changes.items():
            server = self.problem.instances[m]
            bucket = dict(self.items[m])
            for o, interval in edits.items():
                if interval is None:
                    bucket.pop(o, None)
                else:
                    bucket[o] = interval
            new_union = merge(list(bucket.values()))
            kw = server.active_kw - server.idle_kw
            for a, b, sign in difference(self.unions[m], new_union):
                tou += self._add_segment(server.region, a, b, sign * kw, windows[server.region])
        demand = sum(
            region.demand_rate * max(windows[r], default=0.0)
            for r, region in enumerate(self.problem.regions)
        )
        return tou + demand - self.demand()

    def apply(self, changes: Changes) -> None:
        """Commit edits; rebuilds only the touched machines' unions and window sums."""
        for m, edits in changes.items():
            server = self.problem.instances[m]
            for o, interval in edits.items():
                if interval is None:
                    self.items[m].pop(o, None)
                else:
                    self.items[m][o] = interval
            new_union = merge(list(self.items[m].values()))
            kw = server.active_kw - server.idle_kw
            for a, b, sign in difference(self.unions[m], new_union):
                self.tou += self._add_segment(
                    server.region, a, b, sign * kw, self.windows[server.region]
                )
            self.unions[m] = new_union

    def copy(self) -> PowerProfile:
        """Independent bookkeeping copy sharing only immutable problem data."""
        clone = PowerProfile.__new__(PowerProfile)
        clone.problem, clone._tariffs, clone.horizon = self.problem, self._tariffs, self.horizon
        clone.items = [dict(b) for b in self.items]
        clone.unions = [list(u) for u in self.unions]
        clone.windows = [list(w) for w in self.windows]
        clone.tou = self.tou
        return clone

    def with_horizon(self, horizon: float) -> PowerProfile:
        """Return a copy re-integrated over a different batch end (used when T_end moves)."""
        clone = PowerProfile.__new__(PowerProfile)
        clone.problem, clone._tariffs = self.problem, self._tariffs
        clone.items = [dict(b) for b in self.items]
        clone.unions = [list(u) for u in self.unions]
        clone.horizon = horizon
        clone._rebuild()
        return clone
