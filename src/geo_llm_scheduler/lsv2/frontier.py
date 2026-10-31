"""Prototype 1: frontier-gap seeding with a frozen normalization frame.

All quantities use only information available online: the current Archive's own
ideal/nadir define a frame that is frozen for a search window, HV uses the fixed
reference point (1.1, 1.1), and a point's gap score is the unattained area of the
boxes to its two neighbours on the non-dominated front. Endpoints get a bonus and
repeatedly unproductive seeds are discounted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from geo_llm_scheduler.domain.models import Candidate

REF = 1.1


def nondominated(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sorted unique two-objective minimization front."""
    out, best = [], float("inf")
    for x, y in sorted(set(points)):
        if y < best:
            out.append((x, y))
            best = y
    return out


@dataclass(frozen=True)
class FrozenFrame:
    """Min-max normalization fixed for one window; ranges are protected from zero."""

    ideal: tuple[float, float]
    nadir: tuple[float, float]

    @classmethod
    def of(cls, points: list[tuple[float, float]]) -> FrozenFrame:
        front = nondominated(points)
        lo = (min(p[0] for p in front), min(p[1] for p in front))
        hi = (max(p[0] for p in front), max(p[1] for p in front))
        return cls(lo, hi)

    def normalize(self, p: tuple[float, float]) -> tuple[float, float]:
        return tuple(
            (p[k] - self.ideal[k]) / max(self.nadir[k] - self.ideal[k], 1e-12) for k in range(2)
        )  # type: ignore[return-value]

    def hv(self, points: list[tuple[float, float]]) -> float:
        """HV of raw objective points inside the fixed reference box."""
        area, best = 0.0, REF
        for x, y in nondominated([self.normalize(p) for p in points]):
            if x <= REF and y <= REF and y < best:
                area += (REF - x) * (best - y)
                best = y
        return area

    def hv_gain(self, front: list[tuple[float, float]], new: list[tuple[float, float]]) -> float:
        return self.hv(front + new) - self.hv(front)

    def weight_for(self, p: tuple[float, float]) -> tuple[float, float]:
        """Tchebycheff weight whose ray passes through p (lambda_F z_F = lambda_B z_B)."""
        zf, zb = (max(v, 1e-6) for v in self.normalize(p))
        return zb / (zf + zb), zf / (zf + zb)


def gap_scores(front_norm: list[tuple[float, float]], end_bonus: float = 1.0) -> list[float]:
    """Half the unattained box area to each neighbour; endpoints add a share of the largest gap."""
    k = len(front_norm)
    if k == 0:
        return []
    boxes = [
        (front_norm[i + 1][0] - front_norm[i][0]) * (front_norm[i][1] - front_norm[i + 1][1])
        for i in range(k - 1)
    ]
    scores = []
    for i in range(k):
        left = boxes[i - 1] if i > 0 else 0.0
        right = boxes[i] if i < k - 1 else 0.0
        scores.append((left + right) / 2)
    top = max(scores, default=0.0) or 1e-6
    scores[0] += end_bonus * top
    scores[-1] += end_bonus * top
    return scores


@dataclass
class SeedSampler:
    """Choose LS seeds from Archive front members by an explicit rule and RNG.

    Rules: ``uniform``; ``flow_end`` reproducing the observed current allocation
    (most seeds from the Flow third); ``gap`` proportional to gap scores with
    stagnation discount.
    """

    rule: str
    rng: random.Random
    discount: float = 0.5
    shares: tuple[float, float, float] = (0.8, 0.15, 0.05)
    failures: dict[tuple[float, float], int] = field(default_factory=dict)

    def pick(self, frame: FrozenFrame, front: list[tuple[float, float]]) -> int:
        """Return an index into the raw front list (sorted by Flow)."""
        if not front:
            raise ValueError("Empty front")
        if self.rule == "uniform":
            return self.rng.randrange(len(front))
        norm = [frame.normalize(p) for p in front]
        if self.rule == "flow_end":
            thirds = [
                [i for i, q in enumerate(norm) if lo <= q[0] < hi]
                for lo, hi in ((-1e9, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1e9))
            ]
            weights = [s if t else 0.0 for s, t in zip(self.shares, thirds)]
            seg = self.rng.choices(range(3), weights=weights)[0]
            return self.rng.choice(thirds[seg])
        if self.rule == "gap":
            scores = gap_scores(norm)
            weights = [
                s * (self.discount ** self.failures.get(front[i], 0)) + 1e-12
                for i, s in enumerate(scores)
            ]
            return self.rng.choices(range(len(front)), weights=weights)[0]
        raise ValueError("Unknown seed rule")

    def report(self, point: tuple[float, float], productive: bool) -> None:
        """Discount seeds that produced nothing; reset on success."""
        if productive:
            self.failures.pop(point, None)
        else:
            self.failures[point] = self.failures.get(point, 0) + 1


def demand_gate(candidate: Candidate, threshold: float = 0.15) -> bool:
    """A8 availability: some Region's Demand Charge is at least ``threshold`` of the Bill."""
    e = candidate.evaluation
    bill = e.bill
    return bill > 0 and max(e.demand, default=0.0) / bill >= threshold


__all__ = ["FrozenFrame", "SeedSampler", "demand_gate", "gap_scores", "nondominated"]
