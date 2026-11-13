"""Front-gap reproduction allocation for MOEA/D (research arms A1/A0, default off).

Each generation the engine asks for the subproblems to visit. Instead of the
shuffled full traversal, the allocator samples ``population`` slots with
replacement. A ray of weight ``(lambda_F, lambda_B)`` points, in the box-normalized
front plane, along ``f_B / f_F = lambda_F / lambda_B``; it is assigned to the
Archive front gap (box between adjacent nondominated points) it crosses. Each
gap's unattained box area is split among its rays whose incumbent genotype is not
a duplicate of an earlier subproblem's; a gap no ray crosses gives its area to the
nearest ray; the two end rays get ``endpoint_bonus`` times the largest gap area.
Weights are discounted by ``discount ** min(k, cap)`` where ``k`` counts the
subproblem's consecutive slots whose child did not enter the Archive, then mixed
with a uniform ``floor``; no subproblem may exceed ``max_share`` of the slots
(excess is redistributed proportionally to the others). Mode ``gap`` (A1) samples these probabilities;
``shuffled`` (A0) permutes the same probability vector across subproblems first,
matching the non-uniformity while breaking the link to gaps. Draws use the
dedicated ``gap_allocation`` stream; the work is charged to the run clock.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.utils.rng import RNGManager


@dataclass(frozen=True)
class GapOptions:
    """Frozen research settings (provisional values, see open questions)."""

    mode: str = "gap"  # or "shuffled"
    floor: float = 0.2
    discount: float = 0.5
    cap: int = 10
    endpoint_bonus: float = 1.0
    max_share: float = 0.05

    def __post_init__(self) -> None:
        if self.mode not in ("gap", "shuffled"):
            raise ValueError("Gap allocation mode must be gap or shuffled")
        if not (
            0 <= self.floor <= 1 and 0 < self.discount <= 1 and self.cap >= 0 and self.max_share > 0
        ):
            raise ValueError("Invalid gap allocation settings")


def capped(p: list[float], limit: float) -> list[float]:
    """Cap probabilities at ``limit`` and redistribute the excess proportionally."""
    p = list(p)
    if limit * len(p) < 1:
        return [1 / len(p)] * len(p)
    for _ in range(len(p)):
        over = [i for i, x in enumerate(p) if x > limit]
        if not over:
            break
        excess = sum(p[i] - limit for i in over)
        for i in over:
            p[i] = limit
        free = [i for i, x in enumerate(p) if x < limit]
        room = sum(p[i] for i in free)
        for i in free:
            p[i] += excess * (p[i] / room if room > 0 else 1 / len(free))
    return p


def front_points(members: list[Candidate]) -> list[tuple[float, float]]:
    """Distinct nondominated objective vectors sorted by Flow (Bill decreasing)."""
    out: list[tuple[float, float]] = []
    for p in sorted({m.evaluation.objectives for m in members}):
        if not out or p[1] < out[-1][1]:
            out.append(p)
    return out


def ray_weights(
    front: list[tuple[float, float]],
    lambdas: tuple[tuple[float, float], ...],
    duplicate: list[bool],
    endpoint_bonus: float = 1.0,
) -> list[float]:
    """Undiscounted gap weight per subproblem ray (see module docstring)."""
    n = len(lambdas)
    if len(front) < 2:
        return [1.0] * n
    lo = [min(p[k] for p in front) for k in (0, 1)]
    span = [max(max(p[k] for p in front) - lo[k], 1e-12) for k in (0, 1)]
    q = [((p[0] - lo[0]) / span[0], (p[1] - lo[1]) / span[1]) for p in front]
    angle = [math.atan2(y, x) for x, y in q]  # decreasing along the front
    gaps = [(b[0] - a[0]) * (a[1] - b[1]) for a, b in zip(q, q[1:])]
    rays = [math.atan2(lam[0], lam[1]) for lam in lambdas]
    members: list[list[int]] = [[] for _ in gaps]
    ends: dict[int, list[int]] = {0: [], 1: []}
    for i, r in enumerate(rays):
        if r >= angle[0]:
            ends[0].append(i)
        elif r <= angle[-1]:
            ends[1].append(i)
        else:
            k = next(k for k in range(len(gaps)) if angle[k + 1] < r <= angle[k])
            members[k].append(i)
    w = [0.0] * n

    def share(idx: list[int], amount: float) -> None:
        keep = [i for i in idx if not duplicate[i]] or idx
        for i in keep:
            w[i] += amount / len(keep)

    for k, g in enumerate(gaps):
        if members[k]:
            share(members[k], g)
        else:
            mid = (angle[k] + angle[k + 1]) / 2
            share([min(range(n), key=lambda i: (abs(rays[i] - mid), i))], g)
    top = max(gaps) if gaps else 0.0
    for idx in ends.values():
        if idx:
            share(idx, endpoint_bonus * top)
    return w


class GapAllocator:
    """Engine ``traversal`` + ``post_replacement`` seams implementing A1/A0."""

    def __init__(self, options: GapOptions) -> None:
        """Bind frozen options; failure counters start at zero on first use."""
        self.options = options
        self.failures: list[int] = []
        self._issued: list[int] = []
        self._pos = 0
        self.counts = dict(generations=0, fallback=0, slots=0, archive_entries=0)
        self.max_probability: list[float] = []

    def order(
        self,
        generation: int,
        population: list[Candidate],
        lambdas: tuple[tuple[float, float], ...],
        gateway: EvaluationGateway,
        streams: RNGManager,
    ) -> list[int]:
        """Sample this generation's subproblem slots."""
        rng = streams.stream("gap_allocation")
        n = len(population)
        if not self.failures:
            self.failures = [0] * n
        seen: set = set()
        duplicate = []
        for c in population:
            key = (c.genotype.ms, c.genotype.os)
            duplicate.append(key in seen)
            seen.add(key)
        front = front_points(gateway.archive.members)
        if len(front) < 2:
            self.counts["fallback"] += 1
        base = ray_weights(front, lambdas, duplicate, self.options.endpoint_bonus)
        o = self.options
        w = [b * o.discount ** min(k, o.cap) for b, k in zip(base, self.failures)]
        total = sum(w)
        p = [(1 - o.floor) * (x / total if total > 0 else 1 / n) + o.floor / n for x in w]
        p = capped(p, o.max_share)
        if o.mode == "shuffled":
            rng.shuffle(p)
        self._issued = rng.choices(range(n), weights=p, k=n)
        self._pos = 0
        self.counts["generations"] += 1
        self.max_probability.append(max(p))
        return list(self._issued)

    def __call__(
        self,
        population: list[Candidate],
        child: Candidate,
        order: tuple[int, ...],
        lambdas: tuple[tuple[float, float], ...],
        gateway: EvaluationGateway,
        streams: RNGManager,
    ) -> None:
        """After one slot: reset or increase that subproblem's failure counter."""
        i = self._issued[self._pos]
        self._pos += 1
        entered = any(m is child for m in gateway.archive.members)
        self.failures[i] = 0 if entered else self.failures[i] + 1
        self.counts["slots"] += 1
        self.counts["archive_entries"] += entered

    def record(self) -> dict:
        """Settings, counters and the per-generation maximum slot probability."""
        mp = self.max_probability
        return dict(
            options=asdict(self.options),
            counts=dict(self.counts),
            max_probability_mean=sum(mp) / len(mp) if mp else None,
        )
