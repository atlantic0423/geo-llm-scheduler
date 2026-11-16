"""Front-segment isolation for MOEA/D (research arms S3/I3, default off).

Diagnosis (local ``outputs/variance_decomp_20261114``): within one GS4 run every
final-front point shares essentially one region assignment, while different seeds
end in nearly unrelated assignments, and the union of several seeds' fronts
dominates any single run. The neighborhood topology lets one assignment spread
along the whole weight chain.

The topology splits the ``population`` subproblems into ``segments`` groups. Each
subproblem's mating/replacement neighborhood is its ``size`` Euclidean-nearest
weights inside its own group (stable index ties, as ``moead.core.neighborhoods``),
and the global mating pool is its group. Mode ``contiguous`` (S3) uses consecutive
weight blocks, so each front segment may keep its own structure; ``interleaved``
(I3, matched control) uses ``i mod segments``, isolating equally many equally
sized groups that each span the whole front. The same neighborhoods are used for
LS reflow. The archive, normalization and LS seeding stay global. No random
numbers are drawn.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentOptions:
    """Frozen research settings (provisional values, see open questions)."""

    mode: str = "contiguous"  # or "interleaved"
    segments: int = 3

    def __post_init__(self) -> None:
        if self.mode not in ("contiguous", "interleaved"):
            raise ValueError("Segment mode must be contiguous or interleaved")
        if self.segments < 1:
            raise ValueError("At least one segment")


def groups(n: int, options: SegmentOptions) -> list[tuple[int, ...]]:
    """Subproblem index groups; contiguous blocks differ in size by at most one."""
    k = options.segments
    if n < 2 * k:
        raise ValueError("Each segment needs at least two subproblems")
    if options.mode == "interleaved":
        return [tuple(range(g, n, k)) for g in range(k)]
    bounds = [round(g * n / k) for g in range(k + 1)]
    return [tuple(range(bounds[g], bounds[g + 1])) for g in range(k)]


def segmented(
    lambdas: tuple[tuple[float, float], ...], size: int, options: SegmentOptions
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Per-subproblem in-group neighborhoods and global mating pools."""
    neighbors: list[tuple[int, ...]] = [()] * len(lambdas)
    pools: list[tuple[int, ...]] = [()] * len(lambdas)
    for group in groups(len(lambdas), options):
        for i in group:
            w = lambdas[i]
            neighbors[i] = tuple(
                sorted(group, key=lambda j: (sum((a - b) ** 2 for a, b in zip(w, lambdas[j])), j))[
                    : min(size, len(group))
                ]
            )
            pools[i] = group
    return tuple(neighbors), tuple(pools)


class SegmentTopology:
    """Engine ``topology`` seam; also exposes the neighborhoods for LS reflow."""

    def __init__(self, options: SegmentOptions) -> None:
        """Bind frozen options; neighborhoods are built on the first call."""
        self.options = options
        self.neighbors: tuple[tuple[int, ...], ...] = ()

    def __call__(
        self, lambdas: tuple[tuple[float, float], ...], size: int
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        """Return (neighborhoods, global pools) and remember the neighborhoods."""
        self.neighbors, pools = segmented(lambdas, size, self.options)
        return self.neighbors, pools
