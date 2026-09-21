"""Two-objective quality metrics with explicit externally fixed reference data."""

import math


def hypervolume(points: list[tuple[float, float]], reference: tuple[float, float]) -> float:
    """Exact union area of minimizing rectangles bounded by a fixed reference point."""
    area = 0.0
    best_y = reference[1]
    for x, y in sorted(set(points)):
        if x > reference[0] or y > reference[1]:
            raise ValueError("HV reference must weakly dominate every measured point")
        if y < best_y:
            area += (reference[0] - x) * (best_y - y)
            best_y = y
    return area


def igd_plus(
    points: list[tuple[float, float]], reference_front: list[tuple[float, float]]
) -> float:
    """Mean modified distance from a fixed reference front to approximation points."""
    if not reference_front:
        raise ValueError("IGD+ requires a nonempty explicit reference front")
    if not points:
        return math.inf
    return sum(
        min(math.hypot(max(x - rx, 0), max(y - ry, 0)) for x, y in points)
        for rx, ry in reference_front
    ) / len(reference_front)
