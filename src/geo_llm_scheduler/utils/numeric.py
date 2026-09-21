"""Central unit-aware comparisons and stable phenotype identity."""

import math
from dataclasses import dataclass

from geo_llm_scheduler.domain.models import Candidate


@dataclass(frozen=True)
class Tolerances:
    """Coding Contract v1 numerical tolerances, not tuning parameters."""

    relative: float = 1e-9
    time: float = 1e-9
    compute: float = 1e-9
    vram: float = 1e-9
    power: float = 1e-9
    cost: float = 1e-9


TOL = Tolerances()
# Numerical guards are distinct from physical feasibility tolerances.
EPS_NORM = 1e-9
EPS_RATIO = 1e-9
EPS_DIRECTION = 1e-6


def close(a: float, b: float, absolute: float) -> bool:
    """Compare using the relevant physical unit's absolute tolerance."""
    return math.isclose(a, b, rel_tol=TOL.relative, abs_tol=absolute)


def less(a: float, b: float, absolute: float) -> bool:
    """Strict improvement beyond numerical tolerance."""
    return a < b and not close(a, b, absolute)


def identity(candidate: Candidate) -> tuple:
    """Canonical MS/OS/timing identity; intentionally includes phenotype."""
    return (
        candidate.genotype.ms,
        candidate.genotype.os,
        tuple(round(t / TOL.time) for t in candidate.schedule.starts),
    )
