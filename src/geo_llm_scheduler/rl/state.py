"""Fixed 3 x 7 x 2 state encoding and inverse-weight reference directions."""

import math

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.utils.numeric import EPS_DIRECTION

PREFERENCE_LABELS = ("Flow", "Balanced", "Electricity")
CONDITION_LABELS = ("Normal", "Resource", "KV", "Region", "TOU", "Demand", "Compressible")
PROGRESS_LABELS = ("Improving", "Stagnating")


def preference(index: int, count: int) -> int:
    """Low/middle/high Flow-weight thirds map to Electricity/Balanced/Flow."""
    return 2 - min(2, 3 * index // count)


def encode(preference_id: int, dominant: int, stagnant: bool) -> int:
    """Encode Flow/Balanced/Electricity x Normal+six conditions x progress."""
    if preference_id not in range(3) or dominant not in range(7):
        raise ValueError("Invalid state component")
    return (preference_id * 7 + dominant) * 2 + int(stagnant)


def decode(state: int) -> tuple[str, str, str]:
    """Return the three stable labels represented by a 42-state identifier."""
    if state not in range(42):
        raise ValueError("State identifier outside the 42-state space")
    preference_id, remainder = divmod(state, 14)
    dominant, stagnant = divmod(remainder, 2)
    return (
        PREFERENCE_LABELS[preference_id],
        CONDITION_LABELS[dominant],
        PROGRESS_LABELS[stagnant],
    )


def extract(index: int, values: tuple[float, ...], stagnant_count: int, config: Config) -> int:
    """Select largest relative threshold violation; fixed-order severity ties."""
    ratios = [v / t for v, t in zip(values, config.severity_thresholds)]
    dominant = 0 if max(ratios, default=0) <= 1 else 1 + max(range(6), key=lambda k: ratios[k])
    return encode(
        preference(index, config.population),
        dominant,
        stagnant_count >= config.stagnation_threshold,
    )


def direction(weight: tuple[float, float]) -> tuple[float, float]:
    """Unit reciprocal Tchebycheff ray with endpoint zero-weight protection."""
    a, b = (1 / max(w, EPS_DIRECTION) for w in weight)
    length = math.hypot(a, b)
    return a / length, b / length


def associate(vector: tuple[float, float], weights: tuple[tuple[float, float], ...]) -> int:
    """Closest angular direction; index deterministically resolves exact ties."""
    return max(
        range(len(weights)), key=lambda j: sum(a * b for a, b in zip(vector, direction(weights[j])))
    )
