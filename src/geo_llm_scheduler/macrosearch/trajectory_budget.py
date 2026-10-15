"""Explicit research arms and causal, episode-local breadth/depth allocation."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ArmSpec:
    """A bounded trajectory recipe, separate from the adopted algorithm defaults."""

    width: int = 6
    steps: int = 10
    credit: int = 30
    walk: bool = True
    best: bool = True
    metric: str = "max"
    window: int = 2
    signal: str = "endpoint"
    dynamic: bool = False
    reverse: bool = False
    random_budget: bool = False
    aid: bool = False


ARM_SPECS = {
    "S0": ArmSpec(steps=5, walk=False, best=False),
    "S1": ArmSpec(steps=5, walk=False),
    "W0": ArmSpec(steps=5, best=False),
    "W1": ArmSpec(steps=5),
    "F1": ArmSpec(width=3),
    "F2": ArmSpec(width=10, steps=3),
    "D1": ArmSpec(metric="mean", dynamic=True),
    "D2": ArmSpec(dynamic=True),
    "D3": ArmSpec(signal="efficiency", dynamic=True),
    "D4": ArmSpec(metric="action", dynamic=True),
    "D5": ArmSpec(window=1, dynamic=True),
    "D6": ArmSpec(dynamic=True, reverse=True),
    "R0": ArmSpec(random_budget=True),
    "C0": ArmSpec(steps=5, dynamic=True),
    "L0": ArmSpec(steps=10, credit=60),
    "L1": ArmSpec(steps=20, credit=60, dynamic=True),
    "A0": ArmSpec(width=10, aid=True),
    "A1": ArmSpec(dynamic=True, aid=True),
}


def allocate(
    spec: ArmSpec,
    signal: float,
    step: int,
    remaining: int,
    cutoffs: tuple[float, float],
    probabilities: tuple[float, ...],
    rng: random.Random,
) -> int:
    """Choose breadth using only past distance; truncate at the remaining exact credit."""
    if not 0 <= signal <= 1 or remaining < 1 or step < 0:
        raise ValueError("Invalid causal signal, remaining credit or step")
    width = spec.width
    if spec.dynamic:
        rank = 0 if signal <= cutoffs[0] else 1 if signal <= cutoffs[1] else 2
        if spec.reverse:
            rank = 2 - rank
        width = 6 if step == 0 else (3, 6, 10)[rank]
    elif spec.random_budget:
        width = 6 if step == 0 else rng.choices((3, 6, 10), weights=probabilities, k=1)[0]
    return min(width, remaining)
