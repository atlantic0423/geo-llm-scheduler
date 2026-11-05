"""Initial subproblem-slot assignment study on the matched BILL and MAIN_Q references.

Four arms cross two unchanged reference algorithms with two slot policies.
RANDOM keeps the original shuffled correspondence. SORTED only permutes the
already evaluated BILL4 members so that ascending Flow meets descending Flow
weight; initializer draws, members, budgets and every search rule are unchanged.
"""

from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.run import RunResult
from geo_llm_scheduler.experiments.rl23 import run_policy
from geo_llm_scheduler.experiments.rl23_support import study_config as reference_config

ARMS = ("BILL_RANDOM", "BILL_SORTED", "MAIN_RANDOM", "MAIN_SORTED")
REFERENCES = {"BILL": "BILL_PERMUTED", "MAIN": "STRICT_TWO_FOCUS"}


def reference_policy(arm: str) -> str:
    """Map a study arm to its frozen RL23 reference policy name."""
    if arm not in ARMS:
        raise ValueError("Unknown slot-assignment arm")
    return REFERENCES[arm.split("_")[0]]


def slot_policy(arm: str) -> str:
    """Return ``random`` or ``sorted`` for a declared arm."""
    reference_policy(arm)
    return arm.split("_")[1].lower()


def study_config(arm: str, seed: int, seconds: float) -> Config:
    """Both slot policies share the unchanged reference configuration."""
    return reference_config(reference_policy(arm), seed, seconds)


def direction_order(members: list[Candidate]) -> list[Candidate]:
    """Place members by ascending (Flow, Bill) onto slots of descending Flow weight.

    Slot ``j`` carries ``lambda=(j/(N-1), 1-j/(N-1))``, so the last slot is the
    Flow-preferring subproblem. Exact objective ties are ordered by the stable
    sort over the initializer's shuffled list, so the result is deterministic
    per seed.
    """
    ranked = sorted(members, key=lambda c: c.evaluation.objectives)
    return ranked[::-1]


def run_arm(
    problem: ProblemInstance,
    arm: str,
    config: Config,
    evidence: dict,
    *,
    observer: Callable[[str, dict], None] | None = None,
) -> RunResult:
    """Run the reference path, inserting only the declared initial slot permutation."""
    order = direction_order if slot_policy(arm) == "sorted" else None
    return run_policy(
        problem, reference_policy(arm), config, evidence, observer=observer, initial_order=order
    )


__all__ = ["ARMS", "direction_order", "reference_policy", "run_arm", "slot_policy", "study_config"]
