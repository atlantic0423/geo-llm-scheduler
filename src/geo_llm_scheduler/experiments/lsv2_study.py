"""Four-arm LSV2 development study on the matched BILL/MAIN reference paths.

``BILL_RANDOM`` and ``MAIN_RANDOM`` replay the unchanged slot-assignment
references (plain permuted MOEA/D; strict-gate Q42 LS with two-focus Polish).
``LSV2`` keeps the BILL engine (same BILL4 initialization, variation and
replacement) and replaces the trigger-gated LS by the time-share LSV2 search;
``LSV2_NOGAP`` differs from ``LSV2`` only in its seed rule (Flow end 80/15/5).
"""

from __future__ import annotations

from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import ProblemInstance
from geo_llm_scheduler.engine.run import RunResult
from geo_llm_scheduler.experiments.rl23 import run_policy
from geo_llm_scheduler.experiments.rl23_support import study_config as reference_config
from geo_llm_scheduler.experiments.slot_assignment import run_arm as run_reference
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search

ARMS = ("BILL_RANDOM", "MAIN_RANDOM", "LSV2", "LSV2_NOGAP")
# Stress-panel study: LSV2_RANDSEL replaces the bandit by uniform operator choice.
PANEL_ARMS = ("BILL_RANDOM", "MAIN_RANDOM", "LSV2", "LSV2_RANDSEL")
REFERENCE_ARMS = ("BILL_RANDOM", "MAIN_RANDOM")
LSV2_ARMS = ("LSV2", "LSV2_NOGAP", "LSV2_RANDSEL")
ENGINE_POLICY = {
    "BILL_RANDOM": "BILL_PERMUTED",
    "MAIN_RANDOM": "STRICT_TWO_FOCUS",
    "LSV2": "BILL_PERMUTED",
    "LSV2_NOGAP": "BILL_PERMUTED",
    "LSV2_RANDSEL": "BILL_PERMUTED",
}
OPERATOR_POLICY = "STRICT_TWO_FOCUS"


def _check(arm: str) -> None:
    if arm not in ARMS + PANEL_ARMS:
        raise ValueError("Unknown LSV2 study arm")


def study_config(arm: str, seed: int, seconds: float) -> Config:
    """Engine configuration: LSV2 arms share the unchanged BILL_PERMUTED engine."""
    _check(arm)
    return reference_config(ENGINE_POLICY[arm], seed, seconds)


def operator_config(seed: int, seconds: float) -> Config:
    """Operator parameters (A3 tariff policy, A8 limits) of the MAIN reference."""
    return reference_config(OPERATOR_POLICY, seed, seconds)


def lsv2_config(arm: str) -> LSV2Config | None:
    """Declared LSV2 parameters; ``None`` for the two reference arms."""
    _check(arm)
    if arm not in LSV2_ARMS:
        return None
    if arm == "LSV2_RANDSEL":
        return LSV2Config(selector="uniform")
    return LSV2Config(seed_rule="gap" if arm == "LSV2" else "flow_end")


def run_arm(
    problem: ProblemInstance,
    arm: str,
    config: Config,
    evidence: dict,
    *,
    observer: Callable[[str, dict], None] | None = None,
    search_sink: Callable[[LSV2Search], None] | None = None,
) -> RunResult:
    """Run one arm; ``search_sink`` receives the LSV2 search object before the run."""
    _check(arm)
    if arm in REFERENCE_ARMS:
        return run_reference(problem, arm, config, evidence, observer=observer)
    ls = lsv2_config(arm)
    assert ls is not None
    search = LSV2Search(
        ls, config, operator_config(config.seed, config.seconds or 1.0), retain_steps=True
    )
    if search_sink is not None:
        search_sink(search)
    return run_policy(
        problem, ENGINE_POLICY[arm], config, evidence, observer=observer, interleave=search
    )


__all__ = [
    "ARMS",
    "LSV2_ARMS",
    "PANEL_ARMS",
    "REFERENCE_ARMS",
    "lsv2_config",
    "operator_config",
    "run_arm",
    "study_config",
]
