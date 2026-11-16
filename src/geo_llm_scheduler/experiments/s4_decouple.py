"""Decouple the external right shift from MOEA/D replacement (development screen).

Motivation (container batch 2026-11-06, load-dependent power): full S4 lost to
S4 without right shift, while structural offspring entered the Archive about
85% less often. A shifted schedule is not reproducible from its genotype, so
when it is reflowed into the MOEA/D population its subproblem can no longer be
improved by decoded offspring. Two treatment arms remove that write-back:

``B1_ARCHIVE_SHIFT``
    Full S4 with every right-shift entrant kept in the Archive only; population
    replacement by right-shift entrants is disabled. Everything else, including
    the interleaved polish after each Q episode, equals ``A0_FULL_S4``.
``B2_TERMINAL_SHIFT``
    S4 without right shift (``A1_NO_SHIFT``) until the inclusive clock enters
    the final ``TERMINAL_SHARE`` of the budget; from then on, before each
    offspring slot, every Archive front schedule is right-shifted with the
    same FAST COMMON neighborhood until certified empty, Archive only.
``B3_ARCHIVE_TERMINAL``
    ``B1`` plus the terminal front sweep of ``B2`` (the complete decoupled S4).

The control ``A1_NO_SHIFT`` is delegated unchanged to :mod:`s4_ablation`.
"""

from collections.abc import Callable
from dataclasses import asdict

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import ProblemInstance
from geo_llm_scheduler.engine.run import RunResult
from geo_llm_scheduler.experiments import s4_ablation
from geo_llm_scheduler.experiments.rl23 import run_policy
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search

CONTROL = "A1_NO_SHIFT"
FULL = "A0_FULL_S4"
ARCHIVE = "B1_ARCHIVE_SHIFT"
TERMINAL = "B2_TERMINAL_SHIFT"
BOTH = "B3_ARCHIVE_TERMINAL"
EXTRA_ARMS = (ARCHIVE, TERMINAL, BOTH)
TERMINAL_SHARE = 0.05
PROTOCOL = "s4-shift-decoupling/v1"


def _check(arm: str) -> None:
    if arm not in EXTRA_ARMS:
        raise ValueError("Unknown shift-decoupling arm")


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Return the unchanged S4 engine and inclusive wall-clock configuration."""
    _check(arm)
    return s4_ablation.arm_config(FULL, seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """Unchanged full-S4 search fields; arms differ only in :func:`decoupling`."""
    _check(arm)
    return s4_ablation.search_config(FULL)


def decoupling(arm: str) -> dict:
    """The seam settings that distinguish each arm from the full-S4 reference path."""
    _check(arm)
    return dict(
        polish_reflow=False,
        polish_after_episode=arm != TERMINAL,
        terminal_share=TERMINAL_SHARE if arm in (TERMINAL, BOTH) else None,
    )


def make_search(arm: str, config: Config, calibration: dict) -> LSV2Search:
    """Build the same empty online Q learner as S4, then set the decoupling seam."""
    search = s4_ablation.make_search(FULL, config, calibration)
    search.config = search_config(arm)
    settings = decoupling(arm)
    search.polish_reflow = settings["polish_reflow"]
    search.polish_after_episode = settings["polish_after_episode"]
    search.terminal_share = settings["terminal_share"]
    return search


def configuration_record(arm: str, config: Config, calibration: dict) -> dict:
    """Record the S4 boundaries, the search fields and the decoupling seam."""
    record = s4_ablation.configuration_record(FULL, config, calibration)
    record.update(
        arm=arm,
        protocol=PROTOCOL,
        lsv2=asdict(search_config(arm)),
        control=CONTROL,
        decoupling=decoupling(arm),
    )
    return record


def run_arm(
    problem: ProblemInstance,
    arm: str,
    config: Config,
    calibration: dict,
    evidence: dict,
    *,
    observer: Callable[[str, dict], None] | None = None,
    search_sink: Callable[[LSV2Search], None] | None = None,
) -> RunResult:
    """Charge initialization, learning and every right shift to one run."""
    search = make_search(arm, config, calibration)
    if search_sink is not None:
        search_sink(search)
    return run_policy(
        problem, "BILL_PERMUTED", config, evidence, observer=observer, interleave=search
    )


__all__ = [
    "ARCHIVE",
    "BOTH",
    "CONTROL",
    "EXTRA_ARMS",
    "FULL",
    "PROTOCOL",
    "TERMINAL",
    "TERMINAL_SHARE",
    "arm_config",
    "configuration_record",
    "decoupling",
    "make_search",
    "run_arm",
    "search_config",
]
