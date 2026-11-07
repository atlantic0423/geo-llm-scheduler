"""S4 with the demand-envelope common shift (C4) as its only changed component.

The single treatment arm ``S4_C4`` keeps the complete S4 study configuration:
BILL4 initialization, online Q1/NET/G9/B3L3 local search, the shared 20% time
gate, eligibility/region/least-visited input selection and the FAST COMMON
reuse. Only the external right-shift neighborhood switches from ``common_fast``
to ``common_peak``, which adds the knots of the upper envelope of window-average
power to the event-point positions of each common group shift. The control arm
``A0_FULL_S4`` is delegated unchanged to :mod:`s4_ablation`, so the sealed
fourteen-arm registry and its runner are untouched.
"""

from collections.abc import Callable
from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import ProblemInstance
from geo_llm_scheduler.engine.run import RunResult
from geo_llm_scheduler.experiments import s4_ablation
from geo_llm_scheduler.experiments.rl23 import run_policy
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search

CONTROL = "A0_FULL_S4"
ARM = "S4_C4"
EXTRA_ARMS = (ARM,)
PROTOCOL = "s4-c4-envelope/v1"


def _check(arm: str) -> None:
    if arm not in EXTRA_ARMS:
        raise ValueError("Unknown S4+C4 arm")


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Return the unchanged full-S4 engine and inclusive wall-clock configuration."""
    _check(arm)
    return s4_ablation.arm_config(CONTROL, seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """Full-S4 local search with only the right-shift neighborhood set to C4."""
    _check(arm)
    return replace(s4_ablation.search_config(CONTROL), polish_method="common_peak")


def make_search(arm: str, config: Config, calibration: dict) -> LSV2Search:
    """Build the same empty online Q learner as full S4, then swap the neighborhood."""
    search = s4_ablation.make_search(CONTROL, config, calibration)
    search.config = search_config(arm)
    return search


def configuration_record(arm: str, config: Config, calibration: dict) -> dict:
    """Record the full-S4 boundaries plus the single changed neighborhood field."""
    record = s4_ablation.configuration_record(CONTROL, config, calibration)
    record.update(arm=arm, protocol=PROTOCOL, lsv2=asdict(search_config(arm)), control=CONTROL)
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
    """Charge initialization, learning, selection and C4 right shift to one run."""
    search = make_search(arm, config, calibration)
    if search_sink is not None:
        search_sink(search)
    return run_policy(
        problem, "BILL_PERMUTED", config, evidence, observer=observer, interleave=search
    )


__all__ = [
    "ARM",
    "CONTROL",
    "EXTRA_ARMS",
    "PROTOCOL",
    "arm_config",
    "configuration_record",
    "make_search",
    "run_arm",
    "search_config",
]
