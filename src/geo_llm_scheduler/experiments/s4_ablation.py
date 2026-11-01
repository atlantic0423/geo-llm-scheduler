"""User-declared whole-module S4 ablations, with no new algorithm defaults."""

from collections.abc import Callable
from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import ProblemInstance
from geo_llm_scheduler.engine.run import RunResult
from geo_llm_scheduler.experiments import lsv2_eligibility_selection as reference
from geo_llm_scheduler.experiments.lsv2_study import operator_config
from geo_llm_scheduler.experiments.rl23 import run_policy
from geo_llm_scheduler.lsv2.qlearning import QTable
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search

ARMS = ("A0_FULL_S4", "A1_NO_SHIFT", "A2_RANDOM_OPERATOR", "A3_RANDOM_INIT")
BUDGETS = {50: 600, 100: 1200, 200: 2400}
PRIMARY_CONTRASTS = tuple((ARMS[0], arm) for arm in ARMS[1:])


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Return the unchanged S4 engine and inclusive wall-clock configuration."""
    if arm not in ARMS:
        raise ValueError("Unknown S4 ablation arm")
    return reference.arm_config("S4_FRESH", seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """Change one module only, preserving action eligibility and the shared gate."""
    arm_config(arm, 0, 1)
    if arm == "A1_NO_SHIFT":
        return reference.search_config("S0_NONE")
    base = reference.search_config("S4_FRESH")
    return replace(base, selector="uniform") if arm == "A2_RANDOM_OPERATOR" else base


def make_search(arm: str, config: Config, calibration: dict) -> LSV2Search:
    """Use empty online Q per run, or genuinely remove Q in the random arm."""
    ls = search_config(arm)
    if arm == "A2_RANDOM_OPERATOR":
        return LSV2Search(
            ls, config, operator_config(config.seed, config.seconds or 1), retain_steps=True
        )
    search = reference.make_search("S4_FRESH", config, calibration, QTable(), training=True)
    search.config = ls
    return search


def configuration_record(arm: str, config: Config, calibration: dict) -> dict:
    """Record exact ablation boundaries without implying a completed experiment."""
    search_config(arm)
    return {
        "protocol": "s4-whole-module-ablation/v1",
        "arm": arm,
        "engine": asdict(config),
        "lsv2": asdict(search_config(arm)),
        "initialization": "random" if arm == "A3_RANDOM_INIT" else "bill4",
        "controller": "uniform" if arm == "A2_RANDOM_OPERATOR" else "Q1_NET_G9",
        "calibration": calibration,
        "learning_mode": "none" if arm == "A2_RANDOM_OPERATOR" else "online_from_zero",
    }


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
    """Charge initialization, learning, selection and external shift to one run."""
    search = make_search(arm, config, calibration)
    if search_sink is not None:
        search_sink(search)
    return run_policy(
        problem,
        "BILL_PERMUTED",
        config,
        evidence,
        observer=observer,
        interleave=search,
        initialization_arm="EXT_MOEAD" if arm == "A3_RANDOM_INIT" else None,
    )
