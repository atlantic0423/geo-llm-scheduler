"""User-declared whole-module S4 ablations, with no new algorithm defaults."""

from collections.abc import Callable
from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import ProblemInstance
from geo_llm_scheduler.engine.run import RunResult
from geo_llm_scheduler.experiments import lsv2_eligibility_selection as reference
from geo_llm_scheduler.experiments.initial_gap_construction import CONSTRUCTIONS, gap_initializer
from geo_llm_scheduler.experiments.initial_pool_retention import POLICIES, pool_initializer
from geo_llm_scheduler.experiments.lsv2_study import operator_config
from geo_llm_scheduler.experiments.rl23 import run_policy
from geo_llm_scheduler.lsv2.qlearning import QTable
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search

DIRECT = "A4_DIRECT_SHIFT"
ARMS = ("A0_FULL_S4", "A1_NO_SHIFT", "A2_RANDOM_OPERATOR", "A3_RANDOM_INIT", DIRECT)
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
    if arm == DIRECT:
        return replace(
            base,
            selector="direct_shift",
            operators=(),
            trajectory_steps=0,
            polish_selection="endpoint",
        )
    return replace(base, selector="uniform") if arm == "A2_RANDOM_OPERATOR" else base


def make_search(arm: str, config: Config, calibration: dict) -> LSV2Search:
    """Use empty online Q per run, or genuinely remove Q in the random arm."""
    ls = search_config(arm)
    if arm in ("A2_RANDOM_OPERATOR", DIRECT):
        return LSV2Search(
            ls, config, operator_config(config.seed, config.seconds or 1), retain_steps=True
        )
    search = reference.make_search("S4_FRESH", config, calibration, QTable(), training=True)
    search.config = ls
    return search


def configuration_record(
    arm: str,
    config: Config,
    calibration: dict,
    *,
    initial_retention: str = "legacy",
    initial_construction: str = "bill4",
) -> dict:
    """Record exact ablation boundaries without implying a completed experiment."""
    search_config(arm)
    _validate_initialization(arm, initial_retention, initial_construction)
    record = {
        "protocol": "s4-whole-module-ablation/v1",
        "arm": arm,
        "engine": asdict(config),
        "lsv2": asdict(search_config(arm)),
        "initialization": "random" if arm == "A3_RANDOM_INIT" else "bill4",
        "controller": {DIRECT: "direct_shift", "A2_RANDOM_OPERATOR": "uniform"}.get(
            arm, "Q1_NET_G9"
        ),
        "calibration": calibration,
        "learning_mode": "none" if arm in ("A2_RANDOM_OPERATOR", DIRECT) else "online_from_zero",
    }
    if initial_retention != "legacy":
        record.update(protocol="s4-initial-retention/v1", initial_retention=initial_retention)
    if initial_construction == "g1":
        record.update(
            protocol="s4-initial-g1/v1",
            initialization="g1",
            initial_retention=initial_retention,
            construction_protocol="bill-gap-tchebycheff/v1",
        )
    return record


def _validate_initialization(arm: str, policy: str, construction: str) -> None:
    if policy not in POLICIES or (arm == "A3_RANDOM_INIT" and policy != "legacy"):
        raise ValueError("Unknown retention policy or conflict with random-initialization ablation")
    if construction not in CONSTRUCTIONS or (arm == "A3_RANDOM_INIT" and construction != "bill4"):
        raise ValueError("Unknown construction or conflict with random-initialization ablation")


def run_arm(
    problem: ProblemInstance,
    arm: str,
    config: Config,
    calibration: dict,
    evidence: dict,
    *,
    observer: Callable[[str, dict], None] | None = None,
    search_sink: Callable[[LSV2Search], None] | None = None,
    initial_retention: str = "legacy",
    initial_construction: str = "bill4",
) -> RunResult:
    """Charge initialization, learning, selection and external shift to one run."""
    _validate_initialization(arm, initial_retention, initial_construction)
    override = None
    if initial_construction == "g1":
        override = gap_initializer(problem, config, evidence, retention=initial_retention)
    elif initial_retention == "global_pool":
        override = pool_initializer(problem, config, evidence)
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
        initializer_override=override,
    )
