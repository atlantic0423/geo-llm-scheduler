"""Ten-arm fast-right-shift implementation; no campaign is launched by import."""

from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import lsv2_ablation3 as baseline
from geo_llm_scheduler.experiments.lsv2_study import operator_config
from geo_llm_scheduler.lsv2.qlearning import QLearningSearch, QSpec, QTable
from geo_llm_scheduler.lsv2.search import LSV2Config

BASE = baseline.BASE
NO_P3 = BASE + "_NO_P3"
REGIONS = ("all", "flow_end", "middle", "bill_end")
POLISH_ARMS = {
    f"Q1_NET_G9_B3L3_RS_{region.upper()}_{targets.upper()}": (region, targets)
    for region in REGIONS
    for targets in ("partial", "all")
}
ARMS = (BASE, NO_P3, *POLISH_ARMS)
LEARNING_ARMS = ARMS
EXTERNAL_ARMS: tuple[str, ...] = ()
RUN_SEED = baseline.RUN_SEED


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Keep BILL initialization, main search and the total wall budget unchanged."""
    if arm not in ARMS:
        raise ValueError("Unknown fast-right-shift arm")
    return baseline.arm_config(BASE, seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """Construct the two unchanged controls or an external right-shift treatment."""
    if arm not in ARMS:
        raise ValueError("Unknown fast-right-shift arm")
    if arm in (BASE, NO_P3):
        return baseline.search_config(arm)
    region, targets = POLISH_ARMS[arm]
    return replace(
        baseline.search_config(NO_P3),
        polish_region=region,
        polish_targets=targets,
        polish_budget=3,
    )


def make_search(
    arm: str, config: Config, calibration: dict, table: QTable, *, training: bool
) -> QLearningSearch:
    """Build the same online Q1 NET controller, with episode-external polish."""
    spec = QSpec("Q1_NET", 0.9, calibration["rho"], calibration["h"], calibration["w_cutoff"])
    return QLearningSearch(
        config,
        operator_config(config.seed, config.seconds or 1.0),
        spec,
        table,
        training=training,
        lsv2=search_config(arm),
    )


def configuration_record(arm: str, seed: int, seconds: float, calibration: dict) -> dict:
    """Seal both the action mask and independent polish semantics for one run."""
    record = baseline.configuration_record(BASE, seed, seconds, calibration)
    record["lsv2"] = asdict(search_config(arm))
    record["polish_protocol"] = "right-only/post-terminal/shared-ls-share/v2"
    return record
