"""Five-arm COMMON efficiency, regional allocation and demand-knot screen."""

from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import lsv2_group_rightshift as previous
from geo_llm_scheduler.lsv2.qlearning import QLearningSearch, QTable
from geo_llm_scheduler.lsv2.search import LSV2Config

ARMS = ("C0_NONE", "C1_COMMON", "C2_FAST", "C3_REGION", "C4_PEAK")
LEARNING_ARMS = ARMS
EXTERNAL_ARMS: tuple[str, ...] = ()
RUN_SEED = previous.RUN_SEED
PANEL = previous.PANEL
PROTOCOL = "common-refine/equivalent-reuse/region-time/peak-switch/v1"


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Retain original 1200/2400 s including initialization and evolutionary settings."""
    if arm not in ARMS:
        raise ValueError("Unknown COMMON refinement arm")
    return previous.arm_config("NONE_20", seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """Freeze Q1 NET G9 B3L3, no P3, one shared 20% time gate."""
    if arm not in ARMS:
        raise ValueError("Unknown COMMON refinement arm")
    base = previous.search_config("NONE_20" if arm == "C0_NONE" else "COMMON_20")
    if arm in ("C0_NONE", "C1_COMMON"):
        return base
    return replace(
        base,
        polish_method="common_peak" if arm == "C4_PEAK" else "common_fast",
        polish_selection="region_time" if arm == "C3_REGION" else "pool_random",
    )


def make_search(
    arm: str, config: Config, calibration: dict, table: QTable, *, training: bool
) -> QLearningSearch:
    """Build a fresh online learner with episode-external refinement."""
    search = previous.make_search("NONE_20", config, calibration, table, training=training)
    search.config = search_config(arm)
    return search


def configuration_record(arm: str, seed: int, seconds: float, calibration: dict) -> dict:
    """Seal each single-factor contrast and unchanged controller configuration."""
    record = previous.configuration_record("NONE_20", seed, seconds, calibration)
    record["lsv2"] = asdict(search_config(arm))
    record["polish_protocol"] = PROTOCOL
    return record
