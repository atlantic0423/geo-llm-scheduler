"""Eight-arm concurrent right-shift by LS share development screen."""

from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import lsv2_rightshift_selection as previous
from geo_llm_scheduler.lsv2.qlearning import QLearningSearch, QTable
from geo_llm_scheduler.lsv2.search import LSV2Config

ARMS = tuple(
    f"{method}_{share}" for method in ("NONE", "SINGLE", "COMMON", "COORD") for share in (20, 30)
)
LEARNING_ARMS = ARMS
EXTERNAL_ARMS: tuple[str, ...] = ()
RUN_SEED = previous.RUN_SEED
PANEL = previous.PANEL
PROTOCOL = "concurrent-rightshift/four-neighborhoods/two-shares/v1"


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Keep the original total time, initializer and evolutionary backbone."""
    if arm not in ARMS:
        raise ValueError("Unknown concurrent right-shift arm")
    return previous.arm_config("B0", seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """Freeze Q1 NET G9 B3L3 without P3; share includes external polish cost."""
    if arm not in ARMS:
        raise ValueError("Unknown concurrent right-shift arm")
    method, share = arm.split("_")
    base = previous.search_config("B0" if method == "NONE" else "POOL_RANDOM_P")
    return replace(
        base,
        time_share=int(share) / 100,
        polish_method={
            "NONE": "single",
            "SINGLE": "single",
            "COMMON": "common",
            "COORD": "coordinated",
        }[method],
    )


def make_search(
    arm: str, config: Config, calibration: dict, table: QTable, *, training: bool
) -> QLearningSearch:
    """Create a fresh online learner; external timing refinement has no Q credit."""
    search = previous.make_search("B0", config, calibration, table, training=training)
    search.config = search_config(arm)
    return search


def configuration_record(arm: str, seed: int, seconds: float, calibration: dict) -> dict:
    """Seal the exact method, time gate and unchanged per-run Q settings."""
    record = previous.configuration_record("B0", seed, seconds, calibration)
    record["lsv2"] = asdict(search_config(arm))
    record["polish_protocol"] = PROTOCOL
    return record
