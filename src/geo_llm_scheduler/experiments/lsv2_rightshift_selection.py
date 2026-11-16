"""Approved 12-arm right-shift selection screen; importing never launches work."""

from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import lsv2_fast_rightshift as previous
from geo_llm_scheduler.lsv2.qlearning import QLearningSearch, QTable
from geo_llm_scheduler.lsv2.search import LSV2Config

POLICIES = {
    "T_ALL": ("endpoint", "all"),
    "T_BILL": ("endpoint", "bill_end"),
    "POOL_RANDOM": ("pool_random", "all"),
    "DIRECTION": ("direction", "all"),
    "REGION_TIME": ("region_time", "all"),
}
POLISH_ARMS = {
    f"{name}_{suffix}": (selection, region, targets)
    for name, (selection, region) in POLICIES.items()
    for suffix, targets in (("P", "partial"), ("F", "all"))
}
ARMS = ("B0", "B1", *POLISH_ARMS)
LEARNING_ARMS = ARMS
EXTERNAL_ARMS: tuple[str, ...] = ()
RUN_SEED = previous.RUN_SEED
PROTOCOL = "rightshift-selection/shared-physical-pool/post-terminal/v1"
PANEL = {
    100: (6, 9, 12, 13),
    200: (1, 7, 9, 14),
}


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Retain the previous BILL initialization and engine budget for all arms."""
    if arm not in ARMS:
        raise ValueError("Unknown right-shift selection arm")
    return previous.arm_config(previous.BASE, seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """Freeze the two controls or one of five selectors with partial/full targets."""
    if arm not in ARMS:
        raise ValueError("Unknown right-shift selection arm")
    base = previous.search_config(previous.BASE if arm == "B1" else previous.NO_P3)
    if arm in ("B0", "B1"):
        return base
    selection, region, targets = POLISH_ARMS[arm]
    return replace(
        base,
        polish_selection=selection,
        polish_region=region,
        polish_targets=targets,
        polish_diagnostics=True,
    )


def make_search(
    arm: str, config: Config, calibration: dict, table: QTable, *, training: bool
) -> QLearningSearch:
    """Reuse Q1 NET gamma=.9, with a fresh selector and unchanged reward semantics."""
    search = previous.make_search(previous.NO_P3, config, calibration, table, training=training)
    search.config = search_config(arm)
    return search


def configuration_record(arm: str, seed: int, seconds: float, calibration: dict) -> dict:
    """Seal eligibility, selector, diagnostic cost and unchanged controller settings."""
    record = previous.configuration_record(previous.BASE, seed, seconds, calibration)
    record["lsv2"] = asdict(search_config(arm))
    record["polish_protocol"] = PROTOCOL
    record["polish_pool"] = "exact-MS-starts/current-population-plus-terminal/no-history-filter"
    return record
