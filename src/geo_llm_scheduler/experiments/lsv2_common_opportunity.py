"""Eight-arm resource-calendar and input-selection development screen."""

from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import lsv2_common_refine as previous
from geo_llm_scheduler.lsv2.qlearning import QLearningSearch, QTable
from geo_llm_scheduler.lsv2.search import LSV2Config

ARMS = (
    "R0_NONE",
    "R1_FAST",
    "R2_CALENDAR",
    "R3_ELIGIBLE",
    "R4_REGION",
    "R5_FRESH",
    "R6_FRONTIER",
    "R7_DIRECTION",
)
LEARNING_ARMS = ARMS
EXTERNAL_ARMS: tuple[str, ...] = ()
RUN_SEED = previous.RUN_SEED
PANEL = previous.PANEL
PROTOCOL = "common-opportunity/calendar/empty-certificates/fresh-selection/v1"


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Keep the same main search and original total wall-clock budget."""
    if arm not in ARMS:
        raise ValueError("Unknown COMMON opportunity arm")
    return previous.arm_config("C0_NONE", seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """Separate engineering, eligibility, regional and within-region changes."""
    if arm not in ARMS:
        raise ValueError("Unknown COMMON opportunity arm")
    if arm == "R0_NONE":
        return previous.search_config("C0_NONE")
    base = previous.search_config("C2_FAST")
    if arm == "R1_FAST":
        return base
    policies = dict(
        R2_CALENDAR="pool_random",
        R3_ELIGIBLE="eligible_random",
        R4_REGION="eligible_region",
        R5_FRESH="fresh_region",
        R6_FRONTIER="frontier_region",
        R7_DIRECTION="fresh_direction",
    )
    return replace(
        base,
        polish_method="common_calendar",
        polish_selection=policies[arm],
        polish_pool_directions=False,
    )


def make_search(
    arm: str, config: Config, calibration: dict, table: QTable, *, training: bool
) -> QLearningSearch:
    """Build a fresh online Q learner; external refinement receives no Q credit."""
    search = previous.make_search("C0_NONE", config, calibration, table, training=training)
    search.config = search_config(arm)
    return search


def configuration_record(arm: str, seed: int, seconds: float, calibration: dict) -> dict:
    """Seal the selection variant, original budgets and unchanged reward/state."""
    record = previous.configuration_record("C0_NONE", seed, seconds, calibration)
    record["lsv2"] = asdict(search_config(arm))
    record["polish_protocol"] = PROTOCOL
    return record
