"""Eight-arm eligibility/selection screen with a fixed FAST COMMON neighborhood."""

from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import lsv2_common_refine as previous
from geo_llm_scheduler.lsv2.qlearning import QLearningSearch, QTable
from geo_llm_scheduler.lsv2.search import LSV2Config

ARMS = (
    "S0_NONE",
    "S1_RANDOM",
    "S2_ELIGIBLE",
    "S3_REGION",
    "S4_FRESH",
    "S5_FRONTIER",
    "S6_DIRECTION",
    "S7_DIRECTION_FRESH",
)
LEARNING_ARMS = ARMS
EXTERNAL_ARMS: tuple[str, ...] = ()
RUN_SEED = previous.RUN_SEED
PANEL = previous.PANEL
PROTOCOL = "eligibility-selection/fixed-common-fast/original-budgets/v1"
PRIMARY_CONTRASTS = (
    ("S1_RANDOM", "S0_NONE"),
    ("S2_ELIGIBLE", "S1_RANDOM"),
    ("S3_REGION", "S2_ELIGIBLE"),
    ("S4_FRESH", "S3_REGION"),
    ("S5_FRONTIER", "S4_FRESH"),
    ("S6_DIRECTION", "S2_ELIGIBLE"),
    ("S7_DIRECTION_FRESH", "S6_DIRECTION"),
)


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Keep BILL4 initialization, evolutionary backbone and charged wall budget."""
    if arm not in ARMS:
        raise ValueError("Unknown eligibility-selection arm")
    return previous.arm_config("C0_NONE", seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """Use identical refinement; independently vary only input eligibility/selection."""
    if arm not in ARMS:
        raise ValueError("Unknown eligibility-selection arm")
    if arm == "S0_NONE":
        return previous.search_config("C0_NONE")
    policies = dict(
        S1_RANDOM="pool_random",
        S2_ELIGIBLE="eligible_random",
        S3_REGION="eligible_region",
        S4_FRESH="fresh_region",
        S5_FRONTIER="frontier_region",
        S6_DIRECTION="eligible_direction",
        S7_DIRECTION_FRESH="fresh_direction",
    )
    return replace(
        previous.search_config("C2_FAST"),
        polish_selection=policies[arm],
        polish_pool_directions=False,
    )


def make_search(
    arm: str, config: Config, calibration: dict, table: QTable, *, training: bool
) -> QLearningSearch:
    """Construct the unchanged online Q learner with an episode-external selector."""
    search = previous.make_search("C0_NONE", config, calibration, table, training=training)
    search.config = search_config(arm)
    return search


def configuration_record(arm: str, seed: int, seconds: float, calibration: dict) -> dict:
    """Seal selection policy and unchanged neighborhood, Q credit and time rules."""
    record = previous.configuration_record("C0_NONE", seed, seconds, calibration)
    record["lsv2"] = asdict(search_config(arm))
    record["polish_protocol"] = PROTOCOL
    return record
