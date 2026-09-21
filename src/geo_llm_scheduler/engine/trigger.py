"""Selective local-search gate: direction preference and scalar quality."""

import math
import random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.rl.state import associate, preference
from geo_llm_scheduler.utils.numeric import EPS_RATIO


def triggered(
    child: Candidate,
    incumbent: Candidate,
    index: int,
    weights: tuple[tuple[float, float], ...],
    context: NormalizationContext,
    config: Config,
    rng: random.Random,
) -> bool:
    """Gate local effort without suppressing Archive updates or global replacement."""
    if config.trigger_mode == "always":
        return True
    if config.trigger_mode == "fixed":
        return rng.random() < config.fixed_ls_probability
    vector = context.normalize(child)
    if math.hypot(*vector) > EPS_RATIO:
        nearest = associate(vector, weights)
        if config.trigger_mode == "strict":
            if nearest != index:
                return False
        elif preference(nearest, len(weights)) != preference(index, len(weights)):
            return False
    before = context.scalar(incumbent, weights[index])
    return (context.scalar(child, weights[index]) - before) / (
        before + EPS_RATIO
    ) <= config.trigger_delta
