"""Experimental replacement routing; all comparisons remain neighbor-own-weight."""

from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.rl.state import associate
from geo_llm_scheduler.utils.rng import RNGManager


def replacement_neighborhood(
    policy: str,
    birth: int,
    candidate: Candidate,
    neighborhoods: tuple[tuple[int, ...], ...],
    lambdas: tuple[tuple[float, float], ...],
    context: NormalizationContext,
    streams: RNGManager,
) -> tuple[int, ...]:
    """Choose birth neighbors, permute them, or associate the final child direction.

    The context is frozen before replacement. Direction routing uses the existing
    inverse-weight Tchebycheff rays, including stable endpoint/zero-vector ties.
    Only the permuted control consumes the independent replacement RNG stream.
    """
    if policy == "birth":
        return neighborhoods[birth]
    if policy == "direction":
        return neighborhoods[associate(context.normalize(candidate), lambdas)]
    if policy == "permuted":
        order = list(neighborhoods[birth])
        streams.stream("P2:replacement").shuffle(order)
        return tuple(order)
    raise ValueError("Unknown experimental replacement policy")
