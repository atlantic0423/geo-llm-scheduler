"""MOEA/D weights, frozen comparison contexts and neighbor-own-weight replacement."""

from dataclasses import dataclass

from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.utils.numeric import EPS_NORM, TOL, less


@dataclass(frozen=True)
class NormalizationContext:
    """An immutable context shared by every member of a comparison batch."""

    ideal: tuple[float, float]
    maximum: tuple[float, float]

    def normalize(self, candidate: Candidate) -> tuple[float, float]:
        """Use current-population range with a numerical zero-range guard."""
        values = candidate.evaluation.objectives
        return tuple(
            (values[k] - self.ideal[k]) / max(self.maximum[k] - self.ideal[k], EPS_NORM)
            for k in range(2)
        )  # type: ignore[return-value]

    def scalar(self, candidate: Candidate, weight: tuple[float, float]) -> float:
        """Normalized Tchebycheff for the supplied subproblem's own lambda."""
        return max(w * abs(f) for w, f in zip(weight, self.normalize(candidate)))


def weights(n: int) -> tuple[tuple[float, float], ...]:
    """Generate uniformly spaced two-objective weights including endpoints."""
    if n < 2:
        raise ValueError("At least two weights")
    return tuple((i / (n - 1), 1 - i / (n - 1)) for i in range(n))


def neighborhoods(
    lambdas: tuple[tuple[float, float], ...], size: int
) -> tuple[tuple[int, ...], ...]:
    """Euclidean neighbors; stable index breaks equal-distance ties."""
    return tuple(
        tuple(
            sorted(
                range(len(lambdas)),
                key=lambda j: (sum((a - b) ** 2 for a, b in zip(w, lambdas[j])), j),
            )[:size]
        )
        for w in lambdas
    )


def maximum(population: list[Candidate]) -> tuple[float, float]:
    """Recompute scale maximum from the current population only."""
    return tuple(max(c.evaluation.objectives[k] for c in population) for k in range(2))  # type: ignore[return-value]


def replace_neighbors(
    population: list[Candidate],
    candidate: Candidate,
    neighbors: tuple[int, ...],
    lambdas: tuple[tuple[float, float], ...],
    context: NormalizationContext,
    cap: int,
) -> int:
    """Replace at most cap neighbors, each evaluated using its own weight."""
    changed = 0
    for j in neighbors:
        if changed >= cap:
            break
        if less(
            context.scalar(candidate, lambdas[j]),
            context.scalar(population[j], lambdas[j]),
            TOL.cost,
        ):
            population[j] = candidate
            changed += 1
    return changed
