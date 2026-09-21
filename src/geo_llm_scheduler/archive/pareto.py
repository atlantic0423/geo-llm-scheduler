"""External phenotype-aware nondominated archive, independent of scalar acceptance."""

from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.utils.numeric import TOL, close, identity, less


def dominates(a: Candidate, b: Candidate) -> bool:
    """Tolerance-aware Pareto dominance in the two minimizing objectives."""
    x, y = a.evaluation.objectives, b.evaluation.objectives
    tolerances = (TOL.time, TOL.cost)
    return all(v <= w or close(v, w, t) for v, w, t in zip(x, y, tolerances)) and any(
        less(v, w, t) for v, w, t in zip(x, y, tolerances)
    )


class Archive:
    """Unbounded first-version archive; equal objectives may retain distinct timing."""

    def __init__(self) -> None:
        self.members: list[Candidate] = []
        self.attempts = 0
        self.insertions = 0
        self.peak_size = 0

    def consider(self, candidate: Candidate) -> bool:
        """Attempt insertion for every complete feasible exact candidate."""
        if not candidate.evaluation.feasible:
            return False
        self.attempts += 1
        key = identity(candidate)
        if any(identity(c) == key or dominates(c, candidate) for c in self.members):
            return False
        self.members = [c for c in self.members if not dominates(candidate, c)]
        self.members.append(candidate)
        self.members.sort(key=identity)
        self.insertions += 1
        self.peak_size = max(self.peak_size, len(self.members))
        return True
