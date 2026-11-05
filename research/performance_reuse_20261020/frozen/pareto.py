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
        self._identities: dict[int, tuple[Candidate, tuple]] = {}

    def consider(self, candidate: Candidate) -> bool:
        """Attempt insertion for every complete feasible exact candidate."""
        if not candidate.evaluation.feasible:
            return False
        self.attempts += 1
        key = identity(candidate)
        # Members are immutable, but the public member list can be replaced or
        # edited by diagnostics. Reconcile it on every attempt and keep only
        # live members, with strong references preventing Python id reuse.
        identities = {}
        for member in self.members:
            cached = self._identities.get(id(member))
            identities[id(member)] = cached if cached is not None else (member, identity(member))
        self._identities = identities
        if any(identities[id(c)][1] == key or dominates(c, candidate) for c in self.members):
            return False
        self.members = [c for c in self.members if not dominates(candidate, c)]
        self.members.append(candidate)
        identities[id(candidate)] = (candidate, key)
        self.members.sort(key=lambda member: identities[id(member)][1])
        self._identities = {id(member): identities[id(member)] for member in self.members}
        self.insertions += 1
        self.peak_size = max(self.peak_size, len(self.members))
        return True
