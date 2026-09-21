"""Budgeted same-incumbent exact search with order-independent batch comparison."""

from dataclasses import dataclass
from typing import Protocol

from geo_llm_scheduler.domain.models import Candidate, Genotype, Schedule
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.base import ProposalBatch
from geo_llm_scheduler.utils.numeric import TOL, identity


class Evaluator(Protocol):
    """Injected evaluation gateway keeps MacroSearch independent of engine."""

    ideal: tuple[float, float]

    def evaluate(
        self, genotype: Genotype, schedule: Schedule | None = None, origin: str = "structural"
    ) -> Candidate:
        """Exactly evaluate and dispatch every feasible candidate."""
        ...


@dataclass(frozen=True)
class ScoredCandidate:
    """A feasible evaluated candidate paired with its shared-context scalar."""

    candidate: Candidate
    score: float


@dataclass(frozen=True)
class MacroSearchResult:
    """Exact accounting plus the frozen shared comparison context."""

    requested: int
    construction_attempts: int
    exact_evaluations: int
    feasible_count: int
    best: Candidate | None
    evaluated: tuple[Candidate, ...]
    feasible_scored: tuple[ScoredCandidate, ...]
    context: NormalizationContext

    @property
    def effective(self) -> int:
        """B_eff counts distinct complete candidates actually exactly evaluated."""
        return self.exact_evaluations


def execute(
    batch: ProposalBatch,
    incumbent: Candidate,
    budget: int,
    weight: tuple[float, float],
    context: NormalizationContext,
    evaluator: Evaluator,
    origin: str,
) -> MacroSearchResult:
    """Evaluate unique complete proposals then compare all using a temporary ideal."""
    seen = set()
    candidates: list[Candidate] = []
    for proposal in batch.proposals:
        if len(candidates) >= budget:
            break
        key = (
            proposal.genotype.ms,
            proposal.genotype.os,
            None
            if proposal.schedule is None
            else tuple(round(t / TOL.time) for t in proposal.schedule.starts),
        )
        if key in seen:
            continue
        seen.add(key)
        if proposal.schedule is not None:
            if proposal.genotype != incumbent.genotype:
                raise ValueError("Timing proposal changed genotype")
            selected = set(proposal.selected)
            if any(
                a != b
                for o, (a, b) in enumerate(zip(incumbent.schedule.starts, proposal.schedule.starts))
                if o not in selected
            ):
                raise ValueError("Timing proposal changed frozen operation")
        candidates.append(evaluator.evaluate(proposal.genotype, proposal.schedule, origin))
    shared = NormalizationContext(
        (min(context.ideal[0], evaluator.ideal[0]), min(context.ideal[1], evaluator.ideal[1])),
        context.maximum,
    )
    feasible_scored = tuple(
        ScoredCandidate(candidate, shared.scalar(candidate, weight))
        for candidate in candidates
        if candidate.evaluation.feasible
    )
    ordered = sorted(feasible_scored, key=lambda item: (item.score, identity(item.candidate)))
    best = ordered[0].candidate if ordered else None
    return MacroSearchResult(
        budget,
        batch.attempts,
        len(candidates),
        len(feasible_scored),
        best,
        tuple(candidates),
        feasible_scored,
        shared,
    )
