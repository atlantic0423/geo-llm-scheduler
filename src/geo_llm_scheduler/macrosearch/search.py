"""Budgeted same-incumbent exact search with order-independent batch comparison."""

from dataclasses import dataclass
from typing import Callable, Protocol

from geo_llm_scheduler.domain.models import Candidate, Genotype, Schedule
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
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
    sequential: dict[str, float | int | bool | None] | None = None

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
    sequential: bool = False,
    *,
    structural_observer: Callable[[Genotype], None] | None = None,
) -> MacroSearchResult:
    """Evaluate unique complete proposals then compare all using a temporary ideal."""
    seen = set()
    candidates: list[Candidate] = []
    unique: list[Proposal] = []
    for proposal in batch.proposals:
        if len(unique) >= budget:
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
        unique.append(proposal)

    def evaluate_proposal(proposal: Proposal) -> Candidate:
        if proposal.schedule is None and structural_observer is not None:
            structural_observer(proposal.genotype)
        return evaluator.evaluate(proposal.genotype, proposal.schedule, origin)

    audit: dict[str, float | int | bool | None] | None = None
    if sequential:

        def score(items: list[Candidate]) -> float:
            ideal = (
                min(context.ideal[0], evaluator.ideal[0]),
                min(context.ideal[1], evaluator.ideal[1]),
            )
            shared = NormalizationContext(ideal, context.maximum)
            return min(
                (shared.scalar(c, weight) for c in items if c.evaluation.feasible),
                default=float("inf"),
            )

        # Frozen unique proposal order ensures Fixed6 uses the same first six moves.
        for proposal in unique[:3]:
            candidates.append(evaluate_proposal(proposal))
        evaluated3 = len(candidates)
        best3 = score(candidates)
        current3 = NormalizationContext(
            (min(context.ideal[0], evaluator.ideal[0]), min(context.ideal[1], evaluator.ideal[1])),
            context.maximum,
        ).scalar(incumbent, weight)
        to6 = best3 < current3 - TOL.scalar and len(unique) > 3
        if to6:
            for proposal in unique[3:6]:
                candidates.append(evaluate_proposal(proposal))
        evaluated6 = len(candidates)
        best3_shared = score(candidates[:evaluated3])
        best6 = score(candidates)
        to10 = to6 and best6 < best3_shared - TOL.scalar and len(unique) > 6
        if to10:
            for proposal in unique[6:10]:
                candidates.append(evaluate_proposal(proposal))
        best10 = score(candidates)
        denom = current3 + 1e-12
        audit = {
            "candidate_count_available": len(unique),
            "eval_1_3": evaluated3,
            "best3": None if best3 == float("inf") else best3,
            "continue_to6": to6,
            "eval_4_6": evaluated6 - evaluated3,
            "delta_3_6": 0.0 if best3_shared == float("inf") else (best3_shared - best6) / denom,
            "continue_to10": to10,
            "eval_7_10": max(len(candidates) - 6, 0),
            "delta_6_10": 0.0 if best6 == float("inf") else (best6 - best10) / denom,
            "final_B_eff": len(candidates),
        }
    else:
        candidates = [evaluate_proposal(p) for p in unique]
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
        audit,
    )
