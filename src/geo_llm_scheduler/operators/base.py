"""Operator proposals are immutable and never own mutable incumbents."""

import random
from dataclasses import dataclass, field
from typing import Protocol

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance, Schedule


@dataclass(frozen=True)
class Proposal:
    """Structural proposals omit schedule; timing proposals retain phenotype."""

    genotype: Genotype
    schedule: Schedule | None = None
    selected: tuple[int, ...] = ()


@dataclass
class ProposalBatch:
    """Construction attempts and qualifying proposals are separate quantities."""

    proposals: list[Proposal] = field(default_factory=list)
    attempts: int = 0
    diagnostics: dict = field(default_factory=dict)
    instrumentation: dict = field(default_factory=dict)


class Operator(Protocol):
    """Unified bounded candidate generator interface."""

    action: int

    def propose(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        config: Config,
        rng: random.Random,
    ) -> ProposalBatch:
        """Generate independent proposals from one frozen incumbent."""
        ...
