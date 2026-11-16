"""Fast right-only Prefill polish, kept outside the learning action vocabulary."""

from dataclasses import dataclass
from random import Random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.lsv2.flowsafe import FlowSafeBillMoves
from geo_llm_scheduler.operators.base import ProposalBatch


@dataclass(frozen=True)
class FastRightShiftPolish:
    """Use P3's incremental multi-move search, restricted to later Prefill starts.

    ``all_targets`` removes the target-pool truncation, not the successful-move
    cap. Each complete proposal still needs the authoritative exact gateway.
    """

    all_targets: bool = False

    def propose(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        config: Config,
        rng: Random,
    ) -> ProposalBatch:
        """Return up to budget right-only schedules without evaluating or archiving."""
        if type(budget) is not int or budget < 1:
            raise ValueError("Polish candidate budget must be a positive integer")
        return FlowSafeBillMoves(right_only=True, all_targets=self.all_targets).propose(
            problem, incumbent, budget, config, rng
        )
