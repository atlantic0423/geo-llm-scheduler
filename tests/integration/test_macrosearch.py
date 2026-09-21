"""Cross-module candidate lifecycle, archive dispatch and permutation invariants."""

import random

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.io.validation import validate_genotype
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.utils.numeric import identity


@pytest.mark.parametrize("action", range(1, 7))
def test_structural_invariants(problem, action):
    gateway = EvaluationGateway(problem, Archive())
    x = gateway.evaluate(Genotype((0, 1, 0, 1), (0, 1, 0, 1)))
    config = Config()
    batch = StructuralOperator(action).propose(problem, x, 6, config, random.Random(4))
    assert len(batch.proposals) <= 6
    for proposal in batch.proposals:
        assert proposal.schedule is None
        validate_genotype(problem, proposal.genotype)
        if action <= 3:
            assert proposal.genotype.os == x.genotype.os
        if action in (4, 5):
            assert proposal.genotype.ms == x.genotype.ms
        if action == 1:
            assert sum(a != b for a, b in zip(proposal.genotype.ms, x.genotype.ms)) == 1
    before = gateway.counts["ssgs"]
    result = execute(
        batch,
        x,
        6,
        (0.5, 0.5),
        NormalizationContext(gateway.ideal, (1000, 1000)),
        gateway,
        f"A{action}",
    )
    assert gateway.counts["ssgs"] - before == result.effective
    assert gateway.archive.attempts == gateway.counts["feasible"]


def test_order_independent_and_duplicate_budget(problem):
    results = []
    proposals = [
        Proposal(Genotype(ms, (0, 1, 0, 1))) for ms in ((0, 0, 0, 0), (0, 1, 2, 2), (2, 2, 2, 2))
    ]
    for order in (proposals, list(reversed(proposals))):
        gateway = EvaluationGateway(problem, Archive())
        x = gateway.evaluate(Genotype((0, 1, 0, 1), (0, 1, 0, 1)))
        result = execute(
            ProposalBatch(order + order, 6),
            x,
            6,
            (0.5, 0.5),
            NormalizationContext(gateway.ideal, (2000, 2000)),
            gateway,
            "batch",
        )
        assert result.effective == 3
        results.append(identity(result.best))
    assert results[0] == results[1]


def test_timing_frozen_and_no_ssgs(problem):
    gateway = EvaluationGateway(problem, Archive())
    x = gateway.evaluate(Genotype((0, 0, 0, 0), (0, 1, 0, 1)))
    bad = Proposal(x.genotype, Schedule((1, 101, 0, 100)), (0,))
    with pytest.raises(ValueError, match="frozen"):
        execute(
            ProposalBatch([bad]),
            x,
            3,
            (0.5, 0.5),
            NormalizationContext(gateway.ideal, (1000, 1000)),
            gateway,
            "timing",
        )
