"""Nonempty A4/A5 counterexamples verify occurrence identities and legal forward moves."""

import random

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.operators.structural import StructuralOperator, operation_order


def test_a4_targets_decode_after_own_prefill(problem):
    gateway = EvaluationGateway(problem, Archive())
    g = Genotype((0, 1, 0, 1), (0, 1, 0, 1))
    x = gateway.evaluate(g, Schedule((100, 500, 200, 700)))
    batch = StructuralOperator(4).propose(problem, x, 10, Config(), random.Random(2))
    assert batch.proposals
    before = operation_order(g)
    for proposal in batch.proposals:
        after = operation_order(proposal.genotype)
        assert all(after.index(2 * j) < after.index(2 * j + 1) for j in range(2))
        assert proposal.genotype.ms == g.ms
        assert any(after.index(o) < before.index(o) for o in before)


def test_a5_both_occurrences_move_forward(problem):
    gateway = EvaluationGateway(problem, Archive())
    g = Genotype((0, 1, 0, 1), (0, 0, 1, 1))
    x = gateway.evaluate(g, Schedule((100, 500, 200, 700)))
    batch = StructuralOperator(5).propose(problem, x, 10, Config(), random.Random(2))
    assert batch.proposals
    for proposal in batch.proposals:
        after = operation_order(proposal.genotype)
        assert after.index(2) < 2 and after.index(3) < 3
        assert after.index(2) < after.index(3)
        assert proposal.genotype.ms == g.ms
