"""Nonempty A4/A5 counterexamples verify occurrence identities and legal forward moves."""

import random

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments.synthetic import synthetic
from geo_llm_scheduler.operators.structural import StructuralOperator, operation_order
from geo_llm_scheduler.scheduling.ssgs import decode


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


def test_a4_a5_large_position_spaces_are_bounded():
    problem = synthetic(30, 1, 1, 1)
    jobs = len(problem.jobs)
    genotype = Genotype((0,) * (2 * jobs), tuple(range(jobs)) + tuple(range(jobs)))
    earliest = decode(problem, genotype)
    delayed = Schedule(tuple(start + 10000 for start in earliest.starts))
    incumbent = Candidate(genotype, delayed, EvaluationResult(True, 1, 1, (), ()))
    for action in (4, 5):
        batch = StructuralOperator(action).propose(
            problem, incumbent, 3, Config(), random.Random(2)
        )
        assert batch.diagnostics["position_space"] > 2 * 3
        assert batch.attempts <= 2 * 3
        assert batch.diagnostics["qualifying_pool"] <= 2 * 3
        assert len(batch.proposals) <= 3
