"""Golden target-selection and empty-batch cases for A4 and A5."""

import random

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import (
    Candidate,
    EvaluationResult,
    Genotype,
    ProblemInstance,
    Schedule,
)
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.trajectory import improve
from geo_llm_scheduler.experiments.synthetic import synthetic
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.structural import StructuralOperator, operation_order
from geo_llm_scheduler.rl.controller import Controller, reward
from geo_llm_scheduler.utils.rng import RNGManager


def case(
    prefill_waits: tuple[float, ...], decode_waits: tuple[float, ...]
) -> tuple[ProblemInstance, Candidate]:
    """Build a transparent proposal-only schedule with chosen Flow-wait values."""
    problem = synthetic(len(prefill_waits), 1, 1, 1)
    jobs = len(problem.jobs)
    starts = [0.0] * (2 * jobs)
    for i, job in enumerate(problem.jobs):
        starts[2 * i] = job.release + prefill_waits[i]
        starts[2 * i + 1] = starts[2 * i] + job.prefill.duration + decode_waits[i]
    genotype = Genotype((0,) * (2 * jobs), tuple(range(jobs)) * 2)
    incumbent = Candidate(genotype, Schedule(tuple(starts)), EvaluationResult(True, 1, 1, (), ()))
    return problem, incumbent


def test_a4_wait_ranking_and_zero_wait_exclusion():
    problem, incumbent = case((0, 12, 5, 0), (2, 0, 0, 0))
    for budget, expected in ((1, {2}), (2, {2, 4})):
        batch = StructuralOperator(4).propose(
            problem, incumbent, budget, Config(), random.Random(3)
        )
        assert batch.proposals
        assert all(proposal.genotype.ms == incumbent.genotype.ms for proposal in batch.proposals)
        assert {
            next(
                o
                for o in operation_order(incumbent.genotype)
                if operation_order(proposal.genotype).index(o)
                < operation_order(incumbent.genotype).index(o)
            )
            for proposal in batch.proposals
        }.issubset(expected)
        assert all(proposal.genotype != incumbent.genotype for proposal in batch.proposals)


def test_a5_total_wait_ranking_and_single_stage_eligibility():
    problem, incumbent = case((0, 10, 4, 0), (0, 0, 4, 7))
    # Job 1: (10, 0), Job 2: (4, 4), Job 3: (0, 7).
    for budget, expected in ((1, {1}), (2, {1, 2}), (3, {1, 2, 3})):
        batch = StructuralOperator(5).propose(
            problem, incumbent, budget, Config(), random.Random(4)
        )
        assert batch.proposals
        old = operation_order(incumbent.genotype)
        for proposal in batch.proposals:
            new = operation_order(proposal.genotype)
            moved = [i for i in range(4) if new.index(2 * i) < old.index(2 * i)]
            assert len(moved) == 1
            job = moved[0]
            assert job in expected
            assert new.index(2 * job + 1) < old.index(2 * job + 1)
            assert new.index(2 * job) < new.index(2 * job + 1)
            assert proposal.genotype.ms == incumbent.genotype.ms


@pytest.mark.parametrize(
    ("prefill_waits", "decode_waits", "expected_job"),
    [
        ((0, 0, 0, 0), (0, 0, 0, 7), 3),
        ((0, 0, 5, 0), (0, 0, 6, 0), 2),
    ],
)
def test_a5_single_stage_and_both_stage_waits(
    prefill_waits: tuple[float, ...], decode_waits: tuple[float, ...], expected_job: int
):
    problem, incumbent = case(prefill_waits, decode_waits)
    batch = StructuralOperator(5).propose(problem, incumbent, 1, Config(), random.Random(8))
    assert batch.proposals
    old = operation_order(incumbent.genotype)
    assert all(
        operation_order(proposal.genotype).index(2 * expected_job) < old.index(2 * expected_job)
        for proposal in batch.proposals
    )


@pytest.mark.parametrize("action", [4, 5])
def test_empty_wait_does_not_consume_rng_or_evaluation(action):
    problem, incumbent = case((0, 0, 0, 0), (0, 0, 0, 0))
    rng = random.Random(19)
    state = rng.getstate()
    batch = StructuralOperator(action).propose(problem, incumbent, 6, Config(), rng)
    assert rng.getstate() == state
    assert batch.proposals == []
    assert batch.attempts == 0
    gateway = EvaluationGateway(problem, Archive())
    result = execute(
        batch,
        incumbent,
        6,
        (0.5, 0.5),
        NormalizationContext((0, 0), (2, 2)),
        gateway,
        f"A{action}",
    )
    assert result.best is None
    assert result.effective == result.feasible_count == 0
    assert gateway.counts["ssgs"] == gateway.counts["exact"] == 0
    assert reward(1, 1, result.feasible_count) == -1


@pytest.mark.parametrize("action", [4, 5])
def test_empty_action_trajectory_keeps_incumbent(problem, action):
    gateway = EvaluationGateway(problem, Archive())
    incumbent = gateway.evaluate(Genotype((0, 0, 1, 1), (0, 1, 0, 1)))
    before = (gateway.counts["ssgs"], gateway.counts["exact"], gateway.archive.attempts)
    config = Config(enabled_operators=(action,), rl_steps=1, polish=False)
    improved, records = improve(
        incumbent,
        0,
        0,
        0.0,
        ((0.5, 0.5),),
        (incumbent.evaluation.flow + 1, incumbent.evaluation.bill + 1),
        gateway,
        Controller(config),
        config,
        RNGManager(7),
    )
    assert improved == incumbent
    assert len(records) == 1
    assert records[0]["effective"] == records[0]["feasible"] == 0
    assert records[0]["reward"] == -1
    assert (gateway.counts["ssgs"], gateway.counts["exact"], gateway.archive.attempts) == before
