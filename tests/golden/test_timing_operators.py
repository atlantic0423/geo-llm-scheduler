"""A7/A8/Polish adversarial and frozen-phenotype golden cases."""

import random
from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import (
    Genotype,
    Job,
    ProblemInstance,
    Profile,
    Region,
    Schedule,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.operators.active_pack import ActivePack, rank_crowding
from geo_llm_scheduler.operators.peak_coalition import (
    PeakCoalition,
    coalition,
    removal_potential,
    repair_times,
    support_segments,
)
from geo_llm_scheduler.operators.right_shift import polish


def peak_problem(overlap=False):
    jobs = (
        Job("a", 0, Profile(900, 0.4, 1), Profile(1, 0.1, 1)),
        Job("b", 0, Profile(900, 0.4, 1), Profile(1, 0.1, 1)),
        Job("tail", 0, Profile(1, 0.1, 1), Profile(1, 0.1, 1)),
    )
    p = ProblemInstance(
        jobs,
        (Region("r", (Tariff(0, 10000, 0.1),), 1),),
        (ServingInstance("m", 0, 10, 0, 100), ServingInstance("tail", 0, 10, 0, 0)),
    )
    g = Genotype((0, 1, 0, 1, 1, 1), (0, 1, 2, 0, 1, 2))
    s = Schedule((0, 1800, 0 if overlap else 900, 1801, 0, 2699))
    return p, g, s


def test_a7_gain_frozen_and_replay(problem):
    gateway = EvaluationGateway(problem, Archive())
    x = gateway.evaluate(Genotype((0, 1, 0, 1), (0, 1, 0, 1)), Schedule((0, 500, 200, 700)))
    batch = ActivePack().propose(problem, x, 6, Config(), random.Random(1))
    assert batch.proposals
    for proposal in batch.proposals:
        assert sum(a != b for a, b in zip(proposal.schedule.starts, x.schedule.starts)) == 1
        c = gateway.evaluate(proposal.genotype, proposal.schedule, "A7")
        assert c.evaluation.feasible
        if proposal.selected[0] % 2 == 0:
            assert c.evaluation.flow == x.evaluation.flow
    assert gateway.counts["ssgs"] == 0


def test_crowding_zero_dimensions():
    ranks, crowd = rank_crowding([(0, 0, 1, 0), (1, 1, 1, 0), (2, 2, 1, 0)])
    assert set(ranks.values()) == {0}
    assert set(crowd.values()) == {0}


def test_h_zero_coalition_and_small_accumulation():
    p, g, s = peak_problem(True)
    segments = support_segments(p, g, s, 0, (0,))
    assert removal_potential(segments, frozenset({0}), 0) == 0
    assert removal_potential(segments, frozenset({0, 2}), 0) == 100
    assert coalition(segments, (0,), 8, random.Random(1)) == frozenset({0, 2})
    tiny = [(0, frozenset({i}), 0.4e-9) for i in range(4)]
    chosen = coalition(tiny, (0,), 8, random.Random(1))
    assert len(chosen) == 3
    assert len(coalition(tiny, (0,), 2, random.Random(1))) == 2


@pytest.mark.parametrize("overlap", [False, True])
def test_a8_peak_gate_replay_budget_and_horizon(overlap):
    p, g, s = peak_problem(overlap)
    gateway = EvaluationGateway(p, Archive())
    x = gateway.evaluate(g, s)
    a = PeakCoalition().propose(p, x, 10, Config(), random.Random(12))
    b = PeakCoalition().propose(p, x, 10, Config(), random.Random(12))
    assert a == b
    assert a.attempts <= 20 and len(a.proposals) <= 10
    assert a.proposals
    for proposal in a.proposals:
        y = gateway.evaluate(g, proposal.schedule, "A8")
        assert y.evaluation.feasible
        assert max(y.evaluation.windows[0]) < max(x.evaluation.windows[0]) - 1e-9
        assert y.schedule.horizon(p) <= s.horizon(p)
        assert all(
            a == b
            for o, (a, b) in enumerate(zip(s.starts, y.schedule.starts))
            if o not in proposal.selected
        )


def test_midpoint_candidate():
    p, g, s = peak_problem(True)
    partial = replace(s, starts=(-1, s.starts[1], -1, s.starts[3], 0, 2699))
    times = repair_times(p, g, s, partial, frozenset({0, 2}), 0, 0, (0, 900))
    assert 450 in times


def test_polish_bill_only_and_archive(problem):
    regions = (
        replace(problem.regions[0], tariffs=(Tariff(0, 100, 0.5), Tariff(100, 10000, 0.1))),
        problem.regions[1],
    )
    p = replace(problem, regions=regions, jobs=(problem.jobs[0],))
    gateway = EvaluationGateway(p, Archive())
    x = gateway.evaluate(Genotype((0, 1), (0, 0)), Schedule((0, 500)))
    y = polish(p, x, gateway)
    assert y.evaluation.bill < x.evaluation.bill
    assert y.evaluation.flow == x.evaluation.flow
    assert y.schedule.starts[1] == 500
    assert gateway.counts["ssgs"] == 0
    assert gateway.archive.attempts == gateway.counts["feasible"]
    assert gateway.counts["exact:polish"] > 0
