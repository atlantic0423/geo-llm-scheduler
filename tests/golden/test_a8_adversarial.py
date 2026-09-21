"""Strict all-window gating and bounded repair failure cases."""

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
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition, peak_reduced, repair


def test_global_last_decode_shortening_reduces_all_region_tails():
    # Idle power alone makes two equal original peaks. The last D overlaps a
    # zero active-minus-idle server, so direct H=0, yet shortening removes tail.
    p = ProblemInstance(
        (Job("j", 0, Profile(1, 1, 1), Profile(100, 1, 1)),),
        (Region("a", (Tariff(0, 4000, 0.2),), 1), Region("b", (Tariff(0, 4000, 0.1),), 2)),
        (ServingInstance("a0", 0, 2, 100, 100), ServingInstance("b0", 1, 2, 50, 50)),
    )
    g = Genotype((0, 0), (0, 0))
    gateway = EvaluationGateway(p, Archive())
    x = gateway.evaluate(g, Schedule((0, 1700)))
    y = gateway.evaluate(g, Schedule((0, 1)), "A8-tail")
    assert x.evaluation.windows[0] == (100, 100)
    assert peak_reduced(p, g, y.schedule, 0, 100)
    assert y.evaluation.windows[0] == pytest.approx((100 * 101 / 900,))
    assert y.evaluation.windows[1] == pytest.approx((50 * 101 / 900,))
    assert y.evaluation.tou == pytest.approx((100 * 0.2 + 50 * 0.1) * 101 / 3600)
    # The bounded original-support heuristic is allowed to miss this move.
    batch = PeakCoalition().propose(p, x, 3, Config(), random.Random(1))
    assert not batch.proposals
    assert x.schedule.starts == (0, 1700)


def test_a8_no_positive_demand_has_no_proposals(problem):
    p = replace(problem, regions=tuple(replace(r, demand_rate=0) for r in problem.regions))
    gateway = EvaluationGateway(p, Archive())
    x = gateway.evaluate(Genotype((0, 0, 0, 0), (0, 1, 0, 1)))
    assert not PeakCoalition().propose(p, x, 3, Config(), random.Random(1)).proposals


def test_moved_peak_rejected_and_midpoint_accepted():
    p = ProblemInstance(
        (Job("j", 0, Profile(900, 1, 1), Profile(1, 1, 1)),),
        (Region("r", (Tariff(0, 4000, 0.1),), 1),),
        (ServingInstance("p", 0, 2, 0, 100), ServingInstance("d", 0, 2, 0, 0)),
    )
    g = Genotype((0, 1), (0, 0))
    assert not peak_reduced(p, g, Schedule((900, 1800)), 0, 100)
    assert peak_reduced(p, g, Schedule((450, 1800)), 0, 100)
    gateway = EvaluationGateway(p, Archive())
    x = gateway.evaluate(g, Schedule((0, 1800)))
    proposals = PeakCoalition().propose(p, x, 10, Config(), random.Random(3))
    assert proposals.proposals
    assert all(len(q.selected) == 1 for q in proposals.proposals)


def test_no_positions_rolls_back_without_mutating_incumbent(problem):
    gateway = EvaluationGateway(problem, Archive())
    x = gateway.evaluate(Genotype((0, 0, 0, 0), (0, 1, 0, 1)))
    original = x.schedule
    assert repair(problem, x, frozenset({0, 1}), 0, 0, random.Random(1)) is None
    assert x.schedule == original


def test_original_horizon_never_extends(problem):
    gateway = EvaluationGateway(problem, Archive())
    x = gateway.evaluate(Genotype((0, 1, 0, 1), (0, 1, 0, 1)))
    for seed in range(10):
        result = repair(problem, x, frozenset({0, 1, 2, 3}), 0, 6, random.Random(seed))
        if result:
            assert result.horizon(problem) <= x.schedule.horizon(problem)
