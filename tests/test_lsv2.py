"""LS v2 prototypes: incremental Bill equals exact, Flow invariance, valid genotypes, seeding rules."""

import math
import random

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.validation import validate_genotype
from geo_llm_scheduler.lsv2.billing import PowerProfile, difference, merge
from geo_llm_scheduler.lsv2.destroy_repair import MODES, TargetedDestroyRepair, job_scores
from geo_llm_scheduler.lsv2.flowsafe import FlowSafeBillMoves
from geo_llm_scheduler.lsv2.frontier import (
    FrozenFrame,
    SeedSampler,
    demand_gate,
    gap_scores,
    nondominated,
)
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.scheduling.timing import legal_bounds, move


@pytest.fixture(scope="module")
def cases():
    problem = case_problem(20, 3, 6, 202610160020, "arrival", 0)
    out = []
    for g in initial_genotypes(problem, Config(seed=3), random.Random(3))[:6]:
        s = decode(problem, g)
        out.append(Candidate(g, s, evaluate(problem, g, s), "test"))
    return problem, out


def test_merge_and_signed_difference():
    assert merge([(3, 4), (0, 2), (1, 3)]) == [(0, 4)]
    assert difference([(0, 4)], [(1, 5)]) == [(0, 1, -1), (4, 5, 1)]


def test_profile_bill_and_move_deltas_equal_exact(cases):
    problem, members = cases
    rng = random.Random(7)
    for c in members:
        profile = PowerProfile(problem, c.genotype.ms, c.schedule.starts)
        assert math.isclose(profile.bill(), c.evaluation.bill, rel_tol=1e-12)
        for _ in range(25):
            o = 2 * rng.randrange(len(problem.jobs))
            lo, hi = legal_bounds(problem, c.genotype, c.schedule, o)
            if hi <= lo:
                continue
            t = lo + (hi - lo) * rng.random()
            moved = move(problem, c.genotype, c.schedule, o, t)
            if moved is None:
                continue
            change = {c.genotype.ms[o]: {o: (t, t + problem.profile(o).duration)}}
            exact = evaluate(problem, c.genotype, moved)
            assert math.isclose(
                c.evaluation.bill + profile.delta_for(change), exact.bill, rel_tol=1e-9
            )
            copy = profile.copy()
            copy.apply(change)
            assert math.isclose(copy.bill(), exact.bill, rel_tol=1e-9)
            assert math.isclose(profile.bill(), c.evaluation.bill, rel_tol=1e-12)


def test_horizon_extension_restores_removed_last_job(cases):
    problem, members = cases
    c = members[0]
    last = max(range(len(problem.jobs)), key=lambda j: c.schedule.end(problem, 2 * j + 1))
    starts = list(c.schedule.starts)
    starts[2 * last] = starts[2 * last + 1] = -1.0
    partial = PowerProfile(problem, c.genotype.ms, starts)
    end = c.schedule.end(problem, 2 * last + 1)
    longer = partial.with_horizon(end)
    change = {}
    for o in (2 * last, 2 * last + 1):
        change.setdefault(c.genotype.ms[o], {})[o] = (
            c.schedule.starts[o],
            c.schedule.end(problem, o),
        )
    restored = longer.bill() + longer.delta_for(change)
    assert math.isclose(restored, c.evaluation.bill, rel_tol=1e-9)


def test_flowsafe_keeps_flow_and_lowers_bill(cases):
    problem, members = cases
    op = FlowSafeBillMoves()
    found = 0
    for c in members:
        batch = op.propose(problem, c, 4, Config(seed=1), random.Random(1))
        for prop in batch.proposals:
            e = evaluate(problem, prop.genotype, prop.schedule)
            assert e.feasible and math.isclose(e.flow, c.evaluation.flow, rel_tol=1e-12)
            assert e.bill < c.evaluation.bill
            assert all(o % 2 == 0 for o in prop.selected)
            assert all(
                a == b
                for o, (a, b) in enumerate(zip(c.schedule.starts, prop.schedule.starts))
                if o not in prop.selected
            )
            found += 1
        gateway = EvaluationGateway(problem, Archive())
        context = NormalizationContext((0.0, 0.0), c.evaluation.objectives)
        execute(batch, c, 4, (0.5, 0.5), context, gateway, "FS")
    assert found > 0


@pytest.mark.parametrize("mode", MODES)
def test_destroy_repair_returns_valid_decodable_genotypes(cases, mode):
    problem, members = cases
    c = members[1]
    assert len(job_scores(problem, c, mode)) == len(problem.jobs)
    op = TargetedDestroyRepair(mode=mode)
    batch = op.propose(problem, c, 3, Config(seed=1), random.Random(5))
    again = TargetedDestroyRepair(mode=mode).propose(
        problem, c, 3, Config(seed=1), random.Random(5)
    )
    assert [p.genotype for p in batch.proposals] == [p.genotype for p in again.proposals]
    assert batch.proposals
    for prop in batch.proposals:
        assert prop.schedule is None
        validate_genotype(problem, prop.genotype)
        s = decode(problem, prop.genotype)
        assert evaluate(problem, prop.genotype, s).feasible


def test_gap_scores_frame_and_weights():
    front = [(0.0, 1.0), (0.2, 0.5), (1.0, 0.0)]
    scores = gap_scores(front, end_bonus=0.0)
    assert scores == pytest.approx([0.05, 0.25, 0.2])
    frame = FrozenFrame.of([(10, 30), (20, 20), (30, 10), (25, 25)])
    assert frame.ideal == (10, 10) and frame.nadir == (30, 30)
    assert frame.hv_gain([(10, 30), (30, 10)], [(15, 15)]) > 0
    assert frame.hv_gain([(10, 30), (20, 20), (30, 10)], [(25, 25)]) == 0
    wf, wb = frame.weight_for((20, 20))
    assert wf == pytest.approx(0.5) and wb == pytest.approx(0.5)
    assert nondominated([(1, 2), (2, 1), (2, 2)]) == [(1, 2), (2, 1)]


def test_seed_sampler_rules_and_discount():
    frame = FrozenFrame((0, 0), (1, 1))
    front = [(0.05 * i, 1 - 0.05 * i) for i in range(21)]
    flow_end = SeedSampler("flow_end", random.Random(0))
    picks = [frame.normalize(front[flow_end.pick(frame, front)])[0] for _ in range(400)]
    assert sum(x < 1 / 3 for x in picks) / len(picks) > 0.7
    gap = SeedSampler("gap", random.Random(0))
    for _ in range(5):
        gap.report(front[0], False)
    assert gap.failures[front[0]] == 5
    gap.report(front[0], True)
    assert front[0] not in gap.failures
    with pytest.raises(ValueError):
        SeedSampler("other", random.Random(0)).pick(frame, front)


def test_demand_gate(cases):
    problem, members = cases
    c = members[0]
    assert demand_gate(c, 0.0)
    assert not demand_gate(c, 1.01)
