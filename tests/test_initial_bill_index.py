"""Strict reference equivalence, temporary P ordering and exact billing edges."""

import random

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Schedule
from geo_llm_scheduler.experiments.initial_bill_benchmark import benchmark_population
from geo_llm_scheduler.experiments.initial_bill_index import ResourceCalendar, TariffIndex
from geo_llm_scheduler.experiments.initial_bill_mixture import (
    PrefixPower,
    guided_genotype,
    tariff_integral,
)
from geo_llm_scheduler.experiments.initial_frontier import diagnostic_problem
from geo_llm_scheduler.scheduling.resources import earliest
from geo_llm_scheduler.scheduling.ssgs import decode


def test_tariff_index_exact_edges_cache_and_input_scope(problem):
    index = TariffIndex(problem)
    rng = random.Random(192)
    for r in range(len(problem.regions)):
        end = problem.regions[r].tariffs[-1].end
        points = [0.0, end, 900.0, 1000.0] + [rng.random() * end for _ in range(40)]
        for a, b in zip(sorted(points), sorted(points)[1:]):
            assert index.integral(problem, r, a, b) == tariff_integral(problem, r, a, b)
            assert index.integral(problem, r, a, b) == tariff_integral(problem, r, a, b)
        assert index.integral(problem, r, end, end) == 0.0
        with pytest.raises(ValueError, match="outside"):
            index.integral(problem, r, 0.0, end + 1)
    other = diagnostic_problem(6, 45, "T", "mixed")
    with pytest.raises(ValueError, match="another"):
        index.integral(other, 0, 0, 1)


def test_calendar_temporary_and_out_of_order_operation_replay():
    p = diagnostic_problem(20, 19246, "T", "compute_tight")
    g, _ = guided_genotype(p, "BILL", random.Random(83), 0.1)
    schedule = decode(p, g)
    calendar = ResourceCalendar(p)
    starts = [-1.0] * 40
    for job in g.os[::2]:
        po, do = 2 * job, 2 * job + 1
        pm, dm = g.ms[po], g.ms[do]
        ps = earliest(p, g, Schedule(tuple(starts)), po, p.jobs[job].release)
        assert calendar.earliest(pm, po, p.jobs[job].release) == ps
        starts[po] = ps
        est = ps + p.jobs[job].prefill.duration + (pm != dm) * p.jobs[job].kv_delay
        ds = earliest(p, g, Schedule(tuple(starts)), do, est)
        temp = (po, ps, ps + p.jobs[job].prefill.duration) if pm == dm else None
        assert calendar.earliest(dm, do, est, temp) == ds
        starts[do] = ds
        calendar.commit(pm, po, ps)
        calendar.commit(dm, do, ds)
    assert tuple(starts) == schedule.starts


def test_calendar_infeasible_assignment_raises(problem):
    from dataclasses import replace

    job = replace(problem.jobs[0], prefill=replace(problem.jobs[0].prefill, compute=1.1))
    p = replace(problem, jobs=(job, *problem.jobs[1:]))
    with pytest.raises(ValueError, match="cannot fit"):
        ResourceCalendar(p).earliest(0, 0, 0)


def test_prefix_power_exact_overlaps_and_tail(problem):
    slow = PrefixPower(problem)
    fast = PrefixPower(problem, index=TariffIndex(problem))
    for additions, horizon in (
        ({0: [(0.0, 900.0), (100.0, 400.0)]}, 900.0),
        ({0: [(899.5, 1100.0)], 1: [(0.0, 945.0)]}, 1100.0),
        ({}, 1100.0),
    ):
        a, b = slow.trial(additions, horizon), fast.trial(additions, horizon)
        assert a == b
        slow.commit(a)
        fast.commit(b)


@pytest.mark.parametrize(
    "scenario", ("mixed", "burst", "compute_tight", "vram_tight", "demand_high")
)
@pytest.mark.parametrize("weight", (0.0, 0.1, 0.25, 0.5))
def test_guided_exact_genes_scores_and_rng(scenario, weight):
    p = diagnostic_problem(16, 19250, "T", scenario)
    a, b = random.Random(47), random.Random(47)
    ref = guided_genotype(p, "BILL", a, weight)
    fast = guided_genotype(p, "BILL", b, weight, accelerated=True)
    assert ref == fast
    assert a.getstate() == b.getstate()


@pytest.mark.parametrize("family", ("TOU", "DEMAND"))
def test_other_guides_remain_exact(family):
    p = diagnostic_problem(12, 19251, "T", "mixed")
    assert guided_genotype(p, family, random.Random(14), 0.25) == guided_genotype(
        p, family, random.Random(14), 0.25, accelerated=True
    )


def test_bank_all_winners_observations_rng_and_population_exact():
    p = diagnostic_problem(12, 19252, "T", "mixed")
    cfg = Config(seed=571)
    slow = benchmark_population(p, cfg, "BILL10", 911)
    fast = benchmark_population(p, cfg, "BILL10", 911, accelerated=True)
    assert slow[:2] == fast[:2]
    for key in ("observations", "rng_state_sha256", "exact", "trials_per_slot"):
        assert slow[2]["bank_stats"][key] == fast[2]["bank_stats"][key]
    for trials in (1, 2):
        pop, _, stats = benchmark_population(p, cfg, "BILL10", 911, accelerated=True, trials=trials)
        assert len(pop) == len({c.genotype for c in pop}) == 100
        assert stats["new_candidate_exact"] == 10 * trials
        assert stats["final_independent_exact_verified"] == 100
