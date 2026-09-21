"""Independent hand-integrated billing and feasibility oracles."""

from dataclasses import replace

import pytest

from geo_llm_scheduler.domain.models import Genotype, Job, Profile, Schedule, Tariff
from geo_llm_scheduler.evaluation.exact import evaluate, regional_windows, tied_peaks


def test_fixed_tail_and_ties():
    assert regional_windows(((0, 300, 100),), 300) == pytest.approx((100 / 3,))
    assert tied_peaks((30, 30, 20)) == (0, 1)
    assert regional_windows(((0, 1800, 100),), 1800) == (100, 100)


def test_union_and_idle_all_servers(problem):
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    e = evaluate(problem, g, Schedule((0, 100, 0, 100)))
    assert e.feasible
    assert e.flow == 600
    # r0: one active 100 kW + unused idle 10; r1: unused idle 10.
    assert e.tou == pytest.approx((110 * 0.2 + 10 * 0.1) * 300 / 3600)
    assert e.windows[0] == pytest.approx((110 / 3,))
    assert e.demand == pytest.approx((2 * 110 / 3, 3 * 10 / 3))
    assert e.bill == e.tou + sum(e.demand)


def test_cross_tou_boundary(problem):
    region = replace(problem.regions[0], tariffs=(Tariff(0, 150, 0.1), Tariff(150, 10000, 0.3)))
    p = replace(problem, regions=(region, problem.regions[1]))
    e = evaluate(p, Genotype((0, 0, 0, 0), (0, 1, 0, 1)), Schedule((0, 100, 0, 100)))
    assert e.tou == pytest.approx(110 * 150 * (0.1 + 0.3) / 3600 + 10 * 300 * 0.1 / 3600)


def test_precedence_release_and_kv(problem):
    g = Genotype((0, 1, 0, 0), (0, 0, 1, 1))
    assert not evaluate(problem, g, Schedule((0, 100, 0, 100))).feasible
    assert evaluate(problem, g, Schedule((0, 130, 0, 100))).feasible
    assert not evaluate(problem, g, Schedule((-1, 130, 0, 100))).feasible


def test_completion_before_start_and_capacity(problem):
    job = Job("a", 0, Profile(100, 1, 8), Profile(200, 1, 8))
    p = replace(problem, jobs=(job,))
    g = Genotype((0, 0), (0, 0))
    assert evaluate(p, g, Schedule((0, 100))).feasible
    assert not evaluate(p, g, Schedule((0, 99))).feasible
    p = replace(problem, jobs=(job, replace(job, name="b")))
    g = Genotype((0, 0, 0, 0), (0, 0, 1, 1))
    assert evaluate(p, g, Schedule((0, 100, 300, 400))).feasible
    assert not evaluate(p, g, Schedule((0, 100, 299, 400))).feasible


def test_tail_shortening_changes_all_regions(problem):
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    late = evaluate(problem, g, Schedule((0, 500, 0, 500)))
    early = evaluate(problem, g, Schedule((0, 100, 0, 100)))
    assert early.windows[1][0] < late.windows[1][0]
    assert early.demand[1] < late.demand[1]


def test_empty_invalid_and_tariff_coverage(problem):
    empty = replace(problem, jobs=())
    e = evaluate(empty, Genotype((), ()), Schedule(()))
    assert e.feasible and e.bill == 0 and e.flow == 0
    assert not evaluate(
        problem, Genotype((0,) * 4, (0, 1, 0, 1)), Schedule((float("nan"), 0, 0, 0))
    ).feasible
    with pytest.raises(ValueError, match="Tariff"):
        evaluate(problem, Genotype((0,) * 4, (0, 1, 0, 1)), Schedule((0, 10000, 0, 10000)))
