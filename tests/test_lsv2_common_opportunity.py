"""Capacity certificates, half-open boundaries, selection and full gateway regression."""

import math
from dataclasses import asdict, replace
from random import Random

import pytest
from test_lsv2_common_refine import peak_problem
from test_lsv2_fast_rightshift import CAL, _prepared, synthetic
from test_lsv2_group_polish import joint_problem
from test_lsv2_polish_selection import CTX, FRAME, markers

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Job, Profile, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import lsv2_common_opportunity as study
from geo_llm_scheduler.experiments import lsv2_common_refine as previous
from geo_llm_scheduler.lsv2.common_refine import ReusedCommonPolish, _CalendarSearch
from geo_llm_scheduler.lsv2.flowsafe import FlowSafeBillMoves
from geo_llm_scheduler.lsv2.group_polish import _Search
from geo_llm_scheduler.lsv2.opportunity_selection import OPPORTUNITY_POLICIES, OpportunitySelector
from geo_llm_scheduler.lsv2.polish_selection import PolishSelector, physical_key
from geo_llm_scheduler.lsv2.shift_calendar import ShiftCalendar
from geo_llm_scheduler.moead.core import weights
from geo_llm_scheduler.scheduling.resources import _fits_intervals


@pytest.mark.parametrize("peak", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_calendar_preserves_full_proposals_rng_and_exact(peak, seed):
    for problem, c in (joint_problem(), peak_problem(), synthetic(25), synthetic(left_only=True)):
        old, new = (
            ReusedCommonPolish(peak_switch=peak),
            ReusedCommonPolish(peak_switch=peak, resource_calendar=True),
        )
        for _ in range(2):
            a, b = Random(seed), Random(seed)
            x, y = old.propose(problem, c, 3, Config(), a), new.propose(problem, c, 3, Config(), b)
            assert x.proposals == y.proposals and a.getstate() == b.getstate()
            assert x.diagnostics["incremental_deltas"] == y.diagnostics["incremental_deltas"]
            for p in y.proposals:
                ev = evaluate(problem, p.genotype, p.schedule)
                assert ev.feasible and ev.flow == c.evaluation.flow and ev.bill < c.evaluation.bill


@pytest.mark.parametrize("seed", range(10))
def test_calendar_resource_equivalence_on_every_event_midpoint_and_random_shift(seed):
    problem, c = joint_problem()
    prepared = FlowSafeBillMoves(right_only=True)._prepare(problem, c)
    old, new = _Search(problem, c, prepared), _CalendarSearch(problem, c, prepared, False)
    rng = Random(seed)
    for group in old.groups(score_order=False):
        shifts = [0.0, *old._shifts(group)]
        values = shifts + [(x + y) / 2 for x, y in zip(shifts, shifts[1:])]
        values += [rng.uniform(0, shifts[-1]) for _ in range(200)]
        for d in values:
            assert old._fits(group, d) == new._fits(group, d)


def test_open_conflict_intervals_and_compute_vram_tolerances():
    problem, c = joint_problem()
    # Two operations whose combined compute/VRAM can be configured independently.
    for compute, vram, conflict in (
        (0.6, 1, True),
        (0.1, 6, True),
        (0.5, 5, False),
        (0.5 + 1e-12, 5, False),
    ):
        jobs = tuple(
            replace(j, prefill=replace(j.prefill, compute=compute, vram=vram)) for j in problem.jobs
        )
        problem = replace(
            problem, jobs=jobs, instances=tuple(replace(s, vram=10) for s in problem.instances)
        )
        cal = ShiftCalendar(problem, 0, [(0, 0.0, 10.0)], [(2, 20.0, 30.0)], 40.0)
        assert cal.rejects(20.0) == conflict
        assert not cal.rejects(10.0) and not cal.rejects(30.0)
        assert not cal.rejects(10.0 + cal.guard / 2)
        assert not cal.rejects(0.0) and not cal.rejects(40.0)
    assert not ShiftCalendar(problem, 0, [(0, 0, 10)], [], 40).rejects(20)


def test_empty_certificate_scope_invalidation_and_nonempty_not_excluded():
    problem, c = synthetic(left_only=True)
    op = ReusedCommonPolish(resource_calendar=True)
    assert not op.known_empty(problem, c, 3, Config())
    batch = op.propose(problem, c, 3, Config(), Random(4))
    assert not batch.proposals and op.known_empty(problem, c, 3, Config())
    assert not op.known_empty(replace(problem), c, 3, Config())
    assert not op.known_empty(problem, c, 1, Config())
    changed = replace(c, schedule=Schedule((c.schedule.starts[0] + 1e-12, *c.schedule.starts[1:])))
    assert not op.known_empty(problem, changed, 3, Config())
    op.peak_switch = True
    assert not op.known_empty(problem, c, 3, Config())
    op.peak_switch = False
    op.retain([])
    assert not op.known_empty(problem, c, 3, Config())
    problem, c = joint_problem()
    assert op.propose(problem, c, 3, Config(), Random(4)).proposals
    assert not op.known_empty(problem, c, 3, Config())


@pytest.mark.parametrize("policy", OPPORTUNITY_POLICIES)
def test_certificate_exclusion_new_inputs_and_all_empty_are_not_real_calls(policy):
    a, b, c, d = markers()
    selector = OpportunitySelector()
    rng = Random(1)
    pick = selector.choose(policy, [a, b, c], d, FRAME, CTX, weights(100), rng, lambda x: x != b)
    assert pick.candidate == b and selector.diagnostics["certified_empty_excluded"] == 3
    selector.completed(b)
    selector.charge(pick.region, 0.7)
    state = rng.getstate()
    assert (
        selector.choose(policy, [a, b, c], d, FRAME, CTX, weights(100), rng, lambda x: True) is None
    )
    assert state == rng.getstate()
    assert selector.accounting.region_seconds[pick.region] == 0.7
    # Drop old inputs rather than accumulating a new unbounded visit table.
    selector.choose(policy, [a], a, FRAME, CTX, weights(100), rng, lambda x: False)
    assert physical_key(b) not in selector.visits


@pytest.mark.parametrize("policy", ["fresh_region", "frontier_region", "fresh_direction"])
def test_fresh_tier_prevents_repeated_success_starving_dominated_inputs(policy):
    a = markers()[1]
    b = replace(
        a,
        schedule=Schedule((7.0, *a.schedule.starts[1:])),
        evaluation=replace(
            a.evaluation, flow=a.evaluation.flow + 0.01, tou=a.evaluation.tou + 0.01
        ),
    )
    selector = OpportunitySelector()
    selected = []
    for _ in range(4):
        p = selector.choose(policy, [a, b], a, FRAME, CTX, weights(3), Random(4), lambda x: False)
        selected.append(physical_key(p.candidate))
        selector.completed(p.candidate)
    assert set(selected[:2]) == {physical_key(a), physical_key(b)}
    assert sorted(selector.visits.values()) == [2, 2]


def test_lazy_direction_diagnostics_preserve_old_selection_and_rng():
    pool = markers()
    for policy in ("pool_random", "region_time", "direction"):
        a, b = PolishSelector(), PolishSelector()
        ra, rb = Random(2), Random(2)
        for _ in range(20):
            x = a.choose(policy, pool, pool[0], FRAME, CTX, weights(100), ra)
            y = b.choose(
                policy, pool, pool[0], FRAME, CTX, weights(100), rb, diagnose_directions=False
            )
            assert (
                x.candidate == y.candidate
                and x.region == y.region
                and ra.getstate() == rb.getstate()
            )
            a.charge(x.region, 0.3)
            b.charge(y.region, 0.3)
            assert bool(y.direction_sizes) == (policy == "direction")


@pytest.mark.parametrize("arm", study.ARMS)
def test_full_gateway_preserves_terminal_credit_and_scope(arm):
    search, pop, gateway, streams = _prepared("Q1_NET_G9_B3L3_NO_P3")
    search.config = study.search_config(arm)
    reference, rp, rg, rs = _prepared("Q1_NET_G9_B3L3_NO_P3")
    reference._trajectory(rp, rg, rs, lambda: 10)
    search._trajectory(pop, gateway, streams, lambda: 10)
    assert asdict(search.table) == asdict(reference.table)
    assert search.steps[-1]["q_transition"]["terminal"] and search._pending is None
    if search.config.polish_selection in OPPORTUNITY_POLICIES:
        row = search.polish_steps[-1]
        assert sum(search.opportunity_selector.accounting.region_seconds) == pytest.approx(
            row["seconds"]
        )
        assert sum(search.opportunity_selector.visits.values()) == 1
    for c in pop + gateway.archive.members:
        assert c.evaluation == evaluate(gateway.problem, c.genotype, c.schedule)
    record = study.configuration_record(arm, 16901, 1200, CAL)
    assert record["qspec"]["gamma"] == 0.9 and record["lsv2"]["time_share"] == 0.2
    assert record["lsv2"]["trajectory_steps"] == record["lsv2"]["budget"] == 3
    assert "P3" not in search.config.operators


def test_matrix_and_calendar_equivalence_through_next_episode():
    assert study.search_config("R0_NONE") == previous.search_config("C0_NONE")
    assert study.search_config("R1_FAST") == previous.search_config("C2_FAST")
    for fn, args in ((study.search_config, ("bad",)), (study.arm_config, ("bad", 1, 10))):
        with pytest.raises(ValueError):
            fn(*args)
    with pytest.raises(ValueError):
        replace(study.search_config("R3_ELIGIBLE"), polish_method="common")
    a, ap, ag, ar = _prepared("Q1_NET_G9_B3L3_NO_P3")
    b, bp, bg, br = _prepared("Q1_NET_G9_B3L3_NO_P3")
    a.config, b.config = study.search_config("R1_FAST"), study.search_config("R2_CALENDAR")
    for _ in range(4):
        a._trajectory(ap, ag, ar, lambda: 10)
        b._trajectory(bp, bg, br, lambda: 10)
        assert ap == bp and ag.archive.members == bg.archive.members
        assert asdict(a.table) == asdict(b.table)


def test_no_opportunity_keeps_q_and_does_not_mislabel_empty_generation(monkeypatch):
    search, pop, gateway, streams = _prepared("Q1_NET_G9_B3L3_NO_P3")
    search.config = study.search_config("R4_REGION")
    monkeypatch.setattr(search.reused_polish, "known_empty", lambda *a: True)
    monkeypatch.setattr(
        search.reused_polish, "propose", lambda *a: pytest.fail("no input eligible")
    )
    search._trajectory(pop, gateway, streams, lambda: 10)
    row = search.polish_steps[-1]
    assert row["status"] == "no_opportunity" and row["exact"] == 0 and "diagnostics" not in row
    assert search.opportunity_selector.visits == {}
    assert search._pending is None


def test_selection_deadline_does_not_create_failure_or_visit(monkeypatch):
    search, pop, gateway, streams = _prepared("Q1_NET_G9_B3L3_NO_P3")
    search.config = study.search_config("R3_ELIGIBLE")
    monkeypatch.setattr(search.reused_polish, "propose", lambda *a: pytest.fail("past deadline"))
    ticks = iter((10, 101))
    search._polish(pop[0], FRAME, 0, pop, gateway, streams, lambda: next(ticks))
    assert search.polish_steps[-1]["status"] == "deadline_after_selection"
    assert not search.reused_polish._entries and not search.opportunity_selector.visits


@pytest.mark.parametrize("seed", range(12))
def test_calendar_random_profiles_never_reject_reference_feasible_position(seed):
    rng = Random(seed)
    template, _ = joint_problem()
    for offset in (0.0, 1e-10, 1e6, 1e12):
        jobs = tuple(
            Job(
                str(i),
                0,
                Profile(
                    rng.uniform(1, 10),
                    rng.choice((0.25, 0.5, 0.1, 1e-12)),
                    rng.choice((1, 3, 5, 1e-12)),
                ),
                Profile(1, 0.1, 1),
            )
            for i in range(12)
        )
        problem = replace(template, jobs=jobs)
        items = []
        for o in range(0, 24, 2):
            start = offset + rng.uniform(0, 30)
            items.append((o, start, start + problem.profile(o).duration))
        moving, fixed = items[:4], items[4:]
        cal = ShiftCalendar(problem, 0, moving, fixed, 40)
        points = [rng.uniform(0, 40) for _ in range(100)]
        for _, a, b in moving:
            for _, c, d in fixed:
                for t in (c - b, d - a):
                    points += [t, math.nextafter(t, -math.inf), math.nextafter(t, math.inf)]
        for delta in points:
            if not 0 <= delta <= 40 or not cal.rejects(delta):
                continue
            moved = [(o, a + delta, a + delta + problem.profile(o).duration) for o, a, _ in moving]
            merged = sorted(fixed + moved)
            assert not all(
                _fits_intervals(problem, 0, o, t, [v for v in merged if v[0] != o])
                for o, t, _ in moved
            )


def test_selector_validation_and_study_constructor():
    pool = markers()
    with pytest.raises(ValueError):
        OpportunitySelector().choose(
            "bad", pool, pool[0], FRAME, CTX, weights(3), Random(1), lambda c: False
        )
    old, _, _, _ = _prepared("Q1_NET_G9_B3L3_NO_P3")
    for arm in study.ARMS:
        config = study.arm_config(arm, 16901, 1200)
        built = study.make_search(arm, config, CAL, old.table, training=True)
        assert built.config == study.search_config(arm) and built.table is old.table


def test_frontier_preference_is_nonempty_at_near_tolerance_ties():
    a = markers()[1]
    pool = [
        replace(
            a,
            schedule=Schedule((float(i), *a.schedule.starts[1:])),
            evaluation=replace(
                a.evaluation, flow=a.evaluation.flow + dx, tou=a.evaluation.tou + dy
            ),
        )
        for i, (dx, dy) in enumerate(((0, 0), (1e-9, -2e-9), (-2e-9, 1e-9), (-1e-9, -1e-9)))
    ]
    selector = OpportunitySelector()
    for _ in pool:
        pick = selector.choose(
            "frontier_region", pool, pool[0], FRAME, CTX, weights(3), Random(2), lambda c: False
        )
        assert pick is not None
        selector.completed(pick.candidate)
    assert set(selector.visits.values()) == {1}


def test_opportunity_selection_does_not_require_resource_calendar():
    search, pop, gateway, streams = _prepared("Q1_NET_G9_B3L3_NO_P3")
    search.config = replace(study.search_config("R5_FRESH"), polish_method="common_fast")
    search._trajectory(pop, gateway, streams, lambda: 10)
    assert not search.reused_polish.resource_calendar
    assert search.polish_steps[-1]["status"] in ("empty", "evaluated")
    assert sum(search.opportunity_selector.visits.values()) == 1
