"""Shared-pool eligibility, allocation, timing, terminal credit and offline evidence."""

import copy
from dataclasses import asdict, replace
from random import Random

import pytest
from test_lsv2_fast_rightshift import CAL, _prepared, synthetic

from geo_llm_scheduler.domain.models import Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import lsv2_fast_rightshift as previous
from geo_llm_scheduler.experiments import lsv2_rightshift_selection as study
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.lsv2.polish import FastRightShiftPolish
from geo_llm_scheduler.lsv2.polish_diagnostics import inspect_start, repetition_summary
from geo_llm_scheduler.lsv2.polish_selection import (
    POOL_POLICIES,
    PolishSelector,
    candidate_pool,
    physical_fingerprint,
    physical_key,
    segment_of,
)
from geo_llm_scheduler.moead.core import NormalizationContext, weights
from geo_llm_scheduler.rl.state import associate
from geo_llm_scheduler.utils.numeric import TOL, close, identity


def markers():
    """Synthetic objective markers, used only to test selection, not evaluation."""
    _, base = synthetic()
    return [
        replace(
            base,
            schedule=Schedule((float(i), *base.schedule.starts[1:])),
            evaluation=replace(base.evaluation, flow=f, tou=b, demand=(0.0,)),
        )
        for i, (f, b) in enumerate(((0.3, 2.7), (1.3, 1.7), (2.3, 0.7), (1.5, 1.5)))
    ]


FRAME = FrozenFrame((0.0, 0.0), (3.0, 3.0))
CTX = NormalizationContext((0.0, 0.0), (3.0, 3.0))


def test_pool_is_physical_not_objective_or_genotype_and_never_mutates_inputs():
    a, b, _, _ = markers()
    other_os = replace(a, genotype=replace(a.genotype, os=tuple(reversed(a.genotype.os))))
    tiny_timing = replace(a, schedule=replace(a.schedule, starts=(1e-12, *a.schedule.starts[1:])))
    pop = [a] * 100 + [b, other_os, tiny_timing]
    before = copy.deepcopy(pop)
    pool = candidate_pool(pop, other_os)
    assert pop == before and len(pool) == 3
    assert pool == candidate_pool(list(reversed(pop)), other_os)
    assert {physical_key(c) for c in pool} == {physical_key(c) for c in (a, b, tiny_timing)}
    assert physical_fingerprint(a) == physical_fingerprint(other_os)
    assert physical_fingerprint(a) != physical_fingerprint(tiny_timing)
    negative_zero = replace(a, schedule=replace(a.schedule, starts=(-0.0, *a.schedule.starts[1:])))
    assert physical_fingerprint(a) == physical_fingerprint(negative_zero)


def test_random_is_uniform_over_distinct_pool_with_a_dedicated_explicit_rng():
    a, b, c, d = markers()
    pop = [a] * 99 + [b, c]
    pool = candidate_pool(pop, d)
    actual, reference = Random(19), Random(19)
    selector = PolishSelector()
    for _ in range(40):
        pick = selector.choose("pool_random", pop, d, FRAME, CTX, weights(100), actual)
        assert pick.candidate == pool[reference.randrange(4)]
        assert pick.pool_size == 4 and sum(pick.region_sizes) == sum(pick.direction_sizes) == 4
        assert pick.source == ("endpoint" if pick.candidate == d else "population")
    assert actual.getstate() == reference.getstate()
    # Repeated selection is legal; there is no failure/exclusion history.
    for _ in range(2):
        assert (
            selector.choose("pool_random", [a] * 100, a, FRAME, CTX, weights(3), actual).source
            == "both"
        )


def test_direction_rotation_skips_empty_groups_and_uses_own_weight():
    pool = markers()
    lambdas = weights(100)
    groups = {}
    for c in pool:
        groups.setdefault(associate(CTX.normalize(c), lambdas), []).append(c)
    selector = PolishSelector()
    for direction in sorted(groups) * 2:
        result = selector.choose("direction", pool, pool[0], FRAME, CTX, lambdas, Random(1))
        assert result.direction == direction
        scores = [(CTX.scalar(c, lambdas[direction]), c) for c in groups[direction]]
        best = min(s for s, _ in scores)
        expected = min((c for s, c in scores if close(s, best, TOL.scalar)), key=identity)
        assert result.candidate == expected
    # Association uses supplied current MOEA/D context, not the frozen LS frame.
    changed = NormalizationContext((-30, 0), (3, 300))
    result = selector.choose("direction", pool, pool[0], FRAME, changed, lambdas, Random(1))
    assert result.direction == associate(changed.normalize(result.candidate), lambdas)


def test_direction_ties_are_anchored_not_chained_and_stably_ordered():
    a, _, _, _ = markers()
    candidates = [
        replace(
            a,
            schedule=Schedule((float(i), *a.schedule.starts[1:])),
            evaluation=replace(a.evaluation, flow=0, tou=0, demand=(i * 0.75e-9,)),
        )
        for i in range(3)
    ]
    selector = PolishSelector()
    # At the bill axis all three associate to direction zero. Only first two tie with minimum.
    result = selector.choose(
        "direction",
        candidates[::-1],
        candidates[0],
        FRAME,
        NormalizationContext((0, 0), (1, 1)),
        weights(3),
        Random(1),
    )
    assert result.candidate is candidates[0]


def test_region_allocator_balances_seconds_not_calls_and_rotates_exact_ties():
    pool = markers()[:3]
    selector = PolishSelector()
    picks = []
    for _ in range(3):
        result = selector.choose("region_time", pool, pool[0], FRAME, CTX, weights(3), Random(1))
        picks.append(result.region)
    assert picks == [0, 1, 2]
    selector.charge(0, 10)
    selector.charge(1, 1)
    selector.charge(2, 2)
    for _ in range(4):
        pick = selector.choose("region_time", pool, pool[0], FRAME, CTX, weights(3), Random(1))
        assert pick.region == 1
        selector.charge(1, 0.1)
    # An empty region with zero cumulative service never blocks the populated ones.
    pick = PolishSelector().choose(
        "region_time", [pool[2]], pool[2], FRAME, CTX, weights(3), Random(1)
    )
    assert pick.region == 2
    for region, seconds in ((-1, 0), (0, -1), (0, float("nan"))):
        with pytest.raises(ValueError):
            selector.charge(region, seconds)
    with pytest.raises(ValueError):
        selector.choose("bad", pool, pool[0], FRAME, CTX, weights(3), Random(1))


def test_matrix_preserves_controls_and_masks_and_rejects_invalid_policy():
    assert len(study.ARMS) == len(set(study.ARMS)) == 12
    assert study.search_config("B0") == previous.search_config(previous.NO_P3)
    assert study.search_config("B1") == previous.search_config(previous.BASE)
    for arm in study.ARMS:
        config = study.search_config(arm)
        assert config.budget == config.trajectory_steps == config.polish_budget == 3
        assert config.time_share == 0.2 and ("P3" in config.operators) == (arm == "B1")
        record = study.configuration_record(arm, 16901, 1200, CAL)
        assert record["lsv2"] == asdict(config) and record["qspec"]["gamma"] == 0.9
        assert study.arm_config(arm, 16901, 1200) == previous.arm_config(previous.BASE, 16901, 1200)
    for fun, args in ((study.search_config, ("bad",)), (study.arm_config, ("bad", 1, 10))):
        with pytest.raises(ValueError):
            fun(*args)
    for kwargs in (dict(polish_selection="bad"), dict(polish_selection="direction")):
        with pytest.raises(ValueError):
            replace(study.search_config("B0"), **kwargs)


@pytest.mark.parametrize("policy", POOL_POLICIES)
def test_new_selection_is_after_terminal_with_identical_q_credit_and_exact_outputs(
    policy, monkeypatch
):
    search, pop, gateway, streams = _prepared(previous.NO_P3)
    search.config = replace(
        search.config, polish_region="all", polish_selection=policy, polish_diagnostics=True
    )
    original = FastRightShiftPolish.propose
    q_before = []

    def probe(self, *args):
        assert search._pending is None and search.steps[-1]["q_transition"]["terminal"]
        q_before.append(copy.deepcopy(asdict(search.table)))
        return original(self, *args)

    monkeypatch.setattr(FastRightShiftPolish, "propose", probe)
    search._trajectory(pop, gateway, streams, lambda: 10)
    reference, rp, rg, rs = _prepared(previous.NO_P3)
    reference._trajectory(rp, rg, rs, lambda: 10)
    assert q_before == [asdict(search.table)] == [asdict(reference.table)]
    assert search.steps == reference.steps
    row = search.polish_steps[0]
    assert row["seconds"] == pytest.approx(
        row["seconds_selection"] + row["seconds_input_diagnostics"] + row["seconds_service"]
    )
    assert sum(search.polish_selector.region_seconds) == row["seconds_service"]
    assert row["pool_size"] <= len(pop) + 1 and row["physical_fingerprint"]
    for c in pop + gateway.archive.members:
        assert c.evaluation == evaluate(gateway.problem, c.genotype, c.schedule)


def test_selection_deadline_skips_construction_and_does_not_charge_region_service(monkeypatch):
    search, pop, gateway, streams = _prepared(previous.NO_P3)
    search.config = study.search_config("DIRECTION_P")
    ticks = iter((10, 101))
    monkeypatch.setattr(FastRightShiftPolish, "propose", lambda *a: pytest.fail("after deadline"))
    search._polish(
        pop[0],
        FrozenFrame.of([c.evaluation.objectives for c in pop]),
        0,
        pop,
        gateway,
        streams,
        lambda: next(ticks),
    )
    assert search.polish_steps[-1]["status"] == "deadline_after_selection"
    assert search.polish_selector.region_seconds == [0, 0, 0]


def test_exact_replay_and_same_scope_repetition_distinguish_interruptions():
    problem, current = synthetic(25)
    row = dict(
        input_schedule=dict(
            ms=current.genotype.ms, os=current.genotype.os, starts=current.schedule.starts
        ),
        physical_fingerprint=physical_fingerprint(current),
        start_objectives=current.evaluation.objectives,
    )
    scan = inspect_start(problem, row)
    assert scan["first_move_class"] == "improving_first_move"
    assert scan["partial_targets"] == 20 and scan["eligible"] == 25
    assert scan["improves_outside_partial"] and scan["improving_later_times"] > 0
    for move in scan["attempts"]:
        assert move["delta_bill"] == pytest.approx(move["delta_tou"] + move["delta_demand"])
    with pytest.raises(ValueError):
        inspect_start(problem, dict(row, physical_fingerprint="bad"))
    empty = dict(
        row, status="empty", target_scope="partial", complete_empty=True, attempts=0, seconds=2
    )
    rows = [
        dict(empty, status="deadline_partial", complete_empty=False),
        empty,
        empty,
        dict(empty, target_scope="all"),
        dict(status="region_skip", seconds=1),
    ]
    summary = repetition_summary(rows, 20)
    assert summary["calls"] == 4 and summary["unique_inputs"] == 1
    assert summary["after_empty_recalls"] == 1
    assert summary["repeat_fraction_of_called_polish"] == 0.25
    assert summary["repeat_fraction_of_run"] == 0.1
    legacy = repetition_summary([dict(status="empty", attempts=0, seconds=1)])
    assert legacy["calls"] == legacy["uninstrumented_calls"] == 1
    assert not legacy["complete_input_diagnostics"]


def test_replay_no_slack_and_no_benefit_are_separate():
    for left_only, expected in ((True, "no_right_slack"), (False, "no_strict_bill_improvement")):
        problem, current = synthetic(left_only=left_only)
        if not left_only:
            region = problem.regions[0]
            problem = replace(
                problem,
                regions=(
                    replace(region, tariffs=tuple(replace(t, price=1) for t in region.tariffs)),
                ),
            )
            current = replace(
                current, evaluation=evaluate(problem, current.genotype, current.schedule)
            )
        row = dict(
            input_schedule=dict(
                ms=current.genotype.ms, os=current.genotype.os, starts=current.schedule.starts
            ),
            physical_fingerprint=physical_fingerprint(current),
            start_objectives=current.evaluation.objectives,
        )
        assert inspect_start(problem, row)["first_move_class"] == expected


def test_segment_boundaries_unchanged():
    assert [segment_of((v, 0)) for v in (-1, 1 / 3, 2 / 3, 4)] == [0, 1, 2, 2]


def test_offline_scan_identifies_tou_saving_offset_by_demand():
    problem, current = synthetic()
    region = problem.regions[0]
    problem = replace(
        problem,
        regions=(
            replace(
                region,
                demand_rate=100,
                tariffs=(
                    replace(region.tariffs[0], end=900),
                    replace(region.tariffs[1], start=900),
                ),
            ),
        ),
        instances=tuple(replace(m, idle_kw=0) for m in problem.instances),
    )
    schedule = Schedule((100, 1000, 1000, 1300))
    current = replace(
        current, schedule=schedule, evaluation=evaluate(problem, current.genotype, schedule)
    )
    assert current.evaluation.feasible
    row = dict(
        input_schedule=dict(
            ms=current.genotype.ms, os=current.genotype.os, starts=current.schedule.starts
        ),
        physical_fingerprint=physical_fingerprint(current),
        start_objectives=current.evaluation.objectives,
    )
    scan = inspect_start(problem, row)
    assert scan["tou_offset_by_demand"] > 0
    example = next(x for x in scan["attempts"] if x["tou_offset_by_demand"])
    assert example["delta_tou"] < 0 < example["delta_demand"]
    assert example["delta_bill"] >= 0
