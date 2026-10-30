"""Reference equivalence, RNG preservation, scoped reuse and peak-knot counterexamples."""

from dataclasses import asdict, replace
from random import Random

import pytest
from test_lsv2_fast_rightshift import CAL, _prepared, synthetic
from test_lsv2_group_polish import joint_problem

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import (
    Candidate,
    Genotype,
    Job,
    ProblemInstance,
    Profile,
    Region,
    Schedule,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import lsv2_common_refine as study
from geo_llm_scheduler.experiments import lsv2_group_rightshift as previous
from geo_llm_scheduler.lsv2.common_refine import ReusedCommonPolish, _peak_switches
from geo_llm_scheduler.lsv2.flowsafe import FlowSafeBillMoves
from geo_llm_scheduler.lsv2.group_polish import ConcurrentRightShiftPolish, _Search


@pytest.mark.parametrize("scope", [False, True])
@pytest.mark.parametrize("seed", range(12))
def test_exact_proposals_rng_and_repeated_empty_certificate(scope, seed):
    op = ReusedCommonPolish(all_targets=scope)
    for problem, c in (joint_problem(), synthetic(5), synthetic(left_only=True)):
        reference = ConcurrentRightShiftPolish(all_targets=scope)
        for repeat in range(3):
            a, b = Random(seed + repeat), Random(seed + repeat)
            expected = reference.propose(problem, c, 3, Config(), a)
            actual = op.propose(problem, c, 3, Config(), b)
            assert actual.proposals == expected.proposals
            assert (
                actual.diagnostics["incremental_deltas"]
                == expected.diagnostics["incremental_deltas"]
            )
            assert (
                actual.diagnostics["proposal_sources"] == expected.diagnostics["proposal_sources"]
            )
            assert a.getstate() == b.getstate()
            assert actual.diagnostics["group_cache_hit"] == (repeat > 0)
            if repeat and not expected.diagnostics["single"]["proposals"]:
                assert actual.diagnostics["single"]["empty_cache_hit"]
            for prop in actual.proposals:
                ev = evaluate(problem, prop.genotype, prop.schedule)
                assert ev.feasible and ev.flow == c.evaluation.flow and ev.bill < c.evaluation.bill
        op.retain([])
        assert not op._entries


def test_cache_uses_full_physical_input_current_os_problem_and_configuration():
    problem, c = joint_problem()
    op = ReusedCommonPolish()
    op.propose(problem, c, 3, Config(), Random(1))
    different_os = replace(c, genotype=replace(c.genotype, os=tuple(reversed(c.genotype.os))))
    batch = op.propose(problem, different_os, 3, Config(), Random(2))
    assert batch.diagnostics["group_cache_hit"]
    assert all(p.genotype == different_os.genotype for p in batch.proposals)
    changed = replace(
        c, schedule=Schedule(tuple(1 if i == 0 else t for i, t in enumerate(c.schedule.starts)))
    )
    assert not op.propose(problem, changed, 3, Config(), Random(2)).diagnostics["group_cache_hit"]
    op.retain([c])
    assert len(op._entries) == 1
    assert not op.propose(problem, c, 1, Config(), Random(2)).diagnostics["group_cache_hit"]
    other_problem = replace(problem, regions=(replace(problem.regions[0], demand_rate=3),))
    assert not op.propose(other_problem, c, 1, Config(), Random(2)).diagnostics["group_cache_hit"]
    with pytest.raises(ValueError):
        op.propose(problem, c, 0, Config(), Random(1))


def test_group_upper_bounds_preserve_reference_at_tolerated_release_boundary():
    problem, c = joint_problem()
    # Starts may precede release within evaluator tolerance; single and group
    # eligibility comparisons have different lower bounds and must not be merged.
    jobs = tuple(replace(j, release=0.8e-9) for j in problem.jobs)
    problem = replace(problem, jobs=jobs)
    starts = list(c.schedule.starts)
    starts[1] = problem.profile(0).duration + 1.5e-9
    c = replace(c, schedule=Schedule(tuple(starts)))
    prepared = FlowSafeBillMoves(right_only=True)._prepare(problem, c)
    assert _Search(problem, c, prepared).upper == _Search(problem, c).upper


def test_peak_hull_keeps_only_actual_maximum_switches_and_ties():
    assert _peak_switches([10, 6, 0], [6, 10, 0]) == [0.5]
    assert _peak_switches([10, 10, 0], [10, 10, 30]) == [pytest.approx(1 / 3)]
    assert _peak_switches([1, 0, 0], [1, 0, 0]) == []
    assert _peak_switches([], []) == []
    # Three active segments; lower crossing lines must not be emitted.
    assert _peak_switches([3, 2, 0, -100], [0, 2, 3, -90]) == [
        pytest.approx(1 / 3),
        pytest.approx(2 / 3),
    ]


def peak_problem():
    p, d = Profile(600, 0.2, 1), Profile(300, 0.2, 1)
    problem = ProblemInstance(
        (Job("A", 0, p, d), Job("B", 0, p, d)),
        (Region("r", (Tariff(0, 3600, 1),), 1),),
        (ServingInstance("P", 0, 10, 0, 1, (0, 1)), ServingInstance("D", 0, 10, 0, 1, (0, 1))),
    )
    g, s = Genotype((0, 1, 0, 1), (0, 0, 1, 1)), Schedule((0, 1800, 0, 1800))
    return problem, Candidate(g, s, evaluate(problem, g, s))


def test_interior_peak_switch_rescues_empty_common_exact_and_preserves_flow():
    problem, c = peak_problem()
    old = ConcurrentRightShiftPolish().propose(problem, c, 3, Config(), Random(4))
    assert not old.proposals
    op = ReusedCommonPolish(peak_switch=True)
    new = op.propose(problem, c, 3, Config(), Random(4))
    assert new.proposals and new.diagnostics["peak_positions"] > 0
    assert new.proposals[0].schedule.starts[0] == pytest.approx(600)
    for prop, delta in zip(new.proposals, new.diagnostics["incremental_deltas"]):
        ev = evaluate(problem, prop.genotype, prop.schedule)
        assert ev.feasible and ev.flow == c.evaluation.flow
        assert ev.bill - c.evaluation.bill == pytest.approx(delta, abs=1e-9)
        assert ev.bill < c.evaluation.bill and prop.schedule.starts[1::2] == (1800, 1800)
    cached = op.propose(problem, c, 3, Config(), Random(4))
    assert cached.proposals == new.proposals and cached.diagnostics["group_cache_hit"]
    zero = replace(problem, regions=(replace(problem.regions[0], demand_rate=0),))
    batch = ReusedCommonPolish(peak_switch=True).propose(zero, c, 3, Config(), Random(4))
    assert batch.diagnostics["peak_intervals"] == 0


@pytest.mark.parametrize("seed", range(20))
def test_peak_extension_with_nonconstant_prices_exact_and_no_worse_proposal(seed):
    problem, c = peak_problem()
    rng = Random(seed)
    problem = replace(
        problem,
        regions=(
            Region(
                "r",
                (
                    Tariff(0, 450, rng.uniform(0.1, 3)),
                    Tariff(450, 1300, rng.uniform(0.1, 3)),
                    Tariff(1300, 3600, rng.uniform(0.1, 3)),
                ),
                rng.uniform(0, 5),
            ),
        ),
    )
    c = replace(c, evaluation=evaluate(problem, c.genotype, c.schedule))
    base = ReusedCommonPolish().propose(problem, c, 3, Config(), Random(seed))
    peak = ReusedCommonPolish(peak_switch=True).propose(problem, c, 3, Config(), Random(seed))

    def values(batch):
        result = [c.evaluation.bill]
        for prop, delta in zip(batch.proposals, batch.diagnostics["incremental_deltas"]):
            ev = evaluate(problem, prop.genotype, prop.schedule)
            assert ev.feasible and ev.flow == c.evaluation.flow
            assert ev.bill - c.evaluation.bill == pytest.approx(delta, abs=1e-9)
            result.append(ev.bill)
        return result

    assert min(values(peak)) <= min(values(base)) + 1e-9


def test_five_configs_and_unchanged_controls():
    assert len(study.ARMS) == 5 and study.RUN_SEED == 16901
    assert study.search_config("C0_NONE") == previous.search_config("NONE_20")
    assert study.search_config("C1_COMMON") == previous.search_config("COMMON_20")
    base = study.search_config("C2_FAST")
    assert replace(base, polish_selection="region_time") == study.search_config("C3_REGION")
    assert replace(base, polish_method="common_peak") == study.search_config("C4_PEAK")
    for arm in study.ARMS:
        cfg = study.arm_config(arm, 16901, 1200)
        search = study.make_search(
            arm, cfg, CAL, _prepared("Q1_NET_G9_B3L3_NO_P3")[0].table, training=True
        )
        record = study.configuration_record(arm, 16901, 1200, CAL)
        assert record["lsv2"] == asdict(search.config) and record["qspec"]["gamma"] == 0.9
        assert search.config.time_share == 0.2 and "P3" not in search.config.operators
    for fn, args in ((study.search_config, ("bad",)), (study.arm_config, ("bad", 1, 1))):
        with pytest.raises(ValueError):
            fn(*args)


@pytest.mark.parametrize("arm", study.ARMS)
def test_exact_gateway_and_terminal_credit_for_five_arms(arm):
    search, pop, gateway, streams = _prepared("Q1_NET_G9_B3L3_NO_P3")
    search.config = study.search_config(arm)
    reference, rp, rg, rs = _prepared("Q1_NET_G9_B3L3_NO_P3")
    reference._trajectory(rp, rg, rs, lambda: 10)
    search._trajectory(pop, gateway, streams, lambda: 10)
    assert asdict(search.table) == asdict(reference.table)
    assert search._pending is None and search.steps[-1]["q_transition"]["terminal"]
    assert len(search.polish_steps) == (0 if arm == "C0_NONE" else 1)
    for c in pop + gateway.archive.members:
        assert evaluate(gateway.problem, c.genotype, c.schedule) == c.evaluation


def test_reused_path_matches_reference_through_archive_reflow_and_next_q_episode():
    a, ap, ag, ar = _prepared("Q1_NET_G9_B3L3_NO_P3")
    b, bp, bg, br = _prepared("Q1_NET_G9_B3L3_NO_P3")
    a.config, b.config = study.search_config("C1_COMMON"), study.search_config("C2_FAST")
    for _ in range(4):
        a._trajectory(ap, ag, ar, lambda: 10)
        b._trajectory(bp, bg, br, lambda: 10)
        assert ap == bp and ag.archive.members == bg.archive.members
        assert asdict(a.table) == asdict(b.table)
        for stream in ("lsv2:rightshift_selection", "lsv2:fast_rightshift"):
            assert ar.stream(stream).getstate() == br.stream(stream).getstate()
