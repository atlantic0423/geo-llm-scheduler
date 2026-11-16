"""Joint benefit, capacity, timing invariants and unchanged controls."""

from dataclasses import asdict, replace
from random import Random

import pytest
from test_lsv2_fast_rightshift import CAL, _prepared, synthetic

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
from geo_llm_scheduler.experiments import lsv2_group_rightshift as study
from geo_llm_scheduler.experiments import lsv2_rightshift_selection as old
from geo_llm_scheduler.lsv2.group_polish import ConcurrentRightShiftPolish, _Search
from geo_llm_scheduler.lsv2.polish import FastRightShiftPolish
from geo_llm_scheduler.lsv2.search import LSV2Config


def joint_problem():
    p, d = Profile(600, 0.2, 1), Profile(300, 0.2, 1)
    problem = ProblemInstance(
        (
            Job("A", 0, p, d),
            Job("B", 0, p, d),
            Job("C", 0, Profile(1, 0.2, 1), Profile(600, 0.2, 1)),
        ),
        (Region("r", (Tariff(0, 3600, 1),), 0.1),),
        (
            ServingInstance("shared", 0, 10, 0.1, 1, (0, 1)),
            ServingInstance("other", 0, 10, 0.1, 1, (0, 1)),
        ),
    )
    g = Genotype((0, 1, 0, 1, 1, 0), (0, 0, 1, 1, 2, 2))
    s = Schedule((0, 900, 0, 900, 0, 300))
    return problem, Candidate(g, s, evaluate(problem, g, s))


@pytest.mark.parametrize("coordinated", [False, True])
def test_joint_only_improvement_is_not_filtered_by_single_failure(coordinated):
    problem, c = joint_problem()
    assert c.evaluation.feasible
    assert not FastRightShiftPolish().propose(problem, c, 3, Config(), Random(1)).proposals
    op = ConcurrentRightShiftPolish(coordinated)
    result = op.propose(problem, c, 3, Config(), Random(1))
    assert result == op.propose(problem, c, 3, Config(), Random(1))
    assert result.proposals and result.diagnostics["groups"] == 1
    assert len(result.proposals) <= 3
    for prop, delta in zip(result.proposals, result.diagnostics["incremental_deltas"]):
        exact = evaluate(problem, prop.genotype, prop.schedule)
        assert exact.feasible and exact.flow == c.evaluation.flow
        assert exact.bill < c.evaluation.bill
        assert exact.bill - c.evaluation.bill == pytest.approx(delta, abs=1e-9)
        assert prop.selected == (0, 2)
        assert prop.genotype == c.genotype
        assert prop.schedule.starts[1::2] == c.schedule.starts[1::2]


def test_coordinated_members_have_distinct_capacity_feasible_shifts():
    problem, c = joint_problem()
    jobs = (
        Job("A", 0, Profile(10, 0.4, 1), Profile(1, 0.1, 1)),
        Job("B", 0, Profile(20, 0.4, 1), Profile(1, 0.1, 1)),
        Job("C", 0, Profile(1, 0.1, 1), Profile(30, 0.3, 1)),
    )
    problem = replace(problem, jobs=jobs)
    s = Schedule((0, 30, 0, 50, 0, 20))
    c = replace(c, schedule=s, evaluation=evaluate(problem, c.genotype, s))
    ctx = _Search(problem, c)
    ctx.insert((0, 2), (0, 2))
    assert ctx.pool
    starts = min(ctx.pool, key=lambda k: ctx.pool[k][0])
    assert starts[0] != starts[2] and starts[0] >= 0 and starts[2] >= 0
    ev = evaluate(problem, c.genotype, Schedule(starts))
    assert ev.feasible and ev.bill < c.evaluation.bill and ev.flow == c.evaluation.flow


@pytest.mark.parametrize("coordinated", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_exact_invariants_resource_pressure_and_no_mutation(coordinated, seed):
    problem, c = joint_problem()
    rng = Random(seed)
    jobs = tuple(
        replace(
            j, prefill=replace(j.prefill, compute=rng.uniform(0.05, 0.45), vram=rng.uniform(0.5, 4))
        )
        for j in problem.jobs
    )
    problem = replace(
        problem,
        jobs=jobs,
        regions=(
            replace(
                problem.regions[0],
                demand_rate=seed,
                tariffs=(Tariff(0, 450, 3), Tariff(450, 3600, 0.5)),
            ),
        ),
    )
    c = replace(c, evaluation=evaluate(problem, c.genotype, c.schedule))
    assert c.evaluation.feasible
    before = c
    batch = ConcurrentRightShiftPolish(coordinated).propose(problem, c, 3, Config(), Random(seed))
    assert c == before
    for prop, delta in zip(batch.proposals, batch.diagnostics["incremental_deltas"]):
        e = evaluate(problem, prop.genotype, prop.schedule)
        assert e.feasible and e.flow == c.evaluation.flow
        assert e.bill < c.evaluation.bill
        assert e.bill - c.evaluation.bill == pytest.approx(delta, abs=1e-8)
        assert all(a >= b for a, b in zip(prop.schedule.starts, c.schedule.starts))
        assert prop.schedule.starts[1::2] == c.schedule.starts[1::2]
        assert all(
            prop.schedule.starts[o] == c.schedule.starts[o]
            for o in range(problem.operation_count)
            if o not in prop.selected
        )


def test_single_improvements_retained_no_slack_and_validation():
    problem, c = synthetic(4)
    old_result = FastRightShiftPolish().propose(problem, c, 3, Config(), Random(3))
    for mode in (False, True):
        result = ConcurrentRightShiftPolish(mode).propose(problem, c, 3, Config(), Random(3))
        assert result.proposals == old_result.proposals
        problem2, c2 = synthetic(left_only=True)
        empty = ConcurrentRightShiftPolish(mode).propose(problem2, c2, 3, Config(), Random(3))
        assert not empty.proposals and empty.diagnostics["eligible"] == 0
    with pytest.raises(ValueError):
        ConcurrentRightShiftPolish().propose(problem, c, 0, Config(), Random(3))
    with pytest.raises(ValueError):
        LSV2Config(polish_method="invalid")


def test_eight_configs_preserve_control_and_only_change_share_within_pair():
    assert len(study.ARMS) == 8 and study.RUN_SEED == 16901
    assert study.search_config("NONE_20") == old.search_config("B0")
    assert study.search_config("SINGLE_20") == old.search_config("POOL_RANDOM_P")
    for method in ("NONE", "SINGLE", "COMMON", "COORD"):
        a, b = (study.search_config(f"{method}_{share}") for share in (20, 30))
        assert replace(a, time_share=0.3) == b
        assert "P3" not in a.operators and a.budget == a.trajectory_steps == 3
        cfg = study.arm_config(method + "_20", 16901, 1200)
        assert cfg == old.arm_config("B0", 16901, 1200)
        record = study.configuration_record(method + "_20", 16901, 1200, CAL)
        assert record["lsv2"] == asdict(a) and record["qspec"]["gamma"] == 0.9
    for fn, args in ((study.search_config, ("bad",)), (study.arm_config, ("bad", 1, 1))):
        with pytest.raises(ValueError):
            fn(*args)


@pytest.mark.parametrize("arm", study.ARMS)
def test_episode_credit_and_exact_archive_on_all_arms(arm):
    search, pop, gateway, streams = _prepared("Q1_NET_G9_B3L3_NO_P3")
    search.config = study.search_config(arm)
    reference, rp, rg, rs = _prepared("Q1_NET_G9_B3L3_NO_P3")
    reference._trajectory(rp, rg, rs, lambda: 10)
    search._trajectory(pop, gateway, streams, lambda: 10)
    assert asdict(search.table) == asdict(reference.table)
    assert search._pending is None and search.steps[-1]["q_transition"]["terminal"]
    assert len(search.polish_steps) == (0 if arm.startswith("NONE") else 1)
    for c in pop + gateway.archive.members:
        assert evaluate(gateway.problem, c.genotype, c.schedule) == c.evaluation
