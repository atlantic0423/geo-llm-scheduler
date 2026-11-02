"""Retention covers extremes without changing exact candidates or legacy runs."""

from dataclasses import replace
from itertools import permutations

import pytest
from test_lsv2_fast_rightshift import CAL, synthetic

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.initial_pool_retention import pool_initializer
from geo_llm_scheduler.experiments.rl23 import run_policy
from geo_llm_scheduler.experiments.s4_ablation import arm_config, configuration_record, run_arm
from geo_llm_scheduler.initialization.retention import retain_pool
from geo_llm_scheduler.utils.numeric import identity
from geo_llm_scheduler.utils.rng import RNGManager


def point(i, flow, bill, *, ms=None, os=None, starts=None):
    return Candidate(
        Genotype(ms or (i, i), os or (0, 0)),
        Schedule(starts or (float(i), float(i + 1))),
        EvaluationResult(True, flow, bill, (), ()),
        f"point_{i}",
    )


def test_extremes_spacing_and_input_order_invariance():
    pool = [point(1, 0, 10), point(2, 1, 9), point(3, 5, 5), point(4, 10, 0)]
    expected = {identity(pool[i]) for i in (0, 2, 3)}
    for order in permutations(pool):
        selected = retain_pool(order, 3)
        assert {identity(c) for c in selected} == expected
        assert selected == [c for c in order if identity(c) in expected]
    assert retain_pool(pool, 4) == pool
    assert retain_pool(pool, 2) == [pool[0], pool[-1]]


def test_layers_precede_spacing_and_distinct_physical_schedules():
    a, b = point(1, 1, 3), point(2, 3, 1)
    dominated = point(3, 4, 4)
    distant = point(4, 100, 100)
    assert set(retain_pool([distant, dominated, b, a], 3)) == {a, b, dominated}
    same_physical = replace(a, genotype=replace(a.genotype, os=(1, 1)))
    chosen = retain_pool([same_physical, a, b, dominated], 3)
    assert len({(c.genotype.ms, c.schedule.starts) for c in chosen}) == 3
    # If distinct physical schedules cannot fill the requested population,
    # alternate encodings remain useful and may supply the remainder.
    assert len(retain_pool([a, same_physical, b], 3)) == 3
    different_timing = replace(a, schedule=Schedule((9.0, 10.0)))
    assert len(retain_pool([a, different_timing], 2)) == 2


def test_equal_objectives_structure_ties_constant_axis_and_numeric_boundary():
    pool = [point(1, 1, 1, ms=(0, 0)), point(2, 1, 1, ms=(0, 1)), point(3, 1, 1, ms=(1, 1))]
    assert retain_pool(pool, 2) == [pool[0], pool[2]]
    one_axis = [point(i, 1, float(i)) for i in range(1, 5)]
    assert retain_pool(one_axis, 2) == one_axis[:2]
    tiny = [point(1, 1, 2), point(2, 1 + 1e-10, 2 - 1e-10), point(3, 2, 1)]
    assert len(retain_pool(tiny, 2)) == 2
    scaled = [
        replace(c, evaluation=replace(c.evaluation, flow=c.evaluation.flow * 100))
        for c in [point(1, 0, 10), point(2, 1, 9), point(3, 5, 5), point(4, 10, 0)]
    ]
    assert [c.origin for c in retain_pool(scaled, 3)] == ["point_1", "point_3", "point_4"]


@pytest.mark.parametrize("size", [0, 1, 2.0, True, -1])
def test_invalid_size(size):
    with pytest.raises(ValueError):
        retain_pool([point(1, 1, 1)], size)


def test_invalid_pool_and_contract_duplicates():
    a = point(1, 1, 1)
    for pool in (
        [],
        [a, a],
        [a, replace(a, evaluation=replace(a.evaluation, feasible=False))],
        [a, point(2, float("nan"), 1)],
        [a, replace(a, schedule=Schedule((1.0,)))],
        [a, replace(a, genotype=Genotype((0,), (0,)))],
    ):
        with pytest.raises(ValueError):
            retain_pool(pool, 2)


def test_same_generation_archive_exact_and_rng_and_sink_restoration():
    problem, _ = synthetic(4)
    cfg = arm_config("A0_FULL_S4", 1, 30)
    traces, gateways, evidences, streams, populations = [], [], [], [], []
    for global_pool in (False, True):
        gw = EvaluationGateway(problem, Archive())
        trace = []
        sink = trace.append
        gw.candidate_sink = sink
        evidence, rng = {}, RNGManager(1)
        factory = (
            pool_initializer(problem, cfg, evidence)
            if global_pool
            else charged_initializer(problem, cfg, "BILL_MOEAD", evidence)
        )
        population = factory(gw, rng)
        assert gw.candidate_sink is sink
        traces.append(trace)
        gateways.append(gw)
        evidences.append(evidence)
        streams.append(rng)
        populations.append(population)
    assert traces[0] == traces[1] and len(traces[0]) == 140
    assert gateways[0].counts == gateways[1].counts
    assert gateways[0].archive.members == gateways[1].archive.members
    assert gateways[0].ideal == gateways[1].ideal
    assert (
        evidences[1]["legacy_initial_population_sha256"]
        == evidences[0]["initial_population_sha256"]
    )
    assert evidences[1]["initial_archive_sha256"] == evidences[0]["initial_archive_sha256"]
    assert (
        evidences[1]["initialization_seconds"] >= evidences[1]["retention"]["selection_seconds"] > 0
    )
    assert populations[1] == retain_pool(traces[0], 100)
    for name in ("initialization", "variation", "lsv2:seed"):
        assert streams[0].stream(name).getstate() == streams[1].stream(name).getstate()
    for c in populations[1]:
        assert evaluate(problem, c.genotype, c.schedule) == c.evaluation


def test_restore_sink_when_initializer_raises(monkeypatch):
    import geo_llm_scheduler.experiments.initial_pool_retention as module

    problem, _ = synthetic(4)
    cfg = arm_config("A0_FULL_S4", 1, 30)
    with pytest.raises(ValueError):
        pool_initializer(problem, replace(cfg, population=20), {})

    def bad(*args):
        def fail(*args):
            raise RuntimeError("construction failed")

        return fail

    monkeypatch.setattr(module, "charged_initializer", bad)
    gw = EvaluationGateway(problem, Archive())
    sink = [].append
    gw.candidate_sink = sink
    with pytest.raises(RuntimeError):
        pool_initializer(problem, cfg, {})(gw, RNGManager(1))
    assert gw.candidate_sink is sink


def test_s4_integration_and_explicit_conflicts():
    problem, _ = synthetic(4)
    cfg = replace(arm_config("A0_FULL_S4", 1, 30), generations=1)
    evidence, initial = {}, []

    def observe(event, data):
        if event == "initial_ready":
            initial.extend(data["population"])

    result = run_arm(
        problem, "A0_FULL_S4", cfg, CAL, evidence, observer=observe, initial_retention="global_pool"
    )
    assert len(initial) == len(result.population) == 100
    assert evidence["initial_retention"] == "global_pool"
    old = configuration_record("A0_FULL_S4", cfg, CAL)
    assert "initial_retention" not in old
    new = configuration_record("A0_FULL_S4", cfg, CAL, initial_retention="global_pool")
    assert new["lsv2"] == old["lsv2"] and new["engine"] == old["engine"]
    for arm, policy in [("A3_RANDOM_INIT", "global_pool"), ("A0_FULL_S4", "invalid")]:
        with pytest.raises(ValueError):
            configuration_record(arm, cfg, CAL, initial_retention=policy)
        with pytest.raises(ValueError):
            run_arm(problem, arm, cfg, CAL, {}, initial_retention=policy)
    with pytest.raises(ValueError):
        run_policy(
            problem,
            "BILL_PERMUTED",
            cfg,
            {},
            initialization_arm="BILL_MOEAD",
            initializer_override=lambda g, r: [],
        )
