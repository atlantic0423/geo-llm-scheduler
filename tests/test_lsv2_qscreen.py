"""Masked TD, terminal credit, frozen evaluation and reference-path equivalence."""

import copy
from dataclasses import asdict, replace
from random import Random

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.lsv2_delayed import opportunity_features
from geo_llm_scheduler.experiments.lsv2_qscreen import (
    LEARNING_ARMS,
    calibrate,
    make_search,
    run_screen,
)
from geo_llm_scheduler.experiments.lsv2_study import run_arm, study_config
from geo_llm_scheduler.experiments.runner import digest, solution
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.lsv2.qlearning import (
    DESIGNS,
    QLearningSearch,
    QSpec,
    QTable,
    encode_state,
    event_of,
)
from geo_llm_scheduler.utils.numeric import identity
from geo_llm_scheduler.utils.rng import RNGManager

CAL = dict(rho=0.001, h=0.01, p_cutoff=0.5, w_cutoff=1.5)


def test_masked_td_terminal_and_frozen_reads():
    table = QTable(values={"1": {"A8": 1000, "A7": 2}})
    spec = QSpec("Q1_NET", 0.9)
    result = table.update((0,), "A1", 1, (1,), ["A7"], spec, training=True)
    assert result["td_target"] == pytest.approx(2.8)
    assert result["q_new"] == pytest.approx(0.28)
    result = table.update((0,), "A1", -1, None, [], spec, training=True)
    assert result["td_target"] == -1 and result["bootstrap"] == 0
    before = copy.deepcopy(asdict(table))
    table.update((9,), "A1", 5, (1,), ["A7"], spec, training=False)
    assert table.choose((9,), ["A1"], 0, Random(2)) == ("A1", 1)
    assert asdict(table) == before
    with pytest.raises(ValueError):
        table.update((0,), "A1", 1, (1,), [], spec, training=True)
    with pytest.raises(ValueError):
        table.update((0,), "A1", float("nan"), None, [], spec, training=True)
    with pytest.raises(ValueError):
        table.choose((0,), [], 0, Random(2))


def test_epsilon_propensity_and_exact_ties():
    table = QTable(values={"0": {"A1": 2, "A7": 2, "A8": 1}})
    counts = {a: 0 for a in ("A1", "A7", "A8")}
    rng = Random(33)
    for _ in range(6000):
        a, p = table.choose((0,), list(counts), 0.2, rng)
        assert p == pytest.approx(0.2 / 3 + (0.4 if a != "A8" else 0))
        counts[a] += 1
    assert 300 < counts["A8"] < 500
    assert abs(counts["A1"] - counts["A7"]) < 200


@pytest.mark.parametrize("design,size", zip(DESIGNS, (30, 30, 60, 60, 60)))
def test_declared_state_cardinality_and_threshold(design, size):
    spec = QSpec(design, 0.9, cutoff=1.5)
    states = {
        encode_state(spec, s, k, mask, e, feature)
        for s in range(3)
        for k in range(5)
        for mask in (["A7"], ["A7", "A8"])
        for e in range(6)
        for feature in (1.5, 2.0)
    }
    assert len(states) == size
    if design.startswith("Q2"):
        assert encode_state(spec, 0, 0, [], 0, 1.5)[-1] == 0
        assert encode_state(spec, 0, 0, [], 0, 2)[-1] == 1
    with pytest.raises(ValueError):
        encode_state(spec, 3, 0, [], 0)


def test_event_categories_and_invalid_specs():
    assert [event_of(a, True, False) for a in ("A1", "P2_FLOW", "A7", "A8", "P3")] == [
        1,
        1,
        2,
        2,
        3,
    ]
    assert event_of("A1", False, True) == 4
    assert event_of("A1", False, False) == 5
    for kwargs in (
        dict(design="Q4"),
        dict(gamma=1),
        dict(h=0),
        dict(alpha=2),
        dict(rho=float("inf")),
    ):
        with pytest.raises(ValueError):
            QSpec(**(dict(design="Q1_NET", gamma=0.9) | kwargs))


def test_calibration_cannot_read_validation_outcomes():
    rows = [
        dict(key="train", gain=0.1, seconds=2),
        dict(key="train", gain=0.3, seconds=2),
        dict(key="valid", gain=1e99, seconds=1e-9),
    ]
    features = [
        dict(key="train", prefill_window_fraction=0.3, resource_wait_fraction=2),
        dict(key="valid", prefill_window_fraction=99, resource_wait_fraction=99),
    ]
    assert calibrate(rows, features, ["train"]) == dict(
        keys=["train"],
        rows=2,
        feature_rows=1,
        feature_keys=["train"],
        rho=0.1,
        h=0.2,
        p_cutoff=0.3,
        w_cutoff=2,
    )
    with pytest.raises(ValueError):
        calibrate(rows, features, ["missing"])


@pytest.fixture(scope="module")
def problem():
    return case_problem(20, 3, 6, 202610160020, "arrival", 0)


@pytest.mark.parametrize("arm", LEARNING_ARMS)
def test_real_trajectories_terminal_freeze_and_exact(problem, arm):
    cfg = replace(study_config("LSV2", 18001, 100), population=6, neighborhood=4)
    gateway = EvaluationGateway(problem, Archive())
    pop = [gateway.evaluate(g) for g in initial_genotypes(problem, cfg, Random(3))[:6]]
    table = QTable()
    search = make_search(arm, cfg, CAL, table, training=True)
    assert isinstance(search, QLearningSearch)
    streams = RNGManager(18001)
    search(pop, gateway, streams, lambda: 0.0)
    search._trajectory(pop, gateway, streams, lambda: 1.0)
    search._trajectory(pop, gateway, streams, lambda: 1.0)
    assert len(search.steps) == 10
    assert sum(sum(v.values()) for v in table.visits.values()) == 10
    for i, row in enumerate(search.steps):
        assert row["q_transition"]["terminal"] == (i % 5 == 4)
        if i % 5 != 4:
            assert row["q_transition"]["next_state"] == search.steps[i + 1]["q_state"]
            assert row["q_transition"]["next_available"] == search.steps[i + 1]["available"]
        if arm.startswith("Q3") and i % 5 == 0:
            assert row["q_state"][0] == 0
    before = copy.deepcopy(asdict(table))
    search.training = False
    search._trajectory(pop, gateway, streams, lambda: 1.0)
    assert asdict(table) == before
    assert all(not r["q_transition"]["training"] for r in search.steps[-5:])
    for c in pop + gateway.archive.members:
        assert evaluate(problem, c.genotype, c.schedule) == c.evaluation
    # A shortened episode also closes its sole transition, without bootstrapping.
    clock = iter([1.0, 101.0])
    search._trajectory(pop, gateway, streams, lambda: next(clock))
    assert search.steps[-1]["step"] == 0 and search.steps[-1]["q_transition"]["terminal"]


@pytest.mark.parametrize("arm,old", [("UNIFORM", "LSV2_RANDSEL"), ("BANDIT", "LSV2")])
def test_reference_path_equivalence(problem, arm, old):
    cfg = replace(study_config(old, 16601, 1e6), generations=1)

    # Deterministic scheduling timer and call boundary clock remove wall-time noise.
    def controlled(search):
        class Clock:
            t = 0

            def __call__(self):
                self.t += 0.001
                return self.t

        search.timer = Clock()
        original = search.__class__.__call__
        calls = [0]

        class Controlled(search.__class__):
            def __call__(self, pop, gw, streams, budget_elapsed):
                calls[0] += 1
                original(self, pop, gw, streams, lambda: calls[0] * 0.001)

        search.__class__ = Controlled

    search = make_search(arm, cfg, CAL, QTable(), training=False)
    controlled(search)
    result = run_screen(problem, cfg, {}, search)
    baseline = run_arm(problem, old, cfg, {}, search_sink=controlled)
    assert result.gateway.counts == baseline.gateway.counts
    assert digest([solution(c) for c in result.population + result.archive.members]) == digest(
        [solution(c) for c in baseline.population + baseline.archive.members]
    )


def test_fixed_reference_and_unknown_arm():
    cfg = study_config("LSV2", 1, 10)
    search = make_search("FIXED_A7", cfg, {}, QTable(), training=False)
    assert search._choose_operator(0, ["A1", "A7"], RNGManager(1), None, {})[0] == "A7"
    with pytest.raises(ValueError):
        search._choose_operator(0, ["A1"], RNGManager(1), None, {})
    with pytest.raises(ValueError):
        make_search("unknown", cfg, {}, QTable(), training=False)


@pytest.mark.parametrize("n", [20, 100, 200])
def test_fast_features_and_identity_cache_equal_reference(n):
    problem = case_problem(n, 3, 6, 202610160000 + n, "arrival", 0)
    cfg = replace(study_config("LSV2", 18001, 100), population=6, neighborhood=4)
    gw = EvaluationGateway(problem, Archive())
    pop = [gw.evaluate(g) for g in initial_genotypes(problem, cfg, Random(3))[:6]]
    frame = FrozenFrame.of([c.evaluation.objectives for c in pop])
    for design, field in [
        ("Q2_P_NET", "prefill_window_fraction"),
        ("Q2_W_NET", "resource_wait_fraction"),
    ]:
        search = make_search(design + "_G9", cfg, CAL, QTable(), training=True)
        for c in pop:
            expected = opportunity_features(dict(problem=problem, current=c, frame=frame))[field]
            assert search._opportunity(problem, c) == expected
    search = make_search("Q3_NET_G9", cfg, CAL, QTable(), training=True)
    for c in pop:
        assert search._identity_snapshot(c, pop, gw.archive.members) == (
            identity(c),
            [identity(x) for x in pop],
            {identity(x) for x in gw.archive.members},
        )
    # Timing change must invalidate the value even with equal genotype/objectives.
    moved = replace(
        pop[0],
        schedule=replace(
            pop[0].schedule, starts=(pop[0].schedule.starts[0] + 1,) + pop[0].schedule.starts[1:]
        ),
    )
    snapshot = search._identity_snapshot(moved, [moved], [])
    assert snapshot == (identity(moved), [identity(moved)], set())
    assert len(search._identities) == 1 and id(pop[0]) not in search._identities
