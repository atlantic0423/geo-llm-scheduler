"""Hybrid-initialized external baselines with per-generation old Polish."""

import random
from collections import Counter
from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import component_random
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.ext_hybrid import (
    ARMS,
    POLICY,
    REFERENCE_ARMS,
    arm_config,
    base_arm,
    bill_seed,
    hybrid_initializer,
    polish_generation,
    retain_balanced,
    run_arm,
)
from geo_llm_scheduler.experiments.initial_bill_benchmark import traced_original
from geo_llm_scheduler.experiments.initial_bill_mixture import seed_bank
from geo_llm_scheduler.utils.numeric import identity
from geo_llm_scheduler.utils.rng import RNGManager


@pytest.fixture(scope="module")
def problem():
    return case_problem(20, 3, 6, 202610160020, "arrival", 0)


def initial(problem, seed=16101):
    cfg = arm_config("EXTH_MOEAD", seed, 300)
    gateway = EvaluationGateway(problem, Archive())
    evidence: dict = {}
    members = hybrid_initializer(problem, cfg, evidence)(gateway, RNGManager(cfg.seed))
    return members, gateway, evidence


def test_arms_map_to_unchanged_component33_configs():
    assert [base_arm(a) for a in ARMS] == [
        "EXT_MOEAD",
        "EXT_NSGAII",
        "EXT_RK_MOPSO",
        "EXT_SPEA2",
        "EXT_EPS_IBEA",
    ]
    for arm in (*ARMS, *REFERENCE_ARMS):
        assert arm_config(arm, 7, 300) == component_random.arm_config(base_arm(arm), 7, 300)
    with pytest.raises(ValueError):
        base_arm("EXT_MOEAD")


def test_retain_balanced_keeps_construction_order_per_mode(problem):
    cfg = arm_config("EXTH_MOEAD", 16101, 300)
    parents = traced_original(problem, cfg)
    kept = retain_balanced(parents, 2)
    assert len(kept) == 20
    assert Counter(c.origin for c in kept) == {f"initial_mode_{m}": 2 for m in range(10)}
    order = {identity(c): i for i, c in enumerate(parents)}
    assert [order[identity(c)] for c in kept] == sorted(order[identity(c)] for c in kept)
    with pytest.raises(ValueError):
        retain_balanced(parents[:50], 4)


def test_hybrid_initializer_composition_charge_and_determinism(problem):
    members, gateway, evidence = initial(problem)
    assert len(members) == len({c.genotype for c in members}) == 100
    origins = Counter(c.origin.split("_")[0] for c in members)
    assert (origins["initial"], origins["bill"], origins["hybrid"]) == (20, 1, 79)
    assert (evidence["heuristic"], evidence["bill"], evidence["random"]) == (20, 1, 79)
    assert evidence["policy"] == POLICY and evidence["bill_trials"] == 1
    # only retained members pass the gateway: discarded parents never reach the Archive
    assert gateway.counts["exact"] == 100
    assert {identity(c) for c in gateway.archive.members} <= {identity(c) for c in members}
    for c in members:
        assert c.evaluation.feasible
        assert evaluate(problem, c.genotype, c.schedule) == c.evaluation
    again, _, evidence2 = initial(problem)
    assert [identity(c) for c in again] == [identity(c) for c in members]
    assert evidence2["initial_population_sha256"] == evidence["initial_population_sha256"]


def test_bill_seed_is_slot_zero_of_a_single_trial_bill_bank(problem):
    cfg = arm_config("EXTH_MOEAD", 16101, 300)
    parents = traced_original(problem, cfg)
    bank, _ = seed_bank(
        problem,
        parents,
        random.Random(cfg.seed + 100000),
        1,
        families=("BILL",),
        size=10,
        accelerated=True,
        verify_trials=False,
    )
    seen = {c.genotype for c in parents}
    assert bill_seed(problem, random.Random(cfg.seed + 100000), seen) == bank["BILL"][0].genotype


def test_polish_generation_samples_twenty_percent_and_never_hurts(problem):
    members, gateway, _ = initial(problem)
    before = list(members)
    changed = polish_generation(problem, members, gateway, random.Random(3), lambda: False)
    assert gateway.counts["polish_calls"] == 20
    assert set(changed) <= set(range(100))
    for i, (old, new) in enumerate(zip(before, members)):
        assert new.genotype == old.genotype
        assert new.evaluation.flow == old.evaluation.flow
        if i in changed:
            assert new.evaluation.bill < old.evaluation.bill
            assert evaluate(problem, new.genotype, new.schedule) == new.evaluation
        else:
            assert new is old
    stopped = list(before)
    calls = gateway.counts["polish_calls"]
    assert polish_generation(problem, stopped, gateway, random.Random(3), lambda: True) == []
    assert gateway.counts["polish_calls"] == calls and stopped == before
    with pytest.raises(ValueError):
        polish_generation(problem, stopped, gateway, random.Random(3), lambda: False, 0.0)


@pytest.mark.parametrize("arm", ARMS)
def test_hybrid_arm_polishes_after_every_complete_generation(problem, arm):
    cfg = replace(arm_config(arm, 16101, 300), seconds=None, generations=2)
    evidence: dict = {}
    seen: dict = {}

    def observe(event, data):
        if event == "initial_ready":
            seen["initial"] = [identity(c) for c in data["population"]]

    result = run_arm(problem, cfg, arm, evidence, observer=observe)
    _, _, reference = initial(problem)
    assert evidence["initial_population_sha256"] == reference["initial_population_sha256"]
    assert len(seen["initial"]) == 100
    assert result.gateway.counts["polish_calls"] == 2 * 20
    assert len(result.population) == 100
    for c in result.population + result.archive.members:
        assert c.evaluation.feasible
        assert evaluate(problem, c.genotype, c.schedule) == c.evaluation


@pytest.mark.parametrize("arm", REFERENCE_ARMS)
def test_reference_arms_replay_component33_exactly(problem, arm):
    cfg = replace(arm_config(arm, 16101, 300), seconds=None, generations=1)
    replay = run_arm(problem, cfg, arm, {})
    original = component_random.run_arm(problem, cfg, arm, {})
    assert [identity(c) for c in replay.population] == [identity(c) for c in original.population]
    assert [identity(c) for c in replay.archive.members] == [
        identity(c) for c in original.archive.members
    ]
    assert replay.gateway.counts == original.gateway.counts


def test_runner_policies_and_block_orders():
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "run_ext_hybrid", scripts / "run_ext_hybrid.py"
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.remove(str(scripts))
    assert [runner.initializer_policy(a) for a in ARMS] == [POLICY] * 5
    with pytest.raises(ValueError):
        runner.initializer_policy("BILL_MOEAD")
    for index in range(20):
        assert sorted(runner.balanced_order(index)) == sorted(ARMS)
    assert {runner.balanced_order(i)[0] for i in range(5)} == set(ARMS)
