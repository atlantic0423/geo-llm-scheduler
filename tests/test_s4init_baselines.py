"""The opt-in baselines share S4's actual initial population and archive."""

import json

import pytest
from test_lsv2_fast_rightshift import synthetic

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import component_random, ext_hybrid, s4_ablation
from geo_llm_scheduler.experiments.trans_hybrid import HybridSearchContext
from geo_llm_scheduler.experiments.trans_runner import (
    BALANCED_EXTERNAL,
    BALANCED_NATIVE,
    S4INIT_EXTERNAL,
    S4INIT_NATIVE,
    run_case,
)
from geo_llm_scheduler.io.loaders import load_instance, save_instance
from geo_llm_scheduler.utils.rng import RNGManager


def test_s4init_matches_reference_population_archive_and_keys():
    problem, _ = synthetic(4)
    cfg = s4_ablation.arm_config("A0_FULL_S4", 1, 60)
    expected_evidence: dict = {}
    expected_gw = EvaluationGateway(problem, Archive())
    expected = component_random.charged_initializer(problem, cfg, "BILL_MOEAD", expected_evidence)(
        expected_gw, RNGManager(1)
    )
    baseline_evidence: dict = {}
    gw = EvaluationGateway(problem, Archive())
    actual = ext_hybrid.s4_initializer(
        problem, ext_hybrid.arm_config("EXTH_MOEAD", 1, 60), baseline_evidence
    )(gw, RNGManager(1))
    assert actual == expected
    assert gw.archive.members == expected_gw.archive.members
    for field in ("initial_population_sha256", "initial_archive_sha256"):
        assert baseline_evidence[field] == expected_evidence[field]
    assert len(actual) == len({x.genotype for x in actual}) == 100
    ctx = HybridSearchContext(problem, seed=1, seconds=60, s4_init=True)
    first = ctx.initialize(100)
    assert [x.candidate for x in first] == expected
    assert ctx.gateway.archive.members == expected_gw.archive.members
    for x in first:
        replay = ctx.evaluate(x.genes).candidate
        assert replay.genotype == x.candidate.genotype
        assert replay.schedule.starts == x.candidate.schedule.starts
        assert replay.evaluation == evaluate(problem, x.candidate.genotype, x.candidate.schedule)


def test_second_population_uses_independent_s4_seed():
    problem, _ = synthetic(4)
    ctx = HybridSearchContext(problem, seed=1, seconds=60, s4_init=True)
    ctx.initialize(100)
    ctx.initialize(100)
    assert ctx.hybrid_evidence[0]["bank_seed"] == 100001
    assert ctx.hybrid_evidence[1]["bank_seed"] == 1100004
    assert (
        ctx.hybrid_evidence[0]["initial_population_sha256"]
        != ctx.hybrid_evidence[1]["initial_population_sha256"]
    )


def test_conflicting_policies_rejected():
    problem, _ = synthetic(4)
    with pytest.raises(ValueError, match="Choose one"):
        HybridSearchContext(problem, seed=1, seconds=60, balanced=True, s4_init=True)
    with pytest.raises(ValueError, match="Choose one"):
        ext_hybrid.run_arm(
            problem,
            ext_hybrid.arm_config("EXTH_MOEAD", 1, 60),
            "EXTH_MOEAD",
            {},
            balanced=True,
            s4_init=True,
        )


def test_eight_separate_registered_arms():
    assert len(S4INIT_EXTERNAL) == 5 and len(S4INIT_NATIVE) == 3
    assert len(set(S4INIT_EXTERNAL) | set(S4INIT_NATIVE)) == 8
    for name, base in S4INIT_EXTERNAL.items():
        assert BALANCED_EXTERNAL[name.replace("_S4INIT", "_B4050")] == base
    for name, algorithm in S4INIT_NATIVE.items():
        assert BALANCED_NATIVE[name.replace("_S4INIT", "_B4050")] is algorithm


@pytest.mark.parametrize("arm", [*S4INIT_EXTERNAL, *S4INIT_NATIVE])
def test_all_eight_methods_execute_s4_initializer(tmp_path, arm):
    problem, _ = synthetic(4)
    source = tmp_path / "instance.json"
    save_instance(problem, source)
    output = tmp_path / "run"
    summary = run_case(
        arm,
        source,
        output,
        seed=1,
        seconds=60,
        max_generations=1,
        **({"population_size": 6} if arm in S4INIT_NATIVE else {}),
    )
    evidence = json.loads((output / "initialization.json").read_text(encoding="utf-8"))
    initial = evidence[0] if arm in S4INIT_NATIVE else evidence
    problem = load_instance(source)
    reference_gw = EvaluationGateway(problem, Archive())
    expected: dict = {}
    cfg = s4_ablation.arm_config("A0_FULL_S4", 1, 60)
    component_random.charged_initializer(problem, cfg, "BILL_MOEAD", expected)(
        reference_gw, RNGManager(1)
    )
    assert summary["independent_exact_audit"] == "passed"
    assert initial["policy"] == "bill4"
    for field in ("initial_population_sha256", "initial_archive_sha256"):
        assert initial[field] == expected[field]
