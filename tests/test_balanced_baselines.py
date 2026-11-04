"""Pair the stronger initializer and all eight research entry points."""

from test_lsv2_fast_rightshift import synthetic

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import ext_hybrid
from geo_llm_scheduler.experiments.trans_hybrid import HybridSearchContext
from geo_llm_scheduler.experiments.trans_runner import BALANCED_EXTERNAL, BALANCED_NATIVE
from geo_llm_scheduler.utils.rng import RNGManager


def test_balanced_initialization_is_identical_in_both_representations():
    problem, _ = synthetic(4)
    cfg = ext_hybrid.arm_config("EXTH_MOEAD", 1, 60)
    gw = EvaluationGateway(problem, Archive())
    evidence: dict = {}
    original = ext_hybrid.balanced_initializer(problem, cfg, evidence)(gw, RNGManager(1))
    ctx = HybridSearchContext(problem, seed=1, seconds=60, balanced=True)
    keys = ctx.initialize(100)
    assert (
        evidence["heuristic"],
        evidence["bill"],
        evidence["bill_trials"],
        evidence["random"],
    ) == (40, 10, 4, 50)
    assert len(original) == len({x.genotype for x in original}) == 100
    assert [x.candidate for x in keys] == original
    assert (
        ctx.hybrid_evidence[0]["initial_population_sha256"] == evidence["initial_population_sha256"]
    )
    for x in keys:
        assert (
            evaluate(problem, x.candidate.genotype, x.candidate.schedule) == x.candidate.evaluation
        )
        replay = ctx.evaluate(x.genes).candidate
        assert replay.genotype == x.candidate.genotype
        assert replay.schedule.starts == x.candidate.schedule.starts
        assert replay.evaluation == x.candidate.evaluation


def test_eight_independent_arm_names():
    assert len(BALANCED_EXTERNAL) == 5 and len(BALANCED_NATIVE) == 3
    assert len(set(BALANCED_EXTERNAL) | set(BALANCED_NATIVE)) == 8
