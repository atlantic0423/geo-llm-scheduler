"""Whole-module controls and pure random initialization do not alter S4 semantics."""

from dataclasses import asdict, replace

import pytest
from test_lsv2_fast_rightshift import CAL, NO_P3, _prepared, synthetic

from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import lsv2_eligibility_selection as reference
from geo_llm_scheduler.experiments import s4_ablation as study
from geo_llm_scheduler.lsv2.qlearning import QLearningSearch


def test_single_module_configuration_and_empty_learner_per_run():
    assert len(study.ARMS) == 4 and len(study.PRIMARY_CONTRASTS) == 3
    cfg = study.arm_config(study.ARMS[0], 16901, 1200)
    assert study.search_config(study.ARMS[0]) == reference.search_config("S4_FRESH")
    assert study.search_config("A1_NO_SHIFT") == reference.search_config("S0_NONE")
    assert study.search_config("A3_RANDOM_INIT") == study.search_config(study.ARMS[0])
    uniform = study.make_search("A2_RANDOM_OPERATOR", cfg, CAL)
    assert not isinstance(uniform, QLearningSearch) and not hasattr(uniform, "table")
    assert uniform.config.selector == "uniform" and "P3" not in uniform.config.operators
    q1 = study.make_search(study.ARMS[0], cfg, CAL)
    q2 = study.make_search(study.ARMS[0], cfg, CAL)
    assert isinstance(q1, QLearningSearch) and isinstance(q2, QLearningSearch)
    assert q1.table is not q2.table and not q1.table.values
    assert q1.spec.gamma == 0.9 and q1.config.time_share == 0.2
    for arm in study.ARMS:
        assert study.configuration_record(arm, cfg, CAL)["arm"] == arm
    with pytest.raises(ValueError):
        study.search_config("invalid")


def test_full_replays_reference_and_no_shift_never_calls_module():
    a, ap, ag, ar = _prepared(NO_P3)
    b, bp, bg, br = _prepared(NO_P3)
    a.config = reference.search_config("S4_FRESH")
    b.config = study.search_config("A0_FULL_S4")
    a._trajectory(ap, ag, ar, lambda: 10)
    b._trajectory(bp, bg, br, lambda: 10)
    assert ap == bp and ag.archive.members == bg.archive.members
    assert asdict(a.table) == asdict(b.table)
    b.config = study.search_config("A1_NO_SHIFT")
    before = len(b.polish_steps)
    b._trajectory(bp, bg, br, lambda: 10)
    assert len(b.polish_steps) == before
    for candidate in bp + bg.archive.members:
        assert evaluate(bg.problem, candidate.genotype, candidate.schedule) == candidate.evaluation


@pytest.mark.parametrize("arm", study.ARMS)
def test_real_charged_run_initialization_policy_and_exact(arm):
    problem, _ = synthetic(4)
    cfg = replace(study.arm_config(arm, 42, 10), generations=1)
    evidence = {}
    searches = []
    result = study.run_arm(problem, arm, cfg, CAL, evidence, search_sink=searches.append)
    assert len(result.population) == cfg.population
    assert evidence["policy"] == ("random" if arm == "A3_RANDOM_INIT" else "bill4")
    assert evidence["initialization_seconds"] >= 0
    assert len(searches) == 1
    for candidate in result.population:
        assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation
