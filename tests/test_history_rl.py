"""History causality, terminal credit, archive opportunities and reference replay."""

import random
from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments.history_rl import (
    ARMS,
    CONTROLS,
    HistoryPolicy,
    run_policy,
    study_config,
    table_rows,
)
from geo_llm_scheduler.experiments.history_rl_trajectory import archive_value, improve
from geo_llm_scheduler.experiments.rl23 import run_policy as reference_run
from geo_llm_scheduler.moead.core import NormalizationContext, weights
from geo_llm_scheduler.operators.base import ProposalBatch
from geo_llm_scheduler.rl.controller import Controller
from geo_llm_scheduler.utils.rng import RNGManager


def candidate(problem):
    gateway = EvaluationGateway(problem, Archive())
    return gateway, gateway.evaluate(Genotype((0, 0, 1, 1), (0, 1, 0, 1)))


def test_state_encodings_are_bijective_and_reset_start_is_separate():
    assert len(ARMS) == len(set(ARMS)) == 12
    for arm in ARMS:
        cfg = study_config(arm, 9, 300)
        assert cfg.trigger_mode == "strict" and cfg.trigger_delta == 0 and not cfg.polish
        if arm != "BILL_PERMUTED":
            assert cfg.fixed_budget == (3 if arm in ("MACRO_B3", "PREF_HISTORY_B3") else 6)
        if arm in CONTROLS:
            continue
        p = HistoryPolicy(arm)
        indices = (0,) if arm.startswith("HISTORY_") else (0, 50, 99)
        states = {p.state(i, 100) for i in indices}
        for i in indices:
            for a in range(1, 9):
                for o in range(3):
                    states.add(p.state(i, 100, a, o))
        assert states == set(range(table_rows(arm)))
    with pytest.raises(ValueError):
        table_rows("unknown")
    with pytest.raises(ValueError):
        HistoryPolicy("BILL_PERMUTED")
    with pytest.raises(ValueError):
        p.state(100, 100)
    with pytest.raises(ValueError):
        p.state(0, 100, 0, 1)


def test_reward_outcomes_keep_empty_zero_productive_distinct():
    ordinary, frontier = HistoryPolicy("PREF_HISTORY"), HistoryPolicy("PREF_HISTORY_FRONTIER")
    assert [ordinary.outcome(q, n) for q, n in ((-1, 0), (0, 3), (0.001, 1))] == [0, 1, 2]
    assert ordinary.outcome(0, 2, 0.1) == 1
    assert frontier.outcome(0, 2, 0.1) == 2
    assert [
        frontier.signal(q, n, h)
        for q, n, h in ((-1, 0, 0), (0, 1, 0), (2, 1, 0), (0, 1, 1e-3), (2, 1, 1e-3))
    ] == [-1, 0, 1, 1, 2]
    assert ordinary.signal(2.3, 1) == 2.3
    with pytest.raises(ValueError):
        ordinary.outcome(0, -1)


def test_tables_shared_but_terminal_no_bootstrap_and_bandit_no_future():
    for arm in ("PREF_HISTORY", "PREF_HISTORY_BANDIT"):
        p = HistoryPolicy(arm)
        cfg = replace(study_config(arm, 1, 1), alpha=1, gamma=0.7)
        export = Controller(cfg)
        learner = p.learner(export, cfg)
        assert export.q is learner.q and p.learner(export, cfg) is learner
        learner.q[1] = [10.0] * 8
        learner.update(0, 1, 2, 1)
        assert learner.q[0][0] == (2 if cfg.controller == "bandit" else 9)
        learner.update(0, 1, 2, 1, terminal=True)
        assert learner.q[0][0] == 2
        learner.select(0, 0, random.Random(9))
        assert sum(learner.selections[0]) == learner.visits[0] == 1


class Empty:
    def propose(self, *args):
        return ProposalBatch(())


def test_episode_history_is_causal_reset_and_stop_terminal(problem, monkeypatch):
    import geo_llm_scheduler.experiments.history_rl_trajectory as module

    monkeypatch.setattr(module, "StructuralOperator", lambda action: Empty())
    monkeypatch.setattr(module, "ActivePack", Empty)
    monkeypatch.setattr(module, "PeakCoalition", Empty)
    gateway, c = candidate(problem)
    cfg = replace(study_config("PREF_HISTORY_STOP2", 1, 2), population=4, neighborhood=2)
    policy = HistoryPolicy("PREF_HISTORY_STOP2")
    exported = Controller(cfg)
    for _ in range(2):
        _, records = improve(
            c,
            1,
            0,
            0,
            weights(4),
            c.evaluation.objectives,
            gateway,
            exported,
            cfg,
            RNGManager(1),
            policy=policy,
        )
        assert len(records) == 2 and records[-1]["terminal"]
        assert records[-1]["stop_reason"] == "two_unproductive"
        assert records[0]["previous_action"] == 0 and records[0]["previous_reward"] == 0
        assert records[0]["state"] == policy.state(1, 4)
        assert records[1]["state"] == records[0]["next_state"]
        assert records[1]["previous_action"] == records[0]["action"]
        assert records[1]["previous_reward"] == records[0]["reward"]
    assert sum(exported.visits) == 4
    with pytest.raises(ValueError):
        improve(
            c,
            1,
            0,
            0,
            weights(4),
            c.evaluation.objectives,
            gateway,
            exported,
            replace(cfg, budget_policy="random"),
            RNGManager(1),
            policy=policy,
        )


def test_frontier_credit_uses_rectangle_union_and_no_duplicate_credit(problem):
    _, c = candidate(problem)
    context = NormalizationContext((0, 0), c.evaluation.objectives)
    assert archive_value([c], context) == archive_value([c, c], context)
    with pytest.raises(ValueError):
        archive_value([c], None)


@pytest.mark.parametrize("arm", [a for a in ARMS if a not in CONTROLS])
def test_new_trajectories_preserve_exact_and_same_child_history(problem, arm):
    from geo_llm_scheduler.evaluation.exact import evaluate

    gateway, c = candidate(problem)
    cfg = replace(study_config(arm, 7, 300), population=4, neighborhood=2)
    policy = HistoryPolicy(arm)
    policy.initial_context = NormalizationContext((0, 0), c.evaluation.objectives)
    controller = Controller(cfg)
    best, records = improve(
        c,
        1,
        0,
        0,
        weights(4),
        c.evaluation.objectives,
        gateway,
        controller,
        cfg,
        RNGManager(7),
        policy=policy,
    )
    assert records[0]["state"] == policy.state(1, 4) and records[-1]["terminal"]
    assert sum(map(sum, controller.updates)) == len(records)
    assert all(r["exact_evaluations"] <= cfg.fixed_budget for r in records)
    for prior, current in zip(records, records[1:]):
        assert current["state"] == prior["next_state"]
        assert current["previous_action"] == prior["action"]
    for member in [best] + gateway.archive.members:
        assert evaluate(problem, member.genotype, member.schedule) == member.evaluation
        assert member.evaluation.feasible


def test_new_wrapper_initialization_and_no_ls_table(monkeypatch):
    import importlib

    from geo_llm_scheduler.experiments.comparison13 import case_problem

    engine = importlib.import_module("geo_llm_scheduler.engine.run")
    monkeypatch.setattr(engine, "triggered", lambda *args: False)
    problem = case_problem(20, 3, 6, 202610160020, "arrival", 0)
    cfg = replace(study_config("PREF_HISTORY", 16301, 300), generations=1)
    evidence = {}
    result = run_policy(problem, "PREF_HISTORY", cfg, evidence)
    assert len(result.population) == 100 and evidence["policy"] == "bill4"
    assert len(result.controller.q) == 75 and sum(result.controller.visits) == 0
    with pytest.raises(ValueError):
        run_policy(problem, "PREF_HISTORY", replace(cfg, fixed_budget=1), {})


@pytest.mark.parametrize("action", (1, 3, 7, 8))
def test_fixed_action_candidate_archive_and_rng_match_reference(problem, action):
    from geo_llm_scheduler.experiments.rl23_support import StudyPolicy

    cfg = replace(
        study_config("PREF_HISTORY", 18, 300),
        population=4,
        neighborhood=2,
        enabled_operators=(action,),
    )
    outputs = []
    for policy in (HistoryPolicy("PREF_HISTORY"), StudyPolicy("STRICT_TWO_FOCUS")):
        gateway, c = candidate(problem)
        streams = RNGManager(cfg.seed)
        best, records = policy(
            c,
            1,
            0,
            0.1,
            weights(4),
            c.evaluation.objectives,
            gateway,
            Controller(cfg),
            cfg,
            streams,
        )
        outputs.append(
            (
                best,
                gateway.archive.identity_keys(),
                [(r["candidate_objectives"], r["quality_reward"]) for r in records],
                streams.snapshot(),
            )
        )
    assert outputs[0] == outputs[1]


def test_campaign_freezes_original_inputs_new_seeds_and_tables(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    from test_polish_policy_campaign import historical_fixture

    spec = importlib.util.spec_from_file_location(
        "history_runner", Path(__file__).parents[1] / "scripts/run_history_rl.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    old, root = tmp_path / "old", tmp_path / "new"
    historical_fixture(old)
    manifest = runner.prepare(root, old)
    assert manifest["runs"] == 768 and len(manifest["blocks"]) == 64
    assert manifest["independent_parents"] == 16
    assert manifest["memory_reserve_bytes"] == manifest["disk_reserve_bytes"] == 200 * 1024**3
    assert {json.loads(p.read_text())["seed"] for p in (root / "specs").glob("*.json")} == {
        16301,
        26301,
    }
    assert len({runner.sha(p) for p in (root / "inputs").glob("*.json")}) == 32
    runner.validate_inputs(root)
    assert runner.complete(root, manifest["blocks"][0], "PREF_HISTORY", manifest["source"]) is False
    (root / "inputs/b0_v0.json").write_text("mutation")
    with pytest.raises(AssertionError):
        runner.validate_inputs(root)


@pytest.mark.parametrize("arm", CONTROLS)
def test_reference_controls_replay_same_fixed_work(arm):
    from geo_llm_scheduler.experiments.comparison13 import case_problem

    problem = case_problem(20, 3, 12, 202610160023, "arrival", 0)
    cfg = replace(study_config(arm, 23, 100), generations=1, seconds=None)
    first, second = {}, {}
    a = run_policy(problem, arm, cfg, first)
    b = reference_run(problem, arm, cfg, second)
    assert [(x.genotype, x.schedule, x.evaluation) for x in a.population] == [
        (x.genotype, x.schedule, x.evaluation) for x in b.population
    ]
    assert [x.evaluation.objectives for x in a.archive.members] == [
        x.evaluation.objectives for x in b.archive.members
    ]
    assert a.controller.q == b.controller.q
    assert first["initial_population_sha256"] == second["initial_population_sha256"]
