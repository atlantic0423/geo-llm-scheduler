"""Research semantics, terminal credit, states, provenance and unchanged baseline replay."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.engine.trajectory import improve
from geo_llm_scheduler.experiments.rl23 import run_policy
from geo_llm_scheduler.experiments.rl23_support import ARMS, StudyPolicy, study_config
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, weights
from geo_llm_scheduler.operators.base import ProposalBatch
from geo_llm_scheduler.rl.controller import Controller
from geo_llm_scheduler.rl.state import extract
from geo_llm_scheduler.utils.rng import RNGManager


def sample(problem):
    gateway = EvaluationGateway(problem, Archive())
    candidate = gateway.evaluate(Genotype((0, 0, 1, 1), (0, 1, 0, 1)))
    return gateway, candidate


def test_configs_are_strict_and_only_declared_engine_factors_change():
    assert len(ARMS) == len(set(ARMS)) == 23
    for arm in ARMS:
        c = study_config(arm, 16101, 300)
        assert c.trigger_mode == "strict" and c.trigger_delta == 0 and not c.polish
        if not arm.startswith("BILL_"):
            assert c.fixed_budget == (1 if arm == "MACRO_B1" else 3 if arm == "MACRO_B3" else 6)
    assert study_config("BILL_PERMUTED", 1, 1).method == "plain"
    assert study_config("BILL_PERMUTED", 1, 1).replacement_policy == "permuted"
    with pytest.raises(ValueError):
        study_config("unknown", 1, 1)
    with pytest.raises(ValueError):
        StudyPolicy("STRICT_TWO_FOCUS", 0)


def test_rewards_charge_full_cost_without_rewarding_duplicates(problem):
    assert StudyPolicy("REWARD_TIME_PENALTY", 2).signal(0, 4) == -0.5
    assert StudyPolicy("REWARD_BOUNDED_RATE", 2).signal(1, 0) == 10
    assert StudyPolicy("REWARD_BOUNDED_RATE", 2).signal(-1, 2) == -1.25
    policy = StudyPolicy("REWARD_ARCHIVE_TIME", 2)
    assert policy.signal(1, 2, 0) == -0.25
    assert policy.signal(-1, 2, 0) == -1.25
    gateway, candidate = sample(problem)
    with pytest.raises(ValueError):
        policy.archive_value([candidate])
    policy.initial_context = NormalizationContext((0, 0), candidate.evaluation.objectives)
    assert policy.archive_value([candidate]) == policy.archive_value([candidate, candidate])
    assert gateway.counts["exact"] == 1


def test_time_and_binding_states_preserve_size_and_zero_ties(problem):
    _, candidate = sample(problem)
    cfg = Config(population=4, neighborhood=2)
    context = NormalizationContext((0, 0), candidate.evaluation.objectives)
    values = (0.0,) * 6
    args = (1, values, 0, cfg, candidate, context, (0.9, 0.1))
    assert StudyPolicy("STATE_TIME42").state(*args, 0.1) + 1 == StudyPolicy("STATE_TIME42").state(
        *args, 0.9
    )
    assert StudyPolicy("STATE_TIME84").state(*args, 0.1) == 2 * extract(1, values, 0, cfg)
    active = StudyPolicy("STATE_ACTIVE42")
    assert active.state(*args, 0.1) // 14 == 0
    assert active.state(1, values, 0, cfg, candidate, context, (0.1, 0.9), 0.1) // 14 == 2
    zero = NormalizationContext(candidate.evaluation.objectives, candidate.evaluation.objectives)
    assert active.state(1, values, 0, cfg, candidate, zero, (0.1, 0.9), 0.1) // 14 == 1
    exported = Controller(cfg)
    learner = StudyPolicy("STATE_TIME84").learner(exported, cfg)
    assert len(exported.q) == 84 and exported.q is learner.q


def test_duration_bootstrap_terminal_and_masked_future(problem):
    policy = StudyPolicy("TIME_DISCOUNT_Q", 1)
    cfg = Config(gamma=0.5, alpha=1)
    learner = policy.learner(Controller(cfg), cfg)
    learner.q[1] = [4.0] * 8
    policy.update(learner, 0, 1, 2, 1, 2, False)
    assert learner.q[0][0] == 3
    policy.update(learner, 0, 1, 2, 1, 2, True)
    assert learner.q[0][0] == 2
    _, candidate = sample(problem)
    masked = StudyPolicy("LEGAL_OPPORTUNITY_MASK")
    masked.mask(learner, 0, problem, candidate)
    assert 4 not in learner.available_actions(0) and 5 not in learner.available_actions(0)
    assert 8 in learner.available_actions(0)


def test_eligibility_cache_reuses_only_unchanged_candidate_problem_and_actions(
    problem, monkeypatch
):
    from geo_llm_scheduler.experiments import rl23_support

    policy = StudyPolicy("LEGAL_OPPORTUNITY_MASK")
    cfg = Config()
    learner = policy.learner(Controller(cfg), cfg)
    _, candidate = sample(problem)
    original = rl23_support.est_for
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(rl23_support, "est_for", counted)
    policy.mask(learner, 0, problem, candidate)
    first = len(calls)
    policy.mask(learner, 1, problem, candidate)
    assert len(calls) == first and learner.available_actions(0) == learner.available_actions(1)
    policy.mask(learner, 1, replace(problem), candidate)
    assert len(calls) > first
    second = len(calls)
    policy.mask(learner, 1, problem, replace(candidate))
    assert len(calls) > second


def test_baseline_trajectory_replays_genes_archive_q_and_rng(problem):
    cfg = replace(study_config("STRICT_TWO_FOCUS", 18, 300), population=4, neighborhood=2)
    snapshots = []
    for policy in (None, StudyPolicy("STRICT_TWO_FOCUS")):
        gateway, candidate = sample(problem)
        controller = Controller(cfg)
        streams = RNGManager(cfg.seed)
        fn = improve if policy is None else policy
        best, records = fn(
            candidate,
            1,
            0,
            0.1,
            weights(4),
            candidate.evaluation.objectives,
            gateway,
            controller,
            cfg,
            streams,
        )
        snapshots.append(
            (
                best,
                gateway.archive.identity_keys(),
                controller.q,
                [r["action"] for r in records],
                streams.snapshot(),
            )
        )
    assert snapshots[0] == snapshots[1]


class EmptyOperator:
    def propose(self, *args):
        return ProposalBatch((), 0)


def test_early_stop_terminal_and_candidate_accounting(problem):
    cfg = replace(
        study_config("TRAJECTORY_STOP2", 1, 300),
        population=4,
        neighborhood=2,
        enabled_operators=(8,),
    )
    gateway, candidate = sample(problem)
    policy = StudyPolicy("TRAJECTORY_STOP2")
    controller = Controller(cfg)
    _, records = policy(
        candidate,
        1,
        0,
        0.1,
        weights(4),
        maximum([candidate]),
        gateway,
        controller,
        cfg,
        RNGManager(1),
        operator_overrides={8: EmptyOperator()},
    )
    assert len(records) == 2 and not records[0]["terminal"] and records[-1]["terminal"]
    assert sum(map(sum, controller.updates)) == 2


@pytest.mark.parametrize("arm", ARMS)
def test_all_treatments_run_and_preserve_exact_feasibility(problem, arm):
    cfg = replace(
        study_config(arm, 4, 300), population=4, neighborhood=2, generations=1, seconds=None
    )
    policy = StudyPolicy(arm)
    gateway, candidate = sample(problem)
    policy.initial_context = NormalizationContext((0, 0), candidate.evaluation.objectives)
    result = run(problem, cfg, trajectory_override=policy)
    assert all(c.evaluation.feasible for c in result.population + result.archive.members)
    assert result.offspring_count == 4


def test_run_policy_rejects_undeclared_treatment(problem):
    with pytest.raises(ValueError):
        run_policy(problem, "MACRO_B1", study_config("STRICT_TWO_FOCUS", 1, 300), {})


@pytest.mark.parametrize(
    "arm", ("BILL_PERMUTED", "STRICT_TWO_FOCUS", "STATE_TIME84", "REWARD_ARCHIVE_TIME")
)
def test_charged_bill4_wrapper_records_same_initial_fingerprints(arm):
    from geo_llm_scheduler.experiments.comparison13 import case_problem

    problem = case_problem(20, 3, 6, 202610160020, "arrival", 0)
    evidence = {}
    cfg = replace(study_config(arm, 16101, 300), generations=1)
    events = []
    result = run_policy(
        problem, arm, cfg, evidence, observer=lambda event, data: events.append(event)
    )
    assert evidence["policy"] == "bill4" and len(result.population) == 100
    assert "initial_ready" in events and all(c.evaluation.feasible for c in result.population)
    assert len(result.controller.q) == (84 if arm == "STATE_TIME84" else 42)
    assert len(evidence["initial_population_sha256"]) == 64


def test_campaign_original_input_binding_and_queue(tmp_path):
    from test_polish_policy_campaign import historical_fixture

    spec = importlib.util.spec_from_file_location(
        "rl23_runner", Path(__file__).parents[1] / "scripts/run_rl23.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    old = tmp_path / "old"
    historical_fixture(old)
    root = tmp_path / "new"
    manifest = runner.prepare(root, old)
    assert manifest["runs"] == 1472 and len(runner.queue(root, manifest)) == 1472
    assert manifest["window_seconds"] == 28800
    for i, key in enumerate(manifest["blocks"]):
        s = json.loads((root / "specs" / (key + ".json")).read_text())
        assert s["n"] == (20, 50, 100, 200)[i % 4] and set(s["order"]) == set(ARMS)
    runner.validate_inputs(root)
    (root / "inputs/b0_v0.json").write_text("mutated")
    with pytest.raises(AssertionError):
        runner.validate_inputs(root)


def test_time84_table_is_allocated_even_when_strict_gate_never_hits(monkeypatch):
    import importlib

    from geo_llm_scheduler.experiments.comparison13 import case_problem

    engine = importlib.import_module("geo_llm_scheduler.engine.run")
    monkeypatch.setattr(engine, "triggered", lambda *args: False)
    problem = case_problem(20, 3, 6, 202610160020, "arrival", 0)
    cfg = replace(study_config("STATE_TIME84", 16101, 300), generations=1)
    result = run_policy(problem, "STATE_TIME84", cfg, {})
    assert len(result.controller.q) == 84 and sum(result.controller.visits) == 0
