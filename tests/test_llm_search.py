"""LLM execution boundaries, exact opportunities, feature units and replay semantics."""

import json
import random
import time
from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments.llm_features import preference_summaries, static_problem
from geo_llm_scheduler.experiments.llm_focus import FocusStructural, job_scores, sample_destroy
from geo_llm_scheduler.experiments.llm_protocol import POLICY_SCHEMA, default_policies
from geo_llm_scheduler.experiments.llm_search import (
    LLMPolicy,
    ObservedEvaluator,
    run_policy,
    study_config,
)
from geo_llm_scheduler.moead.core import NormalizationContext, weights
from geo_llm_scheduler.operators.base import ProposalBatch
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.utils.rng import RNGManager


def make_candidate(problem):
    gateway = EvaluationGateway(problem, Archive())
    candidate = gateway.evaluate(Genotype((0, 0, 1, 1), (0, 1, 0, 1)))
    return gateway, candidate


def policy_response(payload, timeout):
    assert timeout > 0
    return dict(
        content=json.dumps(
            dict(
                schema_version=POLICY_SCHEMA,
                request_id=payload["request_id"],
                snapshot_id=payload["snapshot_id"],
                policies=[
                    dict(
                        group=g,
                        action_weights={f"A{i}": 1 for i in range(1, 9)},
                        target_focus="MIXED",
                        stop_after_no_gain=2,
                        rationale="test",
                    )
                    for g in default_policies()
                ],
            )
        ),
        returned_model="deepseek-flash",
        finish_reason="stop",
        usage={},
    )


def test_snapshot_denominators_and_same_objectives_do_not_create_coverage(problem):
    gateway, c = make_candidate(problem)
    population = [c] * 6
    context = NormalizationContext(gateway.ideal, c.evaluation.objectives)
    rows = preference_summaries(problem, population, gateway.archive.members, context)
    assert sum(r["member_count"] for r in rows) == 6
    assert sum(r["population_occupied_directions"] for r in rows) == 1
    assert all(r["mean_resource_wait_s"] == sum(c.schedule.resource_wait) / 2 for r in rows)
    assert all(
        r["mean_total_bill_cny"]
        == pytest.approx(r["mean_tou_bill_cny"] + r["mean_demand_bill_cny"])
        for r in rows
    )
    assert all(len(r["diagnostic_schedule_id"]) == 20 for r in rows)
    static = static_problem(problem)
    assert static["regions"][0]["instances"][0]["compute_capacity"] == 1
    assert static["units"]["vram"] == "effective schedulable GB"


@pytest.mark.parametrize("action", [3, 6])
def test_mixed_focus_replays_original_generator_and_rng(problem, action):
    _, c = make_candidate(problem)
    cfg = study_config("LLM_FLASH", 1, 300)
    a, b = random.Random(11), random.Random(11)
    original = StructuralOperator(action).propose(problem, c, 6, cfg, a)
    focused = FocusStructural(action, "MIXED").propose(problem, c, 6, cfg, b)
    assert original.proposals == focused.proposals and a.getstate() == b.getstate()


@pytest.mark.parametrize(
    "focus",
    ["FLOW_WAIT", "RESOURCE_WAIT", "KV_DELAY", "REGION_LOAD", "TOU_EXPOSURE", "DEMAND_PEAK"],
)
@pytest.mark.parametrize("action", [3, 6])
def test_focus_keeps_structural_legality_and_same_incumbent(problem, focus, action):
    gateway, c = make_candidate(problem)
    batch = FocusStructural(action, focus).propose(
        problem, c, 6, study_config("LLM_FLASH", 1, 300), random.Random(8)
    )
    assert len(batch.proposals) <= 6
    for proposal in batch.proposals:
        evaluated = gateway.evaluate(proposal.genotype)
        assert evaluated.evaluation.feasible
        if action == 3:
            assert proposal.genotype.os == c.genotype.os
            changed = [
                i
                for i in range(2)
                if proposal.genotype.ms[2 * i : 2 * i + 2] != c.genotype.ms[2 * i : 2 * i + 2]
            ]
            assert len(changed) == 1


def test_destroy_sampling_is_explicit_without_replacement_and_zero_score_safe():
    assert sample_destroy([0, 0, 0], 2, random.Random(1)) == random.Random(1).sample(range(3), 2)
    assert len(set(sample_destroy([0, 1, 5], 3, random.Random(2)))) == 3
    assert sample_destroy([], 0, random.Random(2)) == []
    with pytest.raises(ValueError):
        sample_destroy([float("nan")], 1, random.Random(2))


def test_unsupported_focus_rejected_and_proxy_kv_zeros_included(problem):
    _, c = make_candidate(problem)
    assert job_scores(problem, c, "KV_DELAY") == [0, 0]
    with pytest.raises(ValueError):
        job_scores(problem, c, "ACTIVE_UNION")


def test_archive_observer_preserves_gateway_dispatch_and_ignores_duplicate_objective(problem):
    gateway, c = make_candidate(problem)
    wrapper = ObservedEvaluator(gateway)
    wrapper.evaluate(c.genotype)
    assert wrapper.new_objectives == 0
    assert gateway.counts["exact"] == 2 and gateway.archive.attempts == 2
    assert wrapper.ideal == gateway.ideal
    wrapper.ideal = gateway.ideal


def test_update_binding_uses_three_group_feedback_and_atomic_failure_retains_policy(problem):
    gateway, c = make_candidate(problem)
    policy = LLMPolicy(300, 1, requester=policy_response)
    policy.begin = time.perf_counter()
    policy.initial_context = NormalizationContext(gateway.ideal, c.evaluation.objectives)
    data = dict(gateway=gateway, population=[c] * 6)
    policy.update(data)
    assert policy.revision == 1 and policy.calls[0]["accepted"]
    assert policy.calls[0]["request"]["recent_feedback"]["history_status"] == "unobserved"
    assert all(
        stats["calls"] is None
        for group in policy.calls[0]["request"]["recent_feedback"]["by_group"].values()
        for stats in group.values()
    )
    policy.begin -= 101
    before = policy.policies
    policy.requester = lambda payload, timeout: dict(
        content="{}", finish_reason="stop", returned_model="deepseek-flash"
    )
    policy.update(data)
    assert policy.revision == 1 and policy.policies is before and len(policy.calls) == 2
    assert not policy.calls[-1]["accepted"]
    assert isinstance(policy.calls[-1]["request"]["previous_policy"][0]["action_weights"], dict)


def test_passed_checkpoints_merge_and_no_model_calls_after_deadline(problem):
    gateway, c = make_candidate(problem)
    policy = LLMPolicy(300, 1, requester=policy_response)
    policy.begin = time.perf_counter() - 210
    policy.update(dict(gateway=gateway, population=[c] * 6))
    assert policy.checkpoint == 3 and len(policy.calls) == 1
    policy.update(dict(gateway=gateway, population=[c] * 6))
    assert len(policy.calls) == 1
    policy.begin = time.perf_counter() - 301
    policy.update(dict(gateway=gateway, population=[c] * 6))
    assert len(policy.calls) == 1


def test_no_candidate_calls_stop_at_policy_threshold_and_no_q_updates(problem, monkeypatch):
    import geo_llm_scheduler.experiments.llm_search as module

    class Empty:
        def propose(self, *args):
            return ProposalBatch()

    monkeypatch.setattr(module, "StructuralOperator", lambda action: Empty())
    monkeypatch.setattr(module, "ActivePack", Empty)
    monkeypatch.setattr(module, "PeakCoalition", Empty)
    monkeypatch.setattr(module, "FocusStructural", lambda *args: Empty())
    gateway, c = make_candidate(problem)
    policy = LLMPolicy(300, 1)
    policy.begin = time.perf_counter()
    policy.policies = {g: replace(p, stop_after_no_gain=2) for g, p in policy.policies.items()}
    cfg = replace(study_config("LLM_FLASH", 1, 300), population=6, neighborhood=2)
    _, records = policy(
        c, 1, 0, 0, weights(6), c.evaluation.objectives, gateway, None, cfg, RNGManager(1)
    )
    assert len(records) == 2 and records[-1]["stop_reason"] == "no_gain_threshold"
    assert sum(s["calls"] for g in policy.stats.values() for s in g.values()) == 2
    assert all(r["exact_evaluations"] == 0 for r in records)


@pytest.mark.parametrize(
    ("arm", "reference"), [("MAIN_Q", "STRICT_TWO_FOCUS"), ("BILL_MOEAD", "BILL_PERMUTED")]
)
def test_reference_controls_are_verbatim_calls(problem, monkeypatch, arm, reference):
    import geo_llm_scheduler.experiments.llm_search as module

    called = []
    sentinel = object()
    monkeypatch.setattr(
        module, "reference_run", lambda *args, **kwargs: called.append((args, kwargs)) or sentinel
    )
    config = study_config(arm, 7, 300)
    assert run_policy(problem, arm, config, {}) is sentinel
    assert called[0][0][1] == reference and called[0][0][2] is config
    with pytest.raises(ValueError):
        study_config("unknown", 1, 300)
