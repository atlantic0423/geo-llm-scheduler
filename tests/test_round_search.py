"""API-free counterexamples for four-layer control, clock and exact-search contracts."""

import copy
import importlib
import json
from dataclasses import replace

import pytest
from test_jev_single import many_jobs

from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.jev_compact import expand_state
from geo_llm_scheduler.experiments.jev_decisions import DecisionDeadline
from geo_llm_scheduler.experiments.jev_single import TopFractionCandidates
from geo_llm_scheduler.experiments.round_decisions import RoundSession, validate_response
from geo_llm_scheduler.experiments.round_prompts import BACKGROUND, ROUND_SELECT
from geo_llm_scheduler.experiments.round_search import (
    ARM,
    RoundPolicy,
    run_policy,
    seed_pool,
    study_config,
)
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.utils.rng import RNGManager


class Flash:
    def __init__(self, action="A1", stop=True):
        self.action, self.stop, self.packets = action, stop, []

    def __call__(self, packet, timeout):
        self.packets.append(copy.deepcopy(packet))
        answer = {k: packet[k] for k in ("request_id", "snapshot_id")}
        if packet["task"] == "ROUND_SELECT":
            answer["selected"] = [
                dict(
                    solution_id=r["solution_id"],
                    direction_id=r["allowed_direction_ids"][0],
                    reason_code="UNDEREXPLORED",
                )
                for r in packet["state"]["candidate_solutions"][
                    : packet["limits"]["selected_count"]
                ]
            ]
        elif packet["task"] == "STEP_STOP":
            answer.update(
                stop=self.stop,
                reason_code="LOW_EXPECTED_GAIN" if self.stop else "CONTINUE_OPPORTUNITY",
            )
        else:
            field = "action_scores" if packet["task"] == "STEP_ACTION" else "option_scores"
            answer[field] = {
                k: int(v) if packet["task"] == "STEP_ACTION" else 1
                for k, v in packet["allowed_options"].items()
            }
            if field == "action_scores" and packet["allowed_options"].get(self.action):
                answer[field] = {k: int(k == self.action) for k in answer[field]}
        return dict(
            content=json.dumps(answer),
            usage=dict(prompt_tokens=100, completion_tokens=50),
            finish_reason="stop",
            returned_model="deepseek-flash",
        )


def test_seed_pool_deduplicates_phenotype_not_objective_and_preserves_bindings(problem):
    _, gateway, seed = many_jobs(problem, 3)
    rows, mapping = seed_pool([seed, seed], 1)
    assert len(rows) == len(mapping) == 1 and rows[0]["allowed_direction_ids"] == [0, 1]
    altered = replace(
        seed, schedule=replace(seed.schedule, starts=tuple(t + 1 for t in seed.schedule.starts))
    )
    rows, _ = seed_pool([seed, altered], 1)
    assert len(rows) == 2  # Same stored objective is intentionally insufficient for deduplication.


@pytest.mark.parametrize(
    "case", ["snapshot", "request", "extra", "direction", "duplicate", "quota"]
)
def test_round_selection_rejects_mismatches(problem, case):
    _, _, seed = many_jobs(problem, 3)
    rows, _ = seed_pool([seed, seed], 1)
    p = dict(
        task="ROUND_SELECT",
        request_id="q",
        snapshot_id="s",
        state=dict(candidate_solutions=rows),
        limits=dict(selected_count=1),
        reason_codes=["UNDEREXPLORED"],
    )
    a = dict(
        request_id="q",
        snapshot_id="s",
        selected=[
            dict(solution_id=rows[0]["solution_id"], direction_id=0, reason_code="UNDEREXPLORED")
        ],
    )
    assert validate_response(p, a) == a
    if case in ("snapshot", "request"):
        a[case + "_id"] = "stale"
    elif case == "extra":
        a["unknown"] = 1
    elif case == "direction":
        a["selected"][0]["direction_id"] = 99
    elif case == "quota":
        a["selected"] = []
    else:
        p["limits"]["selected_count"] = 2
        a["selected"] *= 2
    with pytest.raises(ValueError):
        validate_response(p, a)


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "1", 0])
def test_score_validation_rejects_invalid_or_zero_mass(bad):
    p = dict(
        task="STEP_ACTION", request_id="q", snapshot_id="s", allowed_options=dict(A1=True, A2=False)
    )
    a = dict(request_id="q", snapshot_id="s", action_scores=dict(A1=bad, A2=0))
    with pytest.raises(ValueError):
        validate_response(p, a)
    a["action_scores"] = dict(A1=1, A2=1)
    with pytest.raises(ValueError, match="Masked"):
        validate_response(p, a)
    a["action_scores"] = dict(A1=1)
    with pytest.raises(ValueError, match="cover"):
        validate_response(p, a)


def test_stop_must_be_boolean_with_declared_reason():
    p = dict(task="STEP_STOP", request_id="q", snapshot_id="s", reason_codes=["NO_OPPORTUNITY"])
    a = dict(request_id="q", snapshot_id="s", stop=False, reason_code="NO_OPPORTUNITY")
    assert validate_response(p, a) == a
    for value in (0, 1, "false", None):
        with pytest.raises(ValueError):
            validate_response(p, dict(a, stop=value))


@pytest.mark.parametrize("failed", [False, True])
def test_clock_only_excludes_transport_including_failures(monkeypatch, failed):
    module = importlib.import_module("geo_llm_scheduler.experiments.round_decisions")
    now = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])
    real = module.validate_response

    def validate(*args):
        now[0] += 2
        return real(*args)

    monkeypatch.setattr(module, "validate_response", validate)

    def requester(packet, timeout):
        now[0] += 10
        if failed:
            raise TimeoutError("synthetic")
        return Flash()(packet, timeout)

    def sink(row):
        now[0] += 3

    session = RoundSession(100, requester, sink, max_requests=1)
    answer = session.request("STEP_ACTION", {}, dict(A1=True))
    assert bool(answer) != failed
    assert session.excluded_api_wait_s == 10
    assert session.elapsed() == (3 if failed else 5)
    assert sum(b - a for a, b in session.transport_intervals) == 10
    assert session.records[0]["accepted"] != failed
    assert session.request("STEP_ACTION", {}, dict(A1=True)) is None
    now[0] += 100
    with pytest.raises(DecisionDeadline):
        session.request("STEP_ACTION", {}, dict(A1=True))


@pytest.mark.parametrize("action", range(1, 9))
def test_all_eight_real_generators_use_new_english_scores_and_independent_exact(problem, action):
    p, gateway, seed = many_jobs(problem, 6)
    fake = Flash()
    cfg = replace(study_config(ARM, 31, 100), population=6, neighborhood=2)
    policy = RoundPolicy(cfg, fake)
    policy.initial_context = NormalizationContext(gateway.ideal, seed.evaluation.objectives)
    state = policy.snapshot(gateway, seed, (0.5, 0.5))
    state["current_action"] = f"A{action}"
    batch = TopFractionCandidates(action, policy.session, state).propose(
        p, seed, 1, cfg, RNGManager(99).stream("target")
    )
    assert len(batch.proposals) <= 1
    for proposal in batch.proposals:
        candidate = gateway.evaluate(proposal.genotype, proposal.schedule)
        assert evaluate(p, candidate.genotype, candidate.schedule) == candidate.evaluation
        assert candidate.evaluation.feasible
    for packet in fake.packets:
        assert json.dumps(packet, ensure_ascii=False).isascii()
        assert packet["task"] == "STEP_TARGET"
        assert len(expand_state(packet["state"])["all_jobs"]) == 6
    if action == 6:
        assert len(fake.packets) == 1
        selection = policy.session.selections[-1]
        assert len(selection["selected"]) == len(set(selection["selected"])) == 2


def test_full_round_control_requests_twenty_percent_uses_completed_stop_feedback(problem):
    fake = Flash(stop=False)
    cfg = replace(study_config(ARM, 17, 100), population=10, neighborhood=3, generations=1)
    initial = initial_genotypes(problem, cfg, RNGManager(17).stream("initial"))
    evidence = {}
    events = []
    result = run_policy(
        problem,
        ARM,
        cfg,
        evidence,
        initial=initial,
        requester=fake,
        observer=lambda e, d: events.append((e, d)),
    )
    summary = evidence["round_llm"]
    assert len(summary["rounds"]) == 1
    row = summary["rounds"][0]
    assert row["quota"] == min(2, row["pool_size"])
    assert row["started"] == row["finished"] == row["quota"]
    assert set(summary["valid_by_task"]) == {
        "ROUND_SELECT",
        "STEP_ACTION",
        "STEP_TARGET",
        "STEP_STOP",
    }
    steps = [d for e, d in events if e == "rl_step"]
    assert len(steps) == 5 * row["quota"]
    assert sum(p["task"] == "STEP_STOP" for p in fake.packets) == 4 * row["quota"]
    assert all(p["state"]["trajectory_feedback"] for p in fake.packets if p["task"] == "STEP_STOP")
    assert all(
        d["exact_evaluations"] <= 1 and d["scalar_after"] <= d["scalar_before"] + 1e-9
        for d in steps
    )
    assert all(
        evaluate(problem, c.genotype, c.schedule) == c.evaluation
        for c in result.population + result.archive.members
    )
    assert result.gateway.counts.get("trigger_hits", 0) == 0
    assert not any(any(row) for row in result.controller.updates)


def test_invalid_model_falls_back_without_claiming_valid_control(problem):
    _, gateway, seed = many_jobs(problem, 3)

    def bad(packet, timeout):
        raise RuntimeError("unavailable")

    cfg = replace(study_config(ARM, 3, 100), population=5, neighborhood=2)
    policy = RoundPolicy(cfg, bad)
    policy.initial_context = NormalizationContext(gateway.ideal, seed.evaluation.objectives)
    current = policy.improve_seed(seed, 0, [seed] * 5, gateway, RNGManager(3))
    assert current.evaluation.feasible and policy.session.valid_by_task == {}
    assert policy.session.fallback_by_task["STEP_ACTION"] > 0


def test_generation_seam_only_runs_after_complete_traversal_and_default_is_identical(problem):
    cfg = replace(study_config("BILL_MOEAD", 1, 100), population=4, neighborhood=2, generations=2)
    calls = []
    plain = run(problem, cfg)
    opted = run(problem, cfg, generation_transform=lambda g, p, e, r, s: calls.append((g, len(p))))
    assert calls == [(0, 4), (1, 4)]
    assert plain.population == opted.population and plain.archive.members == opted.archive.members
    assert plain.trace == opted.trace or len(plain.trace) == len(opted.trace)


def test_custom_deepseek_system_prompt_is_validated_and_original_default_retained():
    from geo_llm_scheduler.experiments.llm_global_prompt import SYSTEM_PROMPT
    from geo_llm_scheduler.experiments.llm_protocol import DeepSeekClient

    assert DeepSeekClient("fake").system_prompt == SYSTEM_PROMPT
    assert DeepSeekClient("fake", system_prompt=BACKGROUND).system_prompt == BACKGROUND
    with pytest.raises(ValueError):
        DeepSeekClient("fake", system_prompt="")
    assert BACKGROUND.isascii() and "0.25 hours" in BACKGROUND
    assert "EXACTLY limits.selected_count" in ROUND_SELECT


def test_references_stay_on_the_existing_config_and_unknown_arms_rejected():
    from geo_llm_scheduler.experiments.llm_search import study_config as old

    for arm in ("MAIN_Q", "BILL_MOEAD"):
        assert study_config(arm, 1, 10) == old(arm, 1, 10)
    with pytest.raises(ValueError):
        study_config("other", 1, 10)
