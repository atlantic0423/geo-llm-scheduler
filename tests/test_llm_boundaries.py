"""Production wall-clock, frontier credit and transport failure boundaries."""

import json
import subprocess
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments import llm_search as module
from geo_llm_scheduler.experiments.llm_features import (
    objective_summary,
    preference_summaries,
    static_problem,
    unique_objectives,
)
from geo_llm_scheduler.experiments.llm_focus import FocusStructural, job_scores
from geo_llm_scheduler.experiments.llm_protocol import ACTIONS, GROUPS, POLICY_SCHEMA
from geo_llm_scheduler.experiments.llm_search import LLMPolicy, ObservedEvaluator
from geo_llm_scheduler.moead.core import NormalizationContext, weights
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.utils.rng import RNGManager


def make_data(problem):
    gateway = EvaluationGateway(problem, Archive())
    candidate = gateway.evaluate(Genotype((0, 0, 1, 1), (0, 1, 0, 1)))
    return dict(gateway=gateway, population=[candidate] * 6), candidate


def valid_response(payload, timeout):
    assert timeout > 0
    content = dict(
        schema_version=POLICY_SCHEMA,
        request_id=payload["request_id"],
        snapshot_id=payload["snapshot_id"],
        policies=[
            dict(
                group=group,
                action_weights={a: 20 if a == "A3" else 1 for a in ACTIONS},
                target_focus="RESOURCE_WAIT",
                stop_after_no_gain=1,
                rationale="test only",
            )
            for group in GROUPS
        ],
    )
    return dict(content=json.dumps(content), returned_model="deepseek-flash", finish_reason="stop")


class Process:
    def __init__(self, output, returncode=0, timeout=False):
        self.output, self.returncode, self.timeout = output, returncode, timeout
        self.calls = []
        self.killed = False

    def communicate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.timeout and len(self.calls) == 1:
            raise subprocess.TimeoutExpired("mocked worker", kwargs["timeout"])
        return self.output, "discarded child stderr"

    def kill(self):
        self.killed = True


def install_process(monkeypatch, process):
    captured = []
    monkeypatch.delenv("LLM_REQUEST_LOCK_DIR", raising=False)

    def factory(*args, **kwargs):
        captured.append((args, kwargs))
        return process

    monkeypatch.setattr(module.subprocess, "Popen", factory)
    return captured


def test_isolated_worker_data_only_stdin_and_metadata_response(monkeypatch):
    expected = dict(content="{}", returned_model="deepseek-flash", finish_reason="stop")
    process = Process(json.dumps(dict(ok=True, response=expected)))
    captured = install_process(monkeypatch, process)
    payload = {"request_id": "run01", "jobs": 100}
    assert module.isolated_request(payload, 1) == expected
    assert len(captured) == len(process.calls) == 1
    command = captured[0][0][0]
    assert command[-2:] == ["-m", "geo_llm_scheduler.experiments.llm_http_worker"]
    assert "run01" not in repr(command)
    body = json.loads(process.calls[0][0][0])
    assert body["payload"] == payload and 0 < body["timeout"] <= 1
    assert captured[0][1]["encoding"] == "utf-8" and not process.killed


def test_isolated_worker_hard_timeout_kills_and_reaps_child(monkeypatch):
    process = Process("", timeout=True)
    install_process(monkeypatch, process)
    with pytest.raises(TimeoutError, match="http_hard_timeout"):
        module.isolated_request({}, 1)
    assert process.killed and len(process.calls) == 2
    assert process.calls[1] == ((), {})


@pytest.mark.parametrize("status,ok", [(1, True), (0, False), (1, False)])
def test_isolated_worker_failed_exit_or_provider_response_never_succeeds(monkeypatch, status, ok):
    process = Process(json.dumps(dict(ok=ok, response={})), returncode=status)
    install_process(monkeypatch, process)
    with pytest.raises(RuntimeError, match="provider_request_failed"):
        module.isolated_request({}, 1)
    assert len(process.calls) == 1 and not process.killed


def test_isolated_expired_deadline_does_not_spawn_child(monkeypatch):
    process = Process("unused")
    captured = install_process(monkeypatch, process)
    with pytest.raises(TimeoutError, match="request_deadline"):
        module.isolated_request({}, -1)
    assert captured == []


def test_queue_contention_costs_same_deadline_and_releases_files(monkeypatch, tmp_path):
    clock = [0.0]
    attempted = []
    opened = []
    process = Process("unused")
    captured = install_process(monkeypatch, process)
    monkeypatch.setenv("LLM_REQUEST_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        module.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration)
    )

    def flock(handle, flags):
        attempted.append(handle.name)
        opened.append(handle)
        raise BlockingIOError

    fake = SimpleNamespace(flock=flock, LOCK_EX=1, LOCK_NB=2)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake)
    with pytest.raises(TimeoutError, match="request_queue_timeout"):
        module.isolated_request({}, 0.1)
    assert clock[0] == pytest.approx(0.1) and len(attempted) >= 32
    assert all(handle.closed for handle in opened) and captured == []


@pytest.mark.parametrize("fail", [False, True])
def test_acquired_queue_slot_released_on_success_and_child_failure(monkeypatch, tmp_path, fail):
    process = Process(json.dumps(dict(ok=not fail, response={"content": "{}"})))
    install_process(monkeypatch, process)
    monkeypatch.setenv("LLM_REQUEST_LOCK_DIR", str(tmp_path))
    locked = []
    fake = SimpleNamespace(flock=lambda handle, flags: locked.append(handle), LOCK_EX=1, LOCK_NB=2)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake)
    if fail:
        with pytest.raises(RuntimeError):
            module.isolated_request({}, 1)
    else:
        assert module.isolated_request({}, 1) == {"content": "{}"}
    assert len(locked) == 1 and locked[0].closed


@pytest.mark.parametrize("failure", ["wrong_model", "truncated", "timeout", "late", "bad_json"])
@pytest.mark.parametrize("previous_valid", [False, True])
def test_policy_failure_keeps_complete_previous_or_default_strategy(
    problem, monkeypatch, failure, previous_valid
):
    data, _ = make_data(problem)
    clock = [10.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    sink = []
    policy = LLMPolicy(300, 1, requester=valid_response, sink=sink.append)
    if previous_valid:
        policy.update(data)
        assert policy.revision == 1
        clock[0] = 110
    before = policy.policies

    def requester(payload, timeout):
        if failure == "timeout":
            clock[0] += timeout
            raise TimeoutError("mock timeout")
        value = valid_response(payload, timeout)
        if failure == "wrong_model":
            value["returned_model"] = "another-model"
        elif failure == "truncated":
            value["finish_reason"] = "length"
        elif failure == "late":
            clock[0] = 301
        elif failure == "bad_json":
            value["content"] = "{incomplete"
        return value

    policy.requester = requester
    policy.update(data)
    assert policy.policies is before and policy.revision == int(previous_valid)
    assert not policy.calls[-1]["accepted"]
    assert policy.calls[-1]["failure_reason"] == "invalid_or_unavailable_response"
    assert sink[-1] is policy.calls[-1]
    assert policy.wait_s == pytest.approx(data["gateway"].seconds["llm_wait"])


def test_exhausted_model_wait_cap_does_not_request_or_reset_feedback(problem, monkeypatch):
    data, _ = make_data(problem)
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.0)
    called = []
    policy = LLMPolicy(300, 1, requester=lambda *args: called.append(args))
    policy.wait_s = 15
    policy.observed = True
    policy.stats["Flow"]["A1"] = {"calls": 2, "wall_s": 1}
    policy.update(data)
    assert called == policy.calls == [] and policy.stats["Flow"]["A1"]["calls"] == 2


def test_feature_extraction_budget_exhaustion_prevents_request(problem, monkeypatch):
    data, _ = make_data(problem)
    clock = [10.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    original = module.static_problem
    called = []

    def expensive_features(problem):
        clock[0] = 301
        return original(problem)

    monkeypatch.setattr(module, "static_problem", expensive_features)
    policy = LLMPolicy(300, 1, requester=lambda *args: called.append(args))
    policy.update(data)
    assert called == [] and policy.revision == 0
    assert policy.feature_s == 291 and data["gateway"].seconds["llm_features"] == 291
    assert not policy.calls[0]["accepted"]
    assert policy.calls[0]["request"]["budget"]["remaining_s"] == 0


def test_safe_boundary_before_next_checkpoint_does_not_repeat_request(problem, monkeypatch):
    data, _ = make_data(problem)
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.0)
    policy = LLMPolicy(300, 1, requester=valid_response)
    policy.update(data)
    first = policy.policies
    policy.update(data)
    assert len(policy.calls) == 1 and policy.policies is first


def test_feedback_rates_use_actual_action_and_production_time_denominators(monkeypatch):
    monkeypatch.setattr(module.time, "perf_counter", lambda: 50.0)
    policy = LLMPolicy(300, 1)
    first = policy.feedback()
    assert first["history_status"] == "unobserved" and first["window_s"] is None
    assert first["by_group"]["Flow"]["A1"]["calls"] is None
    policy.observed = True
    policy.activation = 10
    policy.stats["Flow"]["A1"] = dict(
        calls=4,
        wall_s=8,
        local_accepts=1,
        calls_with_archive_gain=2,
        online_hv_gain=0.2,
        dominated_archive_members=3,
        scalar_gain=0.6,
    )
    policy.stats["Flow"]["A2"] = dict(calls=1, wall_s=2)
    policy.stats["Balanced"]["A3"] = dict(calls=1, wall_s=10)
    feedback = policy.feedback()
    a1 = feedback["by_group"]["Flow"]["A1"]
    assert feedback["window_s"] == 40
    assert a1["mean_wall_per_call_s"] == 2
    assert a1["wall_fraction_of_all_action_time"] == 0.4
    assert a1["wall_fraction_of_interval_production"] == 0.2
    assert a1["local_accepts_per_s"] == 0.125
    assert a1["calls_with_archive_gain_per_s"] == 0.25
    assert a1["online_hv_gain_per_s"] == 0.025
    assert a1["dominated_archive_members_per_s"] == 0.375
    assert a1["scalar_gain_per_s"] == pytest.approx(0.075)
    zero = feedback["by_group"]["Flow"]["A7"]
    assert zero["calls"] == 0 and zero["wall_fraction_of_all_action_time"] == 0
    assert zero["mean_wall_per_call_s"] is zero["online_hv_gain_per_s"] is None


def synthetic_candidate(ms, flow, bill, *, starts=(0, 100, 0, 100), os=(0, 1, 0, 1)):
    """Immutable mock-exact points for independent archive/scalar control tests."""
    return Candidate(
        Genotype(ms, os),
        Schedule(starts, (0, 0, 0, 0)),
        EvaluationResult(True, flow, bill, (0, 0), ((1,), (1,))),
        "mock-exact",
    )


class SyntheticGateway:
    """Dispatch mock-exact candidates through the real, unmodified Pareto Archive."""

    def __init__(self, problem, incumbent, candidates):
        self.problem = problem
        self.archive = Archive()
        self.archive.consider(incumbent)
        self.ideal = (0, 0)
        self.seconds = defaultdict(float)
        self.counts = defaultdict(int)
        self.by_gene = {c.genotype: c for c in candidates}
        self.dispatched = []

    def evaluate(self, genotype, schedule=None, origin="structural"):
        candidate = self.by_gene[genotype]
        self.dispatched.append((genotype, schedule, origin))
        self.counts["exact"] += 1
        self.archive.consider(candidate)
        return candidate


def test_archive_new_objectives_keep_locally_rejected_trajectory_alive_and_all_candidates_dispatch(
    problem, monkeypatch
):
    current = synthetic_candidate((0, 0, 0, 0), 100, 100)
    tradeoffs = [
        synthetic_candidate((1, 1, 1, 1), 110, 90),
        synthetic_candidate((2, 2, 2, 2), 120, 80),
        synthetic_candidate((0, 0, 0, 0), 130, 70, os=(1, 0, 1, 0)),
    ]
    gateway = SyntheticGateway(problem, current, tradeoffs)
    proposed_from = []

    class FixedOperator:
        def propose(self, p, incumbent, budget, config, rng):
            proposed_from.append(incumbent)
            return ProposalBatch([Proposal(c.genotype) for c in tradeoffs], attempts=7)

    monkeypatch.setattr(module, "StructuralOperator", lambda action: FixedOperator())
    monkeypatch.setattr(module, "ActivePack", FixedOperator)
    monkeypatch.setattr(module, "PeakCoalition", FixedOperator)
    monkeypatch.setattr(module, "FocusStructural", lambda *args: FixedOperator())
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.0)
    policy = LLMPolicy(300, 1)
    policy.policies = {g: replace(p, stop_after_no_gain=1) for g, p in policy.policies.items()}
    cfg = replace(module.study_config("LLM_FLASH", 1, 300), population=6, neighborhood=2)
    events = []
    best, records = policy(
        current,
        5,
        0,
        0,
        weights(6),
        (200, 200),
        gateway,
        None,
        cfg,
        RNGManager(1),
        observer=lambda name, data: events.append((name, data)),
    )
    assert best is current and len(records) == 2
    assert [r["accepted"] for r in records] == [False, False]
    assert [r["archive_new_objective_events"] for r in records] == [3, 0]
    assert records[0]["consecutive_no_gain"] == 0 and not records[0]["terminal"]
    assert records[1]["stop_reason"] == "no_gain_threshold"
    assert [r["exact_evaluations"] for r in records] == [3, 3]
    assert len(gateway.dispatched) == 6 and gateway.archive.attempts == 7
    assert len(gateway.archive.members) == 4 and all(c is current for c in proposed_from)
    assert [name for name, _ in events].count("structure") == 6
    assert [name for name, _ in events].count("rl_step") == 2


def test_observer_distinguishes_equal_objective_phenotype_from_new_dominating_point(problem):
    original = synthetic_candidate((0, 0, 0, 0), 100, 100)
    equal = synthetic_candidate((1, 1, 1, 1), 100, 100, starts=(1, 101, 1, 101))
    better = synthetic_candidate((2, 2, 2, 2), 90, 90)
    gateway = SyntheticGateway(problem, original, [equal, better])
    measured = ObservedEvaluator(gateway)
    measured.evaluate(equal.genotype)
    assert gateway.archive.insertions == 2 and measured.new_objectives == 0
    measured.evaluate(better.genotype)
    assert measured.new_objectives == 1 and measured.dominated_members == 2
    assert gateway.archive.members == [better]


def test_frozen_hv_proxy_handles_unavailable_context_and_reference_exclusion():
    inside = synthetic_candidate((0, 0, 0, 0), 50, 50)
    outside = synthetic_candidate((1, 1, 1, 1), 120, 1)
    policy = LLMPolicy(300, 1)
    assert policy.archive_value([inside]) == 0
    policy.initial_context = NormalizationContext((0, 0), (100, 100))
    assert policy.archive_value([inside]) == pytest.approx(0.36)
    assert policy.archive_value([inside, outside]) == pytest.approx(0.36)
    assert policy.archive_value([outside]) == 0


def test_focus_diagnostic_failure_uses_original_generator_and_is_audited(problem, monkeypatch):
    data, candidate = make_data(problem)
    source_calls = []

    class Original:
        def propose(self, *args):
            source_calls.append(args[1])
            return ProposalBatch()

    class Broken:
        def propose(self, *args):
            raise ValueError("missing diagnostic")

    class ForcedStream:
        def choices(self, *args, **kwargs):
            return [3]

    streams = SimpleNamespace(stream=lambda name: ForcedStream())
    monkeypatch.setattr(module, "StructuralOperator", lambda action: Original())
    monkeypatch.setattr(module, "FocusStructural", lambda *args: Broken())
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.0)
    policy = LLMPolicy(300, 1)
    cfg = replace(
        module.study_config("LLM_FLASH", 1, 300), population=6, neighborhood=2, rl_steps=1
    )
    _, records = policy(
        candidate,
        1,
        0,
        0,
        weights(6),
        candidate.evaluation.objectives,
        data["gateway"],
        None,
        cfg,
        streams,
    )
    assert source_calls == [candidate]
    assert records[0]["fallback_reason"] == "focus_diagnostic_failure"
    assert records[0]["applied_focus"] == "ORIGINAL"
    with pytest.raises(ValueError, match="undeclared"):
        policy(
            candidate,
            1,
            0,
            0,
            weights(6),
            candidate.evaluation.objectives,
            data["gateway"],
            None,
            cfg,
            streams,
            operator_overrides={3: Original()},
        )


def test_deadline_prevents_additional_trajectory_steps(problem, monkeypatch):
    data, candidate = make_data(problem)
    monkeypatch.setattr(module.time, "perf_counter", lambda: 301.0)
    policy = LLMPolicy(300, 1)
    cfg = replace(module.study_config("LLM_FLASH", 1, 300), population=6, neighborhood=2)
    best, records = policy(
        candidate,
        1,
        0,
        0,
        weights(6),
        candidate.evaluation.objectives,
        data["gateway"],
        None,
        cfg,
        RNGManager(1),
    )
    assert best is candidate and records == []


@pytest.mark.parametrize("action", [1, 7, 8])
def test_exact_scalar_acceptance_resets_failure_and_refreshes_timing_only_diagnostics(
    problem, monkeypatch, action
):
    data, base = make_data(problem)
    gateway = data["gateway"]
    delayed = gateway.evaluate(
        base.genotype,
        Schedule(tuple(t + 100 for t in base.schedule.starts), (0, 0, 0, 0), (100,) * 4),
    )

    class Improving:
        def propose(self, *args):
            if action == 1:
                return ProposalBatch([Proposal(base.genotype)], attempts=1)
            return ProposalBatch([Proposal(base.genotype, base.schedule, (0, 1, 2, 3))], attempts=1)

    class ForcedStream:
        def choices(self, *args, **kwargs):
            return [action]

    streams = SimpleNamespace(stream=lambda name: ForcedStream())
    monkeypatch.setattr(module, "StructuralOperator", lambda action: Improving())
    monkeypatch.setattr(module, "ActivePack", Improving)
    monkeypatch.setattr(module, "PeakCoalition", Improving)
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.0)
    policy = LLMPolicy(300, 1)
    cfg = replace(
        module.study_config("LLM_FLASH", 1, 300), population=6, neighborhood=2, rl_steps=1
    )
    best, records = policy(
        delayed,
        5,
        0,
        0,
        weights(6),
        delayed.evaluation.objectives,
        gateway,
        None,
        cfg,
        streams,
    )
    assert best.evaluation.flow < delayed.evaluation.flow
    assert records[0]["accepted"] and records[0]["consecutive_no_gain"] == 0
    assert records[0]["stop_reason"] == "horizon_or_budget"
    assert best.schedule.starts == base.schedule.starts
    assert len(best.schedule.resource_wait) == problem.operation_count


def test_llm_run_seam_charges_initializer_observes_engine_origin_and_uses_two_focus(
    problem, monkeypatch
):
    clock = [10.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    supplied = []
    outer_events = []
    requests = []
    sinks = []
    evidence = {}
    sentinel = object()
    cfg = module.study_config("LLM_FLASH", 9, 300)

    def initializer_factory(p, config, mode, ev):
        supplied.append((p, config, mode, ev))

        def initialize(gateway, streams):
            clock[0] += 2
            c = gateway.evaluate(Genotype((0, 0, 1, 1), (0, 1, 0, 1)))
            return [c] * 6

        return initialize

    def requester(payload, timeout):
        requests.append(payload)
        clock[0] += 1
        return valid_response(payload, timeout)

    def engine(p, config, *, initializer, observer, trajectory_override, **kwargs):
        gateway = EvaluationGateway(p, Archive())
        members = initializer(gateway, RNGManager(9))
        assert trajectory_override.initial_context is not None
        assert kwargs["retain_trace"] is False
        assert kwargs["offspring_transform"].arm == "TWO_FOCUS"
        observer("initialization", dict(gateway=gateway, population=members, elapsed=4))
        assert trajectory_override.begin == 8
        clock[0] = 109
        observer("before_offspring", dict(gateway=gateway, population=members, processed=1))
        observer("other_event", {})
        return sentinel

    monkeypatch.setattr(module, "charged_initializer", initializer_factory)
    monkeypatch.setattr(module, "run", engine)
    result = module.run_policy(
        problem,
        "LLM_FLASH",
        cfg,
        evidence,
        observer=lambda name, data: outer_events.append(name),
        parameters=dict(requester=requester, policy_sink=sinks.append),
    )
    assert result is sentinel and supplied == [(problem, cfg, "FULL_CURRENT_I4", evidence)]
    assert outer_events == ["initial_ready", "initialization", "before_offspring", "other_event"]
    assert len(requests) == len(sinks) == 2 and requests[0]["budget"]["elapsed_s"] == 4
    assert requests[1]["budget"]["elapsed_s"] == 101
    assert requests[1]["previous_policy"] is not None
    assert evidence["llm"]["requests"] == evidence["llm"]["valid_requests"] == 2
    assert evidence["llm"]["wait_s"] == 2 and evidence["llm"]["thinking"] == "disabled"


def test_undeclared_llm_config_and_treatment_rejected_before_initializer(problem, monkeypatch):
    monkeypatch.setattr(
        module, "charged_initializer", lambda *args: pytest.fail("must not initialize")
    )
    cfg = module.study_config("LLM_FLASH", 1, 300)
    with pytest.raises(ValueError, match="Undeclared"):
        module.run_policy(problem, "LLM_FLASH", replace(cfg, fixed_budget=10), {})
    with pytest.raises(ValueError, match="Unknown treatment"):
        module.run_policy(problem, "unknown", cfg, {})


def test_empty_snapshot_and_small_population_preserve_unobserved_meaning(problem):
    assert objective_summary([]) == dict(
        members=0, unique_objectives=0, flow_range_s=None, bill_range_cny=None
    )
    empty = replace(problem, jobs=())
    assert static_problem(empty)["job_distributions"]["release_s"]["median"] is None
    data, candidate = make_data(problem)
    rows = preference_summaries(
        problem,
        [candidate, candidate],
        data["gateway"].archive.members,
        NormalizationContext(candidate.evaluation.objectives, candidate.evaluation.objectives),
    )
    absent = next(r for r in rows if r["group"] == "Flow")
    assert absent["member_count"] == 0 and absent["representative_points"] == []
    assert unique_objectives([candidate, candidate]) == [candidate]


def test_truncated_peak_prompt_keeps_complete_tied_peak_focus(problem):
    data, candidate = make_data(problem)
    candidate = replace(
        candidate,
        schedule=Schedule((0, 8200, 0, 8200), (0, 0, 0, 0)),
        evaluation=replace(candidate.evaluation, windows=((10,) * 10, (10,) * 10)),
    )
    rows = preference_summaries(
        problem,
        [candidate] * 6,
        [candidate],
        NormalizationContext(data["gateway"].ideal, candidate.evaluation.objectives),
    )
    peak = rows[0]["region_diagnostics"][0]["peak_windows"]
    assert peak["total_count"] == 10 and peak["sent_count"] == 8 and peak["truncated"]
    assert peak["items"][-1]["end_s"] == 7200
    assert job_scores(problem, candidate, "DEMAND_PEAK") == [300, 300]


def test_empty_destroy_set_has_no_candidate_or_implicit_job_generation(problem):
    empty = replace(problem, jobs=())
    candidate = Candidate(
        Genotype((), ()), Schedule((), ()), EvaluationResult(True, 0, 0, (0, 0), ((), ()))
    )
    operator = FocusStructural(6, "RESOURCE_WAIT")
    batch = operator.propose(
        empty, candidate, 6, module.study_config("LLM_FLASH", 1, 300), RNGManager(1).stream("A6")
    )
    assert batch.proposals == [] and batch.attempts == 0
    assert job_scores(problem, make_data(problem)[1], "MIXED") == [0, 0]
    with pytest.raises(ValueError, match="limited"):
        FocusStructural(1, "RESOURCE_WAIT")
