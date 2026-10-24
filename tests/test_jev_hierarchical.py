"""Jev transport contracts, conditional choices and independent exact search audits."""

import json
import math
import random
import subprocess
from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import (
    Genotype,
    Job,
    ProblemInstance,
    Profile,
    Region,
    Schedule,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.jev_candidates import DirectCandidates
from geo_llm_scheduler.experiments.jev_decisions import (
    DecisionDeadline,
    DecisionSession,
    JevSettings,
    epsilon_sample,
    isolated_request,
    validate_answer,
)
from geo_llm_scheduler.experiments.jev_prompts import BACKGROUND_EN, PROMPTS
from geo_llm_scheduler.experiments.jev_search import JevPolicy, run_policy, study_config
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext, weights
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.utils.rng import RNGManager


class FakeJev:
    def __init__(self, action="A1", stop=0.0):
        self.action, self.stop, self.requests = action, stop, []

    def __call__(self, payload, timeout):
        assert timeout > 0
        self.requests.append(payload)
        answers = {}
        for name, q in payload["questions"].items():
            if q["type"] == "noul":
                answers[name] = dict(type="noul", noul=self.stop)
            else:
                keys = list(q["criteria"])
                best = self.action if self.action in keys else keys[0]
                probs = {k: (0.9 if k == best else 0.1 / (len(keys) - 1)) for k in keys}
                if len(keys) == 1:
                    probs = {best: 1.0}
                answers[name] = dict(
                    type="choice", choice=best, probabilities=probs, confidence=0.9
                )
        return dict(
            model=payload["model"], answers=answers, usage=dict(input_tokens=100, output_tokens=30)
        )


def current(problem):
    gateway = EvaluationGateway(problem, Archive())
    g = Genotype((0, 1, 0, 1), (0, 1, 0, 1))
    s = refresh_diagnostics(problem, g, Schedule((100, 300, 200, 500)))
    x = gateway.evaluate(g, s)
    assert x.evaluation.feasible
    return gateway, x


@pytest.mark.parametrize("epsilon,expected", [(0.5, 0.7), (0.9, 0.94)])
def test_sampling_matches_declared_mixture_not_standard_uniform_epsilon(epsilon, expected):
    p = dict(a=0.4, b=0.3, c=0.2, d=0.1)
    rng = random.Random(72)
    n = 20000
    actual = sum(epsilon_sample(p, epsilon, rng) == "a" for _ in range(n)) / n
    assert actual == pytest.approx(expected, abs=0.012)


@pytest.mark.parametrize(
    "p,eps",
    [
        ({}, 0.5),
        ({"a": math.nan}, 0.5),
        ({"a": 0.7}, 0.5),
        ({"a": True}, 0.5),
        ({"a": 1}, -1),
        ({"a": 1}, math.nan),
    ],
)
def test_sampler_rejects_invalid_distributions(p, eps):
    with pytest.raises(ValueError):
        epsilon_sample(p, eps, random.Random(1))


def test_ties_seed_replay_and_rng_stream_independence():
    p = {"a": 0.5, "b": 0.5}
    a, b = RNGManager(8), RNGManager(8)
    a.stream("unrelated").random()
    assert [epsilon_sample(p, 0.9, a.stream("job")) for _ in range(100)] == [
        epsilon_sample(p, 0.9, b.stream("job")) for _ in range(100)
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(epsilon_g=0.7),
        dict(epsilon_g=True),
        dict(model="jev-latest"),
        dict(request_timeout_s=0),
        dict(max_requests=True),
        dict(max_input_tokens=0),
        dict(max_wait_fraction=0),
        dict(stop_threshold=2),
    ],
)
def test_settings_reject_undeclared_or_unsafe_values(kwargs):
    with pytest.raises(ValueError):
        JevSettings(**kwargs)


def test_full_response_validation_and_unknown_option_rejection():
    fake = FakeJev()
    q = dict(type="choice", instructions="Pick", criteria={"a": {}, "b": {}})
    payload = dict(model="jev-1.14.0", questions={"q": q})
    r = fake(payload, 1)
    assert validate_answer(r, payload["model"], "q", q)["choice"] == "a"
    for change in [
        dict(model="jev-latest"),
        dict(answers={}),
        dict(usage=dict(input_tokens=True, output_tokens=0)),
    ]:
        with pytest.raises(ValueError):
            validate_answer(dict(r, **change), payload["model"], "q", q)
    for answer in [
        dict(type="noul", noul=0.1),
        dict(type="choice", choice="c", probabilities={"a": 1, "b": 0}, confidence=1),
        dict(type="choice", choice="b", probabilities={"a": 1, "b": 0}, confidence=1),
        dict(type="choice", choice="a", probabilities={"a": 0.1, "b": 0.1}, confidence=1),
        dict(type="choice", choice="a", probabilities={"a": 1, "b": 0}, confidence=-1),
    ]:
        with pytest.raises(ValueError):
            validate_answer(dict(r, answers={"q": answer}), payload["model"], "q", q)


def test_reservations_success_invalid_late_and_no_retry(monkeypatch):
    fake = FakeJev()
    session = DecisionSession(JevSettings(max_requests=1), 100, fake)
    q = dict(type="noul", instructions="Stop?", criteria={"true": "Stop", "false": "Continue"})
    assert session.ask({}, "q", q)["noul"] == 0
    assert session.input_tokens == session.charged_or_reserved_tokens == 100
    assert session.ask({}, "q", q) is None and len(fake.requests) == 1

    def fails(*args):
        raise RuntimeError("private provider diagnostic")

    session = DecisionSession(JevSettings(), 100, fails)
    assert session.ask({}, "q", q) is None
    assert session.charged_or_reserved_tokens == 65536
    assert "private" not in json.dumps(session.records)
    session = DecisionSession(JevSettings(max_input_tokens=65535), 100, fake)
    assert session.ask({}, "q", q) is None and not session.records
    session.begin -= 101
    with pytest.raises(DecisionDeadline):
        session.ask({}, "q", q)
    session = DecisionSession(JevSettings(), 100, fake)
    session.wait_s = 51
    assert session.ask({}, "q", q) is None

    session = DecisionSession(JevSettings(), 100, lambda payload, timeout: [])
    assert session.ask({}, "q", q) is None
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-boundary-secret")
    with pytest.raises(ValueError):
        DecisionSession(JevSettings(), 100, fake).ask(
            {"accidental": "test-boundary-secret"}, "q", q
        )


def test_all_options_are_reachable_beyond_255_and_singletons_skip_api():
    fake = FakeJev()
    session = DecisionSession(JevSettings(), 100, fake)
    assert session.choose("job", {"only": {}}, {}, random.Random(1)) == "only"
    assert not fake.requests
    options = {str(i): {"job_id": i} for i in range(511)}
    key = session.choose("job", options, {}, random.Random(3))
    assert key in options
    assert all(len(q["criteria"]) <= 255 for r in fake.requests for q in r["questions"].values())
    assert fake.requests[0]["state"]["decision_kind"] == "partition"
    with pytest.raises(ValueError):
        session.choose("job", {}, {}, random.Random(1))


@pytest.mark.parametrize("action", range(1, 9))
@pytest.mark.parametrize("epsilon", [0.5, 0.9])
def test_each_direct_operator_is_legal_bounded_and_does_not_mutate_incumbent(
    problem, action, epsilon
):
    gateway, x = current(problem)
    original = x
    policy = JevPolicy(JevSettings(epsilon_g=epsilon), 1000, 11, FakeJev())
    state = policy.snapshot(gateway, x, (0.5, 0.5))
    state["current_action"] = f"A{action}"
    out = DirectCandidates(action, policy.session, state).propose(
        problem, x, 3, Config(), random.Random(5)
    )
    assert x == original and len(out.proposals) <= 3
    if action <= 6:
        assert out.proposals
    result = execute(
        out,
        x,
        3,
        (0.5, 0.5),
        NormalizationContext(gateway.ideal, x.evaluation.objectives),
        gateway,
        f"A{action}",
    )
    for c in result.evaluated:
        assert c.evaluation.feasible
        assert c.evaluation == evaluate(problem, c.genotype, c.schedule)
    for proposal in out.proposals:
        if proposal.schedule is not None:
            for o in range(problem.operation_count):
                if o not in proposal.selected:
                    assert proposal.schedule.starts[o] == x.schedule.starts[o]
        if action in (1, 2, 3):
            assert proposal.genotype.os == x.genotype.os
    assert (
        not DirectCandidates(action, policy.session, state)
        .propose(problem, x, 0, Config(), random.Random(5))
        .proposals
    )


def test_position_question_uses_actual_sampled_job_not_models_argmax(problem, monkeypatch):
    import geo_llm_scheduler.experiments.jev_decisions as decisions

    problem = replace(problem, jobs=problem.jobs + (replace(problem.jobs[0], name="c"),))
    gateway = EvaluationGateway(problem, Archive())
    g = Genotype((0, 1, 0, 1, 0, 1), (0, 1, 2, 0, 1, 2))
    x = gateway.evaluate(
        g, refresh_diagnostics(problem, g, Schedule((100, 300, 200, 500, 300, 700)))
    )
    fake = FakeJev()
    policy = JevPolicy(JevSettings(), 100, 1, fake)
    # Force a non-argmax choice at every layer to exercise conditional parent binding.
    monkeypatch.setattr(decisions, "epsilon_sample", lambda p, epsilon, rng: list(p)[-1])
    state = policy.snapshot(gateway, x, (0.5, 0.5))
    out = DirectCandidates(4, policy.session, state).propose(
        problem, x, 1, Config(), random.Random(4)
    )
    assert out.proposals
    first = fake.requests[0]
    assert len(first["state"]["all_jobs"]) == 3
    assert set(v["job_id"] for v in first["questions"]["selection"]["criteria"].values()) == {
        0,
        1,
        2,
    }
    position = [r for r in fake.requests if r["state"]["decision_kind"] == "position"]
    assert position and position[-1]["state"]["selected_job"] == 2
    assert position[-1]["state"]["selected_operation"] // 2 == 2


def test_a4_a5_no_zero_wait_fallback_and_deadline(problem):
    gateway = EvaluationGateway(problem, Archive())
    x = gateway.evaluate(Genotype((0, 0, 0, 0), (0, 1, 0, 1)))
    fake = FakeJev()
    policy = JevPolicy(JevSettings(), 100, 1, fake)
    for a in (4, 5):
        assert (
            not DirectCandidates(a, policy.session, {})
            .propose(problem, x, 6, Config(), random.Random(1))
            .proposals
        )
    assert not fake.requests
    policy.session.begin -= 101
    assert (
        not DirectCandidates(1, policy.session, {})
        .propose(problem, x, 6, Config(), random.Random(1))
        .proposals
    )
    with pytest.raises(ValueError):
        DirectCandidates(9, policy.session, {})


def test_a8_moved_peak_rejected_valid_midpoint_and_original_horizon():
    p = ProblemInstance(
        (Job("j", 0, Profile(900, 1, 1), Profile(1, 1, 1)),),
        (Region("r", (Tariff(0, 4000, 0.1),), 1),),
        (ServingInstance("p", 0, 2, 0, 100), ServingInstance("d", 0, 2, 0, 0)),
    )
    g = Genotype((0, 1), (0, 0))
    gateway = EvaluationGateway(p, Archive())
    x = gateway.evaluate(g, refresh_diagnostics(p, g, Schedule((0, 1800))))
    policy = JevPolicy(JevSettings(), 100, 1, FakeJev())
    op = DirectCandidates(8, policy.session, policy.snapshot(gateway, x, (0.5, 0.5)))
    # A stable injected selector picks the valid midpoint from the full start list.
    original = op._pick

    def midpoint(kind, values, rng, **extra):
        if kind == "position" and 450 in values:
            return 450
        return original(kind, values, rng, **extra)

    op._pick = midpoint
    out = op.propose(p, x, 1, Config(), random.Random(3))
    assert out.proposals
    c = gateway.evaluate(g, out.proposals[0].schedule)
    assert c.evaluation.feasible and c.evaluation.windows[0][0] < 100
    assert c.schedule.horizon(p) <= x.schedule.horizon(p)


@pytest.mark.parametrize("p_stop,steps", [(0.9, 1), (0.5, 5), (0.0, 5)])
def test_poststep_noul_timing_no_fifth_query_and_no_q_updates(problem, p_stop, steps):
    gateway, x = current(problem)
    fake = FakeJev(stop=p_stop)
    policy = JevPolicy(JevSettings(), 1000, 1, fake)
    policy.initial_context = NormalizationContext(gateway.ideal, x.evaluation.objectives)
    cfg = replace(study_config("JEV_E05", 1, 1000), population=6, neighborhood=2)
    _, rows = policy(
        x, 1, 0, 0, weights(6), x.evaluation.objectives, gateway, None, cfg, RNGManager(1)
    )
    assert len(rows) == steps
    stops = [r for r in fake.requests if "stop" in r["questions"]]
    assert len(stops) == min(steps, 4)
    assert all(r["state"]["trajectory"]["completed_step"] <= 4 for r in stops)
    assert all("exact_evaluations" in r["state"]["trajectory"] for r in stops)
    assert all(row["exact_evaluations"] <= 6 for row in rows)


def test_three_group_action_batch_updates_and_failure_retains_last(problem):
    gateway, x = current(problem)
    fake = FakeJev()
    policy = JevPolicy(JevSettings(), 1000, 1, fake)
    data = dict(gateway=gateway, population=[x] * 6)
    policy.update(data)
    assert len(fake.requests) == 1 and len(fake.requests[0]["questions"]) == 3
    assert policy.revision == 1
    before = policy.action_probabilities
    policy.begin -= 334
    policy.session.requester = lambda *args: {}
    policy.update(data)
    assert policy.revision == 1 and policy.action_probabilities is before
    policy.begin -= 800
    policy.update(data)
    assert len(policy.session.records) == 2


@pytest.mark.parametrize("arm", ["JEV_E05", "JEV_E09"])
def test_complete_engine_entry_independent_exact_and_english_wire(problem, arm):
    fake = FakeJev(stop=0.8)
    cfg = replace(study_config(arm, 8, 100), population=6, neighborhood=2, generations=1)
    g = Genotype((0, 1, 0, 1), (0, 1, 0, 1))
    evidence = {}
    result = run_policy(problem, arm, cfg, evidence, requester=fake, initial=[g] * 6)
    assert evidence["jev"]["valid_requests"] >= 1
    for c in result.population + result.archive.members:
        assert c.evaluation == evaluate(problem, c.genotype, c.schedule)
        assert c.evaluation.feasible
    for payload in fake.requests:
        assert payload["state"]["problem_contract"] == BACKGROUND_EN
        wire = json.dumps(payload, ensure_ascii=False)
        assert not any("\u4e00" <= c <= "\u9fff" for c in wire)
    assert all(en.isascii() and zh for en, zh in PROMPTS.values())


def test_arm_config_and_mismatch_rejected(problem):
    assert study_config("JEV_E09", 3, 20).fixed_budget == 6
    with pytest.raises(ValueError):
        study_config("bad", 1, 10)
    with pytest.raises(ValueError):
        run_policy(problem, "bad", Config(), {})
    with pytest.raises(ValueError):
        run_policy(
            problem,
            "JEV_E05",
            study_config("JEV_E05", 1, 10),
            {},
            settings=JevSettings(epsilon_g=0.9),
        )
    with pytest.raises(ValueError):
        run_policy(problem, "JEV_E05", Config(), {})


def test_http_parent_kills_child_on_timeout_and_sanitizes_failures(monkeypatch):
    import geo_llm_scheduler.experiments.jev_decisions as decisions

    class Process:
        returncode = 0
        killed = False
        count = 0

        def communicate(self, *args, **kwargs):
            self.count += 1
            if self.count == 1:
                raise subprocess.TimeoutExpired("private", 1)
            return "", ""

        def kill(self):
            self.killed = True

    process = Process()
    monkeypatch.setattr(decisions.subprocess, "Popen", lambda *a, **kw: process)
    with pytest.raises(TimeoutError):
        isolated_request({}, 0.01)
    assert process.killed and process.count == 2
    with pytest.raises(TimeoutError):
        isolated_request({}, 0)


def test_late_answer_is_billed_but_never_used():
    q = dict(type="noul", instructions="Stop?", criteria={"true": "Stop", "false": "Continue"})
    session = DecisionSession(JevSettings(), 10, FakeJev())
    base = session.requester

    def late(payload, timeout):
        result = base(payload, timeout)
        session.begin -= 11
        return result

    session.requester = late
    assert session.ask({}, "stop", q) is None
    assert session.input_tokens == 100 and not session.records[-1]["accepted"]


def test_a8_support_only_positive_original_groups_and_member_cap(problem):
    gateway, x = current(problem)
    policy = JevPolicy(JevSettings(), 100, 1, FakeJev())
    op = DirectCandidates(8, policy.session, policy.snapshot(gateway, x, (0.5, 0.5)))
    segments = [(0, frozenset({0, 1}), 0.5), (1, frozenset({2}), 0.3)]
    members = op._coalition(segments, (0, 1), 3, random.Random(1), 0)
    assert members == frozenset({0, 1, 2})
    members = op._coalition(segments, (0, 1), 1, random.Random(1), 0)
    assert len(members) <= 1
    assert op._repair(problem, x, frozenset({0}), 0, 0, random.Random(1)) is None


def test_two_hundred_jobs_are_all_candidates_with_compact_lossless_rows(problem):
    job = Job("test", 0, Profile(1, 0.5, 1), Profile(2, 0.5, 1))
    p = replace(problem, jobs=(job,) * 200)
    gateway = EvaluationGateway(p, Archive())
    g = Genotype((0, 0) * 200, tuple(range(200)) * 2)
    x = gateway.evaluate(g)
    fake = FakeJev()
    policy = JevPolicy(JevSettings(), 100, 1, fake)
    state = policy.snapshot(gateway, x, (0.5, 0.5))
    out = DirectCandidates(3, policy.session, state).propose(p, x, 1, Config(), random.Random(1))
    assert out.proposals
    first = fake.requests[0]
    assert len(first["questions"]["selection"]["criteria"]) == 200
    assert len(first["state"]["all_jobs"]) == 200
    columns = first["state"]["operation_columns"]
    for row in first["state"]["all_jobs"]:
        for data in row["operations"]:
            op = dict(zip(columns, data))
            assert op["start_s"] == x.schedule.starts[op["operation"]]
    assert not any("name" in row for row in first["state"]["all_jobs"])


def test_bilingual_export_comes_from_wire_constants(tmp_path):
    import runpy

    from geo_llm_scheduler.experiments.jev_prompts import BACKGROUND_ZH

    export = runpy.run_path("scripts/export_jev_prompts.py")["export"]
    export(tmp_path)
    doc = (tmp_path / "PROMPTS_EN_ZH.md").read_text(encoding="utf-8")
    for text in (BACKGROUND_EN, BACKGROUND_ZH):
        assert all("> " + line in doc for line in text.splitlines())
    for name, (english, chinese) in PROMPTS.items():
        for group in ("Flow", "Balanced", "Electricity") if name == "action" else ("",):
            assert (english.format(group=group) if group else english) in doc
            assert (chinese.format(group=group) if group else chinese) in doc
    arms = json.loads((tmp_path / "arms.json").read_text())
    a, b = arms["arms"]["JEV_E05"], arms["arms"]["JEV_E09"]
    assert a.pop("epsilon_g") == 0.5 and b.pop("epsilon_g") == 0.9 and a == b
    assert not any(chr(92) + c in doc for c in "()[]")


def test_cli_exports_independent_exact_audit(problem, tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    import geo_llm_scheduler.cli.jev as cli

    gateway, x = current(problem)
    monkeypatch.setattr(cli, "load_instance", lambda _: problem)
    result = SimpleNamespace(
        population=[x], archive=gateway.archive, elapsed=0.01, termination_reason="generations"
    )

    def injected_run(*args, **kwargs):
        args[3]["jev"] = {"jev_treatment_valid": True}
        return result

    monkeypatch.setattr(cli, "run_policy", injected_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "jev",
            "--instance",
            "test.json",
            "--epsilon-g",
            "0.5",
            "--seconds",
            "10",
            "--seed",
            "1",
            "--output",
            str(tmp_path),
        ],
    )
    cli.main()
    assert json.loads((tmp_path / "result.json").read_text())["final_exact_audit"]
    assert '"final_exact_audit": true' in capsys.readouterr().out
    monkeypatch.setattr(cli, "evaluate", lambda *a: replace(x.evaluation, flow=9999))
    with pytest.raises(ValueError, match="exact audit"):
        cli.main()


def test_worker_uses_fixed_official_url_and_sanitizes_credentials(monkeypatch, capsys):
    import io

    import geo_llm_scheduler.experiments.jev_http_worker as worker

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"model":"jev-1.14.0","answers":{},"usage":{}}'

    urls = []
    monkeypatch.setenv("TYPESAFE_API_KEY", "DUMMY_TEST_KEY")
    monkeypatch.setattr(
        worker, "urlopen", lambda request, timeout: urls.append(request.full_url) or Reply()
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(dict(payload={}, timeout=1))))
    worker.main()
    assert urls == ["https://api.typesafe.ai/v1/systemone"]
    assert json.loads(capsys.readouterr().out)["ok"]
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps(dict(payload={"bad": "DUMMY_TEST_KEY"}, timeout=1)))
    )
    worker.main()
    out = capsys.readouterr().out
    assert not json.loads(out)["ok"] and "DUMMY" not in out


def test_malformed_usage_falls_back_without_leaking_or_crashing():
    session = DecisionSession(JevSettings(), 10, lambda *args: {"usage": None})
    q = dict(type="noul", instructions="Stop?", criteria={"true": "Stop", "false": "Continue"})
    assert session.ask({}, "stop", q) is None
    assert session.charged_or_reserved_tokens == 65536


def test_provider_usage_over_contract_is_counted_and_stops_more_requests():
    def over_contract(payload, timeout):
        response = FakeJev()(payload, timeout)
        response["usage"]["input_tokens"] = 70000
        return response

    session = DecisionSession(JevSettings(max_input_tokens=65536), 10, over_contract)
    q = dict(type="noul", instructions="Stop?", criteria={"true": "Stop", "false": "Continue"})
    assert session.ask({}, "stop", q) is None
    assert session.input_tokens == session.charged_or_reserved_tokens == 70000
    assert session.records[0]["provider_usage_exceeds_context_reservation"]
    assert session.ask({}, "stop", q) is None and len(session.records) == 1


def test_live_run_requires_credential_before_initialization(problem, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        run_policy(problem, "JEV_E05", study_config("JEV_E05", 1, 10), {})


def test_all_failed_injected_policies_are_marked_invalid_treatment(problem):
    g = Genotype((0, 1, 0, 1), (0, 1, 0, 1))
    cfg = replace(study_config("JEV_E05", 1, 30), population=6, neighborhood=2, generations=1)
    evidence = {}
    result = run_policy(
        problem, "JEV_E05", cfg, evidence, requester=lambda *args: {}, initial=[g] * 6
    )
    assert not evidence["jev"]["jev_treatment_valid"]
    assert evidence["jev"]["valid_requests"] == 0
    assert all(c.evaluation.feasible for c in result.population)
