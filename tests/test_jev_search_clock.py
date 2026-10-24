"""API-free checks of both clocks, deadline/progress, costs and frozen pilot binding."""

import importlib
import json
from dataclasses import replace

import pytest
from test_jev_comparison import campaign, historical_fixture
from test_jev_comparison import export_fixture as export_fixture
from test_jev_single import RankingJev, many_jobs

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import jev_decisions as decisions
from geo_llm_scheduler.experiments.jev_clock import CLOCK_EN, PROMPT_VERSION
from geo_llm_scheduler.experiments.jev_search import SINGLE_ARM, run_policy, study_config
from geo_llm_scheduler.experiments.jev_single import SingleJevPolicy
from geo_llm_scheduler.moead.core import weights
from geo_llm_scheduler.utils.rng import RNGManager


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.mark.parametrize("failed", [False, True])
def test_only_blocking_is_free_including_failures_but_local_work_is_charged(monkeypatch, failed):
    clock = Clock()
    monkeypatch.setattr(decisions.time, "perf_counter", clock)
    real = decisions.validate_answer

    def validate(*args):
        clock.advance(1)
        return real(*args)

    monkeypatch.setattr(decisions, "validate_answer", validate)
    fake = RankingJev()

    def requester(payload, timeout):
        assert timeout == 15
        clock.advance(10)
        if failed:
            raise TimeoutError("synthetic timeout")
        return fake(payload, timeout)

    sink = []

    def record(row):
        sink.append(row)
        clock.advance(2)

    session = decisions.DecisionSession(
        decisions.JevSettings(max_requests=2), 10, requester, record, exclude_api_wait=True
    )
    q = {"stop": dict(type="noul", instructions="remaining wall time", criteria={})}
    for _ in range(2):
        assert bool(session.ask_many({}, q)) != failed
    assert session.excluded_api_wait_s == 20
    assert session.elapsed() == (4 if failed else 6)
    assert sum(r["api_blocked_s"] for r in sink) == 20
    assert session.ask_many({}, q) is None  # Request guard remains effective.
    assert len(sink) == 2
    assert sink[0]["request"]["questions"]["stop"]["instructions"] == "remaining search time"
    clock.advance(10)
    with pytest.raises(decisions.DecisionDeadline):
        session.ask_many({}, q)


def test_no_aggregate_wait_cutoff_but_token_reservation_still_limits_calls(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(decisions.time, "perf_counter", clock)
    fake = RankingJev()

    def requester(payload, timeout):
        clock.advance(10)
        return fake(payload, timeout)

    session = decisions.DecisionSession(
        decisions.JevSettings(max_requests=70), 1, requester, exclude_api_wait=True
    )
    q = {"stop": dict(type="noul", instructions="stop", criteria={})}
    for _ in range(65):
        assert session.ask_many({}, q) is not None
    assert session.excluded_api_wait_s == 650 and session.elapsed() == 0
    session.charged_or_reserved_tokens = session.settings.max_input_tokens
    assert session.ask_many({}, q) is None
    old = decisions.DecisionSession(decisions.JevSettings(), 1, requester)
    assert old.ask_many({}, q) is None  # Old end-to-end deadline is unchanged.


def test_engine_uses_search_time_at_safe_boundary_and_for_progress(problem, monkeypatch):
    engine = importlib.import_module("geo_llm_scheduler.engine.run")
    clock, blocked, progress = Clock(), [0.0], []
    monkeypatch.setattr(engine, "perf_counter", clock)

    def observer(event, data):
        if event == "before_offspring":
            clock.advance(10)
            blocked[0] += 7

    def trajectory(child, index, stagnation, fraction, *args, **kwargs):
        progress.append(fraction)
        return child, []

    cfg = Config(
        population=4,
        neighborhood=2,
        generations=20,
        seconds=5,
        trigger_mode="always",
        method="full",
    )
    result = engine.run(
        problem,
        cfg,
        observer=observer,
        trajectory_override=trajectory,
        safe_boundary_deadline=True,
        budget_excluded_seconds=lambda: blocked[0],
    )
    assert result.termination_reason == "time_budget" and result.offspring_count == 1
    assert progress == [0.6]
    assert result.elapsed == 20 and result.elapsed - blocked[0] == 6
    for bad in (-1, float("nan"), 1000):
        with pytest.raises(ValueError, match="excluded"):
            engine.run(problem, cfg, budget_excluded_seconds=lambda: bad)


def test_policy_wire_and_completed_action_rates_exclude_blocking(problem, monkeypatch):
    p, gateway, x = many_jobs(problem, 3)
    clock = Clock()
    monkeypatch.setattr(decisions.time, "perf_counter", clock)
    fake = RankingJev(action="A1")

    def requester(payload, timeout):
        clock.advance(2)
        return fake(payload, timeout)

    real = decisions.validate_answer

    def validate(*args):
        clock.advance(0.01)
        return real(*args)

    monkeypatch.setattr(decisions, "validate_answer", validate)
    policy = SingleJevPolicy(
        decisions.JevSettings(epsilon_g=0.9), 1000, 1, requester, exclude_api_wait=True
    )
    cfg = replace(study_config(SINGLE_ARM, 1, 1000), population=6, neighborhood=2)
    _, rows = policy(
        x, 1, 0, 0, weights(6), x.evaluation.objectives, gateway, None, cfg, RNGManager(1)
    )
    assert len(rows) == 5
    assert all(r["seconds"] > r["action_seconds"] for r in rows)
    assert sum(r["action_seconds"] for r in rows) < 1
    assert policy.elapsed() < 1 and policy.session.excluded_api_wait_s > 10
    for payload in fake.requests:
        assert CLOCK_EN in payload["state"]["problem_contract"]
        assert "consume production wall time" not in payload["state"]["problem_contract"]
        assert payload["state"]["prompt_version"] == PROMPT_VERSION
        assert payload["state"]["budget"]["wait_cap_s"] is None
        assert "wall cost" not in str(payload["questions"])


def test_search_clock_pilot_binding_and_non_b1_rejection(tmp_path, problem):
    old = tmp_path / "old"
    historical_fixture(old)
    root = tmp_path / "new"
    m = campaign.prepare_pilot(
        root,
        old,
        original_key="n200_p0_v0_s16101",
        single_candidate=True,
        exclude_api_wait=True,
    )
    assert campaign.validate_inputs(root) == m
    assert m["clock_basis"] == "search_excluding_api_wait"
    assert m["preallocated_upper_usd"] == pytest.approx(1.78)
    assert m["window_seconds"] == 9480 and m["prompt_version"] == PROMPT_VERSION
    m["clock_basis"] = "wall"
    campaign.atomic_json(root / "manifest.json", m)
    campaign.atomic_json(
        root / "manifest.sha256.json", dict(sha256=campaign.sha(root / "manifest.json"))
    )
    with pytest.raises(AssertionError):
        campaign.validate_inputs(root)
    with pytest.raises(ValueError, match="single B1"):
        campaign.prepare_pilot(tmp_path / "bad", old, exclude_api_wait=True)
    with pytest.raises(ValueError, match="only for the B1"):
        run_policy(problem, "JEV_E09", study_config("JEV_E09", 1, 10), {}, exclude_api_wait=True)


def test_export_keeps_both_clocks_and_validator_rejects_changed_accounting(
    export_fixture, monkeypatch
):
    root, key, m = export_fixture
    m.update(
        kind="single_n200_b1_top10",
        arms=[SINGLE_ARM],
        runs=1,
        epsilon_g=0.9,
        exclude_api_wait=True,
        clock_basis="search_excluding_api_wait",
        prompt_version=PROMPT_VERSION,
        aggregate_api_wait_cap_s=None,
        preallocated_upper_usd=campaign.budget_bound(1, m["max_input_tokens"]),
    )
    spec_path = root / "specs" / (key + ".json")
    spec = json.loads(spec_path.read_text())
    spec.update(n=200, variant=0)
    campaign.atomic_json(spec_path, spec)
    m["files"][spec_path.relative_to(root).as_posix()] = campaign.sha(spec_path)
    campaign.atomic_json(root / "manifest.json", m)
    campaign.atomic_json(
        root / "manifest.sha256.json", dict(sha256=campaign.sha(root / "manifest.json"))
    )
    original = campaign.run_jev

    def fake(*args, exclude_api_wait, observer, request_sink, **kwargs):
        assert exclude_api_wait is True

        def sink(row):
            request_sink(dict(row, api_blocked_s=0.05, clock_basis=m["clock_basis"]))

        result = original(*args, observer=observer, request_sink=sink, **kwargs)
        args[3]["jev"].update(
            prompt_version=PROMPT_VERSION,
            clock_basis=m["clock_basis"],
            search_elapsed_s=result.elapsed - 0.05,
            api_blocked_s=0.05,
            production_wall_s=result.elapsed,
            aggregate_api_wait_cap_s=None,
        )
        return result

    monkeypatch.setattr(campaign, "run_jev", fake)
    campaign._worker(root, key, SINGLE_ARM, 0)
    assert campaign.complete(root, key, SINGLE_ARM, m["source"])
    out = root / "runs" / key / SINGLE_ARM
    summary = json.loads((out / "summary.json").read_text())
    summary["initialization"]["jev"]["api_blocked_s"] = 0.1
    campaign.atomic_json(out / "summary.json", summary)
    marker = json.loads((out / "complete.json").read_text())
    marker["files"]["summary.json"] = dict(
        size=(out / "summary.json").stat().st_size, sha256=campaign.sha(out / "summary.json")
    )
    campaign.atomic_json(out / "complete.json", marker)
    with pytest.raises(AssertionError):
        campaign.complete(root, key, SINGLE_ARM, m["source"])
