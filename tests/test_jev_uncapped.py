"""No-count-cutoff, intact cost guard, and secret-free failure classifications."""

import json
from dataclasses import replace

import pytest
from test_jev_comparison import campaign, historical_fixture
from test_jev_single import RankingJev

from geo_llm_scheduler.experiments.jev_clock import UNCAPPED_PROMPT_VERSION
from geo_llm_scheduler.experiments.jev_decisions import (
    DecisionSession,
    JevSettings,
    JevTransportError,
)
from geo_llm_scheduler.experiments.jev_single import SingleJevPolicy


def test_uncapped_session_passes_512_and_still_stops_at_token_guard():
    session = DecisionSession(
        JevSettings(max_requests=None, max_input_tokens=1_000_000),
        900,
        RankingJev(),
        exclude_api_wait=True,
    )
    q = dict(type="noul", instructions="Stop?", criteria={})
    for _ in range(600):
        assert session.ask({}, "stop", q) is not None
    assert len(session.records) == 600
    session.charged_or_reserved_tokens = 1_000_000
    assert session.ask({}, "stop", q) is None
    assert len(session.records) == 600
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            JevSettings(max_requests=invalid)
    assert JevSettings().max_requests == 512


@pytest.mark.parametrize(
    "failure,code",
    [
        ("sum", "probability_sum_not_one"),
        ("ids", "option_set_mismatch"),
        ("choice", "choice_not_maximum"),
        ("timeout", "transport_timeout"),
        ("http", "http_error"),
        ("provider", "transport_unavailable"),
    ],
)
def test_failure_codes_distinguish_response_and_transport_without_leaking(failure, code):
    fake = RankingJev(action="A1")

    def requester(payload, timeout):
        if failure == "timeout":
            raise TimeoutError("PRIVATE_PROVIDER_DETAIL")
        if failure == "http":
            raise JevTransportError("http_error", 429)
        if failure == "provider":
            raise RuntimeError("PRIVATE_PROVIDER_DETAIL")
        response = fake(payload, timeout)
        answer = response["answers"]["action"]
        if failure == "sum":
            answer["probabilities"]["A1"] = 0.99
        if failure == "ids":
            answer["probabilities"].pop("A2")
        if failure == "choice":
            answer["choice"] = "A2"
        return response

    session = DecisionSession(JevSettings(), 900, requester, exclude_api_wait=True)
    answer = session.ask(
        {}, "action", dict(type="choice", instructions="Action?", criteria={"A1": {}, "A2": {}})
    )
    assert answer is None
    record = session.records[-1]
    assert record["failure_detail"] == code
    assert record["failure_reason"] == "invalid_unavailable_or_late"
    assert "PRIVATE_PROVIDER_DETAIL" not in json.dumps(record)
    if failure in ("sum", "ids", "choice"):
        assert record["reserved_input_tokens"] == 100 and session.input_tokens == 100
    else:
        assert record["reserved_input_tokens"] == 65536
    if failure == "http":
        assert record["http_status"] == 429


def test_transport_classifier_rejects_untrusted_codes_and_status():
    error = JevTransportError("PRIVATE_PROVIDER_DETAIL", 999)
    assert error.code == "transport_unavailable" and error.http_status is None
    assert "PRIVATE_PROVIDER_DETAIL" not in str(error)


def test_uncapped_pilot_freezes_three_controls_search_time_and_cost(tmp_path):
    old = tmp_path / "old"
    historical_fixture(old)
    root = tmp_path / "new"
    m = campaign.prepare_pilot(
        root,
        old,
        original_key="n200_p0_v0_s16101",
        seconds=900,
        max_tokens=80_000_000,
        spend_cap_usd=3.5,
        single_candidate=True,
        exclude_api_wait=True,
        no_request_cap=True,
        include_controls=True,
    )
    assert m["arms"] == ["BILL_MOEAD", "MAIN_Q", campaign.SINGLE_ARM]
    assert m["max_requests"] is None and m["no_request_cap"] is True
    assert m["preallocated_upper_usd"] == pytest.approx(3.46)
    assert campaign.validate_inputs(root) == m
    assert m["prompt_version"] == UNCAPPED_PROMPT_VERSION
    policy = SingleJevPolicy(
        replace(JevSettings(), epsilon_g=0.9, max_requests=None),
        900,
        1,
        RankingJev(),
        exclude_api_wait=True,
    )
    assert policy.session.prompt_version == UNCAPPED_PROMPT_VERSION
    assert "NO request-count cutoff" in policy.session.background
    m["max_requests"] = 512
    campaign.atomic_json(root / "manifest.json", m)
    campaign.atomic_json(
        root / "manifest.sha256.json", dict(sha256=campaign.sha(root / "manifest.json"))
    )
    with pytest.raises(AssertionError):
        campaign.validate_inputs(root)


def test_request_flags_cannot_silently_change_old_arm_semantics(tmp_path):
    for flags in (
        {"no_request_cap": True},
        {"include_controls": True},
        {"single_candidate": True, "no_request_cap": True},
    ):
        with pytest.raises(ValueError, match="requires declared B1"):
            campaign.prepare_pilot(tmp_path / "bad", tmp_path / "missing", **flags)
