"""Restricted policy boundaries and mocked, credential-safe DeepSeek transport."""

import json
import math
from dataclasses import FrozenInstanceError
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from geo_llm_scheduler.experiments import llm_protocol as module
from geo_llm_scheduler.experiments.llm_global_prompt import SYSTEM_PROMPT
from geo_llm_scheduler.experiments.llm_protocol import (
    ACTIONS,
    FOCUSES,
    GROUPS,
    POLICY_SCHEMA,
    DeepSeekClient,
    GroupPolicy,
    default_policies,
    parse_policy,
    probabilities,
)


def response():
    return {
        "schema_version": POLICY_SCHEMA,
        "request_id": "run_01",
        "snapshot_id": "snapshot_02",
        "policies": [
            {
                "group": group,
                "action_weights": dict(zip(ACTIONS, (10, 15, 25, 10, 20, 5, 10, 5))),
                "target_focus": "DEMAND_PEAK",
                "stop_after_no_gain": 2,
                "rationale": "依据真实耗时与前沿反馈调整；代理不是实际节费。",
            }
            for group in GROUPS
        ],
    }


def parse(value):
    return parse_policy(json.dumps(value, ensure_ascii=False), "run_01", "snapshot_02")


def test_valid_policy_atomic_canonical_order_and_immutable():
    value = response()
    value["policies"].reverse()
    parsed = parse(value)
    assert tuple(parsed) == GROUPS
    assert parsed["Balanced"].action_weights == (10, 15, 25, 10, 20, 5, 10, 5)
    assert parsed["Balanced"].target_focus == "DEMAND_PEAK"
    with pytest.raises(FrozenInstanceError):
        parsed["Flow"].target_focus = "MIXED"


def test_defaults_are_fresh_uniform_and_original_l5():
    a, b = default_policies(), default_policies()
    a.pop("Flow")
    assert tuple(b) == GROUPS
    assert all(probabilities(p) == (0.125,) * 8 for p in b.values())
    assert all(p.target_focus == "MIXED" and p.stop_after_no_gain == 5 for p in b.values())


def test_probability_mixture_keeps_positive_exploration_and_expected_ratio():
    policy = parse(response())["Balanced"]
    p = probabilities(policy)
    assert p[2] == pytest.approx(0.2375)
    assert p[5] == pytest.approx(0.0575)
    assert math.fsum(p) == pytest.approx(1)
    one = GroupPolicy("Flow", (100, 0, 0, 0, 0, 0, 0, 0), "MIXED", 5, "")
    assert probabilities(one, 0) == (1, 0, 0, 0, 0, 0, 0, 0)
    assert probabilities(one, 1) == (0.125,) * 8
    assert min(probabilities(one)) == 0.0125


@pytest.mark.parametrize("epsilon", [-0.1, 1.1, math.nan, math.inf, True, None, "0.1"])
def test_invalid_exploration_rejected(epsilon):
    with pytest.raises(ValueError):
        probabilities(default_policies()["Flow"], epsilon)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "search_policy_response_v1"),
        ("request_id", "other"),
        ("snapshot_id", "stale"),
        ("policies", []),
        ("policies", {}),
    ],
)
def test_response_schema_snapshot_and_count_rejected(field, value):
    data = response()
    data[field] = value
    with pytest.raises(ValueError):
        parse(data)


@pytest.mark.parametrize("where", ["top", "policy", "weights"])
def test_unknown_and_missing_fields_rejected(where):
    data = response()
    obj = {
        "top": data,
        "policy": data["policies"][0],
        "weights": data["policies"][0]["action_weights"],
    }[where]
    obj["unknown"] = "ignored?"
    with pytest.raises(ValueError):
        parse(data)
    obj.pop("unknown")
    obj.pop(next(iter(obj)))
    with pytest.raises(ValueError):
        parse(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("group", "Unknown"),
        ("group", ["Flow"]),
        ("target_focus", "ACTIVE_UNION"),
        ("target_focus", "execute_code"),
        ("target_focus", None),
        ("stop_after_no_gain", True),
        ("stop_after_no_gain", 2.0),
        ("stop_after_no_gain", 0),
        ("stop_after_no_gain", 4),
        ("rationale", "x" * 301),
        ("rationale", None),
    ],
)
def test_group_focus_stop_and_rationale_rejected(field, value):
    data = response()
    data["policies"][2][field] = value
    with pytest.raises(ValueError):
        parse(data)


@pytest.mark.parametrize("weight", [True, -1, 101, "25", None, math.nan, math.inf, -math.inf])
def test_invalid_action_weights_rejected(weight):
    data = response()
    data["policies"][0]["action_weights"]["A3"] = weight
    with pytest.raises(ValueError):
        parse(data)


def test_zero_sum_duplicate_groups_and_partial_failure_are_atomic():
    data = response()
    data["policies"][2]["action_weights"] = dict.fromkeys(ACTIONS, 0)
    with pytest.raises(ValueError):
        parse(data)
    data = response()
    data["policies"][2]["group"] = "Flow"
    with pytest.raises(ValueError):
        parse(data)


@pytest.mark.parametrize(
    "content",
    [
        "null",
        "[]",
        "false",
        "{broken",
        "```json\n{}\n```",
        "{} trailing",
        '{"schema_version":"x","schema_version":"y"}',
        '{"nested":{"duplicate":1,"duplicate":2}}',
        '{"nested":NaN}',
    ],
)
def test_json_must_be_plain_unique_finite_and_complete(content):
    with pytest.raises(ValueError, match="Invalid search policy response"):
        parse_policy(content, "run_01", "snapshot_02")


@pytest.mark.parametrize(
    "content,request_id,snapshot_id",
    [
        (None, "run_01", "snapshot_02"),
        ("{}", 1, "snapshot_02"),
        ("{}", "run_01", None),
    ],
)
def test_parser_binding_arguments_are_strings(content, request_id, snapshot_id):
    with pytest.raises(ValueError):
        parse_policy(content, request_id, snapshot_id)


def test_subnormal_positive_weights_and_all_permitted_focus_stop_values():
    for focus in FOCUSES:
        for stop in (1, 2, 3, 5):
            policy = GroupPolicy("Electricity", (1e-320,) * 8, focus, stop, "")
            assert math.fsum(probabilities(policy)) == pytest.approx(1)
    with pytest.raises(ValueError):
        GroupPolicy("Flow", [1.0] * 8, "MIXED", 5, "")
    with pytest.raises(ValueError):
        GroupPolicy("Flow", (1.0,) * 7, "MIXED", 5, "")


class MockOpener:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        return BytesIO(self.result)


def envelope(**overrides):
    data = {
        "id": "response_01",
        "model": "deepseek-flash",
        "choices": [{"message": {"content": json.dumps(response())}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200},
    }
    data.update(overrides)
    return json.dumps(data).encode()


def install_opener(monkeypatch, opener):
    handlers = []

    def factory(*items):
        handlers.extend(items)
        return opener

    monkeypatch.setattr(module, "build_opener", factory)
    return handlers


def test_mocked_client_sends_fixed_global_prompt_json_disabled_thinking_and_no_retry(monkeypatch):
    opener = MockOpener(envelope())
    handlers = install_opener(monkeypatch, opener)
    client = DeepSeekClient("fake-credential-for-test")
    result = client.request({"schema_version": "search_policy_v2", "jobs": 100}, timeout=2.5)
    assert len(opener.calls) == 1
    request, timeout = opener.calls[0]
    assert request.full_url == "https://api.deepseek.com/chat/completions"
    assert 0 < timeout <= 2.5 and request.get_method() == "POST"
    body = json.loads(request.data)
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert json.loads(body["messages"][1]["content"])["jobs"] == 100
    assert body["thinking"] == {"type": "disabled"}
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 2048 and body["stream"] is False
    assert "temperature" not in body
    assert "fake-credential-for-test" not in json.dumps(body)
    assert set(result) == {"content", "usage", "returned_model", "finish_reason", "id"}
    assert result["usage"]["total_tokens"] == 200
    assert tuple(parse_policy(result["content"], "run_01", "snapshot_02")) == GROUPS
    assert handlers[0].redirect_request(request, None, 302, "redirect", {}, "https://evil") is None
    assert "fake-credential-for-test" not in repr(client)


def test_client_preserves_nonstop_finish_for_caller_rejection(monkeypatch):
    opener = MockOpener(
        envelope(choices=[{"message": {"content": "{}"}, "finish_reason": "length"}])
    )
    install_opener(monkeypatch, opener)
    assert DeepSeekClient("fake").request({}, 1)["finish_reason"] == "length"


def test_client_env_loading_is_explicit_constructor_only(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-env-value")
    opener = MockOpener(envelope())
    install_opener(monkeypatch, opener)
    DeepSeekClient().request({}, 1)
    assert opener.calls[0][0].get_header("Authorization") == "Bearer test-env-value"
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(ValueError, match="credential"):
        DeepSeekClient()


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("https://secret.invalid", 401, "credential-secret", {}, BytesIO(b"secret")),
        HTTPError("https://evil.invalid", 307, "redirect-secret", {}, BytesIO(b"secret")),
        URLError("credential-secret"),
        TimeoutError("credential-secret"),
        OSError("credential-secret"),
    ],
)
def test_client_errors_are_generic_no_body_url_key_or_retry(monkeypatch, error):
    opener = MockOpener(error=error)
    install_opener(monkeypatch, opener)
    with pytest.raises(RuntimeError) as caught:
        DeepSeekClient("credential-secret").request({}, 1)
    assert len(opener.calls) == 1
    assert not any(s in str(caught.value) for s in ("secret", "https://", "Authorization"))
    assert caught.value.__suppress_context__


@pytest.mark.parametrize(
    "raw",
    [
        b"not JSON",
        b"[]",
        b"null",
        b'{"choices":[]}',
        envelope(choices=[{"message": {"content": None}, "finish_reason": "stop"}]),
        envelope(choices=[{"message": {"content": "{}"}, "finish_reason": None}]),
        envelope(model=None),
        envelope(id=[]),
        envelope(usage=[]),
        b'{"choices":[],"choices":[]}',
        b'{"value":NaN}',
        b"\xff\xfe",
    ],
)
def test_client_rejects_invalid_envelopes_without_body_echo(monkeypatch, raw):
    opener = MockOpener(raw)
    install_opener(monkeypatch, opener)
    with pytest.raises(RuntimeError, match="invalid response envelope"):
        DeepSeekClient("fake").request({}, 1)
    assert len(opener.calls) == 1


def test_client_redacts_exact_credential_in_all_returned_strings_and_usage_keys(monkeypatch):
    opener = MockOpener(
        envelope(
            id="fake-key",
            model="fake-key",
            usage={"fake-key": ["fake-key", {"echo": "fake-key"}]},
            choices=[{"message": {"content": "fake-key"}, "finish_reason": "fake-key"}],
        )
    )
    install_opener(monkeypatch, opener)
    result = DeepSeekClient("fake-key").request({}, 1)
    assert "fake-key" not in json.dumps(result)
    assert result["content"] == "[REDACTED]"


def test_client_bounds_response_bytes(monkeypatch):
    install_opener(monkeypatch, MockOpener(b"x" * 11))
    monkeypatch.setattr(module, "_MAX_RESPONSE_BYTES", 10)
    with pytest.raises(RuntimeError, match="size limit"):
        DeepSeekClient("fake").request({}, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": ""},
        {"api_key": "two words"},
        {"api_key": "a\nb"},
        {"base_url": "http://api.deepseek.com"},
        {"base_url": "https://user:password@api.deepseek.com"},
        {"base_url": "https://api.deepseek.com?token=secret"},
        {"base_url": "https://api.deepseek.com#fragment"},
        {"base_url": "https://"},
        {"base_url": "https://[broken"},
        {"base_url": "https://api.deepseek.com\n"},
        {"model": ""},
        {"max_tokens": True},
        {"max_tokens": 0},
    ],
)
def test_client_rejects_invalid_credentials_endpoints_and_options(kwargs):
    with pytest.raises(ValueError):
        DeepSeekClient(**({"api_key": "fake"} | kwargs))


@pytest.mark.parametrize(
    "payload,timeout",
    [
        ({}, 0),
        ({}, -1),
        ({}, math.nan),
        ({}, math.inf),
        ({}, True),
        ([], 1),
        ({"bad": math.nan}, 1),
        ({"bad": object()}, 1),
        ({}, 10**1000),
    ],
)
def test_client_argument_errors_do_not_contact_network(monkeypatch, payload, timeout):
    def should_not_open(*args):
        raise AssertionError("network should not be opened")

    monkeypatch.setattr(module, "build_opener", should_not_open)
    with pytest.raises(ValueError):
        DeepSeekClient("fake").request(payload, timeout)


def test_client_read_deadline_expires_even_when_partial_bytes_keep_arriving(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    class SlowBody(BytesIO):
        def read1(self, size):
            clock[0] += 0.4
            return b"x"

    class SlowOpener:
        def open(self, request, timeout):
            return SlowBody()

    install_opener(monkeypatch, SlowOpener())
    with pytest.raises(RuntimeError, match="HTTPS request failed"):
        DeepSeekClient("fake").request({}, 1)
    assert clock[0] == pytest.approx(1.2)


def test_client_updates_socket_timeout_to_remaining_deadline(monkeypatch):
    from types import SimpleNamespace

    clock = [0.0]
    timeouts = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    class ClockBody(BytesIO):
        fp = SimpleNamespace(raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=timeouts.append)))

        def read1(self, size):
            clock[0] += 0.1
            return super().read1(size)

    class ClockOpener:
        def open(self, request, timeout):
            clock[0] += 0.2
            return ClockBody(envelope())

    install_opener(monkeypatch, ClockOpener())
    DeepSeekClient("fake").request({}, 1)
    assert timeouts == pytest.approx([0.8, 0.7])


def test_prompt_contains_problem_cost_contract_and_safe_output_boundaries():
    for fragment in (
        "SAME geographic Region",
        "effective schedulable VRAM",
        "[900b,900(b+1))",
        "full 900 seconds",
        "After T_end",
        "UNION",
        "Tchebycheff",
        "B6=6",
        "L5=5",
        "A1:",
        "A2:",
        "A3:",
        "A4:",
        "A5:",
        "A6:",
        "A7:",
        "A8:",
        "ONLY A3",
        "search_policy_response_v2",
        "search_policy_v2",
        "per second",
        "observational",
        "no counterfactual",
        "not true Pareto",
        "Null means",
    ):
        assert fragment in SYSTEM_PROMPT
