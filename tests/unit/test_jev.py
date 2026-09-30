"""Jev contract, deadline, failure isolation and credential redaction tests."""

import copy
import importlib.util
import io
import json
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from geo_llm_scheduler.controllers import jev
from geo_llm_scheduler.controllers.jev import (
    HttpReply,
    JevClient,
    JevError,
    Question,
    parse_response,
)
from geo_llm_scheduler.controllers.jev_questions import BUDGETS, question_hash, questions

MODEL = "jev-1.13.0"
KEY = "test-only-secret-not-a-real-key"


def payload(qs):
    answers = {}
    for name, question in qs.items():
        if question.kind == "noul":
            answers[name] = {"type": "noul", "noul": 0.4}
        else:
            options = list(question.criteria)
            answers[name] = {
                "type": "choice",
                "choice": options[0],
                "probabilities": {k: float(k == options[0]) for k in options},
                "confidence": 0.9,
            }
    return {"model": MODEL, "answers": answers, "usage": {"input_tokens": 40, "output_tokens": 2}}


class Clock:
    def __init__(self):
        self.time = 0.0

    def __call__(self):
        return self.time

    def sleep(self, duration):
        self.time += duration


def client(transport, clock=None, **kwargs):
    timer = clock or Clock()
    return JevClient(
        MODEL,
        random.Random(16),
        api_key=KEY,
        transport=transport,
        clock=timer,
        sleeper=timer.sleep,
        **kwargs,
    )


def test_combined_choice_and_noul_and_question_freeze():
    qs = questions(budget=True)
    result = parse_response(payload(qs), qs, MODEL)
    assert result.answers["operator"].choice == "A1"
    assert BUDGETS[result.answers["budget"].choice] == 3
    assert result.usage == {"input_tokens": 40, "output_tokens": 2}
    noul = questions(continuation=True)
    assert parse_response(payload(noul), noul, MODEL).answers["continue"].noul == 0.4
    assert len(question_hash()) == 64 and question_hash() == question_hash()
    with pytest.raises(JevError):
        questions(budget=True, continuation=True)


@pytest.mark.parametrize(
    "question",
    [
        Question("choice", "", {"a": "x"}),
        Question("choice", "q", {}),
        Question("choice", "q", {str(i): "x" for i in range(256)}),
        Question("noul", "q", {"yes": "x"}),
        Question("choice", "q", {"": "x"}),
        Question("score", "q", {"a": "x"}),
        Question("choice", "q", {"a": None}),
    ],
)
def test_invalid_question(question):
    with pytest.raises(JevError):
        question.payload()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(model="jev-1.14.0"),
        lambda p: p.update(answers=[]),
        lambda p: p.update(answers={}),
        lambda p: p.update(usage=None),
        lambda p: p["usage"].update(input_tokens=-1),
        lambda p: p["usage"].update(output_tokens=True),
        lambda p: p["answers"]["operator"].update(type="noul"),
        lambda p: p["answers"].update(operator=None),
        lambda p: p["answers"]["operator"].update(choice="A9"),
        lambda p: p["answers"]["operator"].update(probabilities={"A1": 1}),
        lambda p: p["answers"]["operator"].update(confidence=float("nan")),
        lambda p: p["answers"]["operator"].update(confidence="high"),
        lambda p: p["answers"]["operator"].update(confidence=True),
        lambda p: p["answers"]["operator"]["probabilities"].update(A1=0.5),
        lambda p: p["answers"]["operator"]["probabilities"].update(A1=-1),
        lambda p: p["answers"]["operator"].update(choice="A2"),
    ],
)
def test_invalid_schema(mutation):
    qs = questions()
    p = copy.deepcopy(payload(qs))
    mutation(p)
    with pytest.raises(JevError):
        parse_response(p, qs, MODEL)


def test_model_pin_missing_key_and_validation(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevError, match="not configured"):
        JevClient(MODEL, random.Random(1))
    for model in ("jev-latest", "jev-preview", ""):
        with pytest.raises(JevError, match="versioned"):
            JevClient(model, random.Random(1), api_key=KEY)
    for kwargs in ({"timeout": 0}, {"timeout": float("inf")}, {"max_attempts": 4}):
        with pytest.raises(JevError):
            client(lambda *a: None, **kwargs)
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    c = JevClient(MODEL, random.Random(1))
    with pytest.raises(AttributeError):
        c.model = "jev-1.14.0"


@pytest.mark.parametrize("status", [429, 500, 503, 529, 0])
def test_retry_latency_and_combined_request(status):
    clock = Clock()
    qs = questions(budget=True)
    calls = []

    def transport(body, key, timeout):
        calls.append((json.loads(body), key, timeout))
        clock.sleep(0.2)
        if len(calls) == 1:
            return HttpReply(status, retry_after=0.6)
        return HttpReply(200, json.dumps(payload(qs)).encode())

    result = client(transport, clock).decide({"Preference": "Balanced"}, qs)
    assert result.retries == 1 and result.elapsed == pytest.approx(1.0)
    assert result.http_statuses == (status, 200)
    assert set(calls[0][0]["questions"]) == {"operator", "budget"}
    assert "options" not in calls[0][0]["questions"]["operator"]
    assert KEY not in json.dumps(result.__dict__, default=str)


def test_retry_exhaustion_timeout_and_secret_redaction():
    calls = []

    def fail(*args):
        calls.append(args)
        raise TimeoutError(KEY)

    with pytest.raises(JevError) as raised:
        client(fail).decide({}, questions())
    assert len(calls) == 3 and KEY not in str(raised.value)
    assert raised.value.__cause__ is None
    with pytest.raises(JevError, match="HTTP 401"):
        client(lambda *a: HttpReply(401, KEY.encode())).decide({}, questions())
    with pytest.raises(JevError, match="Credential"):
        client(fail).decide({"oops": KEY}, questions())
    with pytest.raises(JevError, match="Credential"):
        client(lambda *a: HttpReply(200, KEY.encode())).decide({}, questions())
    with pytest.raises(JevError, match="one question"):
        client(fail).decide({}, {})
    with pytest.raises(JevError, match="JSON"):
        client(lambda *a: HttpReply(200, b"bad json")).decide({}, questions())


def test_deadline_prevents_new_calls_and_backoff_and_late_answer():
    clock = Clock()
    c = client(lambda *a: pytest.fail("Must not call expired decision"), clock)
    with pytest.raises(JevError, match="deadline"):
        c.decide({}, questions(), deadline=0)
    with pytest.raises(JevError, match="Retry would exceed"):
        client(lambda *a: HttpReply(429), clock).decide({}, questions(), deadline=0.1)
    timeouts = []

    def late(body, key, timeout):
        timeouts.append(timeout)
        clock.sleep(0.2)
        return HttpReply(200, json.dumps(payload(questions())).encode())

    with pytest.raises(JevError, match="after deadline"):
        client(late, clock).decide({}, questions(), deadline=0.1)
    assert timeouts == [0.1]


def test_http_transport_contract_and_error_body_suppression(monkeypatch):
    def ok(request, timeout):
        assert request.full_url == jev.ENDPOINT and timeout == 1
        assert request.get_header("Authorization") == "Bearer " + KEY
        return nullcontext(SimpleNamespace(status=200, read=lambda: b"{}"))

    monkeypatch.setattr(jev, "urlopen", ok)
    assert jev.http_transport(b"{}", KEY, 1) == HttpReply(200, b"{}")

    def error(request, timeout):
        raise HTTPError(jev.ENDPOINT, 429, KEY, {"Retry-After": "bad"}, io.BytesIO(KEY.encode()))

    monkeypatch.setattr(jev, "urlopen", error)
    assert jev.http_transport(b"{}", KEY, 1) == HttpReply(429, b"", 0)

    def disconnected(request, timeout):
        raise URLError(KEY)

    monkeypatch.setattr(jev, "urlopen", disconnected)
    with pytest.raises(JevError) as raised:
        jev.http_transport(b"{}", KEY, 1)
    assert KEY not in str(raised.value)


def test_smoke_missing_key_and_mock_contract_path(monkeypatch, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "e16_smoke", Path(__file__).resolve().parents[2] / "scripts/e16_jev_smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "smoke.json"
    monkeypatch.setattr("sys.argv", ["smoke", "--output", str(target)])
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert module.main() == 2
    assert json.loads(target.read_text())["status"] == "blocked"

    class MockClient:
        def __init__(self, *a):
            pass

        def decide(self, state, qs):
            return parse_response(payload(qs), qs, MODEL)

    monkeypatch.setattr(module, "JevClient", MockClient)
    assert module.main() == 0
    report = json.loads(target.read_text())
    assert not report["formal"] and report["continuation"]["answers"]["continue"]["noul"] == 0.4
    assert KEY not in capsys.readouterr().out
