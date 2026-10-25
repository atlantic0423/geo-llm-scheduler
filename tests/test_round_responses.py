"""Gateway identity, credential boundaries, failure stops and unchanged LS semantics."""

import importlib
import json
import sys
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest
from test_jev_comparison import historical_fixture
from test_llm_protocol import MockOpener
from test_round_search import Flash

from geo_llm_scheduler.experiments import round_responses as module
from geo_llm_scheduler.experiments.round_decisions import InferenceUnavailable, RoundSession
from geo_llm_scheduler.experiments.round_prompts import BACKGROUND
from geo_llm_scheduler.experiments.round_search import ARM, run_policy, study_config
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.utils.rng import RNGManager


def envelope(**changes):
    value = dict(
        id="r1",
        model="gpt-6.1-sol",
        status="completed",
        output=[
            dict(type="reasoning", summary=[]),
            dict(
                type="message",
                role="assistant",
                content=[dict(type="output_text", text='{"ok":true}')],
            ),
        ],
        usage=dict(
            input_tokens=70000, output_tokens=23, input_tokens_details=dict(cached_tokens=1280)
        ),
    )
    return json.dumps(dict(value, **changes)).encode()


def install(monkeypatch, raw=None, error=None):
    opener = MockOpener(raw if raw is not None else envelope(), error)
    handlers = []

    def build(*args):
        handlers.extend(args)
        return opener

    monkeypatch.setattr(module, "build_opener", build)
    monkeypatch.setenv("ROUND_RESPONSES_API_KEY", "synthetic-secret")
    return opener, handlers


def test_responses_wire_preserves_full_english_background_in_input_no_storage_or_deepseek_fields(
    monkeypatch,
):
    opener, handlers = install(monkeypatch)
    profile = module.ResponsesProfile()
    result = module.request(profile, dict(task="STEP_ACTION"), 10)
    http, timeout = opener.calls[0]
    body = json.loads(http.data)
    assert http.full_url == "https://v1.cdks.work/v1/responses" and timeout == 10
    assert body["model"] == "gpt-6.1-sol" and body["store"] is False
    assert body["reasoning"] == dict(effort="high") and body["service_tier"] == "default"
    assert body["input"][0] == dict(role="developer", content=BACKGROUND)
    assert body["instructions"] == BACKGROUND and body["max_output_tokens"] == 8192
    assert "thinking" not in body and "synthetic-secret" not in http.data.decode()
    assert http.get_header("X-openai-actor-authorization") == "local-image-extension"
    assert handlers[0].redirect_request(http, None, 302, "", {}, "https://evil") is None
    assert result["content"] == '{"ok":true}' and result["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 70000 and result["usage"]["completion_tokens"] == 23
    assert result["usage"]["input_tokens_details"]["cached_tokens"] == 1280


@pytest.mark.parametrize(
    "changes",
    [
        dict(model="gpt-5.5"),
        dict(model=None),
        dict(output=[dict(type="message", role="assistant", content=[dict(type="refusal")])]),
        dict(usage=dict(output_tokens=8193)),
        dict(id="synthetic-secret"),
    ],
)
def test_model_mismatch_refusal_over_limit_or_secret_aborts(monkeypatch, changes):
    install(monkeypatch, envelope(**changes))
    result = module.request(module.ResponsesProfile(), {}, 1)
    assert result["fatal_error"]
    assert "synthetic-secret" not in json.dumps(result)


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 429, 500])
def test_http_classifications_do_not_echo_provider_errors_or_retry(monkeypatch, status):
    error = HTTPError("https://private.invalid", status, "synthetic-secret", {}, BytesIO(b"secret"))
    opener, _ = install(monkeypatch, error=error)
    result = module.request(module.ResponsesProfile(), {}, 1)
    assert result["http_status"] == status and len(opener.calls) == 1
    assert bool(result["fatal_error"]) == (status in (400, 401, 402, 403, 404))
    assert "secret" not in json.dumps(result) and "private" not in json.dumps(result)


def test_network_malformed_and_incomplete_results_are_explicit(monkeypatch):
    install(monkeypatch, error=URLError("private"))
    assert module.request(module.ResponsesProfile(), {}, 1)["transport_error"]
    install(monkeypatch, b'{"a":1,"a":2}')
    assert module.request(module.ResponsesProfile(), {}, 1)["transport_error"]
    install(monkeypatch, envelope(status="incomplete"))
    assert module.request(module.ResponsesProfile(), {}, 1)["finish_reason"] == "incomplete"
    with pytest.raises(ValueError):
        module.request(module.ResponsesProfile(), dict(a="中文"), 1)
    with pytest.raises(ValueError):
        module.request(module.ResponsesProfile(), dict(a="synthetic-secret"), 1)


@pytest.mark.parametrize(
    "changes",
    [
        dict(model="gpt-unknown"),
        dict(base_url="https://evil"),
        dict(reasoning_effort="none"),
        dict(request_timeout_s=0),
        dict(max_output_tokens=True),
    ],
)
def test_profile_rejects_implicit_model_or_endpoint_changes(changes):
    with pytest.raises(ValueError):
        module.ResponsesProfile(**changes)


def test_child_timeout_kills_reaps_and_never_retries(monkeypatch):
    calls = []

    class Child:
        def communicate(self, *args, **kwargs):
            calls.append("communicate")
            if len(calls) == 1:
                raise module.subprocess.TimeoutExpired("child", 1)
            return "", ""

        def kill(self):
            calls.append("kill")

    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: Child())
    assert (
        module.isolated_request(module.ResponsesProfile(), {}, 1)["transport_error"]
        == "child_deadline"
    )
    assert calls == ["communicate", "kill", "communicate"]
    with pytest.raises(ValueError):
        module.isolated_request(module.ResponsesProfile(), {}, 0)


def test_fatal_provider_records_excluded_wait_then_stops_no_further_request(monkeypatch):
    decisions = importlib.import_module("geo_llm_scheduler.experiments.round_decisions")
    now = [0.0]
    monkeypatch.setattr(decisions.time, "perf_counter", lambda: now[0])
    calls = []

    def requester(packet, timeout):
        calls.append(packet)
        now[0] += 5
        return dict(fatal_error="http_402", http_status=402)

    session = RoundSession(100, requester, profile=module.ResponsesProfile())
    with pytest.raises(InferenceUnavailable, match="http_402"):
        session.request("STEP_ACTION", {}, dict(A1=True))
    assert session.excluded_api_wait_s == 5 and session.elapsed() == 0
    assert session.records[0]["fatal_error"] == "http_402" and not session.records[0]["accepted"]
    with pytest.raises(InferenceUnavailable):
        session.request("STEP_ACTION", {}, dict(A1=True))
    assert len(calls) == 1


def test_five_transport_failures_or_spend_guard_abort_only_opted_in_runs():
    def requester(*args):
        return dict(transport_error="network")

    session = RoundSession(100, requester, profile=module.ResponsesProfile())
    for _ in range(4):
        assert session.request("STEP_ACTION", {}, dict(A1=True)) is None
    with pytest.raises(InferenceUnavailable, match="consecutive"):
        session.request("STEP_ACTION", {}, dict(A1=True))
    session = RoundSession(100, Flash(), profile=module.ResponsesProfile(), max_requests=1)
    assert session.request("STEP_ACTION", {}, dict(A1=True))
    with pytest.raises(InferenceUnavailable, match="guard"):
        session.request("STEP_ACTION", {}, dict(A1=True))


def test_same_fixed_work_search_and_rngs_with_profile_only_metadata_differs(problem):
    cfg = replace(study_config(ARM, 1, 100), population=10, neighborhood=3, generations=1)
    initial = initial_genotypes(problem, cfg, RNGManager(1).stream("initial"))
    a, b = {}, {}
    old = run_policy(problem, ARM, cfg, a, initial=initial, requester=Flash())
    new = run_policy(
        problem, ARM, cfg, b, initial=initial, requester=Flash(), profile=module.ResponsesProfile()
    )
    assert old.population == new.population and old.archive.members == new.archive.members
    assert old.gateway.counts == new.gateway.counts
    assert b["round_llm"]["model"] == "gpt-6.1-sol"
    assert b["round_llm"]["prompt_version"] == module.ResponsesProfile().version
    assert a["round_llm"]["model"] == "deepseek-flash"


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
restart = importlib.import_module("run_round_responses")
original = importlib.import_module("run_round_llm")


def test_restart_index_reuses_only_eight_complete_references_and_rejects_tampering(
    tmp_path, monkeypatch
):
    old = tmp_path / "historical"
    historical_fixture(old)
    monkeypatch.setattr(original, "ORIGINALS", ("n100_p0_v0_s16101", "n200_p0_v0_s16101"))
    previous = tmp_path / "previous"
    m = original.prepare(previous, old)
    m.update(commit=restart.REFERENCE_HEAD, source=restart.REFERENCE_SOURCE)
    original.atomic_json(previous / "manifest.json", m)
    original.atomic_json(
        previous / "manifest.sha256.json", dict(sha256=original.sha(previous / "manifest.json"))
    )
    for key in m["blocks"]:
        spec = json.loads((previous / "specs" / (key + ".json")).read_text())
        evidence = dict(initial_population_sha256="p", initial_archive_sha256="a")
        original.atomic_json(
            previous / "preflight" / (key + ".json"),
            dict(policies=dict(bill4=dict(evidence=evidence))),
        )
        for arm in ("BILL_MOEAD", "MAIN_Q"):
            path = previous / "runs" / key / arm
            path.mkdir(parents=True)
            for name in restart.REQUIRED:
                (path / name).write_text("fixture")
            original.atomic_json(
                path / "config.json", asdict(study_config(arm, spec["seed"], spec["seconds"]))
            )
            original.atomic_json(
                path / "summary.json",
                dict(
                    termination="time_budget",
                    offspring=1,
                    search_elapsed_s=spec["seconds"],
                    api_blocked_s=0,
                    final_exact_verified=101,
                    population=100,
                    archive=1,
                    initialization=evidence,
                ),
            )
            original.atomic_json(
                path / "complete.json",
                dict(
                    source=m["source"],
                    host=m["host"],
                    arm=arm,
                    key=key,
                    spec_sha256=original.sha(previous / "specs" / (key + ".json")),
                    input_sha256=spec["input_sha256"],
                    files={
                        n: dict(size=(path / n).stat().st_size, sha256=original.sha(path / n))
                        for n in restart.REQUIRED
                    },
                ),
            )
    root = tmp_path / "restart"
    new = restart.prepare(root, previous, module.ResponsesProfile())
    assert new["runs"] == 4 and new["comparison_runs"] == 12 and new["arms"] == [ARM]
    assert len(new["reused_references"]) == 8
    assert not (root / "runs").exists()
    monkeypatch.setattr(restart, "complete", lambda *a: True)
    assert restart.validate(root)
    index_path = root / "formal_execution_index.json"
    index = json.loads(index_path.read_text())
    index["entries"][-1]["path"] = str(previous / "aborted")
    original.atomic_json(index_path, index)
    with pytest.raises(AssertionError):
        restart.validate(root)
    with pytest.raises(FileExistsError):
        restart.prepare(root, previous, module.ResponsesProfile())
    with pytest.raises(ValueError):
        restart.reference_binding(previous, m["blocks"][0], ARM)
    target = Path(new["reused_references"][0]["path"]) / "archive.json.gz"
    target.write_text("changed")
    with pytest.raises(AssertionError):
        restart.reference_binding(previous, m["blocks"][0], "BILL_MOEAD")
