"""Original-input subset, two-clock export and English-only transport regressions."""

import importlib
import json
import sys
from pathlib import Path

import pytest
from test_jev_comparison import historical_fixture
from test_llm_protocol import MockOpener, envelope, install_opener

from geo_llm_scheduler.experiments import round_decisions
from geo_llm_scheduler.experiments.llm_protocol import DeepSeekClient
from geo_llm_scheduler.experiments.round_prompts import BACKGROUND

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
campaign = importlib.import_module("run_round_llm")


def test_subset_copies_two_originals_for_four_paired_seed_blocks(tmp_path, monkeypatch):
    original = tmp_path / "old"
    historical_fixture(original)
    monkeypatch.setattr(campaign, "ORIGINALS", ("n100_p0_v0_s16101", "n200_p0_v0_s16101"))
    root = tmp_path / "new"
    m = campaign.prepare(root, original)
    assert m["runs"] == 12 and len(m["blocks"]) == 4 and len(m["arms"]) == 3
    assert m["seeds"] == [17401, 27401] and m["language"] == "English only"
    assert {
        json.loads((root / "specs" / (key + ".json")).read_text())["n"] for key in m["blocks"]
    } == {100, 200}
    for key in m["blocks"]:
        spec = json.loads((root / "specs" / (key + ".json")).read_text())
        assert (root / spec["input"]).read_bytes() == (original / spec["input"]).read_bytes()
    with pytest.raises(FileExistsError):
        campaign.prepare(root, original)
    assert not campaign.complete(root, m["blocks"][0], "MAIN_Q")


def test_tampered_original_rejected_before_freezing_matrix(tmp_path, monkeypatch):
    original = tmp_path / "old"
    historical_fixture(original)
    monkeypatch.setattr(campaign, "ORIGINALS", ("n100_p0_v0_s16101",))
    (original / "inputs/n100_p0_v0.json").write_text("tampered")
    with pytest.raises(AssertionError):
        campaign.prepare(tmp_path / "new", original)


def test_custom_system_prompt_is_actually_sent_to_transport(monkeypatch):
    opener = MockOpener(envelope())
    install_opener(monkeypatch, opener)
    result = DeepSeekClient("synthetic-secret", system_prompt=BACKGROUND, max_tokens=8192).request(
        {"task": "STEP_ACTION"}, 1
    )
    body = json.loads(opener.calls[0][0].data)
    assert body["messages"][0] == {"role": "system", "content": BACKGROUND}
    assert body["thinking"] == {"type": "disabled"} and body["max_tokens"] == 8192
    assert body["response_format"] == {"type": "json_object"} and result["finish_reason"] == "stop"
    assert "synthetic-secret" not in json.dumps(body)


def test_transport_kills_and_reaps_once_at_deadline(monkeypatch):
    calls = []

    class Child:
        returncode = 0

        def communicate(self, *args, **kwargs):
            calls.append("communicate")
            if len(calls) == 1:
                raise round_decisions.subprocess.TimeoutExpired("child", 1)
            return "", ""

        def kill(self):
            calls.append("kill")

    monkeypatch.setattr(round_decisions.subprocess, "Popen", lambda *a, **k: Child())
    with pytest.raises(TimeoutError):
        round_decisions.isolated_request({}, 1)
    assert calls == ["communicate", "kill", "communicate"]
    with pytest.raises(ValueError):
        round_decisions.isolated_request({}, 0)


@pytest.mark.parametrize(
    "content", ['{"request_id":"q0","request_id":"bad"}', '{"nested":NaN}', "[]", "null"]
)
def test_malformed_json_is_not_accepted_as_a_valid_decision(content):
    def response(packet, timeout):
        return dict(content=content, finish_reason="stop", usage={})

    session = round_decisions.RoundSession(100, response)
    assert session.request("STEP_ACTION", {}, dict(A1=True)) is None
    assert not session.records[0]["accepted"]


def test_no_secret_or_non_english_input_reaches_requester(monkeypatch):
    calls = []
    session = round_decisions.RoundSession(100, lambda *args: calls.append(args))
    with pytest.raises(ValueError):
        session.request("STEP_ACTION", dict(description="中文"), dict(A1=True))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    with pytest.raises(ValueError):
        session.request("STEP_ACTION", dict(description="test-secret"), dict(A1=True))
    assert calls == []
