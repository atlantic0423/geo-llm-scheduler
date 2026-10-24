"""Whole-block selection, immutable original inputs and prepaid API bounds."""

import gzip
import importlib
import json
import sys
import types
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
campaign = importlib.import_module("run_jev_comparison")


def historical_fixture(root):
    root.mkdir()
    files, blocks = {}, []
    for n in (20, 50, 100, 200):
        for parent in range(4):
            for variant in (0, 1):
                key = f"n{n}_p{parent}_v{variant}_s16101"
                relative = f"inputs/n{n}_p{parent}_v{variant}.json"
                target = root / relative
                target.parent.mkdir(exist_ok=True)
                target.write_text(json.dumps(dict(n=n, parent=parent, variant=variant)))
                spec = dict(
                    key=key,
                    n=n,
                    base_seed=n * 100 + parent,
                    variant=variant,
                    seed=16101,
                    seconds=n * 12,
                    regions=2,
                    machines=12,
                    pressure="kv",
                    input=relative,
                    input_sha256=campaign.sha(target),
                )
                path = root / "specs" / (key + ".json")
                campaign.atomic_json(path, spec)
                files[relative] = campaign.sha(target)
                files[path.relative_to(root).as_posix()] = campaign.sha(path)
                blocks.append(key)
    campaign.atomic_json(
        root / "manifest.json",
        dict(
            commit="b148c4addd207fe30a6b2a43d953a04afc35d3a1", runs=1056, blocks=blocks, files=files
        ),
    )
    campaign.atomic_json(
        root / "manifest.sha256.json", dict(sha256=campaign.sha(root / "manifest.json"))
    )


def test_williams_orders_balance_position_and_directed_carryover():
    orders = [campaign.balanced_order(i) for i in range(4)]
    assert all(set(row) == set(campaign.ARMS) for row in orders)
    for position in range(4):
        assert {r[position] for r in orders} == set(campaign.ARMS)
    pairs = Counter((r[i], r[i + 1]) for r in orders for i in range(3))
    assert len(pairs) == 12 and set(pairs.values()) == {1}


def test_budget_bound_counts_all_preallocations_and_rejects_invalid_inputs():
    assert campaign.budget_bound(128, 4_000_000) == pytest.approx(21.604)
    assert campaign.budget_bound(16, 4_000_000) == pytest.approx(2.788)
    for count, tokens, reserve in (
        (0, 4_000_000, 0.1),
        (1, 262143, 0.1),
        (True, 4_000_000, 0.1),
        (1, 4_000_000, float("nan")),
    ):
        with pytest.raises(ValueError):
            campaign.budget_bound(count, tokens, reserve)


def test_prepare_rejects_full_panel_before_writing_when_budget_is_insufficient(tmp_path):
    old = tmp_path / "original"
    historical_fixture(old)
    target = tmp_path / "new"
    with pytest.raises(ValueError, match="exceeds cap"):
        campaign.prepare(target, old)
    assert not target.exists()


def test_preparation_keeps_both_variants_all_scales_and_original_bytes(tmp_path):
    old = tmp_path / "original"
    historical_fixture(old)
    root = tmp_path / "new"
    manifest = campaign.prepare(root, old, parents_per_scale=1, seeds=(17401,))
    assert manifest["runs"] == 32 and manifest["independent_parents"] == 4
    assert manifest["preallocated_upper_usd"] < manifest["spend_cap_usd"] == 4.5
    specs = [json.loads((root / "specs" / (k + ".json")).read_text()) for k in manifest["blocks"]]
    assert Counter(s["n"] for s in specs) == {20: 2, 50: 2, 100: 2, 200: 2}
    for spec in specs:
        assert spec["seed"] == 17401
        assert (root / spec["input"]).read_bytes() == (old / spec["input"]).read_bytes()
    assert campaign.validate_inputs(root) == manifest
    (root / specs[0]["input"]).write_text("tampered")
    with pytest.raises(AssertionError):
        campaign.validate_inputs(root)


def test_single_n200_e09_pilot_uses_three_arms_twenty_minutes_and_under_two_dollars(tmp_path):
    old = tmp_path / "original"
    historical_fixture(old)
    root = tmp_path / "pilot"
    key = "n200_p0_v0_s16101"
    m = campaign.prepare_pilot(root, old, original_key=key)
    assert m["arms"] == ["BILL_MOEAD", "MAIN_Q", "JEV_E09"] and m["runs"] == 3
    assert m["epsilon_g"] == 0.9 and m["preallocated_upper_usd"] == pytest.approx(1.78)
    assert m["spend_cap_usd"] == 2 and m["max_input_tokens"] == 40_000_000
    spec = json.loads((root / "specs" / (m["blocks"][0] + ".json")).read_text())
    assert spec["seconds"] == 1200 and spec["n"] == 200 and spec["seed"] == 17401
    assert (root / spec["input"]).read_bytes() == (old / spec["input"]).read_bytes()
    assert campaign.validate_inputs(root) == m
    m["arms"].append("JEV_E05")
    campaign.atomic_json(root / "manifest.json", m)
    campaign.atomic_json(
        root / "manifest.sha256.json", dict(sha256=campaign.sha(root / "manifest.json"))
    )
    with pytest.raises(AssertionError):
        campaign.validate_inputs(root)


def test_pilot_refuses_overspend_and_non200_inputs_before_creation(tmp_path):
    old = tmp_path / "original"
    historical_fixture(old)
    with pytest.raises(ValueError, match="spend"):
        campaign.prepare_pilot(tmp_path / "bad", old, max_tokens=50_000_000)
    assert not (tmp_path / "bad").exists()
    with pytest.raises(AssertionError):
        campaign.prepare_pilot(tmp_path / "bad", old, original_key="n20_p0_v0_s16101")
    assert not (tmp_path / "bad").exists()


def test_controls_match_existing_config_and_jev_keeps_strict_boundaries():
    for arm in campaign.ARMS:
        cfg = campaign.arm_config(arm, 17401, 300)
        assert cfg.population == 100 and cfg.seconds == 300
        if arm.startswith("JEV_"):
            assert cfg.trigger_mode == "strict" and cfg.trigger_delta == 0
            assert cfg.fixed_budget == 6 and cfg.rl_steps == 5
    with pytest.raises(ValueError):
        campaign.arm_config("LLM_FLASH", 1, 300)


def test_shared_queue_uses_only_four_slots_and_keeps_long_context(monkeypatch, tmp_path):
    import geo_llm_scheduler.experiments.jev_decisions as decisions

    locks = []

    def flock(handle, flags):
        locks.append(handle.fileno())
        if len(locks) < 4:
            raise BlockingIOError

    monkeypatch.setitem(
        sys.modules, "fcntl", types.SimpleNamespace(flock=flock, LOCK_EX=1, LOCK_NB=2)
    )
    monkeypatch.setenv("JEV_REQUEST_LOCK_DIR", str(tmp_path / "locks"))
    sent = []

    def requester(payload, timeout, *, long_context=False):
        sent.append((payload, timeout, long_context))
        return {"model": "jev-1.14.0"}

    monkeypatch.setattr(decisions, "isolated_request", requester)
    assert campaign.queued_request({"questions": {}}, 5) == {"model": "jev-1.14.0"}
    assert len(locks) == 4 and sent[0][0] == {"questions": {}}
    assert 0 < sent[0][1] <= 5 and sent[0][2] is True


def test_expired_queue_never_sends_http_request(monkeypatch, tmp_path):
    import geo_llm_scheduler.experiments.jev_decisions as decisions

    def busy(*args):
        raise BlockingIOError

    monkeypatch.setitem(
        sys.modules, "fcntl", types.SimpleNamespace(flock=busy, LOCK_EX=1, LOCK_NB=2)
    )
    monkeypatch.setenv("JEV_REQUEST_LOCK_DIR", str(tmp_path / "locks"))
    sent = []
    monkeypatch.setattr(decisions, "isolated_request", lambda *a, **k: sent.append(a))
    with pytest.raises(TimeoutError, match="queue"):
        campaign.queued_request({}, 0.001)
    assert not sent


@pytest.fixture
def export_fixture(problem, tmp_path, monkeypatch):
    from geo_llm_scheduler.archive.pareto import Archive
    from geo_llm_scheduler.domain.models import Genotype
    from geo_llm_scheduler.engine.evaluation import EvaluationGateway
    from geo_llm_scheduler.rl.controller import Controller

    root = tmp_path / "synthetic-export"
    root.mkdir()
    path = root / "input.json"
    path.write_text("{}")
    key = "synthetic_s17401"
    spec = dict(
        key=key,
        host=socket_name(),
        seed=17401,
        seconds=300,
        input="input.json",
        input_sha256=campaign.sha(path),
    )
    spec_path = root / "specs" / (key + ".json")
    campaign.atomic_json(spec_path, spec)
    manifest = dict(
        source=campaign.source_hash(),
        host=socket_name(),
        commit="synthetic",
        blocks=[key],
        arms=list(campaign.ARMS),
        runs=4,
        max_input_tokens=4_000_000,
        preallocated_upper_usd=campaign.budget_bound(2, 4_000_000),
        spend_cap_usd=4.5,
        files={"input.json": campaign.sha(path), "specs/" + key + ".json": campaign.sha(spec_path)},
    )
    campaign.atomic_json(root / "manifest.json", manifest)
    campaign.atomic_json(
        root / "manifest.sha256.json", dict(sha256=campaign.sha(root / "manifest.json"))
    )
    initial = dict(
        initial_population_sha256="synthetic-pop", initial_archive_sha256="synthetic-archive"
    )
    campaign.atomic_json(
        root / "preflight" / (key + ".json"),
        dict(
            source=manifest["source"],
            input_sha256=spec["input_sha256"],
            policies=dict(bill4=dict(evidence=initial)),
        ),
    )
    monkeypatch.setattr(campaign, "load_instance", lambda path: problem)

    def fake_run(
        p, arm, cfg, evidence, *, observer, settings=None, requester=None, request_sink=None
    ):
        gateway = EvaluationGateway(p, Archive())
        candidate = gateway.evaluate(Genotype((0, 0, 1, 1), (0, 1, 0, 1)))
        population = [candidate] * 100
        evidence.update(policy="bill4", **initial)
        observer("initial_ready", dict(population=population, gateway=gateway))
        observer(
            "rl_step",
            dict(action=1, action_source="jev_current_step" if arm.startswith("JEV_") else None),
        )
        if arm.startswith("JEV_"):
            request_sink(
                dict(
                    accepted=True, reserved_input_tokens=100, request={"questions": {"action": {}}}
                )
            )
            evidence["jev"] = dict(
                settings=asdict(settings),
                prompt_version=campaign.PROMPT_VERSION,
                provider_transport="injected",
                requests=1,
                valid_requests=1,
                valid_action_requests=1,
                jev_treatment_valid=True,
                billed_or_reserved_input_tokens=100,
                requests_evidence=[],
            )
        return types.SimpleNamespace(
            population=population,
            archive=gateway.archive,
            gateway=gateway,
            elapsed=300.1,
            controller=Controller(cfg),
            termination_reason="time_budget",
            offspring_count=1,
        )

    monkeypatch.setattr(campaign, "run_control", fake_run)
    monkeypatch.setattr(campaign, "run_jev", fake_run)
    return root, key, manifest


def socket_name():
    return campaign.socket.gethostname()


@pytest.mark.parametrize("arm", campaign.ARMS)
def test_four_arm_export_writes_initial_calls_and_fresh_exact_final_artifacts(
    export_fixture, monkeypatch, arm
):
    root, key, manifest = export_fixture
    counts = []
    exact = campaign.evaluate
    monkeypatch.setattr(campaign, "evaluate", lambda *a: counts.append(1) or exact(*a))
    campaign._worker(root, key, arm, 0)
    out = root / "runs" / key / arm
    assert len(counts) == 101 and campaign.complete(root, key, arm, manifest["source"])
    with gzip.open(out / "initial.json.gz", "rt", encoding="utf-8") as handle:
        assert len(json.load(handle)) == 100
    (out / "prompts_en.json").write_text("tampered")
    with pytest.raises(AssertionError):
        campaign.complete(root, key, arm, manifest["source"])


def test_worker_rejects_all_fallback_treatment_and_preserves_failure_evidence(
    export_fixture, monkeypatch
):
    root, key, _ = export_fixture
    original = campaign.run_jev

    def invalid(*args, **kwargs):
        result = original(*args, **kwargs)
        args[3]["jev"].update(jev_treatment_valid=False, valid_action_requests=0)
        return result

    monkeypatch.setattr(campaign, "run_jev", invalid)
    with pytest.raises(AssertionError):
        campaign._worker(root, key, "JEV_E05", 0)
    out = root / "runs" / key / "JEV_E05"
    assert (out / "failure.json").exists() and (out / "policy_calls.jsonl.gz").exists()
