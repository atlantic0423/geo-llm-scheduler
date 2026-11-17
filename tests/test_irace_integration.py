"""Configuration encoding, frozen metric semantics and recoverable target artifacts."""

import json
from pathlib import Path

import pytest
from test_component_search_v4 import panel
from test_lsv2_fast_rightshift import CAL

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.component_search_v4 import NUMERIC, SLOTS, Components
from geo_llm_scheduler.experiments.irace_space import (
    decode_configuration,
    default_table,
    parameter_text,
)
from geo_llm_scheduler.experiments.irace_target import (
    evaluate_request,
    metric_spec,
    score_front,
    sha256,
    validate_case,
)
from geo_llm_scheduler.io.loaders import save_instance


def make_case(tmp_path):
    save_instance(panel(), tmp_path / "instance.json")
    atomic_json(tmp_path / "calibration.json", CAL)
    atomic_json(
        tmp_path / "metric.json",
        dict(objectives=["flow", "bill"], lower=[0, 0], upper=[1000, 100], reference=[1.1, 1.1]),
    )
    case = dict(purpose="smoke", seconds=30, generations=1)
    for name in ("instance", "calibration", "metric"):
        path = tmp_path / f"{name}.json"
        case[name] = dict(path=path.name, sha256=sha256(path))
    atomic_json(tmp_path / "case.json", case)
    return tmp_path / "case.json"


def test_space_matches_all_declared_options():
    assert len(parameter_text().splitlines()) == 16
    assert len(parameter_text("numeric").splitlines()) == 8
    assert decode_configuration({}) == Components()
    for name, options in SLOTS.items():
        for option in options:
            assert getattr(decode_configuration({name: option}), name) == option
    for name, options in NUMERIC.items():
        for i, option in enumerate(options):
            encoded = f"w{i}" if name == "mutation_weights" else str(option)
            assert getattr(decode_configuration({name: encoded}), name) == option
    for mode in ("numeric", "joint"):
        header, row = default_table(mode).splitlines()
        assert decode_configuration(dict(zip(header.split(), row.split()))) == Components()
    for bad in ({"unknown": "1"}, {"ls_width": "2"}, {"mutation_weights": "w9"}):
        with pytest.raises(ValueError):
            decode_configuration(bad)
    with pytest.raises(ValueError):
        parameter_text("bad")


def test_fixed_hv_with_outside_reference_negative_coordinates_and_duplicates():
    metric = dict(objectives=["flow", "bill"], lower=[10, 100], upper=[20, 200], reference=[1, 1])
    # Normalized points (0,.5),(.5,0) have union .75; outside points contribute zero.
    assert score_front([(10, 150), (15, 100), (10, 150), (21, 90)], metric) == -0.75
    assert score_front([(0, 100)], metric) == -2  # No lower clipping.
    assert score_front([], metric) == 0
    with pytest.raises(ValueError):
        score_front([(float("nan"), 1)], metric)
    with pytest.raises(ValueError):
        score_front([], {**metric, "upper": [10, 200]})
    with pytest.raises(ValueError):
        score_front([], {**metric, "objectives": ["bill", "flow"]})
    assert metric_spec([(1, 3), (3, 1)], {})["lower"] == [1, 1]
    for bad in ([], [(1, 1)], [(float("inf"), 1)]):
        with pytest.raises(ValueError):
            metric_spec(bad, {})


def test_target_real_run_replay_and_tamper_detection(tmp_path):
    case = make_case(tmp_path)
    request = dict(configuration={}, seed=7)
    root = tmp_path / "run"
    result = evaluate_request(request, case, root)
    assert result["cost"] < 0 and (root / "result.json").exists()
    assert evaluate_request(request, case, root) == result
    with pytest.raises(ValueError, match="identity"):
        evaluate_request({**request, "seed": 8}, case, root)
    (root / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        evaluate_request(request, case, root)
    (tmp_path / "incomplete").mkdir()
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate_request(request, case, tmp_path / "incomplete")


def test_frozen_case_and_failures_are_not_scores(tmp_path, monkeypatch):
    import geo_llm_scheduler.experiments.irace_target as module

    case = make_case(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("intentional solver failure")

    monkeypatch.setattr(module, "run_components", fail)
    with pytest.raises(RuntimeError):
        evaluate_request(dict(configuration={}, seed=1), case, tmp_path / "failed")
    assert json.loads((tmp_path / "failed/status.json").read_text())["status"] == "failed"
    assert not (tmp_path / "failed/complete.json").exists()
    monkeypatch.undo()
    recovered = evaluate_request(
        dict(configuration={}, seed=1, retry_incomplete=True), case, tmp_path / "failed"
    )
    assert recovered["cost"] < 0
    assert len(list(tmp_path.glob("failed.incomplete_*"))) == 1
    (tmp_path / "metric.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        validate_case(case)


def test_prepare_smoke_and_reference_resume(tmp_path, monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    prepare_module = importlib.import_module("prepare_irace")
    instance = tmp_path / "input.json"
    # Keep the tiny problem but give metadata size 50 to select the frozen fold/budget.
    save_instance(panel(), instance)
    split = tmp_path / "split.json"
    atomic_json(
        split,
        dict(
            instances=[
                dict(
                    instance_id="test",
                    jobs=50,
                    family="F01",
                    replicate=1,
                    split="configuration",
                    path=str(instance),
                    sha256=sha256(instance),
                )
            ]
        ),
    )
    cal = tmp_path / "folds.json"
    atomic_json(cal, {"n50_f0": CAL})
    out = tmp_path / "prepared"
    prepare_module.prepare(out, split, cal, smoke=True)
    before = sha256(out / "cases/test/reference.json")
    prepare_module.prepare(out, split, cal, smoke=True)
    assert sha256(out / "cases/test/reference.json") == before
    assert json.loads((out / "campaign_joint.json").read_text())["max_experiments"] == 48
    with pytest.raises(ValueError, match="identity"):
        prepare_module.prepare(out, split, cal, smoke=False)
    from geo_llm_scheduler.experiments.irace_campaign import seal_campaign, verify_seal

    seal = seal_campaign(out, Path(__file__).resolve().parents[1])
    assert verify_seal(seal)["purpose"] == "smoke"
    assert seal_campaign(out, Path(__file__).resolve().parents[1]) == seal
    with pytest.raises(ValueError, match="not in"):
        verify_seal(seal, tmp_path / "unknown.json")
    campaign = json.loads((out / "campaign_joint_rep3.json").read_text())
    assert campaign["seed"] == 17103
    request = dict(configuration={}, seed=3, seal=str(seal))
    assert evaluate_request(request, out / "cases/test/case.json", out / "trial")["cost"] < 0
    (out / "training.txt").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_seal(seal)


def test_invalid_formal_budget_and_missing_seal(tmp_path):
    case_path = make_case(tmp_path)
    case = json.loads(case_path.read_text())
    case["purpose"] = "formal"
    atomic_json(case_path, case)
    # The tiny fixture is not a supported formal size and must never run as formal.
    with pytest.raises((KeyError, ValueError)):
        evaluate_request(dict(configuration={}, seed=1), case_path, tmp_path / "run")
    case["purpose"] = "unknown"
    atomic_json(case_path, case)
    with pytest.raises(ValueError, match="Unknown"):
        validate_case(case_path)


def test_elite_exports_and_paired_validation_selection(tmp_path, monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("evaluate_irace")
    for arm in ("numeric", "joint"):
        for rep in (1, 2, 3):
            output = tmp_path / f"{arm}_rep{rep}"
            output.mkdir()
            (output / "elites.csv").write_text(".ID.,ls_width,.RANK.\n1,1,1\n2,3,2\n")
    configs = module.candidates(tmp_path)
    assert len(configs) == 3  # One top candidate per race; duplicates never backfill.
    scores = {
        c["id"]: [-2, -2] if c["configuration"].get("ls_width") == "1" else [-1, -1]
        for c in configs
    }
    chosen = module.rank_finalists(configs, scores)
    assert len(chosen) == 3 and chosen[0]["id"] == "default"
    assert all(c["configuration"]["ls_width"] == "1" for c in chosen[1:])
    tied = module.rank_finalists(configs, {c["id"]: [-1, -1] for c in configs})
    assert all(c["fallback"] for c in tied[1:])
    scores["default"] = []
    with pytest.raises(ValueError, match="paired"):
        module.rank_finalists(configs, scores)
