"""Default adoption preserves the frozen mechanism and explicit historical entry points."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_lsv2_fast_rightshift import CAL, synthetic

from geo_llm_scheduler import default_algorithm as default
from geo_llm_scheduler.experiments import s4_moead
from geo_llm_scheduler.experiments.runner import run_config
from geo_llm_scheduler.io.loaders import save_instance


@pytest.mark.parametrize("jobs,budget", [(50, 600), (100, 1200), (200, 2400)])
def test_default_inclusive_budget_and_frozen_arm(monkeypatch, jobs, budget):
    monkeypatch.setattr(default, "load_instance", lambda _: SimpleNamespace(jobs=range(jobs)))
    calls = []
    monkeypatch.setattr(default.trans_runner, "run_case", lambda *a, **kw: calls.append((a, kw)))
    default.run_default("instance.json", "out", calibration=CAL, seed=9)
    args, kwargs = calls[0]
    assert args == ("GS4_GUIDED_RAND_SWITCH_D4", Path("instance.json"), Path("out"))
    assert kwargs == dict(calibration=CAL, seed=9, seconds=budget)


def test_missing_calibration_and_unknown_size_fail_before_output(tmp_path):
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="calibration"):
        default.run_default("missing", output, calibration={})
    instance = tmp_path / "tiny.json"
    save_instance(synthetic()[0], instance)
    with pytest.raises(ValueError, match="default budget"):
        default.run_default(instance, output, calibration=CAL)
    assert not output.exists()


@pytest.mark.parametrize(
    "kwargs",
    [{"seconds": 0}, {"seconds": float("nan")}, {"seconds": True}, {"seconds": 1, "seed": True}],
)
def test_invalid_run_parameters_fail_before_search(kwargs):
    with pytest.raises(ValueError):
        default.run_default("missing", "out", calibration=CAL, **kwargs)


def test_profile_dispatch_and_historical_override_rejection(tmp_path, monkeypatch):
    calibration = tmp_path / "cal.json"
    calibration.write_text(json.dumps({"fold": CAL}))
    calls = []
    monkeypatch.setattr(default, "run_default", lambda *a, **kw: calls.append((a, kw)))
    data = yaml.safe_load(Path("configs/default.yaml").read_text())
    assert data["profile"] == default.DEFAULT_PROFILE
    run_config(
        "configs/default.yaml",
        seed=4,
        output="out",
        instance="input",
        calibration=calibration,
        calibration_key="fold",
        seconds=2,
    )
    assert calls == [(("input", "out"), dict(calibration=CAL, seed=4, seconds=2))]
    with pytest.raises(ValueError, match="overrides require"):
        run_config("configs/smoke.yaml", seconds=1)
    with pytest.raises(ValueError, match="unknown fields"):
        default.run_default_config({"profile": default.DEFAULT_PROFILE, "r1": True})
    with pytest.raises(ValueError, match="requires instance"):
        default.run_default_config(data)


def test_cli_without_config_executes_gs4_and_exact_audit(tmp_path, monkeypatch, capsys):
    instance, calibration, output = (
        tmp_path / "tiny.json",
        tmp_path / "cal.json",
        tmp_path / "result",
    )
    save_instance(synthetic(4)[0], instance)
    calibration.write_text(json.dumps(CAL))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--instance",
            str(instance),
            "--calibration",
            str(calibration),
            "--output",
            str(output),
            "--seconds",
            "1",
        ],
    )
    runpy.run_module("geo_llm_scheduler.cli.run", run_name="__main__")
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "complete" and summary["independent_exact_audit"] == "passed"
    assert summary["algorithm"] == default.DEFAULT_ALGORITHM and summary["budget_seconds"] == 1
    assert summary["archive_size"] > 0
    config = s4_moead.arm_config(default.DEFAULT_ALGORITHM, 1, 1)
    expected = s4_moead.configuration_record(default.DEFAULT_ALGORITHM, config, CAL)
    actual = json.loads((output / "algorithm_config.json").read_text())
    assert actual == json.loads(json.dumps(expected))
    assert actual["guided"]["destroy_count"] == 4 and not actual["guided"]["direction_aware"]
    assert not (output / "r1_online.json").exists()
    assert not (output / "gap_allocation.json").exists()
    with pytest.raises(FileExistsError):
        default.run_default(instance, output, calibration=CAL, seconds=1)


def test_cli_missing_inputs_reports_actionable_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run"])
    with pytest.raises(SystemExit, match="2"):
        runpy.run_module("geo_llm_scheduler.cli.run", run_name="__main__")
    assert "requires instance" in capsys.readouterr().err
