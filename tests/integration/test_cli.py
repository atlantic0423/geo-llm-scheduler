"""Exercise real command entry points and independently re-evaluate saved schedules."""

import json
import runpy
import sys

import pytest


def test_cli_run_validate_inspect(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["validate", "--instance", "examples/smoke.json"])
    runpy.run_module("geo_llm_scheduler.cli.validate_instance", run_name="__main__")
    capsys.readouterr()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--config", "configs/smoke.yaml", "--seed", "1", "--output", str(tmp_path)],
    )
    runpy.run_module("geo_llm_scheduler.cli.run", run_name="__main__")
    capsys.readouterr()
    saved = json.loads((tmp_path / "archive.json").read_text(encoding="utf-8"))[0]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect",
            "--instance",
            "examples/smoke.json",
            "--archive",
            str(tmp_path / "archive.json"),
        ],
    )
    runpy.run_module("geo_llm_scheduler.cli.inspect_schedule", run_name="__main__")
    checked = json.loads(capsys.readouterr().out)
    assert checked["feasible"]
    assert checked["flow"] == pytest.approx(saved["evaluation"]["flow"])
    assert checked["tou"] == pytest.approx(saved["evaluation"]["tou"])
    assert checked["demand"] == pytest.approx(saved["evaluation"]["demand"])
