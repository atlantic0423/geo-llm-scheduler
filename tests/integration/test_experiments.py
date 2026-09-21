"""Research artifact IO and analytical metric checks."""

import csv
import importlib

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.experiments.runner import git_metadata, save_result
from geo_llm_scheduler.experiments.synthetic import synthetic
from geo_llm_scheduler.io.validation import validate_problem


def test_metrics():
    assert hypervolume([(1, 3), (2, 2)], (4, 4)) == 5
    assert igd_plus([(1, 1)], [(2, 2)]) == 0
    assert igd_plus([(3, 2)], [(1, 2)]) == 2
    with pytest.raises(ValueError):
        hypervolume([(5, 1)], (4, 4))


def test_artifacts(problem, tmp_path):
    config = Config(population=4, neighborhood=2, generations=1)
    result = run(problem, config)
    summary = save_result(result, config, "test-instance", tmp_path)
    assert summary["archive_attempts"] == summary["counts"]["feasible"]
    assert summary["archive_insertions"] >= summary["archive_size"]
    assert summary["archive_peak_size"] >= summary["archive_size"]
    assert sum(summary["archive_final_by_origin"].values()) == summary["archive_size"]
    assert summary["termination_reason"] == "generation_limit"
    assert summary["time_budget_seconds"] is None
    assert summary["time_overshoot_seconds"] is None
    assert summary["git_commit"] is None or len(summary["git_commit"]) == 40
    assert summary["git_dirty"] is None or isinstance(summary["git_dirty"], bool)
    assert (tmp_path / "trace.jsonl").exists()
    assert (tmp_path / "qtable.json").exists()
    with (tmp_path / "objectives.csv").open(encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == [
            "flow_seconds",
            "bill_cny",
            "tou_cny",
            "demand_cny",
        ]


def test_synthetic_seed():
    a, b = synthetic(6, 2, 2, 42), synthetic(6, 2, 2, 42)
    assert a == b
    validate_problem(a)


def test_git_metadata_is_explicit_outside_repository(tmp_path):
    assert git_metadata(tmp_path) == {
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }


def test_soft_time_limit_reports_overshoot(problem, monkeypatch, tmp_path):
    run_module = importlib.import_module("geo_llm_scheduler.engine.run")
    ticks = iter((0.0, 0.1, 0.5, 2.0, 2.5))
    monkeypatch.setattr(run_module, "perf_counter", lambda: next(ticks))
    config = Config(population=4, neighborhood=2, generations=2, seconds=1.0)
    result = run_module.run(problem, config)
    assert result.termination_reason == "time_budget"
    summary = save_result(result, config, "timed-instance", tmp_path)
    assert summary["elapsed"] == 2.5
    assert summary["time_overshoot_seconds"] == 1.5
