"""Research artifact IO and analytical metric checks."""

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.experiments.runner import save_result
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
    assert (tmp_path / "trace.jsonl").exists()
    assert (tmp_path / "qtable.json").exists()


def test_synthetic_seed():
    a, b = synthetic(6, 2, 2, 42), synthetic(6, 2, 2, 42)
    assert a == b
    validate_problem(a)
