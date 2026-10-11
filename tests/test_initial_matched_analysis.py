"""Front comparison counterexamples prevent invented Pareto quality conclusions."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "initial_matched", scripts / "analyze_initial_matched.py"
)
assert spec and spec.loader
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_equal_front_keeps_full_hv_and_zero_gap():
    front = [(0, 1), (1, 0)]
    r = analysis.front_metrics(front, front, np.array([0, 0]), np.array([1, 1]))
    assert r["hv_retention_pct"] == pytest.approx(100)
    assert r["igd_plus_to_BASE_final"] == r["mean_direction_gap"] == 0


def test_dominated_initial_front_has_nonzero_distance_and_lower_hv():
    r = analysis.front_metrics([(2, 2)], [(1, 1)], np.array([0, 0]), np.array([2, 2]))
    assert r["hv_retention_pct"] < 100 and r["igd_plus_to_BASE_final"] > 0
    assert r["final_covers_initial"] == 1 and r["initial_covers_final"] == 0
    assert r["flow_gap_pct"] == r["bill_gap_pct"] == 100


def test_decomposition_uses_exact_components_and_rejects_infeasible():
    assert analysis.objective(
        {"evaluation": {"feasible": True, "flow": 1, "tou": 2, "demand": [3, 4]}}
    ) == (1, 9)
    with pytest.raises(ValueError):
        analysis.objective({"evaluation": {"feasible": False}})


def test_source_trace_retains_the_original_population_and_rng(problem):
    from dataclasses import replace

    from geo_llm_scheduler.config import Config
    from geo_llm_scheduler.initialization.generators import initial_genotypes
    from geo_llm_scheduler.utils.rng import RNGManager

    config = replace(Config(), population=20, neighborhood=10, seed=2202)
    trace = analysis.sample_population(problem, config, "MIXED")
    assert trace.original_replay_verified
    assert trace.genotypes == initial_genotypes(
        problem, config, RNGManager(config.seed).stream("initialization")
    )
    assert len(trace.modes) == 20 and set(trace.modes) == set(range(10))


def test_tariff_pair_uses_original_H_constructor_input(tmp_path):
    h = {"jobs": 100, "base_seed": 1315001, "tariff": "H"}
    t = {**h, "tariff": "T"}
    assert analysis.construction_problem_path(tmp_path, h) == analysis.construction_problem_path(
        tmp_path, t
    )
    assert analysis.construction_problem_path(tmp_path, t).name == "100_1315001_H.json"
    with pytest.raises(ValueError):
        analysis.construction_problem_path(tmp_path, {**h, "tariff": "bad"})
