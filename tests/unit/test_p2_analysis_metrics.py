"""Counterexamples for empirical-front geometry and independent-unit aggregation."""

import importlib.util
from pathlib import Path

import pytest

from geo_llm_scheduler.experiments.metrics import igd_plus

_SPEC = importlib.util.spec_from_file_location(
    "p2_analysis", Path(__file__).resolve().parents[2] / "scripts/analyze_p2_results.py"
)
assert _SPEC is not None and _SPEC.loader is not None
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def test_empirical_front_deduplicates_and_handles_equal_coordinates() -> None:
    points = [(0.0, 1.0), (0.0, 0.8), (0.4, 0.8), (0.4, 0.3), (0.4, 0.3), (1.0, 0.0)]
    assert analysis._nd(points) == [(0.0, 0.8), (0.4, 0.3), (1.0, 0.0)]


def test_igd_plus_uses_one_sided_minimization_distance() -> None:
    # A candidate better in Flow pays only its Bill shortfall, not ordinary Euclidean distance.
    points, reference = [(0.0, 0.6)], [(0.5, 0.5)]
    assert analysis._igd(points, reference) == pytest.approx(0.1)
    assert analysis._igd(points, reference) == pytest.approx(igd_plus(points, reference))
    assert analysis._igd([(0.0, 0.0)], reference) == 0.0


def test_hypervolume_excludes_points_outside_reference_box() -> None:
    points = [(0.2, 0.8), (0.8, 0.2), (0.0, 1.2)]
    assert analysis._hv(points, 1.0) == pytest.approx(0.28)
    assert analysis._hv([(0.0, 1.2)], 1.0) == 0.0


def test_tariffs_remain_equal_weight_when_epoch_subset_has_unequal_seed_counts() -> None:
    rows = [
        {"jobs": 50, "base_seed": 1, "tariff": "H", "effect": 0.0},
        {"jobs": 50, "base_seed": 1, "tariff": "H", "effect": 2.0},
        {"jobs": 50, "base_seed": 1, "tariff": "T", "effect": 9.0},
        {"jobs": 100, "base_seed": 2, "tariff": "H", "effect": -2.0},
    ]
    values, strata = analysis._base_values(rows, "effect")
    assert values == [5.0, -2.0]
    assert strata == [50, 100]
