"""Counterexamples for empirical-front geometry and independent-unit aggregation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from geo_llm_scheduler.experiments.metrics import igd_plus
from geo_llm_scheduler.experiments.paired_statistics import holm_adjust, paired_cluster_summary

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


def test_holm_stepdown_preserves_order_and_monotonic_adjustment() -> None:
    assert holm_adjust([0.04, 0.001, 0.03]) == pytest.approx([0.06, 0.003, 0.06])
    with pytest.raises(ValueError):
        holm_adjust([float("nan")])


def test_cluster_summary_retains_zero_clusters_and_requires_matching_strata() -> None:
    result = paired_cluster_summary(
        [0, 0, 0, 0], [50, 50, 100, 100], np.random.default_rng(6), 200, 100
    )
    assert result["mean"] == 0 and result["ci95"] == [0, 0]
    assert result["p_signflip"] == 1 and result["tie_bases"] == 4
    with pytest.raises(ValueError):
        paired_cluster_summary([0, 1], [50], np.random.default_rng(6))
