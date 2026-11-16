"""Analytic counterexamples for D5 reference metrics and clustered inference."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "d5_analysis", Path(__file__).parents[1] / "scripts/analyze_d5_results.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_front_ties_and_degenerate_axis():
    assert analysis.nondominated([(1, 3), (1, 2), (2, 2), (1, 2), (3, 1)]) == [(1, 2), (3, 1)]
    lo, span = analysis.normalization([(5, 7)])
    assert np.array_equal(lo, [5, 7]) and np.array_equal(span, [1, 1])


def test_modified_distance_and_asymmetric_coverage():
    # Better-than-reference points contribute zero IGD+, but nonzero ordinary IGD.
    assert analysis.distance([(0, 0)], [(1, 1)], True) == 0
    assert analysis.distance([(0, 0)], [(1, 1)], False) == pytest.approx(2**0.5)
    assert analysis.coverage([(0, 0)], [(0, 1), (1, 0)]) == 1
    assert analysis.coverage([(0, 1), (1, 0)], [(0, 0)]) == 0


def test_zero_clusters_and_known_holm():
    result = analysis.cluster_summary(
        [0, 0, 0, 0], [50, 50, 100, 100], np.random.default_rng(1), 100, 200
    )
    assert result["ci95"] == [0, 0] and result["p_signflip"] == 1
    assert analysis.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_missing_seed_is_rejected():
    rows = [
        dict(jobs=50, base_seed=1, tariff=t, algorithm_seed=s, value=3)
        for t in ("H", "T")
        for s in (1101, 2202)
    ]
    assert analysis.base_values(rows, "value")[0] == [3]
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.base_values(rows[:-1], "value")
