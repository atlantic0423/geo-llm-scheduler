"""Boundary and design checks for the independent 33-arm analysis."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_component33_results as analysis


def test_hv_outside_box_is_not_clipped_and_ties_are_safe():
    front = [(0, 2), (0.3, 0.9), (0.3, 1), (0.8, 0.2), (2, 0)]
    assert analysis.hv_score(front) == pytest.approx(0.37)
    assert analysis.independent_hv(front) == pytest.approx(0.37)
    assert analysis.distance([(2, 0)], [(1, 0)], True) == 1


def test_frozen_families_include_all_seventy_and_correct_direction():
    assert [len(v) * 2 for v in analysis.FAMILIES.values()] == [10, 24, 22, 14]
    assert len({p for values in analysis.FAMILIES.values() for p in values}) == 35
    assert analysis.FAMILIES["full"][0] == ("FULL_CURRENT_I2", analysis.BASE)
    assert ("FULL_STRICT_NO_POLISH_I4", "BILL_MOEAD") in analysis.FAMILIES["hierarchy"]


def test_cluster_mean_gives_sizes_equal_weight_and_requires_all():
    assert analysis.equal_scale_mean([1, 1, 2, 3, 4], [20, 20, 50, 100, 200]) == 2.5
    with pytest.raises(ValueError):
        analysis.equal_scale_mean([1, 2], [20, 50])
    with pytest.raises(ValueError):
        analysis.equal_scale_mean([1, 2, 3, float("nan")], [20, 50, 100, 200])


def test_explicit_rng_replay_and_zero_null():
    args = ([0.0] * 16, [20, 50, 100, 200] * 4)
    a = analysis.cluster_statistics(*args, np.random.default_rng(4), 300, 1000)
    b = analysis.cluster_statistics(*args, np.random.default_rng(4), 300, 1000)
    assert a == b and a["ci95"] == [0, 0] and a["p_signflip"] == 1


def test_coverage_is_asymmetric_and_equality_counts():
    assert analysis.weak_coverage([(0, 0)], [(0, 0), (1, 1)]) == 1
    assert analysis.weak_coverage([(1, 1)], [(0, 0)]) == 0
    with pytest.raises(ValueError):
        analysis.weak_coverage([], [(0, 0)])
