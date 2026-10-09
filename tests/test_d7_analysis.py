"""Metric counterexamples and frozen D7 multiplicity/cluster guarantees."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "d7_analysis", Path(__file__).parents[1] / "scripts/analyze_d7_results.py"
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


def test_primary_and_secondary_seven_families():
    statistics = {
        a + "-BASE": {m: {"p_signflip": 0.02} for m in ("hv_diff", "igd_plus_gain")}
        for a in analysis.ARMS[1:]
    }
    statistics["B3-BASE"]["igd_plus_gain"]["p_signflip"] = 0.00001
    analysis.correct_families(statistics)
    assert len(analysis.PRIMARY) == len(analysis.SECONDARY) == 7
    for a, m in analysis.PRIMARY:
        assert statistics[a][m]["p_holm_primary_seven"] == pytest.approx(0.14)
        assert "p_holm_secondary_seven" not in statistics[a][m]
    assert statistics["B3-BASE"]["igd_plus_gain"]["p_holm_secondary_seven"] == pytest.approx(
        0.00007
    )


def test_repeated_seeds_do_not_increase_independent_sample_count():
    rows = [
        dict(jobs=j, base_seed=base, tariff=t, algorithm_seed=s, value=base)
        for j, base in ((50, 1), (100, 2))
        for t in ("H", "T")
        for s in (1101, 2202)
    ]
    values, strata, keys = analysis.base_values(rows, "value")
    assert values == [1, 2] and strata == [50, 100] and len(keys) == 2
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.base_values(rows + [rows[0]], "value")
