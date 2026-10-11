"""Metric counterexamples and frozen D8 multiplicity/cluster guarantees."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "d8_analysis", Path(__file__).parents[1] / "scripts/analyze_d8_results.py"
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


def test_three_prespecified_family_sizes_and_separate_adjustment():
    assert (
        len(analysis.PRIMARY) == 8
        and len(analysis.SECONDARY) == 34
        and len(analysis.INTERACTION_FAMILY) == 12
    )
    statistics = {
        a + "-" + b: {m: {"p_signflip": 0.02} for m in ("hv_diff", "igd_plus_gain")}
        for a, b in analysis.CONTRASTS
    }
    statistics["A3-BASE"]["hv_diff"]["p_signflip"] = 0.00001
    analysis.correct_families(statistics)
    assert statistics["A3-BASE"]["hv_diff"]["p_holm_primary_eight"] == pytest.approx(0.00008)
    assert "p_holm_secondary_34" not in statistics["A3-BASE"]["hv_diff"]
    assert statistics[analysis.ARMS[15] + "-BASE"]["hv_diff"][
        "p_holm_secondary_34"
    ] == pytest.approx(0.68)


def test_additive_factorial_has_zero_interaction_and_synergy_is_averaged():
    rows = {
        analysis.ARMS[m]: {
            "jobs": 50,
            "base_seed": 1,
            "tariff": "H",
            "algorithm_seed": 1101,
            "node": 0,
            "hv": sum(bool(m & (1 << i)) * (i + 1) for i in range(4)),
            "igd_plus": -sum(bool(m & (1 << i)) * (i + 1) for i in range(4)),
        }
        for m in range(16)
    }
    assert all(
        r["hv_diff"] == r["igd_plus_gain"] == 0
        for r in analysis.factorial_interactions({"k": rows})
    )
    for m in range(16):
        if m & 3 == 3:
            rows[analysis.ARMS[m]]["hv"] += 2
            rows[analysis.ARMS[m]]["igd_plus"] -= 3
    interactions = {r["contrast"]: r for r in analysis.factorial_interactions({"k": rows})}
    assert interactions["A3*SEQ"]["hv_diff"] == 2 and interactions["A3*SEQ"]["igd_plus_gain"] == 3
    assert all(r["hv_diff"] == 0 for k, r in interactions.items() if k != "A3*SEQ")
    effects = analysis.factorial_effects({"k": rows})
    assert len(effects) == 15
    assert next(r for r in effects if r["mask"] == 3)["hv_diff"] == 2
    assert all(r["hv_diff"] == r["igd_plus_gain"] == 0 for r in effects if r["order"] >= 3)


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
