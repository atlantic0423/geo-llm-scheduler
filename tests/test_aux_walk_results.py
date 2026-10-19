"""Counterexamples for the auxiliary experiment's cluster inference and quality metrics."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "aux_walk_analysis", Path(__file__).parents[1] / "scripts/analyze_aux_walk_results.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_metrics_include_dominated_axis_and_duplicate_points():
    points = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.5, 0.5), (0.8, 0.8)]
    hv, igd = analysis.quality(points, [(0.0, 0.0)])
    assert hv == pytest.approx(0.46)
    assert analysis.integral_hv(points) == pytest.approx(hv)
    assert igd == pytest.approx(0.5**0.5)
    assert analysis.quality([(0.0, 0.0)], [(0.5, 0.5)])[1] == 0


def test_stratified_resampling_preserves_constant_effect_and_replays():
    kwargs = dict(bootstrap=100, signflips=1000)
    first = analysis.inference([2.0] * 8, [20] * 4 + [200] * 4, np.random.default_rng(4), **kwargs)
    assert first == analysis.inference(
        [2.0] * 8, [20] * 4 + [200] * 4, np.random.default_rng(4), **kwargs
    )
    assert first["mean"] == 2 and first["ci95"] == [2.0, 2.0]
    assert first["positive"] == 8 and first["negative"] == 0
    zero = analysis.inference([0.0] * 8, [20] * 8, np.random.default_rng(4), **kwargs)
    assert zero["p"] == 1 and zero["tied"] == 8


def test_holm_retains_input_order_and_stepdown_monotonicity():
    assert analysis.holm([0.03, 0.001, 0.01]) == pytest.approx([0.03, 0.003, 0.02])
    assert analysis.holm([0.04, 0.04, 0.04]) == pytest.approx([0.12] * 3)
    with pytest.raises(ValueError):
        analysis.holm([float("nan")])


def test_rejects_nonfinite_or_incomplete_clusters():
    with pytest.raises(ValueError):
        analysis.inference([1.0, float("inf")], [20, 20], np.random.default_rng(4))
    with pytest.raises(ValueError):
        analysis.inference([1.0], [], np.random.default_rng(4))
    with pytest.raises(ValueError):
        analysis.quality([], [(0.0, 0.0)])
