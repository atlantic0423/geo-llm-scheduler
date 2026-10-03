"""Hand-check correction, clustering and deterministic resampling edge cases."""

import numpy as np
import pytest

from geo_llm_scheduler.experiments.paired_statistics import holm_adjust, paired_cluster_summary


def test_holm_preserves_order_and_monotonic_stepdown() -> None:
    assert holm_adjust([0.04, 0.001, 0.03]) == pytest.approx([0.06, 0.003, 0.06])
    assert holm_adjust([]) == []
    with pytest.raises(ValueError):
        holm_adjust([float("nan")])


def test_stratified_resampling_preserves_constant_stratum_weights() -> None:
    result = paired_cluster_summary(
        [1.0, 1.0, 3.0, 3.0], [50, 50, 100, 100], np.random.default_rng(8), 200, 2000
    )
    assert result["mean"] == 2.0
    assert result["ci95"] == [2.0, 2.0]
    assert result["positive_bases"] == 4
    assert 0.08 < result["p_signflip"] < 0.18


def test_all_ties_and_explicit_rng_are_reproducible() -> None:
    a = paired_cluster_summary([0.0, 0.0], [50, 100], np.random.default_rng(7), 20, 30)
    b = paired_cluster_summary([0.0, 0.0], [50, 100], np.random.default_rng(7), 20, 30)
    assert a == b
    assert a["ci95"] == [0.0, 0.0] and a["p_signflip"] == 1.0
    assert a["tie_bases"] == 2
    with pytest.raises(ValueError):
        paired_cluster_summary([1.0], [], np.random.default_rng(1))
