"""Known-answer numerical cases for the development analysis, independent of solver runs."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "budget12_analysis", Path(__file__).resolve().parents[1] / "research/lsv2_budget12/analyze.py"
)
assert spec and spec.loader
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_hypervolume_union_deduplicates_and_discards_outside_reference():
    # Two overlapping rectangles: .6*.9 + .9*.6 - .6*.6 = .72.
    front = [(0.5, 0.2), (0.2, 0.5), (0.5, 0.2), (0.8, 0.8), (1.2, 0.0)]
    assert analysis.hv(front) == pytest.approx(0.72)
    assert analysis.nd([(2, 3), (1, 3), (2, 2), (1, 3)]) == [(1, 3), (2, 2)]


def test_igd_plus_does_not_penalize_improvement_over_reference():
    assert analysis.igdp([(0, 0)], [(0.5, 0.5), (1, 1)]) == 0
    assert analysis.igdp([(1, 1)], [(0, 0)]) == pytest.approx(2**0.5)


def test_exact_sign_flip_known_null_and_all_positive_extremes():
    sizes = [20] * 4 + [50] * 4 + [100] * 4 + [200] * 4
    positive = analysis.infer([1.0] * 16, sizes, 20261028)
    assert positive["p"] == 2 / 65536
    assert positive["ci95"] == [1, 1]
    assert positive["positive"] == 16
    null = analysis.infer([0.0] * 16, sizes, 20261028)
    assert null["p"] == 1
    assert null["positive"] == null["negative"] == 0
