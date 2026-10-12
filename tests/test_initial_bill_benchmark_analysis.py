"""Test independent indicator integration and preservation of base-cluster units."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "bill_analysis", SCRIPTS / "analyze_initial_bill_benchmark.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


@pytest.mark.parametrize(
    "front,expected",
    [
        ([], 0),
        ([(0, 0)], 1.21),
        ([(0, 1), (1, 0)], 0.21),
        ([(2, 2)], 0),
        ([(0.2, 0.5), (0.8, 0.1)], 0.66),
    ],
)
def test_hv_horizontal_oracle(front, expected):
    assert analysis.hv_oracle(front) == pytest.approx(expected)


def test_fast_bill_enforces_physical_flow_bound():
    front = [(10, 20), (20, 10)]
    assert analysis.fast_bill(front, 9) is None
    assert analysis.fast_bill(front, 15) == 20
    assert analysis.fast_bill(front, 20) == 10


def test_cluster_reduction_is_base_not_population():
    rows = [
        {"jobs": 50, "base_seed": 1, "tariff": t, "seed": s, "gain": g}
        for (t, s), g in zip([(t, s) for t in ("H", "T") for s in (11001, 11002)], [1, 3, 5, 7])
    ]
    values, strata, keys = analysis.base_reduce(rows, "gain")
    assert values == [4] and strata == [50] and keys == [(50, 1)]
    with pytest.raises(AssertionError):
        analysis.base_reduce(rows[:-1], "gain")
    with pytest.raises(AssertionError):
        analysis.base_reduce(rows[:-1] + [rows[0]], "gain")


def test_points_include_every_demand_region():
    assert analysis.points([{"evaluation": {"flow": 10, "tou": 2, "demand": [3, 4]}}]) == [(10, 9)]


def test_artifact_writer_hashes_whole_file(tmp_path):
    path = tmp_path / "nested" / "derived.json"
    analysis.dump(path, {"a": 123})
    assert analysis.read(path) == {"a": 123}
    first = analysis.sha(path)
    path.write_text('{"a":124}', encoding="utf-8")
    assert analysis.sha(path) != first
    analysis.table(tmp_path / "table.csv", [{"x": 1, "y": 2}])
    assert (tmp_path / "table.csv").read_text() == "x,y\n1,2\n"
