"""Boundary and partial-cluster checks for the initialization analysis."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "bill_analysis", ROOT / "research/initial_bill_20261012/analyze.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_hv_strip_oracle_handles_outside_and_duplicate_points():
    """Out-of-reference points contribute zero; ties cannot double count area."""
    assert MODULE.hv_oracle([(0.0, 1.0), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0)]) == pytest.approx(0.21)
    assert MODULE.hv_oracle([(0.0, 0.0), (1.0, 1.0)]) == pytest.approx(1.21)
    assert MODULE.hv_oracle([(2.0, 2.0)]) == 0.0


def test_equal_base_weight_survives_missing_seed_pair():
    """Two observations from a base must not receive half of another base's weight."""
    rows = [{"jobs": 50, "base_seed": 1, "x": 0.0}] * 2 + [
        {"jobs": 100, "base_seed": 2, "x": 10.0}
    ] * 4
    assert MODULE.summarize_by_base(rows, ["x"]) == {"x": 5.0}


def test_direction_metric_does_not_claim_unsupported_front_coverage():
    """Linear direction coverage measures supported directions, not the entire curve."""
    reference = [(0.0, 1.0), (0.6, 0.6), (1.0, 0.0)]
    assert MODULE.directions([(0.0, 1.0), (1.0, 0.0)], reference) == 1.0
    assert MODULE.directions([(1.0, 1.0)], reference) == 0.0


def test_statistical_families_keep_all_hypotheses_when_incomplete(monkeypatch):
    """Partial matrices retain both complete 16-test Holm families."""

    def capture(values, strata, rng):
        return {"n_bases": len(values), "mean": sum(values) / len(values), "p_signflip": 0.5}

    monkeypatch.setattr(MODULE, "cluster_summary", capture)
    pairs = []
    for arm in MODULE.NEW[2:]:
        for other in ("RAW", "RESAMPLE" + arm[-2:]):
            for jobs, base, repetitions, complete in ((50, 1, 2, False), (100, 2, 4, True)):
                for _ in range(repetitions):
                    pairs.append(
                        {
                            "contrast": arm + "-" + other,
                            "jobs": jobs,
                            "base_seed": base,
                            "complete_base": complete,
                            "hv_diff": 0.1,
                            "igd_plus_gain": 0.2,
                        }
                    )
    all_bases = MODULE.statistical_family(pairs)
    complete = MODULE.statistical_family(pairs, True)
    assert [len(f) for f in all_bases["families"].values()] == [16, 16]
    assert all(s["n_bases"] == 2 for c in all_bases["statistics"].values() for s in c.values())
    assert all(s["n_bases"] == 1 for c in complete["statistics"].values() for s in c.values())
