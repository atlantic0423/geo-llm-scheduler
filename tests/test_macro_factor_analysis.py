"""Counterexamples for factorial signs, cluster integrity and frontier diagnostics."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT))
spec = importlib.util.spec_from_file_location(
    "factor_analysis", SCRIPT / "analyze_macro_factor_results.py"
)
assert spec is not None and spec.loader is not None
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def cells() -> list[dict]:
    """Construct one balanced base with four correlated, unequal observations."""
    return [
        dict(jobs=50, base_seed=1, tariff=t, algorithm_seed=s, effect=v)
        for (t, s), v in zip((("H", 1313), ("H", 2313), ("T", 1313), ("T", 2313)), (1, 3, 5, 7))
    ]


def test_base_averaging_uses_independent_unit() -> None:
    """Four correlated H/T/seed observations must produce one effect, not four units."""
    values, strata, keys = analysis.base_values(cells(), "effect")
    assert values == [4.0] and strata == [50] and keys == [(50, 1)]


@pytest.mark.parametrize("kind", ("missing", "duplicate", "wrong_seed", "nonfinite"))
def test_base_counterexamples(kind: str) -> None:
    """Reject incomplete, duplicated, foreign-seed and nonfinite clustered observations."""
    rows = cells()
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = rows[0].copy()
    elif kind == "wrong_seed":
        rows[-1]["algorithm_seed"] = 1101
    else:
        rows[-1]["effect"] = float("nan")
    with pytest.raises(ValueError):
        analysis.base_values(rows, "effect")


def test_interaction_improvement_signs() -> None:
    """A larger CALL generation benefit is positive for both high-HV and low-IGD+."""
    arms = {
        "NATIVE_SHARED": {"hv": 1.0, "igd_plus": 0.5},
        "POOL_SHARED": {"hv": 1.1, "igd_plus": 0.4},
        "NATIVE_CALL": {"hv": 1.2, "igd_plus": 0.35},
        "POOL_CALL": {"hv": 1.5, "igd_plus": 0.1},
    }
    assert analysis.interaction_effect(arms, "hv") == pytest.approx(0.2)
    assert analysis.interaction_effect(arms, "igd_plus") == pytest.approx(0.15)
    arms.pop("NATIVE_CALL")
    with pytest.raises(ValueError):
        analysis.interaction_effect(arms, "hv")


def test_hv_ties_and_reference_clipping() -> None:
    """Equal-x dominated points and out-of-reference points do not change union area."""
    points = [(0.5, 0.6), (0.5, 0.8), (0.8, 0.3), (1.2, 0.0), (0.2, 1.2)]
    assert analysis.integral_hv(points) == pytest.approx(0.39)
    assert analysis.integral_hv([]) == 0.0


def test_coverage_weak_ties_and_empty() -> None:
    """Coverage includes equality but never invents a value for an empty approximation."""
    a, b = [(0.2, 0.4), (0.5, 0.1)], [(0.2, 0.4), (0.6, 0.2), (0.1, 0.8)]
    assert analysis.coverage(a, b) == pytest.approx(2 / 3)
    assert analysis.coverage(a, b) == analysis.shared.coverage(a, b)
    with pytest.raises(ValueError):
        analysis.coverage([], b)


def test_primary_secondary_families_are_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    """A .01 p-value is .06 in a six-test family but .04 in a four-test family."""
    paired = []
    for name in (*analysis.MAIN[:2], *analysis.SECONDARY, analysis.INTERACTION):
        for jobs in (50, 100):
            for base in range(20):
                for row in cells():
                    row.update(jobs=jobs, base_seed=base, contrast=name)
                    row.update({m: 1.0 for m in (*analysis.QUALITY, *analysis.DESCRIPTIVE)})
                    paired.append(row)
    monkeypatch.setattr(
        analysis.shared, "cluster_summary", lambda *args, **kwargs: {"p_signflip": 0.01}
    )
    stats, _ = analysis.inference(paired, 4, 4)
    assert stats[analysis.MAIN[0]]["hv_diff"]["p_holm_primary_six"] == pytest.approx(0.06)
    assert stats[analysis.SECONDARY[0]]["hv_diff"]["p_holm_secondary_four"] == pytest.approx(0.04)
