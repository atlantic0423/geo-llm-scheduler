"""Metric oracles, exact bill sums and independent-base analysis integrity."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_initial_bill_speed as analysis


def test_hv_oracle_known_area_dominance_ties_and_empty():
    assert analysis.hv_oracle([]) == 0.0
    assert analysis.hv_oracle([(0.0, 0.0)]) == pytest.approx(1.21)
    assert analysis.hv_oracle([(0.0, 1.0), (1.0, 0.0), (0.5, 0.5)]) == pytest.approx(0.46)
    assert analysis.hv_oracle([(0.0, 0.0), (0.0, 1.0), (0.0, 0.0), (2.0, 2.0)]) == pytest.approx(
        1.21
    )


def test_points_preserves_original_regional_addition_order():
    assert analysis.points(
        [{"evaluation": {"flow": 3.0, "tou": 0.25, "demand": [1e16, 1.0, 1.0]}}]
    ) == [(3.0, 0.25 + sum([1e16, 1.0, 1.0]))]


def test_synthetic_end_to_end_uses_32_independent_bases_and_four_holm_tests(tmp_path, monkeypatch):
    """Synthetic indicators test statistical plumbing, not algorithm quality."""
    rows = []
    for base in range(32):
        for seed in analysis.SEEDS:
            for arm in analysis.ARMS:
                points = (
                    [(0.0, 1.0), (1.0, 0.0)]
                    if arm == "FAST1"
                    else [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
                )
                ratio = {"REF4": 1.0, "FAST4": 0.4, "FAST2": 0.25, "FAST1": 0.15}[arm]
                rows.append(
                    dict(
                        key=f"{base}_{seed}",
                        arm=arm,
                        n=(20, 50, 100, 200)[base % 4],
                        machines=12 if (base // 4) % 2 == 0 else 24,
                        scenario="synthetic",
                        base_seed=base,
                        seed=seed,
                        points=points,
                        stats=dict(
                            production_wall_seconds=10 * ratio,
                            production_cpu_seconds=10 * ratio,
                            parent_wall_seconds=0.1,
                            bank_wall_seconds=10 * ratio - 0.1,
                            selection_wall_seconds=0.0,
                            audit_wall_seconds=0.2,
                            started=1.0,
                            completed=1.2 + 10 * ratio,
                            max_rss_bytes=1000,
                        ),
                    )
                )
    monkeypatch.setattr(analysis, "validate", lambda raw: ({"source": "synthetic-test-only"}, rows))
    result = analysis.analyze(tmp_path / "raw", tmp_path / "analysis")
    assert result["bases"] == 32 and result["blocks"] == 64 and result["runs"] == 256
    assert len(result["quality"]) == 4
    for r in result["quality"]:
        assert r["n_bases"] == 32 and r["holm_primary_four"] >= r["p_signflip"]
        if r["arm"] == "FAST2":
            assert r["mean"] == 0.0 and r["p_signflip"] == 1.0
    assert all(r["geometric_ratio"] == pytest.approx(0.4) for r in result["cost"])
    assert (tmp_path / "analysis/metrics.csv").exists()
    import plot_initial_bill_speed as plots

    assert all(p.stat().st_size > 1000 for p in plots.plot_costs(tmp_path / "analysis"))


def test_prespecified_front_figure_reads_all_eight_fixed_cases(tmp_path):
    import json

    import plot_initial_bill_speed as plots

    raw = tmp_path / "raw"
    (raw / "specs").mkdir(parents=True)
    out = tmp_path / "out"
    out.mkdir()
    for n in (20, 50, 100, 200):
        for machines in (12, 24):
            key = f"{n}_{machines}"
            (raw / "specs" / f"{key}.json").write_text(
                json.dumps(dict(key=key, n=n, machines=machines, scenario="mixed", seed=15101))
            )
            for arm in analysis.ARMS:
                folder = raw / "runs" / key / arm
                folder.mkdir(parents=True)
                pop = [
                    {"evaluation": {"flow": flow, "tou": bill, "demand": [0.0]}}
                    for flow, bill in ((1.0, 3.0), (2.0, 2.0), (3.0, 1.0))
                ]
                (folder / "population.json").write_text(json.dumps(pop))
    assert all(p.stat().st_size > 1000 for p in plots.plot_fronts(raw, out))
