"""Counterexamples for width inference, offline opportunity and immutable evidence."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "width_analysis", SCRIPTS / "analyze_macro_width_results.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_hv_envelope_is_independent_of_order_duplicates_and_dominated_points():
    points = [(0.2, 0.8), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (2, -0.5), (0.7, 0.9)]
    assert analysis.integral_hv(points) == pytest.approx(0.54)
    assert analysis.integral_hv(list(reversed(points))) == pytest.approx(0.54)
    assert analysis.integral_hv([]) == 0


def complete_cells():
    return [
        dict(jobs=50, base_seed=1, tariff=t, algorithm_seed=s, value=v)
        for t, v in (("H", 0), ("T", 8))
        for s in (1112, 2212)
    ]


def test_base_mean_weights_tariffs_and_seeds_before_inference():
    values, strata, keys = analysis.base_values(complete_cells(), "value")
    assert values == [4] and strata == [50] and keys == [(50, 1)]


@pytest.mark.parametrize("mode", ["missing", "duplicate", "wrong_seed"])
def test_base_mean_rejects_incomplete_pseudoreplicated_cells(mode):
    rows = complete_cells()
    if mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[-1]["algorithm_seed"] = 1101
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.base_values(rows, "value")


def call(gains=(0.0, 0.1, 0.2), feasible=True):
    return {
        "prefix": {
            str(b): {"gain": g, "evaluated": b, "relative_gain": g}
            for b, g in zip((3, 6, 10), gains)
        },
        "online_rng_verified": True,
        "state": 0,
        "action": 1,
        "progress": 0.8,
        "seq_common_gain": gains[0],
        "seq_missed_gain": gains[-1] - gains[0],
        "seq": {"final_B_eff": 3},
        "full_objectives": [(1, 1)] * (10 if feasible else 9),
        "archive_before_objectives": [(2, 2)],
        "late_4_6_after_zero3": True,
        "late_7_10_after_zero6": False,
        "common_generation_seconds": 0.1,
        "total_replay_seconds": 1,
    }


def test_late_improvement_is_not_discarded_after_a_zero_prefix():
    row = analysis.probe_fields(call(), 123)
    assert row["late_4_6"] == 1 and row["seq_missed_gain"] == pytest.approx(0.2)
    assert row["stage"] == 2 and row["objective_additions3"] == 1
    assert row["bucket_calls"] == 123


def test_infeasible_filtering_does_not_shift_prefix_archive_indices():
    row = analysis.probe_fields(call(feasible=False), 100)
    assert row["all_full_feasible"] == 0 and row["objective_additions3"] is None


def test_nonmonotone_gain_and_changed_seq_context_are_rejected():
    with pytest.raises(ValueError, match="Nonmonotone"):
        analysis.probe_fields(call(gains=(0.1, 0.0, 0.2)), 1)
    row = call()
    row["seq_missed_gain"] = 0.7
    with pytest.raises(ValueError, match="SEQ"):
        analysis.probe_fields(row, 1)


def test_six_primary_hypotheses_are_jointly_corrected_and_never_promote_diagnostics():
    paired = []
    for a, b in analysis.CONTRASTS:
        for base in (1, 2):
            for t in ("H", "T"):
                for seed in (1112, 2212):
                    paired.append(
                        {
                            "contrast": a + "-" + b,
                            "jobs": 50,
                            "base_seed": base,
                            "tariff": t,
                            "algorithm_seed": seed,
                            **{m: 0 for m in ("hv_diff", "igd_plus_gain", *analysis.DIAGNOSTICS)},
                        }
                    )
    stats, bases = analysis.inference(paired, 100, 200)
    assert len(analysis.PRIMARY) == 6 and len(bases) == 66
    for contrast, fields in stats.items():
        assert fields["hv_diff"]["p_holm_primary_six"] == 1
        assert "p_holm_primary_six" not in fields["hv_pct"]
        assert fields["hv_diff"]["n_bases"] == 2


def test_group_estimator_weights_visits_within_run_then_gives_bases_equal_weight():
    rows = []
    for base, weights in ((1, (99, 1)), (2, (1, 1))):
        for stage, weight in enumerate(weights):
            rows.append(
                {
                    "arm": "POOL6",
                    "action": 1,
                    "jobs": 50,
                    "base_seed": base,
                    "tariff": "H",
                    "algorithm_seed": 1112,
                    "bucket_calls": weight,
                    "zero3": 1,
                    "zero6": 1,
                    "late_4_6": stage,
                    "late_7_10": stage,
                    "seq_missed_positive": stage,
                    "short_pool": 0,
                    "relative_gain3": 0,
                    "relative_gain6": stage,
                    "relative_gain10": stage,
                    "positive_increment_3_6": stage,
                    "positive_increment_6_10": stage,
                    "seq_effective": 3,
                }
            )
    result = analysis.probe_summary(rows, "action")[0]
    assert result["base_visit_weighted_late_4_6"] == pytest.approx((0.01 + 0.5) / 2)
    assert result["late_given_zero3_bucket_rate"] == 0.5
    assert result["bases"] == 2


def test_common_front_degenerate_axis_and_igd_plus_direction():
    lo, span = analysis.shared.normalization([(2, 3)])
    assert np.array_equal(lo, [2, 3]) and np.array_equal(span, [1, 1])
    assert analysis.shared.distance([(0, 0)], [(1, 1)], True) == 0
    assert analysis.shared.distance([(0, 0)], [(1, 1)], False) == pytest.approx(2**0.5)


def test_independent_audit_rejects_a_modified_saved_pvalue(tmp_path):
    import json

    audit_spec = importlib.util.spec_from_file_location(
        "width_audit", SCRIPTS / "audit_macro_width_results.py"
    )
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    paired = []
    for a, b in analysis.CONTRASTS:
        for base in range(40):
            for tariff in ("H", "T"):
                for seed in (1112, 2212):
                    paired.append(
                        {
                            "contrast": a + "-" + b,
                            "jobs": 50 if base < 20 else 100,
                            "base_seed": base,
                            "tariff": tariff,
                            "algorithm_seed": seed,
                            **{m: 0 for m in ("hv_diff", "igd_plus_gain", *analysis.DIAGNOSTICS)},
                        }
                    )
    stats, bases = analysis.inference(paired, 100, 200)
    stats["POOL3-POOL6"]["hv_diff"]["p_signflip"] = 0.1
    analysis.write_csv(tmp_path / "base_metrics.csv", bases)
    (tmp_path / "analysis_summary.json").write_text(
        json.dumps(
            {"statistics": stats, "seed": analysis.SEED, "bootstrap": 100, "signflips": 200}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sign-flip"):
        audit.audit(tmp_path / "raw", tmp_path)


def test_report_cannot_write_under_immutable_raw_and_requires_verified_sources(tmp_path):
    import json

    report_spec = importlib.util.spec_from_file_location(
        "width_report", SCRIPTS / "report_macro_width_results.py"
    )
    report = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(report)
    raw = tmp_path / "raw"
    with pytest.raises(ValueError, match="immutable raw"):
        report.generate(raw, tmp_path, raw / "report", tmp_path, tmp_path)
    for name, value in (
        ("analysis_summary.json", {}),
        ("independent_audit.json", {"state": "pending"}),
        ("collection.json", {}),
        ("completion.json", {"state": "complete_verified"}),
    ):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Unverified"):
        report.generate(
            raw,
            tmp_path,
            tmp_path / "report",
            tmp_path / "collection.json",
            tmp_path / "completion.json",
        )
