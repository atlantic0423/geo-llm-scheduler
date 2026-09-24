"""D03 v2 severity-scale and transition aggregation tests."""

from geo_llm_scheduler.experiments.d03_analysis import (
    calibration_summary,
    severity_tables,
    transition_rows,
)


def row(run, generation, condition, ratios):
    result = {
        "run_id": run,
        "generation": str(generation),
        "subproblem": "0",
        "trajectory_step": "0",
        "state_id": "0",
        "instance_seed": "1",
        "algorithm_seed": "101",
        "dominant_condition": condition,
    }
    for name, value in zip(("resource", "kv", "region", "tou", "demand", "compressible"), ratios):
        result[f"severity_{name}"] = str(value * 0.2)
        result[f"ratio_{name}"] = str(value)
    return result


def test_severity_scale_reports_threshold_argmax_and_generation_one():
    rows = [
        row("r1", 0, "TOU", (0.5, 0.2, 1.2, 2.0, 1.5, 0.1)),
        row("r1", 10, "Demand", (0.4, 0.3, 1.1, 1.3, 1.8, 0.2)),
    ]
    summary, pairwise, margins = severity_tables(rows, "test")
    tou_all = next(
        item
        for item in summary
        if item["level"] == "suite"
        and item["generation_bin"] == "all"
        and item["severity"] == "tou"
    )
    assert tou_all["p_above_threshold"] == 1
    assert tou_all["p_argmax"] == 0.5
    assert any(item["generation_bin"] == "Generation 1" for item in summary)
    assert len(pairwise) == 36
    assert margins


def test_transition_effective_counts_do_not_credit_singletons():
    rows = [row("r1", 0, "Resource", (2, 0, 0, 0, 0, 0)) for _ in range(10)]
    rows += [row("r1", 1, "KV", (0, 2, 0, 0, 0, 0))]
    summary = transition_rows(rows)[0]
    assert summary["distinct"] == 2
    assert summary["effective_10"] == 1
    assert calibration_summary(rows)["effective_distinct_mean"] == 1
