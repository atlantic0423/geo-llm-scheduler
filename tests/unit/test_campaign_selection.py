"""Paired selector and pooled reference metric invariants."""

import pytest

from geo_llm_scheduler.experiments.campaign.aggregation import _trace_diagnostics
from geo_llm_scheduler.experiments.campaign.selection import common_metrics, select_stage_winner


def test_common_reference_is_shared_and_objective_order_is_respected():
    metrics, reference = common_metrics(
        {
            "a": [(1, 5), (5, 1)],
            "b": [(4, 4)],
        }
    )
    assert reference["pooled_points"] == 3
    assert reference["ideal"] == (1, 1)
    assert metrics["a"]["hv"] > metrics["b"]["hv"]
    assert metrics["a"]["igd_plus"] < metrics["b"]["igd_plus"]


def test_selector_uses_only_complete_paired_cells_and_fallback_on_ambiguity():
    data = [
        {"cell": "one", "arm": "a", "hv": 3, "igd_plus": 1, "elapsed": 5, "exact": 10},
        {"cell": "one", "arm": "b", "hv": 2, "igd_plus": 2, "elapsed": 5, "exact": 10},
        {"cell": "two", "arm": "a", "hv": 100, "igd_plus": 0, "elapsed": 1, "exact": 10},
    ]
    winner = select_stage_winner(data, ("a", "b"), "b")
    assert winner["selected_arm"] == "a"
    assert winner["pair_count"] == 1
    assert winner["missing_pairs"] == ["two"]
    tie = select_stage_winner(
        [
            {**data[0], "hv": 1, "igd_plus": 1},
            {**data[1], "hv": 1, "igd_plus": 1},
        ],
        ("a", "b"),
        "b",
    )
    assert tie["ambiguity"] and tie["fallback_used"]
    assert tie["selected_arm"] == "b"


def test_selector_rejects_unlisted_fallback():
    with pytest.raises(ValueError):
        select_stage_winner([], ("a",), "unknown")


def test_trace_diagnostics_counts_within_trajectory_changes_and_empty_candidates(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"steps":[{"state":1,"dominant_condition":"Resource","budget":6,"effective":0},'
        '{"state":2,"dominant_condition":"KV","budget":6,"effective":2}]}\n'
        '{"steps":[{"state":1,"dominant_condition":"Resource","budget":6,"effective":6}]}\n',
        encoding="utf-8",
    )
    result = _trace_diagnostics(trace)
    assert result["state_coverage"] == 2
    assert result["condition_transition_rate"] == 1
    assert result["no_candidate_rate"] == pytest.approx(1 / 3)
    assert result["requested_budget"] == 18
    assert result["effective_budget"] == 8
