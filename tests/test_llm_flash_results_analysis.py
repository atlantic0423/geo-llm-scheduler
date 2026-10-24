"""Counterexamples for the frozen LLM study's metrics and mechanism interpretation."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).parents[1] / "research/llm_flash_results_20261024"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location("llm_analysis", DIRECTORY / "analyze_results.py")
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)
STATS = importlib.import_module("stats_core")


def test_hv_duplicate_dominance_and_reference_exclusion():
    points = [(0.2, 0.8), (0.8, 0.2), (0.8, 0.2), (0.9, 0.9), (2, 0)]
    assert STATS.volume(points) == pytest.approx(0.45)
    assert STATS.volume_audit(points) == pytest.approx(0.45)
    with pytest.raises(ValueError):
        STATS.nondominated([(float("nan"), 0)])


def test_igd_plus_respects_direction_and_coverage_uses_actual_points():
    assert STATS.distance([(0, 0)], [(1, 1)]) == 0
    assert STATS.distance([(1, 1)], [(0, 0)]) == pytest.approx(2**0.5)
    assert STATS.coverage([(0, 1), (1, 0)], [(0, 1), (0.5, 0.5), (1, 0)]) == pytest.approx(2 / 3)


def test_holm_is_monotone_and_in_original_order():
    assert STATS.holm([0.04, 0.001, 0.02]) == pytest.approx([0.04, 0.003, 0.04])


def test_inference_does_not_treat_64_runs_as_independent_parents():
    with pytest.raises(AssertionError):
        STATS.inference([1] * 64, [20, 50, 100, 200] * 16, 1)
    with pytest.raises(AssertionError):
        STATS.inference([float("nan")] * 16, [20, 50, 100, 200] * 4, 1)
    result = STATS.inference([0.0] * 16, [20, 50, 100, 200] * 4, 1)
    assert result["p"] == result["p_exact_audit"] == 1 and result["ci95"] == [0, 0]


def test_flow_window_never_interpolates_an_unavailable_schedule():
    assert ANALYSIS.flow_window([(1, 10), (1.3, 1)], 1.25) == 10
    assert ANALYSIS.flow_window([(1, 10), (1.3, 1)], 0.9) is None
    assert ANALYSIS.flow_window([(1, 10), (1.3, 1)], 1.3) == 1


def _write(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(step=0, terminal=True, reason="horizon_or_budget"):
    return dict(
        step=step,
        action=3,
        action_seconds=0.5,
        scalar_before=1.0,
        scalar_after=0.9,
        exact_evaluations=1,
        candidate_objectives=[[1, 2]],
        candidate_feasible=[True],
        accepted=True,
        terminal=terminal,
        stop_reason=reason,
        requested_focus="KV_DELAY",
        applied_focus="KV_DELAY",
        archive_new_objective_events=2,
        dominated_archive_members=3,
    )


def test_horizon_or_budget_is_not_misreported_as_no_gain_early_stop(tmp_path):
    path = tmp_path / "steps.gz"
    _write(path, [_row()])
    totals, actions, focus = ANALYSIS.steps(path, "LLM_FLASH")
    assert totals["ls_early_stop_rate"] == 0 and totals["ls_no_gain_terminal_rate"] == 0
    assert actions[0]["new_objective_events"] == 2 and actions[0]["main_archive_insertions"] == 0
    assert focus[(3, "KV_DELAY", "KV_DELAY")] == 1


def test_no_gain_at_fifth_step_is_threshold_terminal_but_not_early(tmp_path):
    path = tmp_path / "steps.gz"
    rows = [_row(i, i == 4, "no_gain_threshold" if i == 4 else None) for i in range(5)]
    _write(path, rows)
    totals, _, _ = ANALYSIS.steps(path, "LLM_FLASH")
    assert totals["ls_early_stop_rate"] == 0 and totals["ls_no_gain_terminal_rate"] == 1
    assert totals["ls_mean_depth"] == 5
    _write(path, [_row(reason="no_gain_threshold")])
    totals, _, _ = ANALYSIS.steps(path, "LLM_FLASH")
    assert totals["ls_early_stop_rate"] == 1 and totals["ls_mean_depth"] == 1


def test_unfinished_or_mismatched_candidate_stream_rejected(tmp_path):
    path = tmp_path / "steps.gz"
    _write(path, [_row(terminal=False)])
    with pytest.raises(AssertionError):
        ANALYSIS.steps(path, "LLM_FLASH")
    row = _row()
    row["candidate_objectives"] = []
    _write(path, [row])
    with pytest.raises(AssertionError):
        ANALYSIS.steps(path, "LLM_FLASH")


def test_saved_exact_objectives_bill_includes_all_demand_regions(tmp_path):
    path = tmp_path / "front.gz"
    with gzip.open(path, "wt") as f:
        json.dump([dict(evaluation=dict(feasible=True, flow=12, tou=2, demand=[3, 4]))], f)
    assert STATS.stored_points(path) == ([(12, 9)], 1)
    p = tmp_path / "unicode.json"
    p.write_text('{"例":1}', encoding="utf-8")
    assert ANALYSIS.json_read(p) == {"例": 1}
