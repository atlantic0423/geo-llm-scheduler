"""Meaningful counterexamples for the independent RL23 result calculations."""

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "rl23_analysis", Path(__file__).parents[1] / "research/rl23_results_20261021/analyze_results.py"
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


def test_hv_union_duplicates_and_reference_exclusion():
    points = [(0.2, 0.8), (0.8, 0.2), (0.8, 0.2), (0.9, 0.9), (2, 0)]
    assert ANALYSIS.volume(points) == pytest.approx(0.45)
    assert ANALYSIS.volume_audit(points) == pytest.approx(0.45)
    with pytest.raises(ValueError):
        ANALYSIS.nondominated([(float("nan"), 1)])


def test_modified_distance_avoids_penalizing_a_dominating_point():
    assert ANALYSIS.distance([(0, 0)], [(1, 1)]) == 0
    assert ANALYSIS.distance([(0, 0)], [(1, 1)], False) == pytest.approx(2**0.5)


def test_coverage_sweep_handles_two_endpoints_and_no_dominator():
    assert ANALYSIS.coverage([(0, 1), (1, 0)], [(0, 1), (0.5, 0.5), (1, 0)]) == pytest.approx(2 / 3)
    assert ANALYSIS.coverage([(1, 1)], [(0, 0)]) == 0


def test_holm_and_parent_unit_are_not_run_level_tests():
    assert ANALYSIS.holm([0.04, 0.001, 0.02]) == pytest.approx([0.04, 0.003, 0.04])
    with pytest.raises(AssertionError):
        ANALYSIS.inference([1] * 64, [20, 50, 100, 200] * 16, 1)
    result = ANALYSIS.inference([0.0] * 16, [20, 50, 100, 200] * 4, 1)
    assert result["p"] == 1 and result["p_exact_audit"] == 1 and result["ci95"] == [0, 0]


def test_step_prefix_and_terminal_audit_rejects_unfinished_episode(tmp_path):
    path = tmp_path / "steps.jsonl.gz"
    row = dict(
        step=0,
        state=0,
        action=1,
        available_actions=[1],
        action_seconds=1,
        seconds=1.01,
        candidate_objectives=[[0.8, 0.8], [0.2, 0.2]],
        candidate_feasible=[True, True],
        exact_evaluations=2,
        reward=10,
        quality_reward=10,
        scalar_before=1,
        scalar_after=0.2,
        accepted=True,
        feasible=2,
        explore=False,
        archive_net_retained=1,
        archive_insertions=1,
        archive_hv_gain=0,
        terminal=True,
        wall_progress=0.1,
        comparison_ideal=[0, 0],
        comparison_maximum=[1, 1],
        weight=[1, 0],
    )
    q = dict(q=[[0] * 8] * 42, selections=np.zeros((42, 8), dtype=int).tolist())
    q["selections"][0][0] = 1
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(row) + "\n")
    result, actions, _, prefixes = ANALYSIS.step_diagnostics(path, "STRICT_TWO_FOCUS", q)
    assert result["trajectories"] == 1 and actions[0]["accepted"] == 1
    assert prefixes[0]["extra_candidate_win"] == 1
    assert prefixes[0]["prefix_scalar_gain"] == pytest.approx(0.2)
    row["terminal"] = False
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(row) + "\n")
    with pytest.raises(AssertionError):
        ANALYSIS.step_diagnostics(path, "STRICT_TWO_FOCUS", q)


def test_state_time42_reads_numeric_bit_and_wall_progress(tmp_path):
    path = tmp_path / "empty.jsonl.gz"
    with gzip.open(path, "wt"):
        pass
    q = dict(q=[[0] * 8] * 84, selections=np.zeros((84, 8), dtype=int).tolist())
    result, actions, depths, prefixes = ANALYSIS.step_diagnostics(path, "STATE_TIME84", q)
    assert result["rl_steps"] == 0 and actions == depths == prefixes == []
