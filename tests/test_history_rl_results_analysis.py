"""Meaningful counterexamples for the independent history-state result calculations."""

import csv
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "history_analysis",
    Path(__file__).parents[1] / "research/history_rl_results_20261023/analyze_results.py",
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


def test_history_replay_uses_terminal_zero_and_resets_child_feedback(tmp_path):
    path = tmp_path / "history.jsonl.gz"
    rows = []
    values = np.zeros((9, 8))
    counts = np.zeros((9, 8), dtype=int)
    for step in range(5):
        state = 0 if step == 0 else 1
        terminal = step == 4
        rows.append(
            dict(
                step=step,
                state=state,
                next_state=1,
                action=1,
                previous_action=0 if step == 0 else 1,
                previous_outcome=0 if step == 0 else 2,
                previous_reward=0 if step == 0 else 1.0,
                preference="Flow",
                outcome=2,
                feasible=1,
                quality_reward=1.0,
                archive_hv_gain=0.0,
                reward=1.0,
                stop_reason="horizon" if terminal else None,
                terminal=terminal,
                accepted=True,
                action_seconds=0.1,
            )
        )
        old = values[state, 0]
        future = 0 if terminal else values[1].max()
        values[state, 0] = old + 0.3 * (1 + 0.7 * future - old)
        counts[state, 0] += 1
    q = dict(
        q=values.tolist(),
        selections=counts.tolist(),
        updates=counts.tolist(),
        visits=counts.sum(1).tolist(),
    )

    def write():
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    write()
    result = ANALYSIS.audit_history(path, "HISTORY_ACTION", q)
    assert sum(r["steps"] for r in result) == 5
    # A terminal bootstrap leak changes a previously learned, nonzero next state.
    q["q"][1][0] += 0.3 * 0.7 * values[1, 0]
    with pytest.raises(AssertionError):
        ANALYSIS.audit_history(path, "HISTORY_ACTION", q)
    q["q"] = values.tolist()
    rows[0]["previous_action"] = 1
    write()
    with pytest.raises(AssertionError):
        ANALYSIS.audit_history(path, "HISTORY_ACTION", q)


def test_flow_window_supplement_uses_actual_points_without_interpolation(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "history_supplement",
        Path(__file__).parents[1] / "research/history_rl_results_20261023/supplement_results.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    data, output = tmp_path / "data", tmp_path / "output"
    output.mkdir()
    fronts, rows, snaps = {}, [], []
    for n in (20, 50, 100, 200):
        for arm in module.ARMS:
            key = f"n{n}"
            rows.append(dict(key=key, arm=arm, base_seed=1, n=n, min_flow=1, min_bill=0.5))
            fronts[key + "|" + arm] = dict(reference="common", archive=[[1, 1], [1.2, 0.5]])
            p = data / "raw/formal/runs" / key / arm
            p.mkdir(parents=True)
            (p / "qtable.json").write_text(json.dumps(dict(selections=[[0] * 8])))
            snaps.append(
                dict(n=n, arm=arm, target={20: 300, 50: 600, 100: 1200, 200: 2400}[n], hv=0.5)
            )
    for name, values in (("run_metrics.csv", rows), ("anytime_metrics.csv", snaps)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (output / "normalized_fronts.json").write_text(json.dumps(fronts))
    (output / "references.json").write_text(
        json.dumps(dict(common=dict(ideal=[0, 0], span=[1, 1])))
    )
    result = module.supplement(data, output)
    assert result["rows"] == 48 and result["descriptive_only"]
    assert all(row["fast_window_bill_saving_pct"] == 0 for row in result["summaries"])
