"""Pooled-count diagnostic visualization calculations."""

import pytest
from matplotlib import pyplot as plt

from geo_llm_scheduler.experiments import diagnostic_visualization as visualization
from geo_llm_scheduler.experiments.diagnostic_visualization import summarize_steps


def row(state, action, reward, condition="Normal", scenario="D0", progress=0.2):
    return {
        "state_id": str(state),
        "selected_action": action,
        "reward": str(reward),
        "dominant_condition": condition,
        "scenario": scenario,
        "normalized_progress": str(progress),
    }


def test_selection_probability_uses_pooled_counts_before_ratio():
    rows = [row(0, "A1", 1.0) for _ in range(9)]
    rows += [row(0, "A2", -1.0)]
    rows += [row(1, "A1", 1.0), row(1, "A2", 1.0)]
    summary = summarize_steps(rows)
    assert summary.visits.tolist()[:2] == [10, 2]
    assert summary.selection_probability[0, 0] == pytest.approx(0.9)
    assert summary.selection_probability[0, 1] == pytest.approx(0.1)
    assert summary.success_rate[0, 0] == pytest.approx(1.0)
    assert summary.success_rate[0, 1] == pytest.approx(0.0)
    assert summary.mean_reward[0, 1] == pytest.approx(-1.0)
    assert summary.total_steps == 12


def test_condition_and_progress_counts_remain_explicit():
    summary = summarize_steps(
        [
            row(2, "A3", 0.5, "Resource", "D1_resource", 0.0),
            row(2, "A3", 0.0, "Resource", "D1_resource", 0.999),
        ]
    )
    assert summary.condition_counts["Resource"] == 2
    assert summary.scenario_conditions["D1_resource"]["Resource"] == 2
    assert summary.progress_counts[0, 2] == 1
    assert summary.progress_counts[9, 2] == 1


def test_dataset_figure_contract_contains_six_png_pdf_pairs(monkeypatch, tmp_path):
    summary = summarize_steps([row(0, "A1", 1.0), row(14, "A2", 0.0), row(28, "A3", -1.0)])
    stems = []

    def capture(figure, directory, stem):
        stems.append((directory, stem))
        plt.close(figure)

    monkeypatch.setattr(visualization, "_save", capture)
    visualization.create_dataset_figures(summary, tmp_path, "test")
    assert {stem for _, stem in stems} == {
        "state_visit_frequency",
        "state_action_selection",
        "state_action_positive_improvement",
        "state_action_mean_reward",
        "dominant_condition_incidence",
        "action_usage_over_progress",
    }
