"""Independent hand-calculated six-threshold relabeling examples."""

from geo_llm_scheduler.experiments.campaign.thresholds import relabel


def test_relabel_normal_relative_winner_and_stable_tie():
    assert relabel((0.2,) * 6, (0.2,) * 6) == 0
    assert relabel((0.1, 0.3, 0, 0, 0, 0), (0.2,) * 6) == 2
    assert relabel((0.4, 0.2, 0, 0, 0, 0), (0.2, 0.1, 0.2, 0.2, 0.2, 0.2)) == 1
