"""Check independent-base weighting and avoid normalization-only prefix gains."""

from pathlib import Path

import pytest

from research.p1.analyze import cluster_interval, metric
from research.p1.reduce import portable_path, prefix_improvement, score


def test_analysis_clusters_do_not_count_repeated_candidates_as_new_bases():
    rows = [{"base_seed": 810001, "x": 0}] * 100 + [{"base_seed": 820001, "x": 1}]
    result = metric(rows, lambda r: r["x"])
    assert result["mean"] == 0.5
    assert result["bases"] == 2
    assert result["ci"] == [0.5, 0.5]


def test_bootstrap_is_reproducible_and_stratified():
    values = {810001: 0.0, 810002: 2.0, 820001: 10.0, 820002: 12.0}
    result = cluster_interval(values)
    assert result == cluster_interval(values)
    assert result["mean"] == 6
    assert result["ci"][0] >= 5
    assert result["ci"][1] <= 7


def test_prefix_gain_uses_the_same_later_context():
    parent = (6.0, 6.0)
    candidates = [(4.0, 4.0), (20.0, 0.0)]
    old_ideal, new_ideal, maximum = (0.0, 3.0), (0.0, 0.0), (10.0, 10.0)
    assert score(candidates[0], old_ideal, maximum, 25) != score(
        candidates[0], new_ideal, maximum, 25
    )
    assert prefix_improvement(parent, candidates, new_ideal, maximum, 25, 1, 2) == 0
    assert prefix_improvement(parent, [(4, 4), (3, 3)], new_ideal, maximum, 25, 1, 2) > 0


def test_relocation_preserves_archive_paths_and_rejects_other_campaigns(tmp_path: Path):
    remote = "/workspace/zhouhanyu/campaign_p1_20261002_node0/inputs/example.json"
    assert portable_path(tmp_path, remote) == (
        tmp_path / "campaign_p1_20261002_node0/inputs/example.json"
    )
    with pytest.raises(ValueError):
        portable_path(tmp_path, "/workspace/zhouhanyu/campaign_e15/input.json")
