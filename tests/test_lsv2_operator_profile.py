"""Known-answer separation of no proposal, no improvement, and negative net reward."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "operator_profile",
    Path(__file__).resolve().parents[1] / "research/lsv2_budget12/operator_profile.py",
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_negative_reward_does_not_imply_empty_or_no_improvement():
    rows = []
    for proposals, gain, reward, seconds in [
        (0, 0, -1, 1),
        (1, 0, -2, 2),
        (1, 0.1, -3, 3),
        (1, 1, 10, 4),
    ]:
        rows.append(
            dict(
                proposals=proposals,
                hv_gain=gain,
                q_reward=reward,
                seconds=seconds,
                entered_archive=int(gain > 0),
                chain="stay",
                available=["A1"],
                bandit={"probability": 1},
            )
        )
    result = module._summary(rows, 10)
    assert result["empty"] == 0.25
    assert result["nonempty_no_hv"] == 0.25
    assert result["positive_hv"] == 0.5
    assert result["negative"] == 0.75
    assert result["positive_hv_but_negative"] == 0.25
    assert result["tail_events"] == 1
    assert result["positive_mass_top1"] == 1
    assert result["empty_seconds"] == 1
    assert result["positive_reward_mean"] == 10


def test_uniform_has_no_invented_net_reward_or_empty_cost():
    rows = [dict(proposals=1, hv_gain=0, seconds=0.1, entered_archive=0, chain="stay")]
    result = module._summary(rows, float("inf"))
    assert "negative" not in result
    assert result["empty_seconds"] is None
    assert result["seconds_mean"] == pytest.approx(0.1)
