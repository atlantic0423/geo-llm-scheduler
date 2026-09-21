"""Reject malformed experiment definitions before consuming search resources."""

import pytest

from geo_llm_scheduler.config import Config, load_config


@pytest.mark.parametrize(
    "options",
    [
        {"method": "unknown"},
        {"controller": "unknown"},
        {"trigger_mode": "unknown"},
        {"budget_policy": "unknown"},
        {"seconds": float("nan")},
        {"seconds": 0},
        {"generations": 1.5},
        {"replacement_cap": 0},
        {"budgets": (10, 6, 3)},
        {"budgets": (3, 6)},
        {"static_budgets": (3,)},
        {"enabled_operators": (1, 1)},
        {"enabled_operators": (9,)},
        {"severity_thresholds": (float("nan"),) * 6},
        {"mutation_weights": (0, 0, 0)},
        {"fixed_ls_probability": 2},
        {"a6_destroy_ratio": 0},
        {"trigger_delta": float("inf")},
        {"neighborhood": 101},
    ],
)
def test_bad_config(options):
    with pytest.raises(ValueError):
        Config(**options)


def test_yaml_unknown_and_roundtrip(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("unknown: 2", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
    path.write_text("budgets: [2, 4, 8]\na6_destroy_ratio: 0.2", encoding="utf-8")
    assert load_config(path).budgets == (2, 4, 8)
    assert load_config(path).a6_destroy_ratio == 0.2
