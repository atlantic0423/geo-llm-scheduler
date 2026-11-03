"""Check false veto handling and call/recipe/candidate denominator separation."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "d4_result_audit", Path(__file__).parents[2] / "scripts/analyze_d4_results.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_empty_recipes_and_inapplicable_nonempty_remain_distinct() -> None:
    row = {
        "empty": True,
        "exact": 0,
        "bill_improved": 0,
        "scalar_improved": 0,
        "gain": 0,
        "construction_seconds": 3,
        "evaluation_seconds": 0,
        "audit_seconds": 1,
        "recipes": [
            {
                "members": [],
                "stage": "empty_members",
                "certificate": {"denied": False, "applicable": True, "seconds": 0.2},
            },
            {
                "members": [1],
                "stage": "resource_positions",
                "repair_seconds": 2,
                "certificate": {"denied": False, "applicable": False, "seconds": 0.3},
            },
            {
                "members": [2],
                "stage": "peak_gate",
                "repair_seconds": 1,
                "certificate": {"denied": True, "applicable": True, "seconds": 0.4},
            },
        ],
    }
    result = module.summarize_calls([row])
    assert result["counts"]["calls"] == result["counts"]["empty"] == 1
    assert result["counts"]["recipes"] == 3
    assert result["counts"]["nonempty_recipes"] == 2
    assert result["counts"]["prunable_recipes"] == 1
    assert result["costs"]["denied_repair_seconds"] == 1
    row["recipes"][-1]["stage"] = "proposal"
    with pytest.raises(ValueError, match="false veto"):
        module.summarize_calls([row])


def test_equal_base_gate_does_not_weight_large_recipe_counts() -> None:
    result = module.base_bootstrap([0.0, 1.0])
    assert result["mean"] == 0.5
    assert result["ci"] == [0.0, 1.0]
    assert result["candidate_gate"] is False
