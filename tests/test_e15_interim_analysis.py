"""Golden and boundary checks for independent-instance interim statistics."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "e15_interim", Path(__file__).resolve().parents[1] / "scripts/e15_interim_analysis.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_base_pairing_averages_seeds_and_tariffs() -> None:
    rows = [
        {"stage": "A_GEN", "base": base, "arm": arm, "hv": base + seed + effect}
        for base in (1, 2)
        for seed in (1, 2, 3)
        for _tariff in ("H", "T")
        for arm, effect in (("F6", 0), ("SEQ", 2))
    ]
    assert MODULE.cluster_differences(rows, "A_GEN", "F6", "SEQ", "hv") == [2, 2]
    with pytest.raises(ValueError, match="Pairing"):
        MODULE.cluster_differences(rows[:-1], "A_GEN", "F6", "SEQ", "hv")


def test_sign_test_bootstrap_and_holm_golden() -> None:
    result = MODULE.summarize_differences([1.0] * 12)
    assert result["sign_p"] == 2 / 4096
    assert result["bootstrap_mean_ci95"] == [1, 1]
    assert MODULE.summarize_differences([0, 0])["sign_p"] == 1
    assert MODULE.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    with pytest.raises(ValueError):
        MODULE.summarize_differences([float("nan")])


def test_analysis_refuses_duplicate_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "COMPLETE", {"A_GEN": 2})
    (tmp_path / "ops").mkdir()
    row = {"stage": "A_GEN", "instance": 1, "seed": 101, "arm": "F6"}
    (tmp_path / "ops/completed_stage_metrics_20261001.json").write_text(json.dumps([row, row]))
    with pytest.raises(ValueError, match="duplicated"):
        MODULE.analyze(tmp_path, tmp_path / "derived")
