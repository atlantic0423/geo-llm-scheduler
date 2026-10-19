"""Counterexamples for scale weighting, factorial effects and HV boundaries."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
analysis = importlib.import_module("analyze_comparison13_results")
diagnosis = importlib.import_module("diagnose_comparison13_failure")


def test_task_scale_weight_is_not_run_weight() -> None:
    """An overrepresented small scale cannot dominate the four-scale estimator."""
    assert analysis.equal_scale_mean([10, 10, 0, 0, 0], [20, 20, 50, 100, 200]) == 2.5
    with pytest.raises(ValueError):
        analysis.equal_scale_mean([1, 2], [20, 50])


def test_factorial_distinguishes_interaction_from_main_effect() -> None:
    scores = dict.fromkeys(analysis.ARMS, 0.0)
    scores.update(M00_I4=1, M10_I4=3, M01_I4=4, M11_I4=10)
    effects = analysis.contrast_effects(scores)
    assert effects["A3"] == 4
    assert effects["RPERM"] == 5
    assert effects["interaction"] == 4


def test_hv_outside_reference_is_zero_without_clipping() -> None:
    front = [(0.2, 0.8), (0.8, 0.2), (1.2, 0), (0, 1.3)]
    assert analysis.hv_score(front) == pytest.approx(0.45)
    assert analysis.hv_score(front) == pytest.approx(analysis.independent_hv(front))
    assert analysis.hv_score([(2, 0), (0, 2)]) == 0
    assert analysis.distance([(2, 2)], [(0, 0)], True) == pytest.approx(8**0.5)


def test_parent_resampling_keeps_scales_and_rng_explicit() -> None:
    sizes = [20, 20, 50, 100, 200]
    values = [1, 1, 2, 3, 4]
    first = analysis.cluster_statistics(values, sizes, np.random.default_rng(9), 100, 200)
    second = analysis.cluster_statistics(values, sizes, np.random.default_rng(9), 100, 200)
    assert first == second
    assert first["mean"] == 2.5
    assert first["ci95"] == [2.5, 2.5]
    assert first["p_signflip"] > 0


def test_serialized_bill_uses_tou_plus_regional_demand(tmp_path: Path) -> None:
    import gzip

    path = tmp_path / "archive.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump([dict(evaluation=dict(feasible=True, flow=10, tou=2, demand=[3, 4]))], stream)
    assert analysis.points(path) == [(10, 9)]


def test_holm_uses_whole_predeclared_family() -> None:
    assert [len(v) for v in analysis.FAMILIES.values()] == [10, 6, 4, 4]
    assert analysis.holm([0.03, 0.01, 0.04]) == pytest.approx([0.06, 0.03, 0.06])


def test_failed_initializer_replays_without_relaxing_constraints(tmp_path: Path) -> None:
    """A real small-case duplicate exhaustion is initialization, before IBEA starts."""
    from geo_llm_scheduler.experiments.comparison13 import case_problem
    from geo_llm_scheduler.io.loaders import save_instance

    problem = case_problem(20, 5, 12, 202610160220, "cheap_capacity", 1)
    save_instance(problem, tmp_path / "input.json")
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "r0_n20_R5M12_cheap_capacity_v1_s16101.json").write_text(
        json.dumps(dict(input="input.json", seed=16101, bank_seed=116101, seconds=300))
    )
    result = diagnosis.diagnose(tmp_path, tmp_path / "diagnosis.json")
    assert result["reproduced"]
    assert result["failure"]["slot"] == 2
    assert result["failure"]["attempt"] == 99
    assert result["failure"]["pool_size"] == 3
    assert result["failure"]["required"] == 4
    assert result["exact"] == 111
    assert result["slots"][-1]["calls"] == 100
