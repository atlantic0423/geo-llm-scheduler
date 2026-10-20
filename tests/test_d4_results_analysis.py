"""Verify paired denominators, label audit and corruption detection in report tooling."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "d4_results", Path(__file__).resolve().parents[1] / "scripts/analyse_d4_results.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_base_bootstrap_and_empty_denominator():
    result = analysis.paired_interval([-1.0, 1.0])
    assert result["mean"] == 0 and result["bases"] == 2
    assert result["ci"] == [-1, 1]
    assert analysis.paired_interval([-1, 1]) == result
    with pytest.raises(ValueError):
        analysis.paired_interval([])
    with pytest.raises(ValueError):
        analysis.paired_interval([float("nan")])
    metrics = analysis.arm_metrics(
        {
            "timing": [{"cpu": 1, "wall": 2}, {"cpu": 3, "wall": 4}],
            "recipes": [{"stage": "resource_positions"}],
            "candidates": [],
            "best_scalar_gain": 0,
            "attempts": 1,
        }
    )
    assert metrics["cpu"] == 2 and metrics["wall"] == 3
    assert metrics["nonempty"] == metrics["positive"] == 0
    assert metrics["recipes"] == 1


def test_scalar_zero_range_uses_original_guard():
    assert analysis.scalar_value([1e-9, 0], [0, 0], [0, 0], [0.5, 0.5]) == 0.5


def fixture(tmp_path: Path):
    root = tmp_path / "raw"
    key = "n50_i875001_H_a1101"
    folder = root / "results" / key
    folder.mkdir(parents=True)
    (root / "inputs").mkdir()
    manifest = {"source_commit": "frozen", "source_hash": "hash", "role": "guard", "keys": [key]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    source = {"evaluation": {"flow": 10, "tou": 2, "demand": [8]}}
    panels = [
        {"sources": [{"candidate": source}], "context": {"maximum": [10, 10]}, "weight": [0.5, 0.5]}
    ]
    (root / "inputs" / f"{key}_panels.json").write_text(json.dumps({"panels": panels}))
    candidate = {
        "genotype": {},
        "schedule": {"starts": [1]},
        "evaluation": {"flow": 8, "tou": 1, "demand": [7]},
    }
    ref = {
        "timing": [{"cpu": 2, "wall": 4}],
        "recipes": [
            {"members": [0], "stage": "peak_gate", "repaired_starts": [0]},
            {"members": [1], "stage": "proposal", "repaired_starts": [1]},
        ],
        "candidates": [candidate],
        "best_scalar_gain": 0.1,
        "attempts": 2,
    }
    guarded = copy.deepcopy(ref)
    guarded["timing"] = [{"cpu": 1, "wall": 2}]
    guarded["recipes"][0] = {"members": [0], "stage": "certificate"}
    unit = {
        "exact_audits": 15,
        "panel": 0,
        "repeat": 0,
        "rows": [
            {
                "source_index": 0,
                "joint_ideal": [0, 0],
                "arms": {
                    "REFERENCE": ref,
                    "LEGACY": copy.deepcopy(ref),
                    "GUARD_ALL": copy.deepcopy(guarded),
                    "GUARD_SINGLE": copy.deepcopy(guarded),
                },
            }
        ],
    }
    path = folder / "p0_s0.json"
    path.write_text(json.dumps(unit))
    marker = {
        "host": "original",
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    (folder / "complete.json").write_text(json.dumps(marker))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({**manifest, "status": "complete_archive_and_each_local_file_verified"})
    )
    return root, receipt, path, marker


def test_full_label_and_counterfactual_audit(tmp_path):
    root, receipt, _, _ = fixture(tmp_path)
    report = analysis.analyse(root, tmp_path / "analysis", receipt)
    assert report["audit_counts"] == {
        "total": 15,
        "source_exact": 1,
        "repaired_exact": 6,
        "proposal_exact": 4,
        "independent_certificate": 4,
    }
    assert report["false_veto"] == 0 and report["output_pairs"] == 2
    result = report["summaries"]["all"]["GUARD_ALL"]
    assert result["cpu_percent_saving"]["mean"] == 50
    assert result["gain_delta"]["mean"] == 0
    with pytest.raises(ValueError, match="outside"):
        analysis.analyse(root, root / "analysis", receipt)


def test_detect_changed_unit_and_wrong_scalar(tmp_path):
    root, receipt, path, marker = fixture(tmp_path)
    data = json.loads(path.read_text())
    data["rows"][0]["arms"]["REFERENCE"]["best_scalar_gain"] = 0.5
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Changed unit"):
        analysis.analyse(root, tmp_path / "a", receipt)
    marker["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (path.parent / "complete.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="Scalar label"):
        analysis.analyse(root, tmp_path / "b", receipt)


def test_peak_proposal_can_have_less_scalar_gain(tmp_path):
    root, receipt, path, marker = fixture(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["role"] = "positions"
    (root / "manifest.json").write_text(json.dumps(manifest))
    unit = json.loads(path.read_text())
    reference = unit["rows"][0]["arms"]["REFERENCE"]
    reference["recipes"] = [reference["recipes"][1]]
    peak = copy.deepcopy(reference)
    peak["candidates"][0]["evaluation"] = {"flow": 6, "tou": 2, "demand": [7]}
    peak["best_scalar_gain"] = 0.05
    unit["rows"][0]["arms"] = {
        "REFERENCE": reference,
        "RANDOM_SCORED": copy.deepcopy(reference),
        "PEAK_RANK": peak,
    }
    unit["exact_audits"] = 7
    path.write_text(json.dumps(unit))
    marker["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (path.parent / "complete.json").write_text(json.dumps(marker))
    report = analysis.analyse(root, tmp_path / "analysis", receipt)
    peak_result = report["summaries"]["all"]["PEAK_RANK"]
    assert peak_result["gain_delta"]["mean"] == pytest.approx(-0.05)
    assert peak_result["nonempty_delta"]["mean"] == 0
    assert report["audit_counts"]["independent_certificate"] == 0
