"""Recovery integrity and hand-calculated analysis regression cases."""

import importlib.util
import json
from pathlib import Path

import pytest


def _module(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_roundtrip_and_tamper_rejection(tmp_path):
    module = _module("recover_20h_results")
    original = tmp_path / "original"
    original.mkdir()
    (original / "finished.json").write_text(json.dumps({"commit": "frozen"}))
    (original / "trace.jsonl").write_bytes(b'{"generation": 1}\n' * 20)
    assets, recovered = tmp_path / "assets", tmp_path / "recovered"
    module.pack(original, assets)
    module.unpack(assets, recovered, tmp_path / "report.json")
    assert (original / "trace.jsonl").read_bytes() == (recovered / "trace.jsonl").read_bytes()
    assert json.loads((tmp_path / "report.json").read_text())["verified_files"] == 2
    (recovered / "trace.jsonl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        module.unpack(assets, recovered, tmp_path / "report.json")
    shard = assets / "raw_part_01.tar.gz"
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Transfer checksum"):
        module.unpack(assets, tmp_path / "other", tmp_path / "report.json")


def test_paired_statistics_and_multiple_testing_golden():
    module = _module("analyze_200gen_results")
    result = module.paired_stats([1.0] * 10)
    assert result["mean"] == result["ci_low"] == result["ci_high"] == 1
    assert result["p_raw"] == 2 / 1024
    assert result["wins"] == 10
    zero = module.paired_stats([0.0] * 10)
    assert zero["p_raw"] == 1 and zero["ties"] == 10
    assert module.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_common_reference_and_dominated_points_golden():
    module = _module("analyze_200gen_results")
    assert module.nondominated([(1, 2), (2, 1), (3, 3), (1, 2)]) == [(1, 2), (2, 1)]
    summary = {
        "elapsed": 1,
        "counts": {},
        "archive_size": 2,
        "archive_peak_size": 2,
        "timings": {},
        "trigger_rate": 0,
        "trigger_success_rate": 0,
    }
    group = [
        {
            "key": "x",
            "scope": "validation",
            "arm": "V1",
            "instance": 111,
            "algorithm_seed": 101,
            "summary": summary,
            "points": [(1, 2), (2, 1)],
        }
    ]
    rows, reference = module.evaluate_group(group)
    assert reference["ideal"] == [1, 1]
    assert reference["scale"] == [1, 1]
    assert rows[0]["hv"] == pytest.approx(0.21)
    assert rows[0]["igd_plus"] == 0
