"""Acceptance of compressed production artifacts, matrix pairing and tamper checks."""

import gzip
import importlib
import json
from pathlib import Path

import pytest
from test_lsv2_fast_rightshift import CAL, synthetic

from geo_llm_scheduler.experiments.trans_runner import digest, run_case
from geo_llm_scheduler.io.loaders import save_instance


def test_production_diagnostics_and_compressed_trace(tmp_path):
    p = tmp_path / "i.json"
    save_instance(synthetic(4)[0], p)
    out = tmp_path / "out"
    run_case(
        "A4_DIRECT_SHIFT",
        p,
        out,
        seconds=10,
        calibration=CAL,
        max_generations=2,
        campaign_output=True,
    )
    assert json.loads((out / "q_table.json").read_text()) is None
    with gzip.open(out / "local_search.json.gz", "rt") as f:
        records = json.load(f)
    assert not records["steps"] and records["polish"]
    assert (out / "checkpoints.jsonl").is_file()
    assert json.loads((out / "heartbeat.json").read_text())["exact"] >= 1
    assert all(
        digest(out / x["path"]) == x["sha256"]
        for x in json.loads((out / "manifest.json").read_text())
    )


def test_seed_one_matrix_has_every_size_fold_and_arm_and_rejects_corruption(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("run_s4_panel")
    panel = tmp_path / "panel"
    (panel / "instances").mkdir(parents=True)
    rows = []
    folds = {}
    for n in (50, 100, 200):
        for f in range(2):
            folds[f"n{n}_f{f}"] = dict(CAL, keys=[])
        for r in range(40):
            key = f"mx_n{n:03d}_{r:02d}"
            rel = f"instances/{key}.json"
            (panel / rel).write_text("{}")
            rows.append(dict(key=key, path=rel, jobs=n, sha256=digest(panel / rel)))
    (panel / "manifest.json").write_text(json.dumps(dict(instances=rows)))
    cal = tmp_path / "folds.json"
    cal.write_text(json.dumps(folds))
    monkeypatch.setattr(runner, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(runner, "CAL", digest(cal))
    root = tmp_path / "campaign"
    runner.prepare(root, panel, cal, "example")
    manifest = runner.validate_inputs(root)
    assert manifest["runs"] == 14 * 120 and manifest["solver_seconds"] == 2352000
    for i, r in enumerate(rows):
        spec = json.loads((root / "specs" / (r["key"] + ".json")).read_text())
        assert spec["calibration_fold"] == f"n{r['jobs']}_f{i % 2}" and spec["seed"] == 1
    (root / rows[0]["path"]).write_text('{"bad":true}')
    with pytest.raises(AssertionError):
        runner.validate_inputs(root)
