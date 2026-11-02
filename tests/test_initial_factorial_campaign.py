"""Four-way provenance and retention invariance of the sealed campaign."""

import importlib
import json
import socket
from pathlib import Path

import pytest
from test_lsv2_fast_rightshift import CAL, synthetic

from geo_llm_scheduler.experiments.trans_runner import digest
from geo_llm_scheduler.io.loaders import save_instance


def test_four_way_campaign_seals_folds_and_preflight_streams(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("run_initial_factorial")
    ref = tmp_path / "ref"
    (ref / "instances").mkdir(parents=True)
    (ref / "specs").mkdir()
    key = "test"
    path = ref / "instances/test.json"
    save_instance(synthetic(4)[0], path)
    (ref / "manifest.json").write_text(json.dumps(dict(blocks=[key])))
    (ref / "panel_manifest.json").write_text("{}")
    (ref / "calibration.json").write_text(json.dumps(dict(fold=CAL)))
    spec = dict(
        key=key,
        n=4,
        seed=1,
        input="instances/test.json",
        input_sha256=digest(path),
        seconds=30,
        calibration_fold="fold",
    )
    (ref / "specs/test.json").write_text(json.dumps(spec))
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(dict(blocks=[key], window_seconds=100, deadline_epoch=9999999999))
    )
    monkeypatch.setattr(runner, "PANEL", digest(ref / "panel_manifest.json"))
    monkeypatch.setattr(runner, "CAL", digest(ref / "calibration.json"))
    monkeypatch.setitem(runner.s4_ablation.BUDGETS, 4, 30)
    root = tmp_path / "campaign"
    runner.prepare(root, ref, selection, socket.gethostname())
    assert runner.validate_inputs(root)["runs"] == 4
    runner.preflight(root, key)
    check = json.loads((root / "preflight/test.json").read_text())
    assert set(check["policies"]) == set(runner.ARMS)
    assert all(p["fresh_exact"] >= 5 for p in check["policies"].values())
    # A regenerated hash cannot conceal a policy change from the explicit config check.
    sp = root / "specs/test.json"
    data = json.loads(sp.read_text())
    data["config"][runner.ARMS[2]]["initial_retention"] = "global_pool"
    sp.write_text(json.dumps(data))
    mp = root / "manifest.json"
    m = json.loads(mp.read_text())
    m["files"]["specs/test.json"] = digest(sp)
    mp.write_text(json.dumps(m))
    (root / "manifest.sha256.json").write_text(json.dumps(dict(sha256=digest(mp))))
    with pytest.raises(AssertionError):
        runner.validate_inputs(root)
