"""Load-power S4 versus no-shift screen: sealed plan, power-model and drift checks."""

import importlib
import json
from pathlib import Path

import pytest

from geo_llm_scheduler.experiments.trans_runner import digest

CAL = dict(rho=0.01, h=1.0, w_cutoff=0.5)


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_s4_load_shift")


def _panel(tmp_path: Path, power: dict) -> tuple[Path, Path, dict]:
    panel = tmp_path / "panel"
    (panel / "instances").mkdir(parents=True)
    rows, folds, index = [], {}, {}
    for n in (50, 100, 200):
        for f in range(2):
            folds[f"n{n}_f{f}"] = dict(CAL, keys=[])
        keys = sorted(
            f"mx_n{n:03d}_f{fam:02d}_r{r:02d}" for fam in range(1, 11) for r in range(1, 5)
        )
        for i, key in enumerate(keys):
            rel = f"instances/{key}.json"
            (panel / rel).write_text("{}")
            rows.append(dict(key=key, path=rel, jobs=n, sha256=digest(panel / rel)))
            index[key] = i
    (panel / "manifest.json").write_text(json.dumps(dict(power_model=power, instances=rows)))
    cal = tmp_path / "folds.json"
    cal.write_text(json.dumps(folds))
    return panel, cal, index


def test_plan_covers_panel_with_both_arms_and_seals_power(tmp_path, monkeypatch, runner):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, index = _panel(tmp_path, power)
    monkeypatch.setattr(runner, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(runner, "CAL", digest(cal))
    root = tmp_path / "campaign"
    runner.prepare(root, panel, cal, "example")
    manifest = runner.validate_inputs(root)
    assert manifest["runs"] == 240 and len(manifest["blocks"]) == 120
    assert manifest["power_model"] == power and manifest["window_seconds"] > 0
    orders = set()
    for block in manifest["blocks"]:
        spec = json.loads((root / "specs" / (block + ".json")).read_text())
        assert spec["calibration_fold"] == f"n{spec['n']}_f{index[spec['instance']] % 2}"
        assert spec["power_model"] == power and spec["seed"] == 1
        assert spec["config"]["A0_FULL_S4"]["lsv2"]["polish_method"] == "common_fast"
        assert spec["config"]["A1_NO_SHIFT"]["lsv2"]["polish_region"] == "off"
        orders.add(tuple(spec["order"]))
    assert orders == {tuple(runner.ARMS), tuple(reversed(runner.ARMS))}
    (root / "specs" / (manifest["blocks"][0] + ".json")).write_text("{}")
    with pytest.raises(AssertionError):
        runner.validate_inputs(root)


def test_prepare_rejects_a_panel_without_the_declared_power_model(tmp_path, monkeypatch, runner):
    panel, cal, _ = _panel(tmp_path, dict(beta=1.0, kappa_prefill=1.0, kappa_decode=1.0))
    monkeypatch.setattr(runner, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(runner, "CAL", digest(cal))
    with pytest.raises(AssertionError):
        runner.prepare(tmp_path / "campaign", panel, cal, "example")
