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


def test_three_shards_partition_the_plan_keep_order_and_balance_budget(
    tmp_path, monkeypatch, runner
):
    power = dict(beta=0.4, kappa_prefill=1.0, kappa_decode=0.25)
    panel, cal, _ = _panel(tmp_path, power)
    monkeypatch.setattr(runner, "PANEL", digest(panel / "manifest.json"))
    monkeypatch.setattr(runner, "CAL", digest(cal))
    full_root = tmp_path / "full"
    runner.prepare(full_root, panel, cal, "example")
    full = runner.validate_inputs(full_root)
    assert full["shard"] == "0/1" and full["runs"] == 240
    seen, seconds = [], []
    for k in range(3):
        root = tmp_path / f"shard{k}"
        runner.prepare(root, panel, cal, f"host{k}", f"{k}/3")
        m = runner.validate_inputs(root)
        assert m["shard"] == f"{k}/3" and m["host"] == f"host{k}"
        assert m["runs"] == 2 * len(m["blocks"])
        seen += m["blocks"]
        seconds.append(m["solver_seconds"])
        sizes = [json.loads((root / "specs" / (b + ".json")).read_text())["n"] for b in m["blocks"]]
        assert all(13 <= sizes.count(n) <= 14 for n in (50, 100, 200))
        for block in m["blocks"]:
            a = (root / "specs" / (block + ".json")).read_text()
            assert a == (full_root / "specs" / (block + ".json")).read_text()
    assert sorted(seen) == sorted(full["blocks"]) and len(set(seen)) == 120
    assert sum(seconds) == full["solver_seconds"] and max(seconds) - min(seconds) <= 2 * 2400
    with pytest.raises(AssertionError):
        runner.shard_blocks([], "3/3")


def test_memory_gate_uses_container_limit_and_keeps_shared_host_reserve(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    flat = importlib.import_module("slot_assignment_dispatch")
    gib = 1024**3
    # Small container: the 8GiB cgroup limit binds, host reserve of 220GiB does not.
    assert flat.memory_gate(100 * gib, 900 * gib, 5 * gib, 8 * gib)
    assert not flat.memory_gate(100 * gib, 900 * gib, int(5.7 * gib), 8 * gib)
    assert not flat.memory_gate(100 * gib, 40 * gib, 1 * gib, 8 * gib)
    assert not flat.memory_gate(4 * gib, 900 * gib, 1 * gib, 8 * gib)
    assert not flat.memory_gate(100 * gib, 900 * gib, None, 8 * gib)
    # Shared host or large unit limit: original 220GiB memory and disk reserve.
    assert flat.memory_gate(221 * gib, 221 * gib, 3 * gib, 80 * gib)
    assert not flat.memory_gate(219 * gib, 900 * gib, 3 * gib, 80 * gib)
    assert not flat.memory_gate(221 * gib, 219 * gib, None, None)
    assert flat.memory_gate(221 * gib, 221 * gib, None, None)


def test_container_charge_excludes_reclaimable_file_cache(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    flat = importlib.import_module("slot_assignment_dispatch")
    gib = 1024**3
    # Observed on the 253 containers: 3.1GiB cache from earlier batches plus 18 workers.
    rows = dict(anon=int(2.7 * gib), file=int(3.1 * gib), active_file=int(2.6 * gib), shmem=0)
    stat = "".join(f"{k} {v}" + chr(10) for k, v in rows.items())
    current = int(5.8 * gib)
    assert not flat.memory_gate(100 * gib, 900 * gib, current, 8 * gib)
    charged = flat.charged_memory(current, stat)
    assert abs(charged - 2.7 * gib) < 1024
    assert flat.memory_gate(100 * gib, 900 * gib, charged, 8 * gib)
    # Shared memory sits inside ``file`` but is not reclaimable.
    assert flat.charged_memory(6 * gib, f"file {4 * gib}" + chr(10) + f"shmem {3 * gib}") == 5 * gib
    assert flat.charged_memory(gib, f"file {2 * gib}") == 0
    assert flat.charged_memory(gib, "") == gib
