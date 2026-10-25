"""Matrix isolation, balanced order, native prices and sealed run corruption."""

import json
import socket
import sys
from collections import Counter
from pathlib import Path

import pytest

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.initial_bill_speed import ARMS, speed_problem, speed_specs
from geo_llm_scheduler.io.loaders import save_instance


def test_matrix_independent_cases_pairing_scale_and_order():
    specs = speed_specs()
    assert len(specs) == 64 and len({s["base_seed"] for s in specs}) == 32
    assert {s["n"] for s in specs} == {20, 50, 100, 200}
    assert {s["machines"] for s in specs} == {12, 24}
    assert len({s["key"] for s in specs}) == 64
    for pos in range(4):
        assert Counter(s["order"][pos] for s in specs) == dict.fromkeys(ARMS, 16)
    assert not ({s["base_seed"] for s in specs} & {s["base_seed"] for s in speed_specs(pilot=True)})
    for seed in {s["base_seed"] for s in specs}:
        pair = [s for s in specs if s["base_seed"] == seed]
        assert len(pair) == 2 and len({(s["n"], s["scenario"], s["machines"]) for s in pair}) == 1


def test_native_prices_and_pressure():
    for scenario in ("mixed", "burst", "vram_tight", "demand_high"):
        p = speed_problem(20, 531, 24, scenario)
        assert len({tuple(t.price for t in r.tariffs) for r in p.regions}) == 3
        assert p == speed_problem(20, 531, 24, scenario)
        if scenario == "burst":
            assert all(j.release == 0 for j in p.jobs)
        if scenario == "vram_tight":
            assert all(m.vram == 8 for m in p.instances)
        if scenario == "demand_high":
            assert p.regions[0].demand_rate == 20
    with pytest.raises(ValueError):
        speed_problem(500, 5, 9, "mixed")


def test_frozen_matrix_tamper_rejected(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_initial_bill_speed as runner

    tiny = speed_specs(pilot=True)[:1]
    monkeypatch.setattr(runner, "speed_specs", lambda **kw: tiny)
    m = runner.prepare(tmp_path, pilot=True)
    assert m["runs"] == 4 and runner.validate_inputs(tmp_path) == m
    with pytest.raises(ValueError, match="overwrite"):
        runner.prepare(tmp_path, pilot=True)
    path = tmp_path / next(name for name in m["files"] if name.startswith("inputs"))
    path.write_text("{}")
    with pytest.raises(AssertionError):
        runner.validate_inputs(tmp_path)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux resource/affinity campaign")
def test_worker_complete_strict_block_and_tamper(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_initial_bill_speed as runner

    spec = dict(
        speed_specs(pilot=True)[0], n=6, key="tiny", host=socket.gethostname(), input="input.json"
    )
    save_instance(speed_problem(6, spec["base_seed"], 12, "mixed"), tmp_path / "input.json")
    spec["input_sha256"] = runner.digest(tmp_path / "input.json")
    atomic_json(tmp_path / "specs/tiny.json", spec)
    source = runner.source_digest()
    atomic_json(
        tmp_path / "manifest.json",
        {
            "source": source,
            "files": {"specs/tiny.json": runner.digest(tmp_path / "specs/tiny.json")},
        },
    )
    # Run isolated child processes: worker's deliberate RLIMIT_AS never affects pytest.
    import subprocess

    for arm in ARMS:
        assert not runner.complete(tmp_path, "tiny", arm, source)
        subprocess.run(
            [
                sys.executable,
                runner.__file__,
                "worker",
                "--root",
                str(tmp_path),
                "--key",
                "tiny",
                "--arm",
                arm,
            ],
            check=True,
        )
    assert runner.validate_block(tmp_path, "tiny", source)
    summary = json.loads((tmp_path / "runs/tiny/FAST2/summary.json").read_text())
    assert summary["new_candidate_exact"] == 20
    (tmp_path / "runs/tiny/FAST2/population.json").write_text("[]")
    with pytest.raises(AssertionError, match="population"):
        runner.complete(tmp_path, "tiny", "FAST2", source)
