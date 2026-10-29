"""Sealed right-shift campaign inherits input and calibration validation."""

import argparse
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

from geo_llm_scheduler.experiments import lsv2_fast_rightshift as previous
from geo_llm_scheduler.experiments import lsv2_rightshift_selection as selection
from geo_llm_scheduler.experiments.lsv2_fast_rightshift import RUN_SEED

CAL = dict(rho=0.001, h=0.01, w_cutoff=1.5)


@pytest.fixture(scope="module")
def runner():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "rightshift_campaign_runner", scripts / "run_lsv2_fast_rightshift_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(scripts))


@pytest.mark.parametrize("study,count", [(previous, 80), (selection, 96)])
def test_prepare_reuses_calibration_and_old_seed_and_seals_configs(tmp_path, runner, study, count):
    old, prior = tmp_path / "old", tmp_path / "q"
    files, cals = {}, {}
    for n in (20, 50, 100, 200):
        for i in selection.PANEL.get(n, range(4)):
            key = f"st_n{n}_d{i:02d}_s16901"
            name = f"inputs/st_n{n}_d{i:02d}.json"
            p = old / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
            files[name] = runner.sha(p)
            s = dict(
                key=key,
                seed=RUN_SEED,
                panel="dev",
                n=n,
                input=name,
                input_sha256=files[name],
                seconds={20: 300, 50: 600, 100: 1200, 200: 2400}[n],
            )
            sn = "specs/" + key + ".json"
            runner.atomic_json(old / sn, s)
            files[sn] = runner.sha(old / sn)
            cals[key] = dict(CAL, keys=["other"], validation_keys=[key])
            out = old / "runs" / key / "EXTH_NSGAII"
            runner.atomic_json(
                out / "summary.json",
                dict(
                    initialization=dict(initial_population_sha256="p", initial_archive_sha256="a")
                ),
            )
            runner._seal(out, ["summary.json"])
    runner.atomic_json(old / "manifest.json", dict(files=files))
    runner.atomic_json(old / "manifest.sha256.json", dict(sha256=runner.sha(old / "manifest.json")))
    runner.atomic_json(prior / "calibration.json", cals)
    runner.atomic_json(
        prior / "manifest.json",
        dict(files={"calibration.json": runner.sha(prior / "calibration.json")}),
    )
    runner.atomic_json(
        prior / "manifest.sha256.json", dict(sha256=runner.sha(prior / "manifest.json"))
    )
    root = tmp_path / "new"
    runner._prepare(
        argparse.Namespace(root=root, historical=old, prior_q=prior, host=socket.gethostname()),
        study,
    )
    m = json.loads((root / "manifest.json").read_text())
    assert len(m["jobs"]) == count and len(m["blocks"]) == 8
    assert m["run_seeds"] == [16901]
    assert json.loads((root / "calibration.json").read_text()) == cals
    assert len({j["id"] for j in m["jobs"]}) == count
    assert runner._validate(root, study) == m
    with pytest.raises(AssertionError):
        runner._validate(root, selection if study is previous else previous)
    for job in m["jobs"]:
        assert job["runs"][0]["training"] == (job["arm"] in study.LEARNING_ARMS)
        assert runner.sha(root / job["config_file"]) == m["files"][job["config_file"]]
    # Re-sealing a modified setting cannot bypass semantic configuration validation.
    name = m["jobs"][0]["config_file"]
    config = runner._read(root / name)
    config["lsv2"]["polish_budget"] = 99
    runner.atomic_json(root / name, config)
    m["files"][name] = runner.sha(root / name)
    runner.atomic_json(root / "manifest.json", m)
    runner.atomic_json(
        root / "manifest.sha256.json", dict(sha256=runner.sha(root / "manifest.json"))
    )
    with pytest.raises(AssertionError):
        runner._validate(root, study)
