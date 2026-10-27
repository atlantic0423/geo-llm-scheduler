"""Sealed checkpoint corruption and instance-crossfit task preparation."""

import argparse
import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def runner():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("qscreen_runner", scripts / "run_lsv2_qscreen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(scripts))


def test_seal_detects_missing_or_corrupted_checkpoint(tmp_path, runner):
    assert runner._sealed(tmp_path) is None
    (tmp_path / "q.json").write_text('{"values":{}}')
    runner._seal(tmp_path, ["q.json"], commit="abc")
    assert runner._sealed(tmp_path)["commit"] == "abc"
    (tmp_path / "q.json").write_text('{"values":{1}}')
    with pytest.raises(AssertionError):
        runner._sealed(tmp_path)


def test_prepare_separates_whole_instances_and_persists_train_only_calibration(tmp_path, runner):
    old = tmp_path / "historical"
    files, features, rows = {}, [], []
    for n in (20, 50, 100, 200):
        for i in range(4):
            key = f"st_n{n}_d{i:02d}_s16901"
            name = f"inputs/st_n{n}_d{i:02d}.json"
            path = old / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            files[name] = runner.sha(path)
            spec = dict(
                key=key,
                n=n,
                panel="dev",
                input=name,
                input_sha256=files[name],
                base_seed=f"st_n{n}_d{i:02d}",
                seconds={20: 300, 50: 600, 100: 1200, 200: 2400}[n],
            )
            sp = old / "specs" / (key + ".json")
            runner.atomic_json(sp, spec)
            files[sp.relative_to(old).as_posix()] = runner.sha(sp)
            features.append(
                dict(key=key, prefill_window_fraction=i / 4, resource_wait_fraction=i + 0.1)
            )
            rows.append(dict(key=key, gain=i + 1, seconds=2))
    runner.atomic_json(old / "manifest.json", dict(files=files))
    runner.atomic_json(old / "manifest.sha256.json", dict(sha256=runner.sha(old / "manifest.json")))
    steps = tmp_path / "steps.csv.gz"
    with gzip.open(steps, "wt", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    feature_path = tmp_path / "features.csv"
    with feature_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)
    root = tmp_path / "campaign"
    runner._prepare(
        argparse.Namespace(
            root=root, historical=old, steps=steps, features=feature_path, host="test"
        )
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["jobs"]) == 176
    assert len(manifest["blocks"]) == 64
    assert sum(len(j["runs"]) for j in manifest["jobs"]) == 736
    for job in manifest["jobs"]:
        if job["arm"] not in runner.LEARNING_ARMS:
            continue
        assert [r["training"] for r in job["runs"]] == [True] * 4 + [False] * 4
        train = {r["key"].rsplit("_s", 1)[0] for r in job["runs"][:4]}
        valid = {r["key"].rsplit("_s", 1)[0] for r in job["runs"][4:]}
        assert len(train) == len(valid) == 2 and not train & valid
        assert {s.rsplit("_s", 1)[0] for s in job["calibration"]["keys"]} == train
