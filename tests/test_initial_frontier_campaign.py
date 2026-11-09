"""Transactional completion refuses tampering and mismatched source identities."""

import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import pytest

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.io.loaders import save_instance

SCRIPT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT))
spec = importlib.util.spec_from_file_location(
    "initial_frontier_campaign", SCRIPT / "run_initial_frontier.py"
)
assert spec and spec.loader
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_completion_marker_rejects_tampered_artifact(tmp_path):
    root = tmp_path
    spec = {"source_commit": "a", "source_hash": "b"}
    atomic_json(root / "specs/k.json", spec)
    folder = root / "runs/k/MIXED"
    atomic_json(folder / "summary.json", spec)
    atomic_json(
        folder / "complete.json",
        {
            "host": platform.node(),
            "spec_sha256": file_hash(root / "specs/k.json"),
            "files": {"summary.json": file_hash(folder / "summary.json")},
        },
    )
    assert campaign.complete(root, "k", "MIXED", spec)
    atomic_json(folder / "summary.json", {"source_commit": "bad"})
    with pytest.raises(ValueError, match="artifact"):
        campaign.complete(root, "k", "MIXED", spec)


def test_completion_marker_requires_same_host(tmp_path):
    atomic_json(tmp_path / "specs/k.json", {})
    atomic_json(
        tmp_path / "runs/k/MIXED/complete.json",
        {"host": "another-host", "spec_sha256": file_hash(tmp_path / "specs/k.json"), "files": {}},
    )
    with pytest.raises(ValueError, match="hosts"):
        campaign.complete(tmp_path, "k", "MIXED", {})


def test_missing_completion_is_not_success(tmp_path):
    assert not campaign.complete(tmp_path, "k", "MIXED", {})


def test_freeze_pairing_disjoint_bases_and_repeat_order(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(campaign, "git_metadata", lambda: {"git_dirty": False, "git_commit": "a"})
    monkeypatch.setattr(campaign, "canonical_source_hash", lambda: "b")
    # The unit fixture only tests the design; real 50/100/200 inputs are validated separately.
    monkeypatch.setattr(campaign, "diagnostic_problem", lambda *args: problem)
    matrices = [campaign.prepare(tmp_path / str(node), node, seed_count=3) for node in (0, 1)]
    assert sum(m["block_count"] for m in matrices) == 216
    assert sum(m["population_count"] for m in matrices) == 2376
    bases = []
    for node, matrix in enumerate(matrices):
        rows = [
            json.loads((tmp_path / str(node) / "specs" / (k + ".json")).read_text())
            for k in matrix["blocks"]
        ]
        assert [r["seed"] for r in rows] == sorted(r["seed"] for r in rows)
        assert all(set(r["methods"]) == set(campaign.METHODS) for r in rows)
        bases.append({r["base_seed"] for r in rows})
        assert len(bases[-1]) == 18
        assert {r["jobs"] for r in rows} == {50, 100, 200}
        assert {r["scenario"] for r in rows} == set(campaign.SCENARIOS)
        for name, digest in matrix["input_files"].items():
            assert file_hash(tmp_path / str(node) / name) == digest
    assert bases[0].isdisjoint(bases[1])


def test_worker_publishes_exact_checked_paired_methods(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(campaign, "git_metadata", lambda: {"git_dirty": False, "git_commit": "a"})
    monkeypatch.setattr(campaign, "canonical_source_hash", lambda: "b")
    save_instance(problem, tmp_path / "input.json")
    row = {
        "key": "k",
        "instance": "input.json",
        "instance_sha256": file_hash(tmp_path / "input.json"),
        "source_commit": "a",
        "source_hash": "b",
        "seed": 17,
        "population": 10,
        "perturbation": 0.1,
        "attempt_limit": 10000,
        "methods": campaign.METHODS,
    }
    atomic_json(tmp_path / "specs/k.json", row)
    campaign.worker(tmp_path, "k")
    for method in campaign.METHODS:
        assert campaign.complete(tmp_path, "k", method, row)
        summary = json.loads((tmp_path / "runs/k" / method / "summary.json").read_text())
        assert summary["independent_exact_verified"] == 10 and summary["population_complete"]
    # Resumption reuses same-host, checked artifacts without generating new candidates.
    before = file_hash(tmp_path / "runs/k/MIXED/summary.json")
    campaign.worker(tmp_path, "k")
    assert file_hash(tmp_path / "runs/k/MIXED/summary.json") == before


def test_failed_worker_stops_both_shards_without_retry(tmp_path, monkeypatch):
    root = tmp_path / "node0"
    atomic_json(
        root / "manifest.json",
        {"source_commit": "a", "source_hash": "b", "input_files": {}, "blocks": ["k", "j"]},
    )
    atomic_json(root / "manifest.sha256.json", {"sha256": file_hash(root / "manifest.json")})
    for k in ("k", "j"):
        atomic_json(root / "specs" / (k + ".json"), {"methods": ["MIXED"]})
    monkeypatch.setattr(campaign, "canonical_source_hash", lambda: "b")
    monkeypatch.setattr(campaign, "git_metadata", lambda: {"git_commit": "a"})
    monkeypatch.setattr(
        campaign, "resources", lambda _: {"available_memory_gib": 7, "free_disk_gib": 30}
    )
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    launches = []

    class Failed:
        pid = 1

        def poll(self):
            return 1

    def launch(*args, **kwargs):
        launches.append(args)
        return Failed()

    monkeypatch.setattr(campaign.subprocess, "Popen", launch)
    campaign.run(root, 1, time.time() + 60)
    assert len(launches) == 1
    assert (tmp_path / "stop.request").exists()
    assert (root / "ops/failures/k.json").exists()
    result = json.loads((root / "ops/exit.json").read_text())
    assert result["state"] == "failed" and result["pending_blocks"] == 1
