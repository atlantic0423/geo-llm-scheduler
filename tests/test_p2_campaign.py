"""Portable artifact identity, exact worker results, and interrupted-triad recovery."""

import json
import runpy
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import p2_worker, runner
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.utils.rng import RNGManager

SCRIPT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_p2_campaign.py"))


def test_canonical_source_newlines_and_relative_paths(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "module.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    (b / "module.py").write_bytes(b"a = 1\nb = 2\n")
    assert p2_worker.canonical_source_hash(a) == p2_worker.canonical_source_hash(b)
    assert p2_worker.campaign_path(a, "inputs/i.json") == a / "inputs/i.json"
    with pytest.raises(ValueError):
        p2_worker.campaign_path(a, "../escape.json")
    with pytest.raises(ValueError):
        p2_worker.campaign_path(a, str(b.resolve()))


@pytest.mark.parametrize("policy", ["variation", "genotype_clone", "phenotype_clone"])
def test_light_worker_exact_result_and_portable_resume(problem, tmp_path, monkeypatch, policy):
    root = tmp_path / "campaign"
    (root / "inputs").mkdir(parents=True)
    for directory in ("specs", "runs", "ops"):
        (root / directory).mkdir()
    config = Config(
        population=4,
        neighborhood=2,
        generations=1,
        seed=17,
        instance="inputs/i.json",
        output="runs/k",
        method="full",
        offspring_policy=policy,
    )
    save_instance(problem, root / "inputs/i.json")
    initial = initial_genotypes(problem, config, RNGManager(17).stream("initialization"))
    atomic_json(root / "inputs/initial.json", [asdict(g) for g in initial])
    provenance = {"git_commit": "frozen", "git_dirty": False, "git_branch": None}
    monkeypatch.setattr(p2_worker, "git_metadata", lambda: provenance)
    monkeypatch.setattr(runner, "git_metadata", lambda: provenance)
    spec = {
        "key": "k",
        "source_hash": p2_worker.canonical_source_hash(),
        "source_commit": "frozen",
        "config": asdict(config),
        "instance": "inputs/i.json",
        "instance_hash": file_hash(root / "inputs/i.json"),
        "initial": "inputs/initial.json",
        "initial_hash": file_hash(root / "inputs/initial.json"),
    }
    atomic_json(root / "specs/k.json", spec)
    monkeypatch.chdir(root)
    p2_worker.execute(root, "k")
    assert p2_worker.validate_complete(root / "runs/k", spec) == (True, "complete")
    summary = json.loads((root / "runs/k/summary.json").read_text())
    assert summary["offspring_count"] == 4
    assert summary["counts"]["exact"] >= 4
    assert (root / "runs/k/trace.jsonl").stat().st_size == 0
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert p2_worker.validate_complete(relocated / "runs/k", spec)[0]
    (relocated / "runs/k/archive.json").write_text("[]")
    assert not p2_worker.validate_complete(relocated / "runs/k", spec)[0]
    with pytest.raises(FileExistsError):
        p2_worker.execute(root, "k")


def test_foreign_partial_triad_preserved_before_rerun(tmp_path, monkeypatch):
    root = tmp_path
    (root / "specs").mkdir()
    (root / "runs").mkdir()
    block = {"key": "block", "runs": ["F6", "CG", "CT"]}
    for key in block["runs"]:
        atomic_json(root / "specs" / f"{key}.json", {"key": key})
    atomic_json(root / "runs/F6/complete.json", {"host": "old-host"})
    monkeypatch.setitem(
        SCRIPT["recover_block"].__globals__, "validate_complete", lambda *_: (True, "ok")
    )
    assert not SCRIPT["recover_block"](root, block, "old-host")
    assert (root / "runs/F6/complete.json").exists()
    assert not SCRIPT["recover_block"](root, block, "new-host")
    assert not (root / "runs/F6").exists()
    assert list((root / "quarantine").glob("*/F6/complete.json"))
    (root / "runs/.CG.attempt-123").mkdir()
    assert not SCRIPT["recover_block"](root, block, "new-host")
    assert list((root / "quarantine").glob("*/.CG.attempt-123"))
    atomic_json(root / "ops/failures/supervisor_CT.json", {"exit_code": 1})
    with pytest.raises(RuntimeError, match="Recorded crash"):
        SCRIPT["recover_block"](root, block, "new-host")


def test_supervisor_lock_excludes_concurrent_owner(tmp_path):
    with SCRIPT["lease"](tmp_path / "lock"):
        with pytest.raises(OSError):
            with SCRIPT["lease"](tmp_path / "lock"):
                pass
    with SCRIPT["lease"](tmp_path / "lock"):
        pass


def test_resource_guard_uses_available_memory(tmp_path):
    value = SCRIPT["resources"](tmp_path)
    assert value["available_memory_gib"] > 0
    assert value["logical_cpus"] >= 1
    assert value["free_disk_gib"] > 0


def test_stop_request_drains_started_triad_and_resume_skips_it(tmp_path, monkeypatch):
    globals_ = SCRIPT["supervise"].__globals__
    blocks = [
        {"key": f"b{i}", "runs": [f"b{i}_{arm}" for arm in ("F6", "CG", "CT")]} for i in range(2)
    ]
    manifest = {
        "blocks": blocks,
        "source_hash": p2_worker.canonical_source_hash(),
        "input_files": {},
        "run_count": 6,
    }
    for directory in ("specs", "runs", "ops", "logs"):
        (tmp_path / directory).mkdir()
    atomic_json(tmp_path / "manifest.json", manifest)
    atomic_json(
        tmp_path / "manifest.sha256.json",
        {"manifest_sha256": file_hash(tmp_path / "manifest.json")},
    )
    for block in blocks:
        for key in block["runs"]:
            atomic_json(tmp_path / "specs" / f"{key}.json", {"config": {"seconds": 100}})
    launched = []

    class CompletedProcess:
        pid = 987654321

        def __init__(self, command, **kwargs):
            key = command[-1]
            launched.append(key)
            atomic_json(tmp_path / "runs" / key / "complete.json", {"host": "local"})
            (tmp_path / "stop.request").touch()

        def poll(self):
            return 0

    monkeypatch.setattr(globals_["subprocess"], "Popen", CompletedProcess)
    monkeypatch.setattr(globals_["platform"], "node", lambda: "local")
    monkeypatch.setattr(globals_["time"], "sleep", lambda _: None)
    monkeypatch.setitem(globals_, "keep_awake", lambda _: None)
    monkeypatch.setitem(
        globals_,
        "resources",
        lambda _: {"logical_cpus": 1, "available_memory_gib": 12, "free_disk_gib": 100},
    )
    monkeypatch.setitem(
        globals_,
        "validate_complete",
        lambda output, spec: ((output / "complete.json").exists(), "mock completed"),
    )
    SCRIPT["supervise"](tmp_path, 1)
    assert launched == blocks[0]["runs"]
    assert json.loads((tmp_path / "ops/exit.json").read_text())["state"] == "paused"
    (tmp_path / "stop.request").unlink()
    SCRIPT["supervise"](tmp_path, 1)
    assert launched == blocks[0]["runs"] + blocks[1]["runs"]
    assert json.loads((tmp_path / "ops/exit.json").read_text())["state"] == "complete"
