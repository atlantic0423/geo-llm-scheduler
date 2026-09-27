"""Generation-only worker artifacts and idempotent server-style batch execution."""

import importlib.util
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash, validate_result
from geo_llm_scheduler.experiments.runner import source_digest
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.utils.rng import RNGManager


def _campaign_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_20h_campaign.py"
    spec = importlib.util.spec_from_file_location("generation20_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_only_batch_resumes_without_repeating_result(tmp_path):
    module = _campaign_module()
    repo = Path(__file__).resolve().parents[2]
    instance = repo / "examples" / "smoke.json"
    config = Config(
        population=10,
        neighborhood=3,
        generations=2,
        seconds=None,
        method="full",
        seed=41,
        instance=str(instance),
    )
    initial = initial_genotypes(
        load_instance(instance), config, RNGManager(41).stream("initialization")
    )
    initial_file = tmp_path / "initial.json"
    atomic_json(initial_file, [asdict(g) for g in initial])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    job = JobSpec(
        "mini",
        "full",
        1,
        41,
        asdict(config),
        str(instance),
        file_hash(instance),
        str(initial_file),
        file_hash(initial_file),
        source_digest(),
        commit,
    )
    atomic_json(tmp_path / "launch.json", {"deadline_epoch": time.time() + 60})
    assert module.run_batch(tmp_path, [job], 1, "mini")
    result = tmp_path / "runs" / job.key
    assert validate_result(result, job) == (True, "complete")
    before = file_hash(result / "summary.json")
    assert module.run_batch(tmp_path, [job], 1, "mini")
    assert file_hash(result / "summary.json") == before
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert summary["termination_reason"] == "generation_limit"
    assert summary["time_budget_seconds"] is None


def test_dispatch_memory_guard_ignores_reclaimable_file_cache(tmp_path, monkeypatch):
    module = _campaign_module()
    group = tmp_path / "sys" / "fs" / "cgroup"
    group.mkdir(parents=True)
    (group / "memory.max").write_text("1000")
    (group / "memory.current").write_text("900")
    (group / "memory.stat").write_text("inactive_file 800\n")
    monkeypatch.setattr(module, "Path", lambda text: tmp_path / text.lstrip("/"))
    assert module._memory_ok()
    (group / "memory.stat").write_text("inactive_file 10\n")
    assert not module._memory_ok()
