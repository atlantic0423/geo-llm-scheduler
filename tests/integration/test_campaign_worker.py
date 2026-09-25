"""A real small worker writes only complete, hash-validated results."""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash, validate_result
from geo_llm_scheduler.experiments.campaign.worker import execute_job
from geo_llm_scheduler.experiments.runner import source_digest
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.utils.rng import RNGManager


def test_worker_complete_artifact_and_corruption_detection(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    instance = repository / "examples" / "smoke.json"
    config = Config(
        population=10,
        neighborhood=3,
        generations=3,
        seconds=1.0,
        method="plain",
        seed=17,
        instance=str(instance),
    )
    initial = initial_genotypes(
        load_instance(instance), config, RNGManager(17).stream("initialization")
    )
    initial_path = tmp_path / "initial.json"
    atomic_json(initial_path, [asdict(g) for g in initial])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    spec = JobSpec(
        "stage",
        "plain",
        1,
        17,
        asdict(config),
        str(instance),
        file_hash(instance),
        str(initial_path),
        file_hash(initial_path),
        source_digest(),
        commit,
    )
    spec_path = tmp_path / "spec.json"
    atomic_json(spec_path, asdict(spec))
    output = tmp_path / "run"
    execute_job(spec_path, output)
    assert validate_result(output, spec) == (True, "complete")
    atomic_json(output.with_suffix(".failure.json"), {"message": "worker exit failure"})
    assert validate_result(output, spec) == (False, "worker-failure-marker")
    output.with_suffix(".failure.json").unlink()
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["instance_hash"] = "tampered"
    atomic_json(summary_path, summary)
    assert not validate_result(output, spec)[0]
