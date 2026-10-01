"""Hash-frozen P0/P1 workers; failed attempts remain visible and are never retried."""

from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from pathlib import Path
from time import perf_counter

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.support import atomic_json, process_peak_rss_gb
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.e15_worker import TUPLE_FIELDS
from geo_llm_scheduler.experiments.nsga2 import run_nsga2
from geo_llm_scheduler.experiments.p1_diagnostics import diagnose_run
from geo_llm_scheduler.experiments.p1_observation import P1Observer, SemanticRecorder
from geo_llm_scheduler.experiments.runner import digest, source_digest
from geo_llm_scheduler.io.loaders import load_instance


def execute_spec(spec_path: Path, output: Path) -> None:
    """Run the frozen spec, publish sampling evidence, then bounded offline diagnosis."""
    spec = JobSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    for path, expected in (
        (spec.instance_path, spec.instance_hash),
        (spec.initial_path, spec.initial_hash),
    ):
        if file_hash(Path(path)) != expected:
            raise ValueError("Frozen input changed")
    if source_digest() != spec.source_hash:
        raise ValueError("Frozen source changed")
    data = dict(spec.config)
    for key in TUPLE_FIELDS:
        data[key] = tuple(data[key])
    config = Config(**data)
    problem = load_instance(spec.instance_path)
    initial = [
        Genotype(tuple(g["ms"]), tuple(g["os"]))
        for g in json.loads(Path(spec.initial_path).read_text(encoding="utf-8"))
    ]
    start = perf_counter()
    observer = None
    recorder = SemanticRecorder()
    if spec.stage.startswith("P1") or spec.stage == "RSS":
        observer = P1Observer(output, config)
    else:
        output.mkdir(parents=True, exist_ok=False)
    try:
        runner = run_nsga2 if config.method == "nsga2" else run
        result = runner(problem, config, initial, retain_trace=False, observer=observer or recorder)
    finally:
        if observer is not None:
            observer.close()
    signature = (observer.recorder if observer else recorder).signature(result)
    atomic_json(output / "semantic_signature.json", signature)
    summary = {
        "job_key": spec.key,
        "input_hash": spec.input_hash,
        "source_commit": spec.source_commit,
        "source_hash": spec.source_hash,
        "config_hash": digest(spec.config),
        "instance_hash": spec.instance_hash,
        "initial_hash": spec.initial_hash,
        "host": platform.node(),
        "pid": os.getpid(),
        "elapsed": result.elapsed,
        "offspring_count": result.offspring_count,
        "termination_reason": result.termination_reason,
        "counts": dict(result.gateway.counts),
        "timings": dict(result.gateway.seconds),
        "peak_rss_gb_sampling": process_peak_rss_gb(),
        "archive_size": len(result.archive.members),
        "archive_peak_size": result.archive.peak_size,
        "observer_seconds": observer.seconds if observer else 0.0,
    }
    if observer is not None:
        if result.termination_reason != "time_budget":
            raise ValueError("P1 ended before its time budget")
        if len(observer.checkpoints) != 3:
            raise ValueError("P1 missing prescribed checkpoints")
        atomic_json(
            output / "sampling_complete.json", {**summary, "checkpoints": observer.checkpoints}
        )
        del result
        summary["diagnostics"] = diagnose_run(problem, output, config)
    summary["process_elapsed"] = perf_counter() - start
    summary["peak_rss_gb"] = process_peak_rss_gb()
    atomic_json(output / "summary.json", summary)
    files = {
        p.name: {"size": p.stat().st_size, "sha256": file_hash(p)}
        for p in output.iterdir()
        if p.is_file() and p.name != "complete.json"
    }
    atomic_json(output / "complete.json", {"input_hash": spec.input_hash, "files": files})


def validate_spec(output: Path, spec: JobSpec) -> bool:
    """Accept only a complete marker whose inputs and every artifact checksum agree."""
    try:
        marker = json.loads((output / "complete.json").read_text(encoding="utf-8"))
        if marker["input_hash"] != spec.input_hash:
            return False
        for name, entry in marker["files"].items():
            p = output / name
            if p.stat().st_size != entry["size"] or file_hash(p) != entry["sha256"]:
                return False
        return "summary.json" in marker["files"] and "semantic_signature.json" in marker["files"]
    except (OSError, KeyError, ValueError, TypeError):
        return False


def main() -> int:
    """Execute exactly one worker; preserve a failure marker without automatic retry."""
    spec_path, output = map(Path, sys.argv[1:])
    try:
        execute_spec(spec_path, output)
    except Exception as error:
        atomic_json(
            output.with_suffix(".failure.json"),
            {
                "error": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
