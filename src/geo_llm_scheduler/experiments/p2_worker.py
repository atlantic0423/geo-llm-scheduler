"""Portable, bounded-observation P2 worker with transactional completion markers."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json, process_peak_rss_gb
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.campaign.worker import TUPLE_FIELDS
from geo_llm_scheduler.experiments.runner import digest, git_metadata, save_result
from geo_llm_scheduler.io.loaders import load_instance

REQUIRED = (
    "summary.json",
    "config.json",
    "archive.json",
    "population.json",
    "qtable.json",
    "objectives.csv",
    "trace.jsonl",
    "anytime.json",
)


def canonical_source_hash(root: Path | None = None) -> str:
    """Hash UTF-8 source after newline normalization, allowing Windows/Linux relocation."""
    root = root or Path(__file__).resolve().parents[1]
    return digest(
        {
            p.relative_to(root).as_posix(): digest(p.read_text(encoding="utf-8"))
            for p in sorted(root.rglob("*.py"))
        }
    )


def campaign_path(root: Path, relative: str) -> Path:
    """Resolve a frozen relative path and reject absolute paths or directory escape."""
    candidate = root / relative
    if Path(relative).is_absolute() or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("Campaign paths must stay relative and inside the campaign")
    return candidate


def validate_complete(output: Path, spec: dict) -> tuple[bool, str]:
    """Validate portable identity and every published artifact before reusing a run."""
    try:
        marker = json.loads((output / "complete.json").read_text(encoding="utf-8"))
        if marker["input_hash"] != digest(spec):
            return False, "input hash mismatch"
        if set(marker["files"]) != set(REQUIRED):
            return False, "incomplete artifact manifest"
        if any(file_hash(output / name) != sha for name, sha in marker["files"].items()):
            return False, "artifact hash mismatch"
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if (
            summary["canonical_source_hash"] != spec["source_hash"]
            or summary["git_commit"] != spec["source_commit"]
            or summary["git_dirty"] is not False
            or summary["instance_hash"] != spec["instance_hash"]
            or digest(config) != digest(spec["config"])
            or summary["offspring_count"] < 1
        ):
            return False, "provenance/configuration/progress mismatch"
        population = json.loads((output / "population.json").read_text(encoding="utf-8"))
        archive = json.loads((output / "archive.json").read_text(encoding="utf-8"))
        if len(population) != config["population"] or not archive:
            return False, "missing final solutions"
        if any(not c["evaluation"]["feasible"] for c in population + archive):
            return False, "infeasible final solution"
        if (output / "trace.jsonl").stat().st_size:
            return False, "unbounded historical trace enabled"
        return True, "complete"
    except (OSError, ValueError, KeyError, TypeError):
        return False, "missing or unreadable completion artifacts"


def execute(root: Path, key: str) -> None:
    """Execute one frozen job; preserve unfinished files and publish only checked results."""
    spec = json.loads(campaign_path(root, f"specs/{key}.json").read_text(encoding="utf-8"))
    if spec["key"] != key or canonical_source_hash() != spec["source_hash"]:
        raise ValueError("Worker key or source mismatch")
    provenance = git_metadata()
    if provenance["git_commit"] != spec["source_commit"] or provenance["git_dirty"]:
        raise ValueError("Worker requires the clean frozen Git commit")
    instance = campaign_path(root, spec["instance"])
    initial_file = campaign_path(root, spec["initial"])
    if (
        file_hash(instance) != spec["instance_hash"]
        or file_hash(initial_file) != spec["initial_hash"]
    ):
        raise ValueError("Frozen input hash mismatch")
    data = dict(spec["config"])
    for field in TUPLE_FIELDS:
        data[field] = tuple(data[field])
    config = Config(**data)
    initial = [
        Genotype(tuple(g["ms"]), tuple(g["os"]))
        for g in json.loads(initial_file.read_text(encoding="utf-8"))
    ]
    output = campaign_path(root, f"runs/{key}")
    temporary = campaign_path(root, f"runs/.{key}.attempt-{os.getpid()}")
    if output.exists() or temporary.exists():
        raise FileExistsError("Unrecovered output; supervisor must inspect before dispatch")
    temporary.mkdir(parents=True)
    heartbeat = campaign_path(root, f"ops/heartbeats/{key}.json")
    last = -10.0
    snapshots: list[dict] = []
    thresholds = [config.seconds * f for f in (0.1, 0.25, 0.5, 0.75, 0.9)] if config.seconds else []
    counters: dict[str, dict[str, float]] = {}

    def observe(event: str, context: dict) -> None:
        nonlocal last
        if event == "candidate":
            return
        elapsed = context["elapsed"]
        gateway = context["gateway"]
        if event == "offspring":
            for step in context["row"]["steps"]:
                stats = counters.setdefault(str(step["action"]), {})
                for field in ("attempts", "effective", "feasible", "accepted", "reward", "seconds"):
                    stats[field] = stats.get(field, 0.0) + step[field]
        if thresholds and elapsed >= thresholds[0]:
            threshold = thresholds.pop(0)
            snapshots.append(
                {
                    "target_seconds": threshold,
                    "elapsed": elapsed,
                    "objectives": [c.evaluation.objectives for c in gateway.archive.members],
                }
            )
        if elapsed - last >= 10:
            atomic_json(
                heartbeat,
                {
                    "pid": os.getpid(),
                    "host": platform.node(),
                    "timestamp": time.time(),
                    "elapsed": elapsed,
                    "exact": gateway.counts["exact"],
                    "offspring": context.get("processed", 0),
                    "archive_size": len(gateway.archive.members),
                    "rss_peak_gib": process_peak_rss_gb(),
                },
            )
            last = elapsed

    problem = load_instance(instance)
    result = run(problem, config, initial, retain_trace=False, observer=observe)
    # Verification and export are outside the engine clock and separately accounted.
    verification_start = time.perf_counter()
    for candidate in result.population + result.archive.members:
        if evaluate(problem, candidate.genotype, candidate.schedule) != candidate.evaluation:
            raise ValueError("Final solution failed independent exact re-evaluation")
    snapshots.append(
        {
            "target_seconds": config.seconds,
            "elapsed": result.elapsed,
            "objectives": [c.evaluation.objectives for c in result.archive.members],
        }
    )
    summary = save_result(result, config, spec["instance_hash"], temporary)
    summary.update(
        canonical_source_hash=spec["source_hash"],
        initial_hash=spec["initial_hash"],
        input_hash=digest(spec),
        host=platform.node(),
        offspring_count=result.offspring_count,
        rss_peak_gib=process_peak_rss_gb(),
        light_operator_totals=counters,
        verification_and_export_seconds=time.perf_counter() - verification_start,
    )
    atomic_json(temporary / "summary.json", summary)
    atomic_json(temporary / "anytime.json", snapshots)
    atomic_json(
        temporary / "complete.json",
        {
            "input_hash": digest(spec),
            "host": platform.node(),
            "files": {name: file_hash(temporary / name) for name in REQUIRED},
        },
    )
    valid, reason = validate_complete(temporary, spec)
    if not valid:
        raise ValueError(reason)
    temporary.rename(output)
    atomic_json(
        heartbeat,
        {
            "pid": os.getpid(),
            "host": platform.node(),
            "timestamp": time.time(),
            "state": "complete",
            "elapsed": result.elapsed,
            "offspring": result.offspring_count,
            "exact": result.gateway.counts["exact"],
            "rss_peak_gib": process_peak_rss_gb(),
        },
    )


def main() -> None:
    """Run the isolated worker and leave a persistent failure record on any exception."""
    root, key = Path(sys.argv[1]).resolve(), sys.argv[2]
    try:
        os.chdir(root)
        execute(root, key)
    except Exception:
        atomic_json(
            root / "ops" / "failures" / f"{key}.json",
            {
                "key": key,
                "pid": os.getpid(),
                "host": platform.node(),
                "timestamp": time.time(),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
