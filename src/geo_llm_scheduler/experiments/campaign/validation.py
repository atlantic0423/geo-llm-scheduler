"""Independent completeness checks for resumable experiment results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.runner import digest

REQUIRED = (
    "summary.json",
    "config.json",
    "archive.json",
    "population.json",
    "qtable.json",
    "trace.jsonl",
    "objectives.csv",
    "complete.json",
)


def file_hash(path: Path) -> str:
    """Hash a file in bounded chunks without loading a large artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result(directory: Path, spec: JobSpec) -> tuple[bool, str]:
    """Require full artifacts, exact identity, finite metrics and a valid termination."""
    if not directory.is_dir():
        return False, "missing-directory"
    if directory.with_suffix(".failure.json").exists():
        return False, "worker-failure-marker"
    for name in REQUIRED:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            return False, f"missing-or-empty:{name}"
    try:
        marker = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        archive = json.loads((directory / "archive.json").read_text(encoding="utf-8"))
        population = json.loads((directory / "population.json").read_text(encoding="utf-8"))
        qtable = json.loads((directory / "qtable.json").read_text(encoding="utf-8"))
        if marker.get("job_key") != spec.key or marker.get("input_hash") != spec.input_hash:
            return False, "identity-mismatch"
        if marker.get("summary_hash") != file_hash(directory / "summary.json"):
            return False, "summary-checksum"
        if summary.get("source_hash") != spec.source_hash:
            return False, "source-hash-mismatch"
        if summary.get("git_commit") != spec.source_commit:
            return False, "commit-mismatch"
        if summary.get("instance_hash") != spec.instance_hash:
            return False, "instance-hash-mismatch"
        if summary.get("config_hash") != digest(config):
            return False, "config-hash-mismatch"
        if summary.get("seed") != spec.algorithm_seed:
            return False, "seed-mismatch"
        if summary.get("method") != spec.config["method"]:
            return False, "method-mismatch"
        if summary.get("termination_reason") not in ("time_budget", "generation_limit"):
            return False, "invalid-termination"
        if spec.config["seconds"] is None:
            if summary["termination_reason"] != "generation_limit":
                return False, "not-complete-generations"
        if not isinstance(population, list) or len(population) != spec.config["population"]:
            return False, "population-size"
        if not isinstance(archive, list) or not archive or not isinstance(qtable, dict):
            return False, "archive-or-qtable"
        for key in ("elapsed", "archive_size", "archive_peak_size"):
            value = summary.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return False, f"nonfinite:{key}"
        if summary["archive_size"] != len(archive):
            return False, "archive-size"
        if spec.config["seconds"] is None:
            if (
                summary["time_budget_seconds"] is not None
                or summary["time_overshoot_seconds"] is not None
            ):
                return False, "generation-budget"
        elif (
            not math.isclose(summary["time_budget_seconds"], spec.config["seconds"])
            or not isinstance(summary["time_overshoot_seconds"], (int, float))
            or not math.isfinite(summary["time_overshoot_seconds"])
            or summary["time_overshoot_seconds"] < 0
        ):
            return False, "time-budget"
        expected_rows = spec.config["generations"] * spec.config["population"]
        if spec.config["seconds"] is None:
            if sum(1 for _ in (directory / "trace.jsonl").open(encoding="utf-8")) != expected_rows:
                return False, "generation-count"
        with (directory / "objectives.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(archive):
            return False, "objectives-count"
        for row in rows:
            if not all(
                math.isfinite(float(row[key]))
                for key in ("flow_seconds", "bill_cny", "tou_cny", "demand_cny")
            ):
                return False, "nonfinite-objective"
        trace = directory / "trace.jsonl"
        if trace.stat().st_size:
            with trace.open("rb") as handle:
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    return False, "truncated-trace"
        if not trace.stat().st_size:
            return False, "empty-trace"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return False, f"parse-error:{error}"
    return True, "complete"
