"""Research run artifacts: raw traces, seed/config/source hashes and replay digest."""

import csv
import hashlib
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.engine.run import RunResult, run
from geo_llm_scheduler.io.loaders import load_instance


def digest(value: object) -> str:
    """Stable UTF-8 JSON hash independent of dictionary insertion order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def solution(candidate: Candidate) -> dict:
    """Export assignment through genotype only, alongside retained timing and objectives."""
    return {
        "genotype": asdict(candidate.genotype),
        "schedule": asdict(candidate.schedule),
        "evaluation": asdict(candidate.evaluation),
        "origin": candidate.origin,
    }


def deterministic_trace(trace: list[dict]) -> list[dict]:
    """Exclude runtime measurements while retaining the full search decisions."""
    return [
        {
            **{k: v for k, v in row.items() if k != "elapsed"},
            "steps": [
                {k: v for k, v in step.items() if not k.endswith("seconds")}
                for step in row["steps"]
            ],
        }
        for row in trace
    ]


def source_digest() -> str:
    """Hash installed Python module sources, without caches or workspace artifacts."""
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))
        }
    )


def save_result(result: RunResult, config: Config, instance_hash: str, output: Path) -> dict:
    """Write reproducible machine-readable artifacts and detailed accounting."""
    output.mkdir(parents=True, exist_ok=True)
    config_data = asdict(config)
    summary = {
        "seed": config.seed,
        "instance_hash": instance_hash,
        "config_hash": digest(config_data),
        "source_hash": source_digest(),
        "python": platform.python_version(),
        "method": config.method,
        "controller": config.controller,
        "budget_policy": config.budget_policy,
        "elapsed": result.elapsed,
        "counts": dict(result.gateway.counts),
        "timings": dict(result.gateway.seconds),
        "archive_size": len(result.archive.members),
        "archive_attempts": result.archive.attempts,
        "archive_contributions": result.archive.contributions,
        "trace_hash": digest(deterministic_trace(result.trace)),
        "hv": None,
        "igd_plus": None,
        "metric_status": "requires explicit common reference point/front",
    }
    counts = result.gateway.counts
    summary["trigger_rate"] = counts["trigger_hits"] / max(1, counts["trigger_attempts"])
    summary["trigger_success_rate"] = counts["trigger_successes"] / max(1, counts["trigger_hits"])
    operators: dict[str, dict[str, float]] = {}
    for row in result.trace:
        for step in row["steps"]:
            key = f"A{step['action']}"
            stat = operators.setdefault(
                key, {"calls": 0, "attempts": 0, "exact": 0, "accepted": 0, "seconds": 0.0}
            )
            stat["calls"] += 1
            stat["attempts"] += step["attempts"]
            stat["exact"] += step["effective"]
            stat["accepted"] += int(step["accepted"])
            stat["seconds"] += step["seconds"]
    summary["operators"] = operators
    artifacts = {
        "summary.json": summary,
        "config.json": config_data,
        "archive.json": [solution(c) for c in result.archive.members],
        "population.json": [solution(c) for c in result.population],
        "qtable.json": {
            "q": result.controller.q,
            "visits": result.controller.visits,
            "updates": result.controller.updates,
            "selections": result.controller.selections,
        },
    }
    for name, data in artifacts.items():
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "trace.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result.trace),
        encoding="utf-8",
    )
    with (output / "objectives.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["flow_seconds", "bill_usd", "tou_usd", "demand_usd"])
        writer.writerows(
            (c.evaluation.flow, c.evaluation.bill, c.evaluation.tou, sum(c.evaluation.demand))
            for c in result.archive.members
        )
    return summary


def run_config(path: str | Path, seed: int | None = None, output: str | None = None) -> dict:
    """Load config and instance, run once, and save artifacts to an identifiable directory."""
    config = load_config(path)
    if seed is not None:
        config = replace(config, seed=seed)
    problem = load_instance(config.instance)
    result = run(problem, config)
    destination = (
        Path(output)
        if output
        else Path(config.output)
        / f"{config.method}_{config.controller}_{config.budget_policy}_seed{config.seed}"
    )
    instance_hash = hashlib.sha256(Path(config.instance).read_bytes()).hexdigest()
    return save_result(result, config, instance_hash, destination)
