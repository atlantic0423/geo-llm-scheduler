"""Research run artifacts: raw traces, seed/config/source hashes and replay digest."""

import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.engine.run import RunResult, run
from geo_llm_scheduler.experiments.nsga2 import run_nsga2
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
    instrumentation = {
        "preference",
        "dominant_condition",
        "search_progress",
        "progress",
        "severities",
        "proposals",
        "exact_evaluations",
        "operator_instrumentation",
    }
    return [
        {
            **{k: v for k, v in row.items() if k != "elapsed"},
            "steps": [
                {
                    k: v
                    for k, v in step.items()
                    if not k.endswith("seconds") and k not in instrumentation
                }
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


def git_metadata(root: Path | None = None) -> dict[str, str | bool | None]:
    """Return explicit repository provenance, or null fields outside a Git checkout."""
    repository = root or Path(__file__).resolve().parents[3]

    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repository), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"git_commit": None, "git_branch": None, "git_dirty": None}
    branch = git("branch", "--show-current")
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {
        "git_commit": commit,
        "git_branch": branch or None,
        "git_dirty": None if status is None else bool(status),
    }


def save_result(result: RunResult, config: Config, instance_hash: str, output: Path) -> dict:
    """Write reproducible machine-readable artifacts and detailed accounting."""
    output.mkdir(parents=True, exist_ok=True)
    config_data = asdict(config)
    provenance = git_metadata()
    final_by_origin: dict[str, int] = {}
    for candidate in result.archive.members:
        final_by_origin[candidate.origin] = final_by_origin.get(candidate.origin, 0) + 1
    summary: dict[str, object] = {
        "seed": config.seed,
        "instance_hash": instance_hash,
        "config_hash": digest(config_data),
        "source_hash": source_digest(),
        **provenance,
        "python": platform.python_version(),
        "method": config.method,
        "controller": config.controller,
        "budget_policy": config.budget_policy,
        "elapsed": result.elapsed,
        "termination_reason": result.termination_reason,
        "time_budget_seconds": config.seconds,
        "time_overshoot_seconds": (
            None if config.seconds is None else max(0.0, result.elapsed - config.seconds)
        ),
        "counts": dict(result.gateway.counts),
        "timings": dict(result.gateway.seconds),
        "archive_size": len(result.archive.members),
        "archive_peak_size": result.archive.peak_size,
        "archive_attempts": result.archive.attempts,
        "archive_insertions": result.archive.insertions,
        "archive_final_by_origin": final_by_origin,
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
                key,
                {
                    "calls": 0,
                    "attempts": 0,
                    "proposals": 0,
                    "exact": 0,
                    "feasible": 0,
                    "accepted": 0,
                    "positive_reward": 0,
                    "reward_sum": 0.0,
                    "scalar_improvement": 0.0,
                    "delta_flow": 0.0,
                    "delta_bill": 0.0,
                    "archive_insertions": 0,
                    "archive_net_retained": 0,
                    "seconds": 0.0,
                    "construction_seconds": 0.0,
                },
            )
            stat["calls"] += 1
            stat["attempts"] += step["attempts"]
            stat["proposals"] += step.get("proposals", step["effective"])
            stat["exact"] += step["effective"]
            stat["feasible"] += step["feasible"]
            stat["accepted"] += int(step["accepted"])
            stat["positive_reward"] += int(step["reward"] > 0)
            stat["reward_sum"] += step["reward"]
            stat["scalar_improvement"] += step["scalar_before"] - step["scalar_after"]
            stat["delta_flow"] += step["delta_flow"]
            stat["delta_bill"] += step["delta_bill"]
            stat["archive_insertions"] += step["archive_insertions"]
            stat["archive_net_retained"] += step["archive_net_retained"]
            stat["seconds"] += step["seconds"]
            stat["construction_seconds"] += step["construction_seconds"]
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
        writer.writerow(["flow_seconds", "bill_cny", "tou_cny", "demand_cny"])
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
    result = run_nsga2(problem, config) if config.method == "nsga2" else run(problem, config)
    destination = (
        Path(output)
        if output
        else Path(config.output)
        / f"{config.method}_{config.controller}_{config.budget_policy}_seed{config.seed}"
    )
    instance_hash = hashlib.sha256(Path(config.instance).read_bytes()).hexdigest()
    return save_result(result, config, instance_hash, destination)
