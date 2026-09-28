"""Isolated E15 worker with compact, hash-checked run artifacts."""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.nsga2 import run_nsga2
from geo_llm_scheduler.experiments.runner import digest, source_digest
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.utils.numeric import identity

TUPLE_FIELDS = (
    "mutation_weights",
    "severity_thresholds",
    "severity_budget_cutoffs",
    "budgets",
    "static_budgets",
    "enabled_operators",
)


def validate_e15_result(path: Path, spec: JobSpec) -> tuple[bool, str]:
    """Validate identity, termination, exact cap, endpoints and atomic marker."""
    names = (
        "summary.json",
        "archive_objectives.json",
        "archive_identities.json",
        "anytime.json",
        "budget_funnel.json",
        "complete.json",
    )
    if not path.is_dir() or any(not (path / n).is_file() for n in names):
        return False, "missing-artifact"
    try:
        marker = json.loads((path / "complete.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        objectives = json.loads((path / "archive_objectives.json").read_text(encoding="utf-8"))
        if marker["input_hash"] != spec.input_hash or marker["job_key"] != spec.key:
            return False, "identity"
        if marker["summary_sha256"] != file_hash(path / "summary.json"):
            return False, "checksum"
        if (
            summary["source_hash"] != spec.source_hash
            or summary["source_commit"] != spec.source_commit
        ):
            return False, "source"
        if (
            summary["instance_hash"] != spec.instance_hash
            or summary["initial_hash"] != spec.initial_hash
        ):
            return False, "inputs"
        if summary["config_hash"] != digest(spec.config):
            return False, "config"
        if summary["archive_size"] != len(objectives) or not objectives:
            return False, "archive"
        reason = summary["termination_reason"]
        if spec.config["exact_evaluation_cap"] is not None:
            if (
                reason != "exact_evaluation_cap"
                or summary["counts"]["exact"] != spec.config["exact_evaluation_cap"]
            ):
                return False, "exact-cap"
        elif spec.config["seconds"] is not None:
            if reason != "time_budget":
                return False, "time-limit"
        elif (
            reason != "generation_limit"
            or summary["generations_completed"] != spec.config["generations"]
        ):
            return False, "generation-limit"
        if summary["counts"]["exact"] < spec.config["population"]:
            return False, "initialization"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return False, f"parse:{error}"
    return True, "complete"


def execute_job(spec_path: Path, output: Path) -> None:
    """Run one frozen experiment and publish only complete validated results."""
    spec = JobSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    if file_hash(Path(spec.instance_path)) != spec.instance_hash:
        raise ValueError("Instance changed")
    if file_hash(Path(spec.initial_path)) != spec.initial_hash:
        raise ValueError("Initial population changed")
    if source_digest() != spec.source_hash:
        raise ValueError("Source changed")
    data = dict(spec.config)
    for key in TUPLE_FIELDS:
        data[key] = tuple(data[key])
    config = Config(**data)
    initial = [
        Genotype(tuple(g["ms"]), tuple(g["os"]))
        for g in json.loads(Path(spec.initial_path).read_text(encoding="utf-8"))
    ]
    if len(initial) != config.population:
        raise ValueError("Initial population size mismatch")
    problem = load_instance(spec.instance_path)
    result = (
        run_nsga2(problem, config, initial)
        if config.method in ("nsga2", "nsga2_memetic")
        else run(problem, config, initial)
    )
    tmp = output.with_name(f".{output.name}.attempt-{os.getpid()}")
    if tmp.exists():
        raise FileExistsError("Prior attempt needs quarantine")
    tmp.mkdir(parents=True)
    final_archive = result.archive.members
    state_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    budget_counts: Counter[str] = Counter()
    action_budget_counts: Counter[str] = Counter()
    funnel: dict[str, dict[str, float]] = {}
    anytime = []
    for row in result.trace:
        if row["subproblem"] == config.population - 1 and (
            row["generation"] % 10 == 9 or row["generation"] == config.generations - 1
        ):
            anytime.append(
                {
                    "generation": row["generation"] + 1,
                    "elapsed": row["elapsed"],
                    "archive_size": len(row["archive_objectives"]),
                    "exact": row["exact_count"],
                }
            )
        for step in row["steps"]:
            state_counts[str(step["state"])] += 1
            action_counts[str(step["action"])] += 1
            budget_counts[str(step["budget"])] += 1
            action_budget_counts[f"A{step['action']}:B{step['budget']}"] += 1
            action = str(step["action"])
            entry = funnel.setdefault(
                action,
                {
                    "calls": 0,
                    "requested": 0,
                    "effective": 0,
                    "no_candidate": 0,
                    "reward_sum": 0.0,
                    "positive": 0,
                    "budget_seconds": 0.0,
                },
            )
            entry["calls"] += 1
            entry["requested"] += step["budget"]
            entry["effective"] += step["effective"]
            entry["no_candidate"] += int(step["effective"] == 0)
            entry["reward_sum"] += step["reward"]
            entry["positive"] += int(step["reward"] > 0)
            entry["budget_seconds"] += step.get("budget_seconds", 0.0)
    full_trace = spec.algorithm_seed == 101 and spec.instance_seed // 10 % 3 == 0
    summary = {
        "job_key": spec.key,
        "source_hash": spec.source_hash,
        "source_commit": spec.source_commit,
        "instance_hash": spec.instance_hash,
        "initial_hash": spec.initial_hash,
        "config_hash": digest(spec.config),
        "seed": spec.algorithm_seed,
        "method": config.method,
        "budget_policy": config.budget_policy,
        "termination_reason": result.termination_reason,
        "generations_completed": len(result.trace) // config.population,
        "partial_generation_offspring": len(result.trace) % config.population,
        "elapsed": result.elapsed,
        "counts": dict(result.gateway.counts),
        "timings": dict(result.gateway.seconds),
        "archive_size": len(final_archive),
        "archive_peak_size": result.archive.peak_size,
        "archive_insertions": result.archive.insertions,
        "state_counts": dict(state_counts),
        "action_counts": dict(action_counts),
        "budget_counts": dict(budget_counts),
        "action_budget_counts": dict(action_budget_counts),
        "archive_final_by_origin": dict(Counter(c.origin for c in final_archive)),
        "adaptive_counts": dict(result.adaptive.counts) if result.adaptive else {},
        "adaptive_cpu_seconds": result.adaptive.cpu_seconds if result.adaptive else 0.0,
        "trace_sample": full_trace,
    }
    atomic_json(tmp / "summary.json", summary)
    atomic_json(tmp / "archive_objectives.json", [c.evaluation.objectives for c in final_archive])
    atomic_json(tmp / "archive_identities.json", [digest(identity(c)) for c in final_archive])
    atomic_json(tmp / "budget_funnel.json", funnel)
    atomic_json(tmp / "anytime.json", anytime)
    if full_trace:
        with (tmp / "trace.jsonl").open("w", encoding="utf-8") as handle:
            for row in result.trace:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(
        tmp / "complete.json",
        {
            "job_key": spec.key,
            "input_hash": spec.input_hash,
            "summary_sha256": file_hash(tmp / "summary.json"),
        },
    )
    valid, reason = validate_e15_result(tmp, spec)
    if not valid:
        raise ValueError(f"Incomplete result: {reason}")
    if output.exists():
        prior_valid, _ = validate_e15_result(output, spec)
        if prior_valid:
            shutil.rmtree(tmp)
            return
        raise FileExistsError("Incomplete prior result needs quarantine")
    os.replace(tmp, output)


def main() -> int:
    """CLI worker entry point."""
    if len(sys.argv) != 3:
        raise SystemExit("python -m geo_llm_scheduler.experiments.e15_worker SPEC OUTPUT")
    spec_path, output = map(Path, sys.argv[1:])
    try:
        execute_job(spec_path, output)
    except Exception as error:
        atomic_json(
            output.with_suffix(".failure.json"),
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
