"""One isolated campaign run; supervisors own retries and final checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash, validate_result
from geo_llm_scheduler.experiments.nsga2 import run_nsga2
from geo_llm_scheduler.experiments.runner import save_result, source_digest
from geo_llm_scheduler.io.loaders import load_instance

TUPLE_FIELDS = (
    "mutation_weights",
    "severity_thresholds",
    "budgets",
    "static_budgets",
    "enabled_operators",
)


def execute_job(spec_path: Path, output: Path) -> None:
    """Run a frozen job and atomically publish only a validated complete result."""
    spec = JobSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    if file_hash(Path(spec.instance_path)) != spec.instance_hash:
        raise ValueError("Instance hash changed after campaign freeze")
    if file_hash(Path(spec.initial_path)) != spec.initial_hash:
        raise ValueError("Initial population changed after campaign freeze")
    if source_digest() != spec.source_hash:
        raise ValueError("Source hash changed after campaign freeze")
    config_data = dict(spec.config)
    for key in TUPLE_FIELDS:
        config_data[key] = tuple(config_data[key])
    config = Config(**config_data)
    frozen = json.loads(Path(spec.initial_path).read_text(encoding="utf-8"))
    initial = [Genotype(tuple(row["ms"]), tuple(row["os"])) for row in frozen]
    if len(initial) != config.population:
        raise ValueError("Frozen initial population has the wrong size")
    problem = load_instance(spec.instance_path)
    result = (
        run_nsga2(problem, config, initial)
        if config.method == "nsga2"
        else run(problem, config, initial)
    )
    temporary = output.with_name(f".{output.name}.attempt-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Unrecovered attempt directory: {temporary}")
    summary = save_result(result, config, spec.instance_hash, temporary)
    atomic_json(
        temporary / "complete.json",
        {
            "job_key": spec.key,
            "input_hash": spec.input_hash,
            "summary_hash": file_hash(temporary / "summary.json"),
            "elapsed": summary["elapsed"],
        },
    )
    valid, reason = validate_result(temporary, spec)
    if not valid:
        raise ValueError(f"Run artifacts invalid: {reason}")
    if output.exists():
        old_valid, _ = validate_result(output, spec)
        if old_valid:
            shutil.rmtree(temporary)
            return
        raise FileExistsError("Incomplete output needs supervisor quarantine")
    os.replace(temporary, output)


def main(args: list[str] | None = None) -> int:
    """CLI worker entry point with an explicit failure artifact for the supervisor."""
    values = args if args is not None else sys.argv[1:]
    if len(values) != 2:
        raise SystemExit(
            "Usage: python -m geo_llm_scheduler.experiments.campaign.worker SPEC OUTPUT"
        )
    spec_path, output = map(Path, values)
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
