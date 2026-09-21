"""Resume D01, run independent D02, then aggregate and visualize both datasets."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from geo_llm_scheduler.experiments.diagnostic_instances import SCENARIOS
from geo_llm_scheduler.experiments.diagnostic_visualization import (
    create_comparison_figures,
    create_dataset_figures,
    summarize_steps,
    write_reports,
)
from geo_llm_scheduler.experiments.diagnostics import aggregate_diagnostics
from geo_llm_scheduler.experiments.mixed_instances import (
    MIXED_ALGORITHM_SEEDS,
    MIXED_INSTANCE_SEEDS,
    MIXED_SCENARIO,
)

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPOSITORY / "outputs" / "diagnostics"
PIPELINE_ROOT = OUTPUT_ROOT / "pipeline"
STATE_PATH = PIPELINE_ROOT / "night_pipeline_state.json"
CHECKPOINT_PATH = PIPELINE_ROOT / "night_resume_checkpoint.json"
INCOMPLETE_ARCHIVE = REPOSITORY / "outputs" / "diagnostic_incomplete_archive"
REPORT_ROOT = REPOSITORY / "docs" / "reports"
REQUIRED_ARTIFACTS = (
    "summary.json",
    "trace.jsonl",
    "qtable.json",
    "diagnostic.json",
    "rl_steps.csv",
    "rl_steps.jsonl",
)
D01_VARIANTS = {1: "single_peak", 2: "multi_tied_peak", 3: "mixed_peak"}


@dataclass(frozen=True)
class RunIdentity:
    """Expected identity and artifact location for one resumable diagnostic run."""

    phase: str
    scenario: str
    jobs: int
    instance_seed: int
    algorithm_seed: int
    generations: int
    variant: str | None = None

    @property
    def run_id(self) -> str:
        variant = f"_{self.variant}" if self.variant else ""
        return (
            f"{self.phase}_{self.scenario}{variant}_j{self.jobs}_i{self.instance_seed}"
            f"_a{self.algorithm_seed}_g{self.generations}"
        )

    @property
    def directory(self) -> Path:
        return OUTPUT_ROOT / self.phase / "runs" / self.run_id


def d01_identities() -> tuple[RunIdentity, ...]:
    """Return the fixed 63-run controlled diagnostic matrix."""
    return tuple(
        RunIdentity(
            "main",
            scenario,
            50,
            instance_seed,
            algorithm_seed,
            30,
            D01_VARIANTS[instance_seed] if scenario == "D5_demand" else None,
        )
        for scenario in SCENARIOS
        for instance_seed in (1, 2, 3)
        for algorithm_seed in (101, 202, 303)
    )


def d02_identities() -> tuple[RunIdentity, ...]:
    """Return the independent 15-run mixed validation matrix."""
    return tuple(
        RunIdentity("mixed", MIXED_SCENARIO, 50, instance_seed, algorithm_seed, 30)
        for instance_seed in MIXED_INSTANCE_SEEDS
        for algorithm_seed in MIXED_ALGORITHM_SEEDS
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_run(identity: RunIdentity) -> tuple[bool, str]:
    """Validate identity, termination, and all required nonempty artifacts."""
    directory = identity.directory
    if not directory.is_dir():
        return False, "missing"
    for name in REQUIRED_ARTIFACTS:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            return False, f"missing-or-empty:{name}"
    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "diagnostic.json").read_text(encoding="utf-8"))
        json.loads((directory / "qtable.json").read_text(encoding="utf-8"))
        if summary.get("termination_reason") != "generation_limit":
            return False, f"termination:{summary.get('termination_reason')}"
        expected = {
            "run_id": identity.run_id,
            "phase": identity.phase,
            "scenario": identity.scenario,
            "jobs": identity.jobs,
            "instance_seed": identity.instance_seed,
            "algorithm_seed": identity.algorithm_seed,
            "generations": identity.generations,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False, "identity-mismatch"
        with (directory / "rl_steps.csv").open(newline="", encoding="utf-8") as handle:
            if next(csv.DictReader(handle), None) is None:
                return False, "empty-rl-steps"
        for name in ("trace.jsonl", "rl_steps.jsonl"):
            with (directory / name).open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    return False, f"truncated:{name}"
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return False, f"parse-error:{error}"
    return True, "complete"


def scan(identities: tuple[RunIdentity, ...]) -> tuple[list[str], dict[str, str]]:
    """Split an expected matrix into complete and missing/incomplete runs."""
    complete: list[str] = []
    incomplete: dict[str, str] = {}
    for identity in identities:
        valid, reason = validate_run(identity)
        if valid:
            complete.append(identity.run_id)
        else:
            incomplete[identity.run_id] = reason
    return complete, incomplete


def _checkpoint(stage: str, **extra: object) -> None:
    d01_complete, d01_incomplete = scan(d01_identities())
    d02_complete, d02_incomplete = scan(d02_identities())
    payload = {
        "updated_at": _timestamp(),
        "pipeline_pid": os.getpid(),
        "status": "running",
        "stage": stage,
        "d01": {
            "complete": len(d01_complete),
            "expected": 63,
            "missing_or_incomplete": d01_incomplete,
        },
        "d02": {
            "complete": len(d02_complete),
            "expected": 15,
            "missing_or_incomplete": d02_incomplete,
        },
        **extra,
    }
    _atomic_json(STATE_PATH, payload)
    _atomic_json(
        CHECKPOINT_PATH,
        {
            "updated_at": payload["updated_at"],
            "stage": stage,
            "d01_complete_runs": d01_complete,
            "d01_remaining_runs": list(d01_incomplete),
            "d02_complete_runs": d02_complete,
            "d02_remaining_runs": list(d02_incomplete),
        },
    )


def _quarantine_incomplete(label: str, identities: tuple[RunIdentity, ...]) -> list[dict[str, str]]:
    """Move incomplete artifacts aside while preserving every complete run."""
    _, incomplete = scan(identities)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    moved: list[dict[str, str]] = []
    for identity in identities:
        reason = incomplete.get(identity.run_id)
        if reason is None or reason == "missing" or not identity.directory.exists():
            continue
        destination = INCOMPLETE_ARCHIVE / label / f"{identity.run_id}__{stamp}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(identity.directory), str(destination))
        moved.append(
            {"run_id": identity.run_id, "reason": reason, "preserved_at": str(destination)}
        )
    return moved


def _run_until_complete(
    label: str,
    identities: tuple[RunIdentity, ...],
    command: list[str],
) -> None:
    """Run at most three resume passes, retrying only invalid or absent identities."""
    initial_complete, _ = scan(identities)
    for attempt in range(1, 4):
        moved = _quarantine_incomplete(label, identities)
        _checkpoint(
            f"{label}-resume-pass-{attempt}",
            resume_effective=bool(initial_complete),
            initial_complete={label: len(initial_complete)},
            preserved_incomplete=moved,
            child_command=command,
        )
        completed = subprocess.run(command, cwd=REPOSITORY, check=False)
        complete, incomplete = scan(identities)
        _checkpoint(
            f"{label}-resume-pass-{attempt}-finished",
            child_exit_code=completed.returncode,
            complete_after_pass={label: len(complete)},
        )
        if len(complete) == len(identities):
            return
        if attempt == 3:
            raise RuntimeError(
                f"{label} incomplete after three resume passes: "
                f"{len(complete)}/{len(identities)}; {incomplete}"
            )


def _read_expected_steps(identities: tuple[RunIdentity, ...]) -> list[dict[str, str]]:
    """Read exactly one validated matrix and no calibration or foreign phase data."""
    rows: list[dict[str, str]] = []
    for identity in identities:
        valid, reason = validate_run(identity)
        if not valid:
            raise RuntimeError(f"Cannot visualize incomplete run {identity.run_id}: {reason}")
        with (identity.directory / "rl_steps.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _verify_aggregate(path: Path, expected: int, phase: str) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected or {row["phase"] for row in rows} != {phase}:
        raise RuntimeError(f"Aggregate {path} expected {expected} {phase} runs; got {len(rows)}")


def run_pipeline() -> None:
    """Execute the complete persistent D01→D02→comparison workflow."""
    d01 = d01_identities()
    d02 = d02_identities()
    initial_d01, initial_d01_incomplete = scan(d01)
    initial_d02, initial_d02_incomplete = scan(d02)
    _checkpoint(
        "started",
        started_at=_timestamp(),
        resume_effective=bool(initial_d01),
        initial_complete={"d01": len(initial_d01), "d02": len(initial_d02)},
        initial_missing_or_incomplete={
            "d01": initial_d01_incomplete,
            "d02": initial_d02_incomplete,
        },
    )

    d01_command = [
        sys.executable,
        "-u",
        "scripts/run_diagnostics.py",
        "--stage",
        "main",
        "--config",
        "configs/diagnostics/main/base.yaml",
        "--instances",
        "instances/diagnostic",
        "--output",
        "outputs/diagnostics",
        "--jobs",
        "50",
        "--instance-seeds",
        "1",
        "2",
        "3",
        "--algorithm-seeds",
        "101",
        "202",
        "303",
        "--generations",
        "30",
        "--resume",
    ]
    _run_until_complete("d01", d01, d01_command)
    _checkpoint("d01-integrity-verified-63-of-63")
    d01_aggregate = aggregate_diagnostics(
        OUTPUT_ROOT,
        phases=("main",),
        manifest=REPOSITORY / "instances" / "diagnostic" / "manifest.csv",
        aggregate_subdir="aggregate/d01",
    )
    _verify_aggregate(OUTPUT_ROOT / "aggregate" / "d01" / "run_summary.csv", 63, "main")
    controlled = summarize_steps(_read_expected_steps(d01))
    d01_figures = create_dataset_figures(
        controlled, REPORT_ROOT / "figures" / "d01", "D01 controlled diagnostic"
    )
    _checkpoint(
        "d01-aggregate-and-figures-complete",
        d01_aggregate=d01_aggregate,
        d01_figure_count=len(d01_figures),
    )

    d02_command = [
        sys.executable,
        "-u",
        "scripts/run_mixed_diagnostics.py",
        "--stage",
        "main",
        "--config",
        "configs/diagnostics/mixed/base.yaml",
        "--instances",
        "instances/diagnostic_mixed",
        "--output",
        "outputs/diagnostics",
        "--jobs",
        "50",
        "--instance-seeds",
        *[str(value) for value in MIXED_INSTANCE_SEEDS],
        "--algorithm-seeds",
        *[str(value) for value in MIXED_ALGORITHM_SEEDS],
        "--generations",
        "30",
        "--resume",
    ]
    _run_until_complete("d02", d02, d02_command)
    _checkpoint("d02-integrity-verified-15-of-15")
    d02_aggregate = aggregate_diagnostics(
        OUTPUT_ROOT,
        phases=("mixed",),
        manifest=REPOSITORY / "instances" / "diagnostic_mixed" / "manifest.csv",
        aggregate_subdir="aggregate/d02",
    )
    _verify_aggregate(OUTPUT_ROOT / "aggregate" / "d02" / "run_summary.csv", 15, "mixed")
    mixed = summarize_steps(_read_expected_steps(d02))
    d02_figures = create_dataset_figures(
        mixed, REPORT_ROOT / "figures" / "d02", "D02 mixed random validation"
    )
    comparison_figures = create_comparison_figures(
        controlled, mixed, REPORT_ROOT / "figures" / "comparison"
    )
    reports = write_reports(controlled, mixed, OUTPUT_ROOT, REPORT_ROOT)
    _checkpoint(
        "complete",
        status="complete",
        finished_at=_timestamp(),
        d02_aggregate=d02_aggregate,
        d02_figure_count=len(d02_figures),
        comparison_figure_count=len(comparison_figures),
        reports=[str(path) for path in reports],
    )


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as error:
        _checkpoint("failed", status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
