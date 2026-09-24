"""Persistent resumable D03 v2 calibration, frozen evaluation, and reporting pipeline."""

from __future__ import annotations

import csv
import json
import os
import shutil
import statistics
import subprocess
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from geo_llm_scheduler.config import load_config
from geo_llm_scheduler.experiments.d03_analysis import (
    aggregate_suite,
    calibration_summary,
    create_representative_timelines,
    read_steps,
    write_csv,
    write_initial_severity_summary,
    write_policy_consensus,
)
from geo_llm_scheduler.experiments.d03_instances import (
    ALGORITHM_SEEDS,
    CONDITIONS,
    TYPE_I_CALIBRATION_SEEDS,
    TYPE_I_EVALUATION_SEEDS,
    TYPE_I_PROFILES,
    TYPE_II_CALIBRATION_SEEDS,
    TYPE_II_EVALUATION_SEEDS,
    TYPE_II_PROFILES,
    materialize,
)
from geo_llm_scheduler.experiments.diagnostic_visualization import (
    create_dataset_figures,
    summarize_steps,
)
from geo_llm_scheduler.experiments.diagnostics import DiagnosticRunSpec, run_diagnostic

REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / "outputs" / "diagnostics" / "d03_v2"
RUN_ROOT = ROOT / "runs"
STATE = ROOT / "pipeline_state.json"
CHECKPOINT = ROOT / "resume_checkpoint.json"
INSTANCE_ROOT = REPOSITORY / "instances" / "diagnostic_d03"
REPORT = ROOT / "D03_SUMMARY.md"
FIGURES = ROOT / "figures"
FROZEN = REPOSITORY / "configs" / "diagnostics" / "d03" / "frozen_selection.json"
REQUIRED = (
    "summary.json",
    "trace.jsonl",
    "qtable.json",
    "diagnostic.json",
    "rl_steps.csv",
    "rl_steps.jsonl",
    "trigger_funnel.csv",
    "initial_severity.csv",
)


@dataclass(frozen=True)
class PlannedRun:
    """A complete D03 run identity plus its generator fields."""

    phase: str
    suite_type: str
    family: str
    profile: str
    role: str
    jobs: int
    instance_seed: int
    algorithm_seed: int
    generations: int

    @property
    def scenario(self) -> str:
        if self.role == "calibration":
            return f"D03C_{self.suite_type}_{self.family}_{self.profile}"
        return f"D03I_{self.family}" if self.suite_type == "type_i" else f"D03II_{self.profile}"

    @property
    def instance_path(self) -> Path:
        generated = (
            "D03I_" + self.family if self.suite_type == "type_i" else "D03II_" + self.profile
        )
        return (
            INSTANCE_ROOT / self.role / generated / f"jobs{self.jobs}_seed{self.instance_seed}.json"
        )

    @property
    def spec(self) -> DiagnosticRunSpec:
        return DiagnosticRunSpec(
            self.phase,
            self.scenario,
            self.jobs,
            self.instance_seed,
            self.algorithm_seed,
            self.generations,
            str(self.instance_path),
            self.profile,
        )

    @property
    def directory(self) -> Path:
        return RUN_ROOT / self.phase / "runs" / self.spec.run_id


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _checkpoint(stage: str, **extra: object) -> None:
    payload = {
        "updated_at": _timestamp(),
        "pid": os.getpid(),
        "status": "running",
        "stage": stage,
        **extra,
    }
    _atomic(STATE, payload)
    _atomic(CHECKPOINT, payload)


def validate_run(run: PlannedRun) -> tuple[bool, str]:
    """Accept only complete, generation-limited, identity-matching artifacts."""
    directory = run.directory
    if not directory.is_dir():
        return False, "missing"
    for name in REQUIRED:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            return False, f"missing-or-empty:{name}"
    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "diagnostic.json").read_text(encoding="utf-8"))
        if summary["termination_reason"] != "generation_limit":
            return False, "termination"
        if metadata["run_id"] != run.spec.run_id or metadata["generations"] != run.generations:
            return False, "identity"
        with (directory / "rl_steps.csv").open(newline="", encoding="utf-8") as handle:
            if next(csv.DictReader(handle), None) is None:
                return False, "empty-steps"
        for name in ("trace.jsonl", "rl_steps.jsonl"):
            with (directory / name).open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    return False, f"truncated:{name}"
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        return False, f"parse:{error}"
    return True, "complete"


def _materialize(run: PlannedRun) -> dict[str, object]:
    return materialize(
        INSTANCE_ROOT,
        run.suite_type,
        run.family,
        run.profile,
        run.jobs,
        run.instance_seed,
        run.role,
    )


def _run_matrix(runs: list[PlannedRun], config_path: Path, stage: str) -> None:
    config = load_config(config_path)
    complete = sum(validate_run(run)[0] for run in runs)
    for index, planned in enumerate(runs, start=1):
        valid, reason = validate_run(planned)
        if valid:
            continue
        if planned.directory.exists():
            archived = (
                ROOT / "incomplete" / f"{planned.spec.run_id}__{datetime.now():%Y%m%d-%H%M%S}"
            )
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(planned.directory), str(archived))
        _materialize(planned)
        _checkpoint(
            stage,
            completed=complete,
            expected=len(runs),
            current=planned.spec.run_id,
            resume_effective=complete > 0,
            prior_reason=reason,
        )
        summary = run_diagnostic(planned.spec, config, RUN_ROOT, resume=False)
        valid, reason = validate_run(planned)
        if not valid:
            raise RuntimeError(f"Invalid completed run {planned.spec.run_id}: {reason}")
        complete += 1
        print(
            json.dumps(
                {
                    "stage": stage,
                    "completed": f"{complete}/{len(runs)}",
                    "run_id": planned.spec.run_id,
                    "elapsed": summary["elapsed"],
                }
            ),
            flush=True,
        )


def _d02_precalibration() -> dict[str, object]:
    directories = sorted((REPOSITORY / "outputs" / "diagnostics" / "mixed" / "runs").glob("*"))
    if len(directories) != 15:
        raise RuntimeError(f"D02 raw matrix must contain exactly 15 runs, found {len(directories)}")
    rows = read_steps(directories)
    result = aggregate_suite(
        rows, "D02", ROOT / "precalibration" / "d02", FIGURES / "precalibration_d02"
    )
    _checkpoint("d02-severity-scale-complete", d02=result)
    return result


def calibration_runs() -> tuple[list[PlannedRun], list[PlannedRun]]:
    type_i = [
        PlannedRun("calibration_i", "type_i", condition, profile, "calibration", 50, seed, 101, 15)
        for condition in CONDITIONS
        for profile in TYPE_I_PROFILES
        for seed in TYPE_I_CALIBRATION_SEEDS
    ]
    type_ii = [
        PlannedRun(
            "calibration_ii", "type_ii", "transition", profile, "calibration", 50, seed, 101, 15
        )
        for profile in TYPE_II_PROFILES
        for seed in TYPE_II_CALIBRATION_SEEDS
    ]
    return type_i, type_ii


def _select_profiles(type_i: list[PlannedRun], type_ii: list[PlannedRun]) -> dict[str, object]:
    records = []
    selected_i = {}
    for condition in CONDITIONS:
        candidates = []
        for order, profile in enumerate(TYPE_I_PROFILES):
            subset = [run for run in type_i if run.family == condition and run.profile == profile]
            result = calibration_summary(read_steps([run.directory for run in subset]), condition)
            candidates.append((float(cast(float, result["target_share"])), -order, profile, result))
            records.append(
                {
                    "suite_type": "type_i",
                    "family": condition,
                    "profile": profile,
                    **{key: value for key, value in result.items() if key != "counts"},
                    "counts_json": json.dumps(result["counts"], sort_keys=True),
                }
            )
        selected_i[condition] = max(candidates)[2]
    candidates_ii = []
    for order, profile in enumerate(TYPE_II_PROFILES):
        subset = [run for run in type_ii if run.profile == profile]
        result = calibration_summary(read_steps([run.directory for run in subset]))
        candidates_ii.append(
            (
                float(cast(float, result["effective_distinct_mean"])),
                float(cast(float, result["distinct_mean"])),
                int(cast(int, result["transition_count"])),
                -order,
                profile,
                result,
            )
        )
        records.append(
            {
                "suite_type": "type_ii",
                "family": "transition",
                "profile": profile,
                **{key: value for key, value in result.items() if key != "counts"},
                "counts_json": json.dumps(result["counts"], sort_keys=True),
            }
        )
    selected_ii = max(candidates_ii)[4]
    write_csv(ROOT / "calibration" / "candidate_profiles.csv", records)
    payload = {
        "frozen_at": _timestamp(),
        "selection_rule": {
            "type_i": "maximum target-condition share; predefined profile order breaks ties",
            "type_ii": "effective distinct >=10, distinct, transitions; predefined order breaks ties",
        },
        "calibration_seeds": {
            "type_i": TYPE_I_CALIBRATION_SEEDS,
            "type_ii": TYPE_II_CALIBRATION_SEEDS,
        },
        "evaluation_seeds": {
            "type_i": TYPE_I_EVALUATION_SEEDS,
            "type_ii": TYPE_II_EVALUATION_SEEDS,
        },
        "type_i_profiles": selected_i,
        "type_ii_profile": selected_ii,
        "algorithm_seeds": ALGORITHM_SEEDS,
        "jobs": 50,
        "generations": 50,
        "calibration_records": records,
    }
    FROZEN.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _freeze_commit() -> str:
    subprocess.run(["git", "add", str(FROZEN.relative_to(REPOSITORY))], cwd=REPOSITORY, check=True)
    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPOSITORY, check=False)
    if status.returncode:
        subprocess.run(
            ["git", "commit", "-m", "Freeze D03 v2 calibration selection"],
            cwd=REPOSITORY,
            check=True,
        )
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=REPOSITORY, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True, capture_output=True, text=True
    ).stdout.strip()
    return commit


def evaluation_runs(frozen: dict[str, object]) -> tuple[list[PlannedRun], list[PlannedRun]]:
    profiles = cast(dict[str, str], frozen["type_i_profiles"])
    type_i = [
        PlannedRun(
            "type_i",
            "type_i",
            condition,
            profiles[condition],
            "evaluation",
            50,
            seed,
            algorithm,
            50,
        )
        for condition in CONDITIONS
        for seed in TYPE_I_EVALUATION_SEEDS
        for algorithm in ALGORITHM_SEEDS
    ]
    profile = str(frozen["type_ii_profile"])
    type_ii = [
        PlannedRun(
            "type_ii", "type_ii", "transition", profile, "evaluation", 50, seed, algorithm, 50
        )
        for seed in TYPE_II_EVALUATION_SEEDS
        for algorithm in ALGORITHM_SEEDS
    ]
    return type_i, type_ii


def _write_manifest(runs: list[PlannedRun], path: Path) -> None:
    records = []
    seen = set()
    for run in runs:
        key = (run.suite_type, run.family, run.profile, run.instance_seed, run.role)
        if key not in seen:
            records.append(_materialize(run))
            seen.add(key)
    write_csv(path, records)


def _read_named(run_directories: list[Path], name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for directory in run_directories:
        with (directory / name).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _cross_suite(type_i_rows: list[dict[str, str]], type_ii_rows: list[dict[str, str]]) -> None:
    suites = {
        "D01 controlled": read_steps(
            sorted((REPOSITORY / "outputs/diagnostics/main/runs").glob("*"))
        ),
        "D02 mixed": read_steps(sorted((REPOSITORY / "outputs/diagnostics/mixed/runs").glob("*"))),
        "D03 Type-I": type_i_rows,
        "D03 Type-II": type_ii_rows,
    }
    rows = []
    for label, values in suites.items():
        counts = Counter(row["dominant_condition"] for row in values)
        rows.append(
            {
                "suite": label,
                "steps": len(values),
                "state_coverage": len({row["state_id"] for row in values}),
                **{f"condition_{name}": counts[name] for name in CONDITIONS},
            }
        )
    write_csv(ROOT / "aggregate/cross_suite_summary.csv", rows)


def _report(
    d02: dict[str, object],
    first: dict[str, object],
    second: dict[str, object],
    frozen: dict[str, object],
) -> None:
    type_i_rows = read_steps([run.directory for run in evaluation_runs(frozen)[0]])
    type_ii_rows = read_steps([run.directory for run in evaluation_runs(frozen)[1]])
    from geo_llm_scheduler.experiments.d03_analysis import transition_rows

    transitions = transition_rows(type_ii_rows)
    effective = [int(cast(int, row["effective_10"])) for row in transitions]
    type_i_counts = Counter(row["dominant_condition"] for row in type_i_rows)
    nonzero = [value for value in type_i_counts.values() if value]
    imbalance = max(nonzero) / min(nonzero) if nonzero else float("inf")
    REPORT.write_text(
        f"""# D03 DominantCondition Balanced Diagnostic Summary

状态：后台流水线完成后自动生成的 **D03 controlled diagnostic**。D03 Type-I、Type-II、D02 数据始终分开聚合；本报告不构成 Q-learning 优于 Random/Bandit 的性能结论。

## 冻结与完整性

- Frozen profiles：`{json.dumps(frozen["type_i_profiles"], ensure_ascii=False)}`；Type-II `{frozen["type_ii_profile"]}`。
- Type-I：{first["runs"]}/42 runs；Type-II：{second["runs"]}/15 runs；每条 50 generations。
- D02 severity 回溯在任何 D03 calibration 前完成；calibration seeds 与 evaluation seeds 不相交。

## 覆盖

- D02 state coverage：{d02["state_coverage"]}/42。
- Type-I state coverage：{first["state_coverage"]}/42；condition counts `{json.dumps(type_i_counts, ensure_ascii=False)}`；imbalance ratio {imbalance:.3f}。
- Type-II state coverage：{second["state_coverage"]}/42。
- Type-II effective distinct at n>=10：min {min(effective)}，median {statistics.median(effective)}，max {max(effective)}。

## 解释边界

Type-I 仅用于获得诊断样本，不能反推真实 workload 中七类 condition 均匀。Severity-scale 风险、condition transition、action effectiveness、state aliasing 与 Trigger sampling bias 应结合 `aggregate/` CSV 和 `figures/d03/` 图进行人工复核。现有 severity 公式和阈值在本轮完全冻结。
""",
        encoding="utf-8",
    )


def run_pipeline() -> None:
    """Run every D03 v2 stage with resume and integrity checks."""
    _checkpoint(
        "started",
        source_branch=subprocess.run(
            ["git", "branch", "--show-current"], cwd=REPOSITORY, capture_output=True, text=True
        ).stdout.strip(),
    )
    d02 = _d02_precalibration()
    type_i_cal, type_ii_cal = calibration_runs()
    _run_matrix(
        type_i_cal, REPOSITORY / "configs/diagnostics/d03/calibration.yaml", "type-i-calibration"
    )
    _run_matrix(
        type_ii_cal, REPOSITORY / "configs/diagnostics/d03/calibration.yaml", "type-ii-calibration"
    )
    frozen = _select_profiles(type_i_cal, type_ii_cal)
    freeze_commit = _freeze_commit()
    frozen["freeze_commit"] = freeze_commit
    _atomic(ROOT / "frozen_selection.json", frozen)
    _checkpoint("frozen", freeze_commit=freeze_commit, selection=frozen)
    type_i, type_ii = evaluation_runs(frozen)
    _write_manifest(type_i, ROOT / "manifests/type_i_evaluation.csv")
    _write_manifest(type_ii, ROOT / "manifests/type_ii_evaluation.csv")
    _run_matrix(type_i, REPOSITORY / "configs/diagnostics/d03/evaluation.yaml", "type-i-evaluation")
    _run_matrix(
        type_ii, REPOSITORY / "configs/diagnostics/d03/evaluation.yaml", "type-ii-evaluation"
    )
    type_i_directories = [run.directory for run in type_i]
    type_ii_directories = [run.directory for run in type_ii]
    type_i_rows = read_steps(type_i_directories)
    type_ii_rows = read_steps(type_ii_directories)
    first = aggregate_suite(
        type_i_rows, "D03 Type-I", ROOT / "aggregate/type_i", FIGURES / "type_i"
    )
    second = aggregate_suite(
        type_ii_rows, "D03 Type-II", ROOT / "aggregate/type_ii", FIGURES / "type_ii"
    )
    write_initial_severity_summary(
        _read_named(type_i_directories, "initial_severity.csv"),
        "D03 Type-I",
        ROOT / "aggregate/type_i",
    )
    write_initial_severity_summary(
        _read_named(type_ii_directories, "initial_severity.csv"),
        "D03 Type-II",
        ROOT / "aggregate/type_ii",
    )
    write_policy_consensus(type_i_directories, ROOT / "aggregate/type_i")
    write_policy_consensus(type_ii_directories, ROOT / "aggregate/type_ii")
    create_representative_timelines(type_ii_rows, ROOT / "aggregate/type_ii", FIGURES / "type_ii")
    create_dataset_figures(
        summarize_steps(type_i_rows), FIGURES / "type_i/state_action", "D03 Type-I"
    )
    create_dataset_figures(
        summarize_steps(type_ii_rows), FIGURES / "type_ii/state_action", "D03 Type-II"
    )
    _cross_suite(type_i_rows, type_ii_rows)
    _report(d02, first, second, frozen)
    complete = sum(validate_run(run)[0] for run in type_i + type_ii)
    _atomic(
        STATE,
        {
            "updated_at": _timestamp(),
            "pid": os.getpid(),
            "status": "complete",
            "stage": "complete",
            "formal_complete": complete,
            "formal_expected": 57,
            "type_i": first,
            "type_ii": second,
            "report": str(REPORT),
            "freeze_commit": freeze_commit,
        },
    )


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as error:
        _atomic(
            STATE,
            {
                "updated_at": _timestamp(),
                "pid": os.getpid(),
                "status": "failed",
                "stage": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
