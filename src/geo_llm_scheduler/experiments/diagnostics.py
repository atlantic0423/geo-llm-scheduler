"""D01 run orchestration, step flattening and heatmap-ready aggregation."""

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments.runner import save_result
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.rl.state import PREFERENCE_LABELS, decode, preference

SEVERITY_NAMES = ("resource", "kv", "region", "tou", "demand", "compressible")
TARGET_CONDITION = {
    "D0_balanced": "Normal",
    "D1_resource": "Resource",
    "D2_kv": "KV",
    "D3_region": "Region",
    "D4_tou": "TOU",
    "D5_demand": "Demand",
    "D6_compressible": "Compressible",
}


@dataclass(frozen=True)
class DiagnosticRunSpec:
    """Identity and immutable inputs for one independent diagnostic run."""

    phase: str
    scenario: str
    jobs: int
    instance_seed: int
    algorithm_seed: int
    generations: int
    instance_path: str
    variant: str | None = None

    @property
    def run_id(self) -> str:
        """Return a collision-resistant human-readable run directory name."""
        variant = f"_{self.variant}" if self.variant else ""
        return (
            f"{self.phase}_{self.scenario}{variant}_j{self.jobs}"
            f"_i{self.instance_seed}_a{self.algorithm_seed}_g{self.generations}"
        )


STEP_FIELDS = (
    "run_id",
    "phase",
    "scenario",
    "variant",
    "jobs",
    "instance_seed",
    "algorithm_seed",
    "generation",
    "subproblem",
    "trajectory_step",
    "normalized_progress",
    "state_id",
    "preference",
    "dominant_condition",
    "search_progress",
    *tuple(f"severity_{name}" for name in SEVERITY_NAMES),
    *tuple(f"threshold_{name}" for name in SEVERITY_NAMES),
    *tuple(f"ratio_{name}" for name in SEVERITY_NAMES),
    "selected_action",
    "explore_or_greedy",
    "q_selected_before",
    "q_selected_after",
    "argmax_q_action",
    "B_act",
    "B_eff",
    "raw_construction_attempts",
    "proposals",
    "exact_evaluations",
    "feasible_count",
    "accepted",
    "positive_improvement",
    "reward",
    "scalar_before",
    "scalar_after",
    "delta_flow",
    "delta_bill",
    "archive_insertions",
    "archive_net_retained",
    "step_seconds",
    "construction_seconds",
    "budget_seconds",
    "operator_instrumentation",
)

FUNNEL_FIELDS = (
    "run_id",
    "generation",
    "preference",
    "processed_subproblems",
    "trigger_passes",
    "rl_steps",
)

INITIAL_SEVERITY_FIELDS = (
    "run_id",
    "phase",
    "scenario",
    "instance_seed",
    "algorithm_seed",
    "generation",
    "subproblem",
    "state_id",
    "preference",
    "dominant_condition",
    *tuple(f"severity_{name}" for name in SEVERITY_NAMES),
    *tuple(f"threshold_{name}" for name in SEVERITY_NAMES),
    *tuple(f"ratio_{name}" for name in SEVERITY_NAMES),
)


def flatten_steps(trace: list[dict], spec: DiagnosticRunSpec) -> list[dict[str, object]]:
    """Flatten nested engine trace rows without duplicating state extraction logic."""
    rows: list[dict[str, object]] = []
    for event in trace:
        for step in event["steps"]:
            severities = tuple(step["severities"])
            thresholds = tuple(step.get("severity_thresholds", (0.2,) * len(SEVERITY_NAMES)))
            ratios = tuple(
                step.get(
                    "severity_ratios",
                    tuple(value / threshold for value, threshold in zip(severities, thresholds)),
                )
            )
            if len(severities) != len(SEVERITY_NAMES):
                raise ValueError("Diagnostic trace requires all six severity values")
            if len(thresholds) != 6 or len(ratios) != 6:
                raise ValueError("Diagnostic trace requires six thresholds and ratios")
            row: dict[str, object] = {
                "run_id": spec.run_id,
                "phase": spec.phase,
                "scenario": spec.scenario,
                "variant": spec.variant or "",
                "jobs": spec.jobs,
                "instance_seed": spec.instance_seed,
                "algorithm_seed": spec.algorithm_seed,
                "generation": event["generation"],
                "subproblem": event["subproblem"],
                "trajectory_step": step.get("step", 0),
                "normalized_progress": step["progress"],
                "state_id": step["state"],
                "preference": step["preference"],
                "dominant_condition": step["dominant_condition"],
                "search_progress": step["search_progress"],
                "selected_action": f"A{step['action']}",
                "explore_or_greedy": "explore" if step["explore"] else "greedy",
                "q_selected_before": step.get("q_selected_before", 0.0),
                "q_selected_after": step.get("q_selected_after", 0.0),
                "argmax_q_action": f"A{step.get('argmax_q_action', step['action'])}",
                "B_act": step["budget"],
                "B_eff": step.get("effective_budget", step["exact_evaluations"]),
                "raw_construction_attempts": step["attempts"],
                "proposals": step["proposals"],
                "exact_evaluations": step["exact_evaluations"],
                "feasible_count": step["feasible"],
                "accepted": int(step["accepted"]),
                "positive_improvement": int(step.get("positive_improvement", step["reward"] > 0)),
                "reward": step["reward"],
                "scalar_before": step["scalar_before"],
                "scalar_after": step["scalar_after"],
                "delta_flow": step["delta_flow"],
                "delta_bill": step["delta_bill"],
                "archive_insertions": step["archive_insertions"],
                "archive_net_retained": step["archive_net_retained"],
                "step_seconds": step["seconds"],
                "construction_seconds": step["construction_seconds"],
                "budget_seconds": step["budget_seconds"],
                "operator_instrumentation": json.dumps(
                    step["operator_instrumentation"], ensure_ascii=False, sort_keys=True
                ),
            }
            row.update(
                {f"severity_{name}": value for name, value in zip(SEVERITY_NAMES, severities)}
            )
            row.update(
                {f"threshold_{name}": value for name, value in zip(SEVERITY_NAMES, thresholds)}
            )
            row.update({f"ratio_{name}": value for name, value in zip(SEVERITY_NAMES, ratios)})
            rows.append(row)
    return rows


def _write_csv(
    path: Path, rows: list[dict[str, object]], fields: Iterable[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or sorted({key for row in rows for key in row}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_trigger_funnel(
    trace: list[dict], spec: DiagnosticRunSpec, population: int
) -> list[dict[str, object]]:
    """Count processed, triggered, and RL-step events by generation and preference."""
    counts: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for event in trace:
        label = PREFERENCE_LABELS[preference(event["subproblem"], population)]
        key = (int(event["generation"]), label)
        counts[key][0] += 1
        counts[key][1] += int(bool(event["trigger"]))
        counts[key][2] += len(event["steps"])
    return [
        {
            "run_id": spec.run_id,
            "generation": generation,
            "preference": label,
            "processed_subproblems": values[0],
            "trigger_passes": values[1],
            "rl_steps": values[2],
        }
        for (generation, label), values in sorted(counts.items())
    ]


def flatten_initial_severities(
    snapshot: list[dict], spec: DiagnosticRunSpec
) -> list[dict[str, object]]:
    """Flatten generation-zero severity observations without creating RL visits."""
    rows = []
    for item in snapshot:
        values = tuple(item["severities"])
        thresholds = tuple(item["severity_thresholds"])
        ratios = tuple(item["severity_ratios"])
        row: dict[str, object] = {
            "run_id": spec.run_id,
            "phase": spec.phase,
            "scenario": spec.scenario,
            "instance_seed": spec.instance_seed,
            "algorithm_seed": spec.algorithm_seed,
            "generation": 0,
            "subproblem": item["subproblem"],
            "state_id": item["state"],
            "preference": item["preference"],
            "dominant_condition": item["dominant_condition"],
        }
        row.update({f"severity_{name}": value for name, value in zip(SEVERITY_NAMES, values)})
        row.update({f"threshold_{name}": value for name, value in zip(SEVERITY_NAMES, thresholds)})
        row.update({f"ratio_{name}": value for name, value in zip(SEVERITY_NAMES, ratios)})
        rows.append(row)
    return rows


def run_diagnostic(
    spec: DiagnosticRunSpec,
    base_config: Config,
    output_root: str | Path,
    *,
    resume: bool = False,
) -> dict[str, object]:
    """Run one isolated diagnostic and save standard plus step-level artifacts."""
    destination = Path(output_root) / spec.phase / "runs" / spec.run_id
    summary_path = destination / "summary.json"
    if destination.exists() and any(destination.iterdir()):
        if resume and summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"Diagnostic output already exists: {destination}")
    problem = load_instance(spec.instance_path)
    config = replace(
        base_config,
        instance=spec.instance_path,
        seed=spec.algorithm_seed,
        generations=spec.generations,
        output=str(destination),
    )
    result = run(problem, config)
    instance_hash = hashlib.sha256(Path(spec.instance_path).read_bytes()).hexdigest()
    summary = save_result(result, config, instance_hash, destination)
    metadata = {
        **asdict(spec),
        "run_id": spec.run_id,
        "instance_hash": instance_hash,
        "output_directory": destination.as_posix(),
    }
    summary["diagnostic"] = metadata
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "diagnostic.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = flatten_steps(result.trace, spec)
    _write_csv(destination / "rl_steps.csv", rows, STEP_FIELDS)
    (destination / "rl_steps.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    _write_csv(
        destination / "trigger_funnel.csv",
        flatten_trigger_funnel(result.trace, spec, config.population),
        FUNNEL_FIELDS,
    )
    _write_csv(
        destination / "initial_severity.csv",
        flatten_initial_severities(result.initial_severity_snapshot, spec),
        INITIAL_SEVERITY_FIELDS,
    )
    return summary


def _read_steps(path: Path) -> list[dict[str, object]]:
    numeric_int = {
        "jobs",
        "instance_seed",
        "algorithm_seed",
        "generation",
        "subproblem",
        "state_id",
        "B_act",
        "raw_construction_attempts",
        "proposals",
        "exact_evaluations",
        "feasible_count",
        "accepted",
        "archive_insertions",
        "archive_net_retained",
        "trajectory_step",
        "B_eff",
        "positive_improvement",
    }
    numeric_float = {
        "normalized_progress",
        "reward",
        "scalar_before",
        "scalar_after",
        "delta_flow",
        "delta_bill",
        "step_seconds",
        "construction_seconds",
        "budget_seconds",
        "q_selected_before",
        "q_selected_after",
        *{f"severity_{name}" for name in SEVERITY_NAMES},
        *{f"threshold_{name}" for name in SEVERITY_NAMES},
        *{f"ratio_{name}" for name in SEVERITY_NAMES},
    }
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for key in numeric_int:
                row[key] = int(raw[key])
            for key in numeric_float:
                row[key] = float(raw[key])
            rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _as_int(value: object) -> int:
    return int(str(value))


def _as_float(value: object) -> float:
    return float(str(value))


def aggregate_diagnostics(
    output_root: str | Path,
    *,
    phases: tuple[str, ...] | None = None,
    manifest: str | Path | None = None,
    aggregate_subdir: str = "aggregate",
) -> dict[str, int]:
    """Aggregate raw diagnostic artifacts into direct heatmap and profiling tables."""
    root = Path(output_root)
    aggregate = root / aggregate_subdir
    run_rows: list[dict[str, object]] = []
    all_steps: list[dict[str, object]] = []
    summaries: list[tuple[dict, dict]] = []
    for metadata_path in sorted(root.rglob("diagnostic.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if phases is not None and metadata["phase"] not in phases:
            continue
        summary = json.loads((metadata_path.parent / "summary.json").read_text(encoding="utf-8"))
        steps = _read_steps(metadata_path.parent / "rl_steps.csv")
        all_steps.extend(steps)
        summaries.append((metadata, summary))
        run_rows.append(
            {
                **{
                    key: metadata[key]
                    for key in (
                        "run_id",
                        "phase",
                        "scenario",
                        "variant",
                        "jobs",
                        "instance_seed",
                        "algorithm_seed",
                        "generations",
                        "instance_hash",
                    )
                },
                "elapsed": summary["elapsed"],
                "termination_reason": summary["termination_reason"],
                "time_budget_seconds": summary["time_budget_seconds"],
                "time_overshoot_seconds": summary["time_overshoot_seconds"],
                "rl_steps": len(steps),
                "trigger_rate": summary["trigger_rate"],
                "trigger_success_rate": summary["trigger_success_rate"],
                "exact_evaluations": summary["counts"].get("exact", 0),
                "ssgs_rebuilds": summary["counts"].get("ssgs", 0),
                "archive_final_size": summary["archive_size"],
                "archive_peak_size": summary["archive_peak_size"],
                "trace_hash": summary["trace_hash"],
                "git_commit": summary["git_commit"],
                "git_branch": summary["git_branch"],
                "git_dirty": summary["git_dirty"],
                "config_hash": summary["config_hash"],
                "source_hash": summary["source_hash"],
            }
        )
    if not summaries:
        raise ValueError("No diagnostic runs found for aggregation")
    _write_csv(aggregate / "run_summary.csv", run_rows)

    scenarios = sorted({(str(row["phase"]), str(row["scenario"])) for row in all_steps})
    state_groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    action_groups: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    operator_groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    scenario_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in all_steps:
        phase, scenario, state, action = (
            str(row["phase"]),
            str(row["scenario"]),
            _as_int(row["state_id"]),
            str(row["selected_action"]),
        )
        state_groups[phase, scenario, state].append(row)
        action_groups[phase, scenario, state, action].append(row)
        operator_groups[phase, scenario, action].append(row)
        scenario_groups[phase, scenario].append(row)

    state_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    success_rows: list[dict[str, object]] = []
    reward_rows: list[dict[str, object]] = []
    for phase, scenario in scenarios:
        for state in range(42):
            preference, condition, progress = decode(state)
            visits = state_groups[phase, scenario, state]
            state_rows.append(
                {
                    "phase": phase,
                    "scenario": scenario,
                    "state_id": state,
                    "preference": preference,
                    "dominant_condition": condition,
                    "search_progress": progress,
                    "visits": len(visits),
                }
            )
            for action_id in range(1, 9):
                action = f"A{action_id}"
                values = action_groups[phase, scenario, state, action]
                rewards = [_as_float(value["reward"]) for value in values]
                count_rows.append(
                    {
                        "phase": phase,
                        "scenario": scenario,
                        "state_id": state,
                        "selected_action": action,
                        "selections": len(values),
                        "updates": len(values),
                    }
                )
                success_rows.append(
                    {
                        "phase": phase,
                        "scenario": scenario,
                        "state_id": state,
                        "selected_action": action,
                        "selections": len(values),
                        "accepted": sum(_as_int(value["accepted"]) for value in values),
                        "positive_reward": sum(_as_float(value["reward"]) > 0 for value in values),
                        "acceptance_rate": sum(_as_int(value["accepted"]) for value in values)
                        / max(1, len(values)),
                        "positive_reward_rate": sum(
                            _as_float(value["reward"]) > 0 for value in values
                        )
                        / max(1, len(values)),
                    }
                )
                reward_rows.append(
                    {
                        "phase": phase,
                        "scenario": scenario,
                        "state_id": state,
                        "selected_action": action,
                        "selections": len(values),
                        "mean_reward": _mean(rewards),
                        "median_reward": _median(rewards),
                        "mean_scalar_improvement": _mean(
                            [
                                _as_float(value["scalar_before"]) - _as_float(value["scalar_after"])
                                for value in values
                            ]
                        ),
                    }
                )
    _write_csv(aggregate / "state_visits.csv", state_rows)
    _write_csv(aggregate / "state_action_counts.csv", count_rows)
    _write_csv(aggregate / "state_action_success.csv", success_rows)
    _write_csv(aggregate / "state_action_reward.csv", reward_rows)

    operator_rows: list[dict[str, object]] = []
    specialty_rows: list[dict[str, object]] = []
    for phase, scenario in scenarios:
        for action_id in range(1, 9):
            action = f"A{action_id}"
            values = operator_groups[phase, scenario, action]
            calls = len(values)
            feasible = sum(_as_int(value["feasible_count"]) for value in values)
            exact = sum(_as_int(value["exact_evaluations"]) for value in values)
            rewards = [_as_float(value["reward"]) for value in values]
            operator_rows.append(
                {
                    "phase": phase,
                    "scenario": scenario,
                    "action": action,
                    "selections": calls,
                    "calls": calls,
                    "raw_construction_attempts": sum(
                        _as_int(value["raw_construction_attempts"]) for value in values
                    ),
                    "proposals": sum(_as_int(value["proposals"]) for value in values),
                    "exact_evaluations": exact,
                    "feasible_count": feasible,
                    "feasible_rate": feasible / max(1, exact),
                    "accepted": sum(_as_int(value["accepted"]) for value in values),
                    "local_acceptance_rate": sum(_as_int(value["accepted"]) for value in values)
                    / max(1, calls),
                    "positive_reward": sum(reward > 0 for reward in rewards),
                    "positive_reward_rate": sum(reward > 0 for reward in rewards) / max(1, calls),
                    "mean_reward": _mean(rewards),
                    "median_reward": _median(rewards),
                    "mean_scalar_improvement": _mean(
                        [
                            _as_float(value["scalar_before"]) - _as_float(value["scalar_after"])
                            for value in values
                        ]
                    ),
                    "mean_delta_flow": _mean([_as_float(value["delta_flow"]) for value in values]),
                    "mean_delta_bill": _mean([_as_float(value["delta_bill"]) for value in values]),
                    "archive_insertions": sum(
                        _as_int(value["archive_insertions"]) for value in values
                    ),
                    "archive_net_retained": sum(
                        _as_int(value["archive_net_retained"]) for value in values
                    ),
                    "cpu_seconds": sum(_as_float(value["step_seconds"]) for value in values),
                    "cpu_seconds_per_call": sum(
                        _as_float(value["step_seconds"]) for value in values
                    )
                    / max(1, calls),
                    "exact_per_call": exact / max(1, calls),
                }
            )
            instruments = [json.loads(str(row["operator_instrumentation"])) for row in values]
            if action == "A7":
                gains = [
                    float(gain) for item in instruments for gain in item.get("g_pack_values", [])
                ]
                specialty_rows.append(
                    {
                        "phase": phase,
                        "scenario": scenario,
                        "action": action,
                        "calls": calls,
                        "proposals": sum(_as_int(value["proposals"]) for value in values),
                        "exact_evaluations": exact,
                        "accepted": sum(_as_int(value["accepted"]) for value in values),
                        "cpu_seconds": sum(_as_float(value["step_seconds"]) for value in values),
                        "positive_compression_operations": sum(
                            int(item.get("positive_compression_operations", 0))
                            for item in instruments
                        ),
                        "positive_compression_moves": sum(
                            int(item.get("positive_compression_moves", 0)) for item in instruments
                        ),
                        "g_pack_count": len(gains),
                        "g_pack_mean": _mean(gains),
                        "g_pack_median": _median(gains),
                        "target_region_observations": "",
                    }
                )
            elif action == "A8":
                regions = [
                    str(item["target_region"]) for item in instruments if "target_region" in item
                ]
                specialty_rows.append(
                    {
                        "phase": phase,
                        "scenario": scenario,
                        "action": action,
                        "calls": calls,
                        "proposals": sum(_as_int(value["proposals"]) for value in values),
                        "exact_evaluations": exact,
                        "accepted": sum(_as_int(value["accepted"]) for value in values),
                        "cpu_seconds": sum(_as_float(value["step_seconds"]) for value in values),
                        "singleton_attempts": sum(
                            int(item.get("singleton_attempts", 0)) for item in instruments
                        ),
                        "coalition_attempts": sum(
                            int(item.get("coalition_attempts", 0)) for item in instruments
                        ),
                        "repair_successes": sum(
                            int(item.get("repair_successes", 0)) for item in instruments
                        ),
                        "strict_peak_reductions": sum(
                            int(item.get("strict_peak_reductions", 0)) for item in instruments
                        ),
                        "mean_tied_peak_count": _mean(
                            [float(item.get("tied_peak_count", 0)) for item in instruments]
                        ),
                        "mean_original_peak": _mean(
                            [float(item.get("original_peak", 0)) for item in instruments]
                        ),
                        "target_region_observations": json.dumps(regions),
                    }
                )
    _write_csv(aggregate / "operator_diagnostics.csv", operator_rows)
    _write_csv(aggregate / "a7_a8_diagnostics.csv", specialty_rows)

    incidence_rows: list[dict[str, object]] = []
    for phase, scenario in scenarios:
        values = scenario_groups[phase, scenario]
        incidence: dict[str, object] = {
            "phase": phase,
            "scenario": scenario,
            "rl_steps": len(values),
        }
        for name in SEVERITY_NAMES:
            samples = [_as_float(value[f"severity_{name}"]) for value in values]
            incidence[f"{name}_mean"] = _mean(samples)
            incidence[f"{name}_median"] = _median(samples)
        for condition in ("Normal", "Resource", "KV", "Region", "TOU", "Demand", "Compressible"):
            incidence[f"condition_{condition}"] = sum(
                value["dominant_condition"] == condition for value in values
            ) / max(1, len(values))
        incidence_rows.append(incidence)
    baseline = {
        (str(item["phase"]), "D0_balanced"): item
        for item in incidence_rows
        if item["scenario"] == "D0_balanced"
    }
    for item in incidence_rows:
        target_condition = TARGET_CONDITION.get(str(item["scenario"]))
        if target_condition is None:
            item["target_condition"] = ""
            item["target_condition_proportion"] = None
            item["target_vs_d0_delta"] = None
            item["target_vs_d0_ratio"] = None
            continue
        proportion = _as_float(item[f"condition_{target_condition}"])
        base_row = baseline.get((str(item["phase"]), "D0_balanced"))
        base = _as_float(base_row[f"condition_{target_condition}"]) if base_row else 0.0
        item["target_condition"] = target_condition
        item["target_condition_proportion"] = proportion
        item["target_vs_d0_delta"] = proportion - base
        item["target_vs_d0_ratio"] = proportion / base if base > 0 else None
    _write_csv(aggregate / "scenario_condition_incidence.csv", incidence_rows)

    runtime_rows: list[dict[str, object]] = []
    archive_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    for metadata, summary in summaries:
        timings = summary["timings"]
        modules = {
            "initialization": timings.get("initialization", 0.0),
            "ssgs": timings.get("ssgs", 0.0),
            "exact_evaluator": timings.get("exact", 0.0),
            "A1_A6_construction": sum(
                timings.get(f"construction:A{action}", 0.0) for action in range(1, 7)
            ),
            "A7": timings.get("construction:A7", 0.0),
            "A8": timings.get("construction:A8", 0.0),
            "RightShiftPolish": timings.get("polish", 0.0),
            "Archive": timings.get("archive", 0.0),
            "budget_bookkeeping": timings.get("budget", 0.0),
            "total": summary["elapsed"],
        }
        for module, seconds in modules.items():
            runtime_rows.append(
                {
                    "run_id": metadata["run_id"],
                    "phase": metadata["phase"],
                    "scenario": metadata["scenario"],
                    "module": module,
                    "seconds": seconds,
                    "share_of_total": float(seconds) / max(float(summary["elapsed"]), 1e-15),
                }
            )
        archive_rows.append(
            {
                "run_id": metadata["run_id"],
                "phase": metadata["phase"],
                "scenario": metadata["scenario"],
                "final_size": summary["archive_size"],
                "peak_size": summary["archive_peak_size"],
                "attempts": summary["archive_attempts"],
                "insertions": summary["archive_insertions"],
                "growth_ratio": float(summary["archive_peak_size"])
                / max(1, int(summary["archive_size"])),
            }
        )
        if metadata["phase"] == "scale":
            scale_rows.append(
                {
                    "run_id": metadata["run_id"],
                    "jobs": metadata["jobs"],
                    "generations": metadata["generations"],
                    "elapsed": summary["elapsed"],
                    "rl_steps": sum(1 for row in all_steps if row["run_id"] == metadata["run_id"]),
                    "trigger_hits": summary["counts"].get("trigger_hits", 0),
                    "exact_evaluations": summary["counts"].get("exact", 0),
                    "ssgs_rebuilds": summary["counts"].get("ssgs", 0),
                    "ssgs_seconds": timings.get("ssgs", 0.0),
                    "exact_seconds": timings.get("exact", 0.0),
                    "A7_seconds": timings.get("construction:A7", 0.0),
                    "A8_seconds": timings.get("construction:A8", 0.0),
                    "polish_seconds": timings.get("polish", 0.0),
                    "archive_final_size": summary["archive_size"],
                    "archive_peak_size": summary["archive_peak_size"],
                    "trace_bytes": (Path(metadata["output_directory"]) / "trace.jsonl")
                    .stat()
                    .st_size,
                }
            )
    _write_csv(aggregate / "runtime_breakdown.csv", runtime_rows)
    _write_csv(aggregate / "archive_diagnostics.csv", archive_rows)
    _write_csv(aggregate / "scale_calibration.csv", scale_rows)

    manifest_path = Path(manifest) if manifest is not None else None
    if manifest_path is not None and manifest_path.exists():
        (aggregate / "diagnostic_manifest.csv").write_bytes(manifest_path.read_bytes())
    return {
        "runs": len(summaries),
        "rl_steps": len(all_steps),
        "state_action_rows": len(count_rows),
    }
