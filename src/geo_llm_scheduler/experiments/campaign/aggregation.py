"""Stage and final aggregation with paired common-reference Flow/CNY metrics."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from geo_llm_scheduler.experiments.campaign.selection import (
    common_metrics,
    read_objectives,
    select_stage_winner,
)
from geo_llm_scheduler.experiments.campaign.state_store import StateStore
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Emit a CSV with a stable header even when the stage has no observations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _trace_diagnostics(path: Path) -> dict[str, Any]:
    """Summarize state coverage, transitions and no-candidate attempts from raw steps."""
    states: set[int] = set()
    conditions: Counter[str] = Counter()
    transitions = opportunities = no_candidate = calls = requested = effective = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            previous: str | None = None
            for step in event.get("steps", []):
                condition = str(step["dominant_condition"])
                states.add(int(step["state"]))
                conditions[condition] += 1
                if previous is not None:
                    opportunities += 1
                    transitions += int(previous != condition)
                previous = condition
                calls += 1
                requested += int(step.get("budget", 0))
                count = int(step.get("effective", 0))
                effective += count
                no_candidate += int(count == 0)
    return {
        "state_coverage": len(states),
        "condition_counts": json.dumps(dict(sorted(conditions.items())), ensure_ascii=False),
        "condition_transition_rate": transitions / opportunities if opportunities else 0.0,
        "no_candidate_rate": no_candidate / calls if calls else 0.0,
        "local_search_calls": calls,
        "requested_budget": requested,
        "effective_budget": effective,
    }


def stage_aggregate(
    root: Path,
    stage: str,
    arms: tuple[str, ...],
    fallback: str,
    store: StateStore,
) -> dict[str, Any]:
    """Validate paired arm cells, save all accounting and choose a provisional arm."""
    rows = store.rows(stage)
    output = root / stage
    output.mkdir(parents=True, exist_ok=True)
    by_cell: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    failures: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in rows:
        cell = f"i{row['instance_seed']}_a{row['algorithm_seed']}"
        if row["state"] == "SUCCEEDED":
            by_cell[cell][row["arm"]] = row
        elif row["state"] == "FAILED_PERMANENT":
            failures.append({"key": row["key"], "reason": row["reason"]})
        else:
            missing.append({"key": row["key"], "state": row["state"]})
    observations: list[dict[str, Any]] = []
    for cell, members in sorted(by_cell.items()):
        if not set(arms) <= set(members):
            missing.extend(
                {"key": f"{cell}:{arm}", "state": "unpaired"} for arm in arms if arm not in members
            )
            continue
        points = {
            arm: read_objectives(root / stage / "runs" / members[arm]["key"] / "objectives.csv")
            for arm in arms
        }
        metrics, provenance = common_metrics(points)
        atomic_json(output / "references" / f"{cell}.json", provenance)
        for arm in arms:
            row = members[arm]
            summary = json.loads(
                (root / stage / "runs" / row["key"] / "summary.json").read_text(encoding="utf-8")
            )
            observations.append(
                {
                    "cell": cell,
                    "arm": arm,
                    "hv": metrics[arm]["hv"],
                    "igd_plus": metrics[arm]["igd_plus"],
                    "elapsed": summary["elapsed"],
                    "exact": summary["counts"].get("exact", 0),
                    "archive_peak": summary["archive_peak_size"],
                    "termination_reason": summary["termination_reason"],
                    "overshoot": summary["time_overshoot_seconds"],
                    "run_key": row["key"],
                    **_trace_diagnostics(root / stage / "runs" / row["key"] / "trace.jsonl"),
                }
            )
    fields = [
        "cell",
        "arm",
        "hv",
        "igd_plus",
        "elapsed",
        "exact",
        "archive_peak",
        "termination_reason",
        "overshoot",
        "run_key",
        "state_coverage",
        "condition_counts",
        "condition_transition_rate",
        "no_candidate_rate",
        "local_search_calls",
        "requested_budget",
        "effective_budget",
    ]
    write_csv(output / "paired_metrics.csv", observations, fields)
    write_csv(output / "failed_jobs.csv", failures, ["key", "reason"])
    write_csv(output / "missing_jobs.csv", missing, ["key", "state"])
    selected = select_stage_winner(observations, arms, fallback)
    counts = dict(Counter(row["state"] for row in rows))
    summary = {
        "stage": stage,
        "expected_jobs": len(rows),
        "counts": counts,
        "paired_cells": selected["pair_count"],
        "selected": selected,
        "status": "complete" if not failures and not missing else "partial",
        "statistically_concluded": False,
    }
    atomic_json(output / "stage_summary.json", summary)
    (output / "selection_report.md").write_text(
        f"# {stage} 临时筛选\n\n"
        f"完成配对单元：{selected['pair_count']}；缺失：{len(missing)}；失败：{len(failures)}。\n\n"
        f"下游临时选项：`{selected['selected_arm']}`。原因：{selected['reason']}。"
        "此处仅为流水线自动解锁，不构成统计显著性或最终论文结论。\n",
        encoding="utf-8",
    )
    return summary


def final_aggregate(root: Path, store: StateStore, finish_reason: str) -> dict[str, Any]:
    """Report all completed/partial runs and compute E13 only from paired trios."""
    rows = store.rows()
    report_root = root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    e13 = [
        row for row in rows if row["stage"] in ("e13_moead", "e13_nsga2", "e13_full", "e13_extra")
    ]
    trios: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in e13:
        if row["state"] == "SUCCEEDED":
            cell = f"i{row['instance_seed']}_a{row['algorithm_seed']}"
            arm = (
                "moead"
                if row["stage"] == "e13_moead"
                else ("nsga2" if row["stage"] == "e13_nsga2" else row["arm"])
            )
            trios[cell][arm] = row
    metrics: list[dict[str, Any]] = []
    runtime: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    paired = {
        cell: members
        for cell, members in trios.items()
        if {"full", "moead", "nsga2"} <= set(members)
    }
    points_by_cell = {
        cell: {
            arm: read_objectives(root / row["stage"] / "runs" / row["key"] / "objectives.csv")
            for arm, row in members.items()
        }
        for cell, members in paired.items()
    }
    by_instance: dict[str, list[str]] = defaultdict(list)
    for cell in paired:
        by_instance[cell.split("_a", 1)[0]].append(cell)
    for instance, cells in sorted(by_instance.items()):
        pooled = {
            f"{cell}:{arm}": points
            for cell in cells
            for arm, points in points_by_cell[cell].items()
        }
        _, reference = common_metrics(pooled)
        reference_path = root / "metrics" / f"e13_{instance}_reference.json"
        atomic_json(reference_path, reference)
        lo, scale = reference["ideal"], reference["scale"]
        front = [tuple(point) for point in reference["reference_front_normalized"]]
        reference_point = tuple(reference["reference_point_normalized"])
        for cell in sorted(cells):
            members = paired[cell]
            for arm, row in members.items():
                points = [
                    ((point[0] - lo[0]) / scale[0], (point[1] - lo[1]) / scale[1])
                    for point in points_by_cell[cell][arm]
                ]
                summary = json.loads(
                    (root / row["stage"] / "runs" / row["key"] / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                metrics.append(
                    {
                        "cell": cell,
                        "arm": arm,
                        "hv": hypervolume(points, reference_point),
                        "igd_plus": igd_plus(points, front),
                        "reference": reference_path.relative_to(root).as_posix(),
                        "source_commit": summary["git_commit"],
                        "instance_hash": summary["instance_hash"],
                    }
                )
                runtime.append(
                    {
                        "cell": cell,
                        "arm": arm,
                        "elapsed": summary["elapsed"],
                        "overshoot": summary["time_overshoot_seconds"],
                        "exact": summary["counts"].get("exact", 0),
                        "generations": max(
                            (
                                json.loads(line)["generation"]
                                for line in (
                                    root / row["stage"] / "runs" / row["key"] / "trace.jsonl"
                                )
                                .read_text(encoding="utf-8")
                                .splitlines()
                            ),
                            default=-1,
                        )
                        + 1,
                        "archive_size": summary["archive_size"],
                        "archive_peak": summary["archive_peak_size"],
                    }
                )
                provenance.append(
                    {
                        "cell": cell,
                        "arm": arm,
                        "job": row["key"],
                        "input_hash": row["input_hash"],
                        "result_hash": row["result_hash"],
                        "source_commit": summary["git_commit"],
                        "source_hash": summary["source_hash"],
                        "instance_hash": summary["instance_hash"],
                        "config_hash": summary["config_hash"],
                    }
                )
    write_csv(
        root / "e13_hv_igd.csv",
        metrics,
        ["cell", "arm", "hv", "igd_plus", "reference", "source_commit", "instance_hash"],
    )
    write_csv(
        root / "e13_paired_metrics.csv",
        metrics,
        ["cell", "arm", "hv", "igd_plus", "reference", "source_commit", "instance_hash"],
    )
    write_csv(
        root / "runtime_accounting.csv",
        runtime,
        [
            "cell",
            "arm",
            "elapsed",
            "overshoot",
            "exact",
            "generations",
            "archive_size",
            "archive_peak",
        ],
    )
    write_csv(
        root / "provenance.csv",
        provenance,
        [
            "cell",
            "arm",
            "job",
            "input_hash",
            "result_hash",
            "source_commit",
            "source_hash",
            "instance_hash",
            "config_hash",
        ],
    )
    counts = dict(Counter(row["state"] for row in rows))
    selection_records: list[dict[str, Any]] = []
    for path in sorted((root / "selections").glob("selected_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        selection_records.append(
            {
                "stage": record["stage"],
                "arm": record["selected_arm"],
                "fallback": bool(record["fallback_used"]),
                "reason": record["reason"],
                "evidence": record.get("evidence", "not applicable"),
                "selection_file": path.relative_to(root).as_posix(),
            }
        )
    stage_counts = {
        stage: dict(Counter(row["state"] for row in rows if row["stage"] == stage))
        for stage in sorted({row["stage"] for row in rows})
    }
    incomplete = [
        {"key": row["key"], "stage": row["stage"], "state": row["state"], "reason": row["reason"]}
        for row in rows
        if row["state"] != "SUCCEEDED"
    ]
    summary = {
        "status": finish_reason,
        "job_counts": counts,
        "stage_counts": stage_counts,
        "incomplete_jobs": incomplete,
        "selected_for_downstream": selection_records,
        "e13_paired_cells": len(metrics) // 3,
        "e13_unpaired_cells": len(trios) - len(metrics) // 3,
        "fallback_files": [
            record["selection_file"] for record in selection_records if record["fallback"]
        ],
        "statistically_concluded": False,
    }
    atomic_json(root / "campaign_summary.json", summary)
    selections_table = "\n".join(
        f"| {record['stage']} | {record['arm']} | "
        f"{'是' if record['fallback'] else '否'} | {record['reason']} |"
        for record in selection_records
    )
    stages_table = "\n".join(
        f"| {stage} | {states.get('SUCCEEDED', 0)} | "
        f"{sum(n for state, n in states.items() if state != 'SUCCEEDED')} |"
        for stage, states in stage_counts.items()
    )
    (report_root / "FINAL_CAMPAIGN_REPORT.md").write_text(
        "# 60 小时实验批次结果\n\n"
        f"结束原因：`{finish_reason}`。成功 {counts.get('SUCCEEDED', 0)} 条；"
        f"其他状态 {len(incomplete)} 条。完整 E13 配对单元："
        f"{summary['e13_paired_cells']}；不完整单元：{summary['e13_unpaired_cells']}。\n\n"
        "## 分阶段完成情况\n\n"
        "| 阶段 | 成功 | 未成功 |\n| --- | ---: | ---: |\n"
        f"{stages_table}\n\n"
        "## 下游临时选择与回退\n\n"
        "| 阶段 | 选项 | 使用回退 | 原因 |\n| --- | --- | --- | --- |\n"
        f"{selections_table}\n\n"
        "这些选择只用于本批次下游配置，并非统计显著性结论。"
        "全部选择证据在各阶段 `stage_summary.json` 和 `paired_metrics.csv`。\n\n"
        "## 最终三算法比较与可复现性\n\n"
        "`e13_hv_igd.csv` 与 `e13_paired_metrics.csv` 只包含完整配对三元组，"
        "HV/IGD+ 使用同一实例全部完整配对结果构造共同参考。"
        "`runtime_accounting.csv` 给出用时、超时、精确评价与 Archive 规模；"
        "`provenance.csv`、`campaign_manifest.json`、`checksums.sha256` 记录来源。"
        "未成功 job 的明细在 `campaign_summary.json`，不可并入完整配对统计。\n",
        encoding="utf-8",
    )
    checksum_paths = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.name != "checksums.sha256"
        and "state.sqlite3" not in p.name
        and not p.name.endswith(".log")
    ]
    (root / "checksums.sha256").write_text(
        "".join(
            f"{file_hash(p)}  {p.relative_to(root).as_posix()}\n" for p in sorted(checksum_paths)
        ),
        encoding="utf-8",
    )
    return summary
