"""Prospective base-level inference for completed real-path confirmation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.d1_holdout_analysis import holm_adjust, paired_inference
from geo_llm_scheduler.experiments.d1_realpath import (
    MASTER_SEED,
    frozen_manifest,
    validate_realpath_job,
)
from geo_llm_scheduler.experiments.d1_routing import GROUPS


def weighted_run_values(rows: list[dict], buckets: dict[str, int], field: str) -> dict[str, float]:
    """Use event-frequency weights within phase, then equal three phase weights.

    Reservoir inclusion is uniform within phase/preference. Empty phases are
    retained as zero opportunity; a nonempty unsampled bucket is an error.
    """
    values = {g: 0.0 for g in GROUPS}
    for stage in range(3):
        population = sum(v for k, v in buckets.items() if k.startswith(f"{stage}:"))
        subset = [r for r in rows if r["stage"] == stage]
        for p in range(3):
            if buckets[f"{stage}:{p}"] > 0 and not any(r["preference"] == p for r in subset):
                raise ValueError("Nonempty reservoir is missing")
        for g in GROUPS:
            total = sum(r[field][g] * r["bucket_events"] / r["bucket_sampled"] for r in subset)
            values[g] += (total / population if population else 0.0) / 3
    return values


def analyse_realpath(root: Path, draws: int = 50000) -> dict:
    """Validate all paired artifacts before publishing pilot resources or formal inference."""
    manifest = frozen_manifest(root)
    per_run = []
    costs: dict[str, dict[str, float]] = {g: defaultdict(float) for g in GROUPS}
    natural: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    flat = []
    summaries = []
    blocks = []
    missing_phases = 0
    for key in manifest["keys"]:
        for role in ("samples", "probes"):
            valid, reason = validate_realpath_job(root, role, key)
            if not valid:
                raise ValueError(f"{role}/{key}: {reason}")
        source_summary = json.loads(
            (root / "samples" / key / "summary.json").read_text(encoding="utf-8")
        )
        summary = json.loads((root / "probes" / key / "summary.json").read_text(encoding="utf-8"))
        summaries.append({"source": source_summary, "probe": summary})
        data = json.loads((root / "probes" / key / "data.json").read_text(encoding="utf-8"))
        rows = data["rows"]
        blocks.extend(data["allocation"])
        missing_phases += sum(not any(r["stage"] == s for r in rows) for s in range(3))
        for label, counts in data["source_counts"].items():
            for name, value in counts.items():
                natural[label][name] += value
        values = weighted_run_values(rows, data["buckets"], "retained_gains")
        frozen = weighted_run_values(rows, data["buckets"], "frozen_retained_gains")
        metrics = {
            m: weighted_run_values(rows, data["buckets"], f"{m}_gains") for m in ("flow", "bill")
        }
        per_run.append(
            {
                "key": key,
                "base": summary["base_seed"],
                "jobs": summary["jobs"],
                **values,
                **{f"{g}_frozen": frozen[g] for g in GROUPS},
                **{f"{g}_{m}": metrics[m][g] for g in GROUPS for m in metrics},
            }
        )
        for row in rows:
            item = {
                "run": key,
                "base": summary["base_seed"],
                "jobs": summary["jobs"],
                "tariff": summary["tariff"],
                **{
                    k: row[k]
                    for k in (
                        "event",
                        "stage",
                        "preference",
                        "path",
                        "subproblem",
                        "probability",
                        "ms_distance",
                        "os_distance",
                    )
                },
            }
            for group in GROUPS:
                arm = row["groups"][group]
                item[f"{group}_gain"] = row["retained_gains"][group]
                item[f"{group}_route"] = arm["route"]
                item[f"{group}_flow"] = arm["objectives"][0]
                item[f"{group}_bill"] = arm["objectives"][1]
                for name in (
                    "seconds",
                    "extraction_seconds",
                    "projection_seconds",
                    "construction_seconds",
                    "exact_seconds",
                    "exact",
                    "failed_projection",
                    "duplicates",
                    "overshoot_seconds",
                ):
                    costs[group][name] += arm[name]
                costs[group]["TRUE_calls"] += arm["route"] == "TRUE"
                costs[group]["late_exact"] += arm["exact"] - arm["on_time_exact"]
                costs[group]["infeasible_exact"] += arm["exact"] - arm["feasible"]
                costs[group]["shared_seconds"] += row["shared_seconds"]
            flat.append(item)
    report = {
        "verification_status": "PILOT_VALIDATED" if manifest["pilot"] else "ANALYZED",
        "scope": manifest["scope"],
        "source_commit": manifest["source_commit"],
        "runs": len(per_run),
        "bases": len(manifest["bases"]),
        "cases": len(flat),
        "missing_phases": missing_phases,
        "natural_counts": dict(natural),
        "pooled_natural_gate_rate": natural["all"]["gate_true"] / max(1, natural["all"]["events"]),
        "source_worker_seconds": sum(s["source"]["elapsed"] for s in summaries),
        "probe_worker_seconds": sum(s["probe"]["elapsed"] for s in summaries),
        "observer_seconds": sum(s["source"]["observer_seconds"] for s in summaries),
        "peak_rss_gib": max(s[r]["rss_peak_gib"] for s in summaries for r in ("source", "probe")),
        "exact_audit_checks": sum(s["probe"]["audit_checks"] for s in summaries),
        "planning_seconds": sum(s["probe"]["planning_seconds"] for s in summaries),
        "replay_seconds": sum(s["probe"]["replay_seconds"] for s in summaries),
        "replayed_cases": sum(s["probe"]["replayed_cases"] for s in summaries),
        "replay_budget_binding": sum(s["probe"]["replay_budget_binding"] for s in summaries),
        "costs": {g: dict(v) for g, v in costs.items()},
        "random_blocks": {
            "total": len(blocks),
            "degenerate": sum(b["degenerate"] for b in blocks),
            "nondegenerate": sum(not b["degenerate"] for b in blocks),
            "symmetric_difference": sum(b["symmetric_difference"] for b in blocks),
        },
        "normalization": manifest["normalization"],
        "aggregation": manifest["aggregation"],
    }
    if not manifest["pilot"]:
        base_rows = []
        for base in manifest["bases"]:
            runs = [r for r in per_run if r["base"] == base]
            if len(runs) != 6:
                raise ValueError("Each new base requires both tariffs and three seeds")
            base_rows.append(
                {
                    "base": base,
                    "jobs": runs[0]["jobs"],
                    **{
                        field: float(np.mean([r[field] for r in runs]))
                        for field in (
                            *GROUPS,
                            *(f"{g}_{m}" for g in GROUPS for m in ("frozen", "flow", "bill")),
                        )
                    },
                }
            )
        primary = {
            f"CONDITIONAL-{g}": paired_inference(
                [r["CONDITIONAL"] - r[g] for r in base_rows],
                [r["jobs"] for r in base_rows],
                draws,
                MASTER_SEED,
            )
            for g in ("ALWAYS_A7", "RANDOM")
        }
        for result, adjusted in zip(
            primary.values(), holm_adjust([r["p_exact"] for r in primary.values()])
        ):
            result["p_holm"] = adjusted
        report.update(
            base_rows=base_rows,
            quality={g: float(np.mean([r[g] for r in base_rows])) for g in GROUPS},
            primary=primary,
            frozen_retained_sensitivity={
                f"CONDITIONAL-{g}": float(
                    np.mean([r["CONDITIONAL_frozen"] - r[f"{g}_frozen"] for r in base_rows])
                )
                for g in ("ALWAYS_A7", "RANDOM")
            },
            passes_quality_screen=all(
                r["mean"] > 0 and r["ci95"][0] > 0 and r["p_holm"] < 0.05 for r in primary.values()
            ),
            flow_bill_improvement={
                g: {m: float(np.mean([r[f"{g}_{m}"] for r in base_rows])) for m in ("flow", "bill")}
                for g in GROUPS
            },
        )
        if flat:
            with (root / "analysis/paired_cases.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
        with (root / "analysis/base_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(base_rows[0]))
            writer.writeheader()
            writer.writerows(base_rows)
    atomic_json(root / "analysis/report.json", report)
    lines = [
        "# D1 第三步：真实结构路径",
        "",
        "Material Passport: academic-research-suite / experiment-agent; run+validate; 2026-10-04.",
        f"Verification Status: {report['verification_status']}; Version: D1_realpath_v1.",
        "",
        f"{report['runs']} 次来源运行，{report['cases']} 条真实目标；{report['exact_audit_checks']} 次独立 exact 核对。",
        f"来源观测累计 {report['observer_seconds']:.3f} 秒，最大 RSS {report['peak_rss_gib']:.3f} GiB；缺失阶段 {missing_phases}。",
        "",
        "资源预跑不得用于选择质量规则；采样路径频率只代表本次被观测 F6。默认算法不变，完整线上收益尚未验证。",
    ]
    if not manifest["pilot"]:
        lines.extend(["", "## 主比较（基础实例等权）", ""])
        for label, result in report["primary"].items():
            lines.append(
                f"- {label}: {100 * result['mean']:+.4f} 个百分点，95% 区间 [{100 * result['ci95'][0]:+.4f}, {100 * result['ci95'][1]:+.4f}]；Holm p={result['p_holm']:.6g}。"
            )
        lines.extend(
            [
                "",
                "各策略用自己实际已评价候选更新 ideal 选择保留解；主评估用三组共同更新 ideal 评价保留解，未用其他组候选重新挑解。另附固定 ideal 敏感性。",
                "统计独立单位为16个基础实例，种子/电价/阶段/提案是重复测量；两个主比较做 Holm 校正。分叉路径、幸存者、伪重复及基准率等11项已检查。",
                "自然机会频率包含未改动提案；随机池退化与空等待全部保留。此阶段只确认机制适用性，第四步等墙钟 HV/IGD+ 尚未进行。",
            ]
        )
    (root / "analysis/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
