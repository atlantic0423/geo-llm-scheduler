"""Prespecified, instance-level statistics for the D1 terminal internal holdout."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.d1_holdout import MASTER_SEED, validate_holdout_job
from geo_llm_scheduler.experiments.d1_routing import GROUPS
from geo_llm_scheduler.utils.numeric import EPS_NORM


def _updated_ideal_gains(row: dict) -> dict[str, float]:
    objectives = (
        [row["base_objectives"]]
        + [
            p["objectives"]
            for arm in row["groups"].values()
            for p in arm["proposals"]
            if p["on_time"] and p["feasible"]
        ]
        + [a["objectives"] for a in row["groups"].values()]
    )
    ideal = [min(row["context"]["ideal"][m], *(v[m] for v in objectives)) for m in range(2)]
    maximum, weight = row["context"]["maximum"], row["weight"]

    def scalar(values: list) -> float:
        return max(
            weight[m] * abs((values[m] - ideal[m]) / max(maximum[m] - ideal[m], EPS_NORM))
            for m in range(2)
        )

    base = scalar(row["base_objectives"])
    gains = {}
    for group, arm in row["groups"].items():
        values = [base, scalar(arm["objectives"])]
        values.extend(
            scalar(p["objectives"]) for p in arm["proposals"] if p["on_time"] and p["feasible"]
        )
        gains[group] = (base - min(values)) / max(base, 1e-12)
    return gains


def paired_inference(
    differences: list[float], sizes: list[int], draws: int = 50000, seed: int = MASTER_SEED
) -> dict:
    """Bootstrap bases within size and enumerate exact base sign flips, two-sided."""
    if not differences or len(differences) != len(sizes) or len(differences) > 20 or draws < 1:
        raise ValueError("Require one paired effect per base and at most twenty bases")
    values = np.asarray(differences, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite paired effect")
    rng = np.random.default_rng(seed)
    sampled = np.zeros(draws)
    for size in sorted(set(sizes)):
        group = values[np.asarray(sizes) == size]
        sampled += group[rng.integers(0, len(group), (draws, len(group)))].sum(axis=1)
    sampled /= len(values)
    observed = float(values.mean())
    extreme = 0
    total = 1 << len(values)
    for begin in range(0, total, 4096):
        masks = np.arange(begin, min(total, begin + 4096), dtype=np.uint32)
        signs = (
            2 * ((masks[:, None] >> np.arange(len(values), dtype=np.uint32)) & 1).astype(float) - 1
        )
        statistics = (signs * values).mean(axis=1)
        extreme += int(np.sum(np.abs(statistics) >= abs(observed) - 1e-14))
    return {
        "mean": observed,
        "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "p_exact": extreme / total,
        "bases_positive": int(np.sum(values > 0)),
        "bases_negative": int(np.sum(values < 0)),
        "n_bases": len(values),
        "bootstrap_draws": draws,
    }


def holm_adjust(values: list[float]) -> list[float]:
    """Return monotone Holm family-wise adjusted p-values in original order."""
    if any(not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid p-value")
    result = [0.0] * len(values)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=lambda i: values[i])):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        result[index] = previous
    return result


def analyse_holdout(root: Path, draws: int = 50000) -> dict:
    """Publish complete paired tables and base-level effects without selecting subgroups."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["pilot"]:
        raise ValueError("Resource pilot is not the formal quality confirmation")
    summaries = []
    flat: list[dict] = []
    per_run: dict[str, dict] = {}
    costs: dict[str, dict[str, float]] = {g: defaultdict(float) for g in GROUPS}
    for key in manifest["keys"]:
        valid, reason = validate_holdout_job(root, key)
        if not valid:
            raise ValueError(f"{key}: {reason}")
        folder = root / "jobs" / key
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        summaries.append(summary)
        rows = [
            r
            for r in json.loads((folder / "data.json").read_text(encoding="utf-8"))["rows"]
            if "groups" in r
        ]
        if not rows:
            raise ValueError("A whole source run has no usable paired cases")
        values = {}
        updated_rows = [_updated_ideal_gains(r) for r in rows]
        for group in GROUPS:
            values[group] = float(np.mean([r["groups"][group]["relative_gain"] for r in rows]))
            values[f"{group}_updated"] = float(np.mean([r[group] for r in updated_rows]))
            for m, metric in enumerate(("flow", "bill")):
                values[f"{group}_{metric}"] = float(
                    np.mean(
                        [
                            (r["base_objectives"][m] - r["groups"][group]["objectives"][m])
                            / max(r["base_objectives"][m], 1e-12)
                            for r in rows
                        ]
                    )
                )
        per_run[key] = {"base": summary["base_seed"], "jobs": summary["jobs"], **values}
        for row in rows:
            item = {
                "run": key,
                "base": summary["base_seed"],
                "jobs": summary["jobs"],
                "tariff": summary["tariff"],
                "sample_index": row["sample_index"],
                "kind": row["kind"],
                "magnitude": row["magnitude"],
                "preference": row["preference"],
                "base_scalar": row["base_scalar"],
                "shared_seconds": row["shared_seconds"],
                "positive_intent": row["groups"]["CONDITIONAL"]["intent_nonzero"] > 0,
            }
            for group in GROUPS:
                arm = row["groups"][group]
                item[f"{group}_gain"] = arm["relative_gain"]
                item[f"{group}_route"] = arm["route"]
                item[f"{group}_seconds"] = arm["seconds"]
                item[f"{group}_flow"] = arm["objectives"][0]
                item[f"{group}_bill"] = arm["objectives"][1]
                for name in (
                    "seconds",
                    "exact",
                    "extraction_seconds",
                    "projection_seconds",
                    "construction_seconds",
                    "exact_seconds",
                    "failed_projection",
                    "overshoot_seconds",
                ):
                    costs[group][name] += arm[name]
                costs[group]["late_exact"] += arm["exact"] - arm["on_time_exact"]
                costs[group]["infeasible_exact"] += arm["exact"] - arm["feasible"]
                costs[group]["TRUE_calls"] += arm["route"] == "TRUE"
                costs[group]["successes"] += arm["relative_gain"] > 1e-12
                costs[group]["below_frozen_ideal"] += sum(
                    p["below_frozen_ideal"] for p in arm["proposals"]
                )
            flat.append(item)
    base_rows = []
    for base in sorted(manifest["bases"]):
        runs = [v for v in per_run.values() if v["base"] == base]
        if len(runs) != 6:
            raise ValueError("Formal base requires all tariff/seed source runs")
        base_rows.append(
            {
                "base": base,
                "jobs": runs[0]["jobs"],
                **{g: float(np.mean([r[g] for r in runs])) for g in GROUPS},
                **{
                    f"{g}_{m}": float(np.mean([r[f"{g}_{m}"] for r in runs]))
                    for g in GROUPS
                    for m in ("updated", "flow", "bill")
                },
            }
        )
    primary = {}
    for control in ("ALWAYS_A7", "RANDOM"):
        effects = [r["CONDITIONAL"] - r[control] for r in base_rows]
        primary[f"CONDITIONAL-{control}"] = paired_inference(
            effects, [r["jobs"] for r in base_rows], draws
        )
    adjusted = holm_adjust([r["p_exact"] for r in primary.values()])
    for result, p in zip(primary.values(), adjusted):
        result["p_holm"] = p
    sensitivity = {}
    for control in ("ALWAYS_A7", "RANDOM"):
        result = paired_inference(
            [r["CONDITIONAL_updated"] - r[f"{control}_updated"] for r in base_rows],
            [r["jobs"] for r in base_rows],
            draws,
        )
        sensitivity[f"CONDITIONAL-{control}"] = {
            k: result[k] for k in ("mean", "ci95", "bases_positive", "n_bases")
        }
    descriptive: dict[str, dict] = {}
    for field in ("jobs", "tariff", "kind"):
        for level in sorted({r[field] for r in flat}):
            subset = [r for r in flat if r[field] == level]
            means = {
                b: {
                    g: float(np.mean([r[f"{g}_gain"] for r in subset if r["base"] == b]))
                    for g in GROUPS
                }
                for b in sorted({r["base"] for r in subset})
            }
            descriptive[f"{field}:{level}"] = {
                "n_bases": len(means),
                "quality": {g: float(np.mean([v[g] for v in means.values()])) for g in GROUPS},
                "conditional_minus_a7": float(
                    np.mean([v["CONDITIONAL"] - v["ALWAYS_A7"] for v in means.values()])
                ),
            }
    all_cases = len(flat)
    common_seconds = sum(r["shared_seconds"] for r in flat)
    planning_seconds = sum(s["planning_seconds"] for s in summaries)
    report = {
        "verification_status": "ANALYZED",
        "scope": manifest["scope"],
        "cases": all_cases,
        "skipped": sum(s["skipped"] for s in summaries),
        "source_snapshots": sum(s["source_snapshots"] for s in summaries),
        "n_bases": len(base_rows),
        "quality": {g: float(np.mean([r[g] for r in base_rows])) for g in GROUPS},
        "primary": primary,
        "updated_ideal_sensitivity": sensitivity,
        "updated_ideal_quality": {
            g: float(np.mean([r[f"{g}_updated"] for r in base_rows])) for g in GROUPS
        },
        "flow_bill_improvement": {
            g: {m: float(np.mean([r[f"{g}_{m}"] for r in base_rows])) for m in ("flow", "bill")}
            for g in GROUPS
        },
        "descriptive_groups": descriptive,
        "positive_intent_sources": len(
            {(r["run"], r["sample_index"]) for r in flat if r["positive_intent"]}
        ),
        "passes_quality_screen": all(
            r["mean"] > 0 and r["ci95"][0] > 0 and r["p_holm"] < 0.05 for r in primary.values()
        ),
        "costs": {
            g: {
                **costs[g],
                "mean_marginal_seconds": costs[g]["seconds"] / all_cases,
                "mean_with_shared_seconds": (
                    costs[g]["seconds"] + common_seconds + planning_seconds
                )
                / all_cases,
                "success_rate": costs[g]["successes"] / all_cases,
            }
            for g in GROUPS
        },
        "common_seconds": common_seconds,
        "planning_seconds": planning_seconds,
        "worker_seconds": sum(s["elapsed"] for s in summaries),
        "peak_rss_gib": max(s["rss_peak_gib"] for s in summaries),
        "base_rows": base_rows,
        "normalization": manifest["normalization"],
        "fallacy_scan": "11/11 reviewed; internal holdout, synthetic targets and fixed-context limitations remain",
    }
    for name, records in (("paired_cases.csv", flat), ("base_summary.csv", base_rows)):
        with (root / "analysis" / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    atomic_json(root / "analysis/report.json", report)
    lines = [
        "# D1 条件迁移：16 实例终态留出验证",
        "",
        "## Material Passport",
        "",
        "Origin Skill: academic-research-suite / experiment-agent; Mode: run+validate; Date: 2026-10-04.",
        "Verification Status: ANALYZED; Version: D1_terminal_routing_holdout_v1.",
        "",
        "## 范围与完整性",
        "",
        f"{len(summaries)} 条终态来源运行，{len(base_rows)} 个基础实例，{all_cases} 个三组配对目标，{report['skipped']} 个缺失/未改变结构槽位。",
        "该集合是未参与D1规则选择的P2内部留出；不是完全未见的新实例，不代表早中晚或完整MOEA/D。",
        "源权重与候选生成前的归一化固定；每组实际执行所选路线，随机组在同一来源内匹配TRUE调用数。",
        "",
        "## 质量及实际成本",
        "",
        "| 组别 | 基础实例等权平均标量改善 | 成功比例 | 每目标边际耗时 | 含共同工作及分配规划 |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in GROUPS:
        c = report["costs"][g]
        lines.append(
            f"| {g} | {100 * report['quality'][g]:.4f}% | {100 * c['success_rate']:.2f}% | {1000 * c['mean_marginal_seconds']:.2f} ms | {1000 * c['mean_with_shared_seconds']:.2f} ms |"
        )
    lines.extend(["", "## 两项事前主比较", ""])
    for label, r in primary.items():
        lines.append(
            f"- {label}: {100 * r['mean']:+.4f} 个百分点，95% base bootstrap区间 [{100 * r['ci95'][0]:+.4f}, {100 * r['ci95'][1]:+.4f}]；精确双侧符号翻转p={r['p_exact']:.6g}，Holm p={r['p_holm']:.6g}，{r['bases_positive']}/{r['n_bases']} base正向。"
        )
    lines.extend(["", "## 共同更新ideal敏感性（描述性）", ""])
    for label, r in sensitivity.items():
        lines.append(
            f"- {label}: {100 * r['mean']:+.4f} 个百分点，95%区间 [{100 * r['ci95'][0]:+.4f}, {100 * r['ci95'][1]:+.4f}]。只重排实际已完成的候选，不是新执行方法。"
        )
    lines.extend(["", "## 实际目标与分层（描述性）", ""])
    for g, metrics in report["flow_bill_improvement"].items():
        lines.append(
            f"- {g}: Flow相对SSGS改善 {100 * metrics['flow']:+.4f}%，Bill改善 {100 * metrics['bill']:+.4f}%；负值表示变差。"
        )
    for label, subgroup in descriptive.items():
        lines.append(
            f"- {label}: 条件组−A7 {100 * subgroup['conditional_minus_a7']:+.4f} 个百分点，{subgroup['n_bases']} base。"
        )
    lines.extend(
        [
            "",
            "## 预定质量门及边界",
            "",
            f"同时优于A7及随机对照的预定质量筛选门：{'通过' if report['passes_quality_screen'] else '未通过'}。",
            "重复种子/快照/候选不当作独立样本；两种规模内重采样，两个主质量比较Holm校正，分层只作诊断。",
            "相同quota或最多3exact不等于同成本；资源pilot不用于选择质量规则。所有晚完成及失败成本保留。",
            "固定ideal下出现新候选低于ideal时，必须报告并做共同更新ideal的敏感性检查；不能只挑有利归一化结论。",
            "11/11统计谬误检查覆盖：分层反转、生态推断、选择/碰撞偏差、基准率、均值回归、幸存者、多重探索、分叉路径、因果过推及伪重复。",
            "完整线上收益仍需新实例及等wall-clock的HV/IGD+/覆盖检验。当前默认算法不变。",
        ]
    )
    (root / "analysis/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
