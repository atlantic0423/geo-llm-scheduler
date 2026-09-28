"""E15 automatic paired aggregation, figures and final reproducibility checks."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _paired(rows: list[dict], left: str, right: str, metric: str = "hv") -> list[dict]:
    """Collapse algorithm seeds first, then pair H/T within each base structure."""
    grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["base"], row["tariff"], row["arm"])].append(row[metric])
    result = []
    for stage, base, tariff, arm in sorted(grouped):
        if arm != right:
            continue
        a = grouped.get((stage, base, tariff, left))
        b = grouped.get((stage, base, tariff, right))
        if a and b:
            result.append(
                {
                    "stage": stage,
                    "base": base,
                    "tariff": tariff,
                    "comparison": f"{right}-{left}",
                    "metric": metric,
                    "left": mean(a),
                    "right": mean(b),
                    "difference": mean(b) - mean(a),
                }
            )
    return result


def _figure(
    path: Path, title: str, labels: list[str], values: list[float], ylabel: str = "Value"
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(values)), values, color="#3b75a8")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="#333333", lw=0.8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=300)
    plt.close(fig)


def _mean_by(rows: list[dict], key: str, value: str) -> tuple[list[str], list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(float(row[value]))
    labels = sorted(grouped)
    return labels, [mean(grouped[label]) for label in labels]


def analyze(root: Path, specs: list[JobSpec]) -> None:
    """Aggregate only complete mandatory runs, with no cross-protocol metric pooling."""
    from run_e15_campaign import _metric_rows

    by_stage: dict[str, list[JobSpec]] = defaultdict(list)
    for spec in specs:
        by_stage[spec.stage].append(spec)
    rows = []
    for stage, group in by_stage.items():
        stage_rows = _metric_rows(root, group)
        if len(stage_rows) != len(group):
            raise RuntimeError(f"Incomplete {stage} metric aggregation")
        rows.extend(stage_rows)
    analysis = root / "analysis"
    reports = root / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _csv(analysis / "all_metrics.csv", rows)
    a = [r for r in rows if r["stage"].startswith("A_")]
    b = [r for r in rows if r["stage"].startswith("B_")]
    c = [r for r in rows if r["stage"].startswith("C_")]
    d = [r for r in rows if r["stage"].startswith("D_")]
    _csv(analysis / "macrosearch_comparison.csv", a)
    _csv(analysis / "backbone_2x2.csv", b)
    _csv(analysis / "fairness_generation.csv", [r for r in rows if r["stage"].endswith("GEN")])
    _csv(analysis / "fairness_wallclock.csv", [r for r in rows if r["stage"].endswith("TIME")])
    _csv(analysis / "fairness_exact_eval.csv", c)
    _csv(analysis / "scale_100job.csv", d)
    budget_rows = []
    runtime_rows = []
    archive_rows = []
    sequential_rows = []
    coverage_rows = []
    severity_rows = []
    for spec in specs:
        run = root / "runs" / spec.key
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        funnel = json.loads((run / "budget_funnel.json").read_text(encoding="utf-8"))
        for budget, calls in summary["budget_counts"].items():
            budget_rows.append(
                {
                    "stage": spec.stage,
                    "arm": spec.arm,
                    "instance": spec.instance_seed,
                    "seed": spec.algorithm_seed,
                    "budget": budget,
                    "calls": calls,
                }
            )
        for action_budget, calls in summary.get("action_budget_counts", {}).items():
            budget_rows.append(
                {
                    "stage": spec.stage,
                    "arm": spec.arm,
                    "instance": spec.instance_seed,
                    "seed": spec.algorithm_seed,
                    "budget": action_budget,
                    "calls": calls,
                }
            )
        for action, values in funnel.items():
            budget_rows.append(
                {
                    "stage": spec.stage,
                    "arm": spec.arm,
                    "instance": spec.instance_seed,
                    "seed": spec.algorithm_seed,
                    "budget": f"A{action}",
                    "calls": values["calls"],
                }
            )
        elapsed = max(summary["elapsed"], 1e-9)
        runtime_rows.append(
            {
                "stage": spec.stage,
                "arm": spec.arm,
                "instance": spec.instance_seed,
                "seed": spec.algorithm_seed,
                "elapsed": elapsed,
                "exact": summary["counts"]["exact"],
                "evaluator_fraction": summary["timings"].get("exact", 0) / elapsed,
                "ssgs_fraction": summary["timings"].get("ssgs", 0) / elapsed,
                "macrosearch_fraction": sum(
                    v for k, v in summary["timings"].items() if k.startswith("construction:")
                )
                / elapsed,
                "archive_fraction": summary["timings"].get("archive", 0) / elapsed,
            }
        )
        archive_rows.append(
            {
                "stage": spec.stage,
                "arm": spec.arm,
                "instance": spec.instance_seed,
                "seed": spec.algorithm_seed,
                "archive_size": summary["archive_size"],
                "archive_peak": summary["archive_peak_size"],
                "archive_seconds": summary["timings"].get("archive", 0),
            }
        )
        # Detailed per-invocation records are available only in frozen trace strata.
        trace = run / "trace.jsonl"
        if trace.exists():
            with trace.open(encoding="utf-8") as handle:
                for line in handle:
                    for step in json.loads(line)["steps"]:
                        details = step.get("budget_details", {})
                        common = {
                            "stage": spec.stage,
                            "arm": spec.arm,
                            "instance": spec.instance_seed,
                            "seed": spec.algorithm_seed,
                            "action": step["action"],
                            "budget": step["budget"],
                        }
                        if step.get("sequential"):
                            sequential_rows.append({**common, **step["sequential"]})
                        if "gap" in details:
                            coverage_rows.append({**common, **details})
                        if "action_percentile" in details:
                            severity_rows.append({**common, **details})
    _csv(analysis / "macrosearch_budget_activation.csv", budget_rows)
    _csv(analysis / "runtime_breakdown.csv", runtime_rows)
    _csv(analysis / "archive_cost.csv", archive_rows)
    for name, table in (
        ("sequential_margins.csv", sequential_rows),
        ("coverage_gap.csv", coverage_rows),
        ("severity_percentile.csv", severity_rows),
    ):
        if table:
            _csv(analysis / name, table)
    paired = []
    for left, right in (
        ("F6", "C2"),
        ("F6", "S2"),
        ("F6", "SEQ"),
        ("M0", "N0"),
        ("M0", "M1"),
        ("N0", "N1"),
        ("M1", "N1"),
    ):
        paired += _paired(rows, left, right, "hv")
        paired += _paired(rows, left, right, "igd_plus")
    _csv(analysis / "paired_differences.csv", paired)
    _csv(
        analysis / "tariff_paired_analysis.csv",
        [
            {
                "stage": stage,
                "base": base,
                "arm": arm,
                "metric": "hv",
                "difference_T_minus_H": mean(v["T"]) - mean(v["H"]),
            }
            for (stage, base, arm), v in _tariff_groups(rows).items()
            if v["H"] and v["T"]
        ],
    )
    selected = json.loads((root / "selected_adaptive_macrosearch.json").read_text())
    evidence: dict[str, dict[str, float]] = {}
    for stage in ("B_GEN", "B_TIME", "C_EVAL", "D_GEN", "D_TIME"):
        evidence[stage] = {}
        for contrast in ("N0-M0", "N1-M1", "M1-M0", "N1-N0"):
            subset = [
                r["difference"]
                for r in paired
                if r["stage"] == stage and r["metric"] == "hv" and r["comparison"] == contrast
            ]
            evidence[stage][contrast] = median(subset) if subset else float("nan")
    switch_support = all(
        evidence[stage]["N0-M0"] > 0 and evidence[stage]["N1-M1"] >= 0 for stage in evidence
    )
    reverse_support = all(
        evidence[stage]["N0-M0"] < 0 and evidence[stage]["N1-M1"] < 0 for stage in evidence
    )
    atomic_json(
        analysis / "selection_evidence.json",
        {
            "selected_adaptive": selected,
            "evidence_supports_switch": True
            if switch_support
            else False
            if reverse_support
            else "ambiguous",
            "stage_median_base_paired_hv_differences": evidence,
            "status": "descriptive evidence only; no automatic algorithm decision",
        },
    )
    # Sixteen paired diagnostic figures; every PNG has a PDF twin at 300 dpi.
    charts = [
        (
            "01_stageA_hv_paired",
            "Stage A paired HV",
            [r for r in paired if r["stage"] == "A_GEN" and r["metric"] == "hv"],
            "comparison",
            "difference",
        ),
        (
            "02_budget_3_6_10",
            "Budget 3/6/10 calls",
            [r for r in budget_rows if r["budget"] in ("3", "6", "10")],
            "budget",
            "calls",
        ),
        (
            "03_per_action_budget",
            "Per-action calls",
            [r for r in budget_rows if ":B" in str(r["budget"])],
            "budget",
            "calls",
        ),
        ("04_runtime_vs_hv", "Runtime by arm", rows, "arm", "elapsed"),
        ("05_exact_vs_hv", "Exact evaluations by arm", rows, "arm", "exact"),
        ("06_sequential_margins", "Sequential marginal gains", sequential_rows, "arm", "delta_3_6"),
        ("07_coverage_gap", "Coverage gap by budget", coverage_rows, "budget", "gap"),
        (
            "08_severity_percentile",
            "Severity percentile by budget",
            severity_rows,
            "budget",
            "action_percentile",
        ),
        ("09_backbone_interaction", "2x2 backbone interaction", b, "arm", "hv"),
        (
            "10_equal_generation",
            "Equal-generation HV",
            [r for r in b if r["stage"] == "B_GEN"],
            "arm",
            "hv",
        ),
        (
            "11_equal_wallclock",
            "Equal-time HV",
            [r for r in b if r["stage"] == "B_TIME"],
            "arm",
            "hv",
        ),
        ("12_equal_exact", "Equal-exact HV", c, "arm", "hv"),
        ("13_tariff_pair", "H/T paired HV", rows, "tariff", "hv"),
        ("14_scale_50_100", "50/100-job HV", [*b, *d], "stage", "hv"),
        ("15_archive_fraction", "Archive time fraction", runtime_rows, "arm", "archive_fraction"),
        (
            "16_runtime_components",
            "Evaluator time fraction",
            runtime_rows,
            "arm",
            "evaluator_fraction",
        ),
    ]
    for name, title, table, key, value in charts:
        if table:
            labels, values = _mean_by(table, key, value)
        else:
            labels, values = ["no sampled calls"], [0.0]
        _figure(figures / name, title, labels, values, value)
    report = [
        "# E15 自动结果报告（待人工研究复核）",
        "",
        f"源码提交：`{specs[0].source_commit}`。强制矩阵完整 {len(specs)} 条。",
        "",
        "Stage A：Fixed6、Coverage-v2、Severity-v2、Sequential；"
        f"按预注册规则选出的实验候选为 **{selected['selected']}**，"
        f"相对 Fixed6 的描述性标记为 `{selected['adaptive_beats_fixed']}`。",
        "",
        "Stage B/C/D 的各主干对比、三公平口径、H/T 配对与 100-job 扩展"
        "见 `analysis/` 下同名 CSV 和 `reports/figures/` 的 16 组 PNG/PDF。",
        "",
        "## 架构证据（各 stage 的 base 配对 HV 差中位数）",
        "",
        "| 阶段 | N0−M0 | N1−M1 | M1−M0 | N1−N0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, values in evidence.items():
        report.append(
            f"| {stage} | {values['N0-M0']:.5f} | {values['N1-M1']:.5f} | "
            f"{values['M1-M0']:.5f} | {values['N1-N0']:.5f} |"
        )
    report += [
        "",
        "这一自动判断只整理证据，不改变当前 MOEA/D 主线或默认预算策略。",
        "请结合配对差分分布、IGD+、运行时间、Exact 次数和 Archive 开销进行人工复核。",
        "",
    ]
    (reports / "FINAL_E15_REPORT_CN.md").write_text("\n".join(report), encoding="utf-8")
    atomic_json(
        root / "campaign_manifest.json",
        {
            "source_commit": specs[0].source_commit,
            "selected_adaptive": selected["selected"],
            "matrix": {stage: len(group) for stage, group in by_stage.items()},
            "pilot": json.loads((root / "resource_plan.json").read_text()),
        },
    )
    with (root / "checksums.sha256").open("w", encoding="utf-8") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256" and "logs" not in path.parts:
                handle.write(f"{file_hash(path)}  {path.relative_to(root).as_posix()}\n")


def _tariff_groups(rows: list[dict]) -> dict[tuple[str, int, str], dict[str, list[float]]]:
    groups: dict[tuple[str, int, str], dict[str, list[float]]] = defaultdict(
        lambda: {"H": [], "T": []}
    )
    for row in rows:
        groups[(row["stage"], row["base"], row["arm"])][row["tariff"]].append(row["hv"])
    return groups
