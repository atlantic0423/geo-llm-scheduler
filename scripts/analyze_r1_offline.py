"""Conservative replay analysis: subtract all existing final-population value.

The preterminal checkpoint may precede the exact end of structural search. Its
union with the final population is therefore the baseline for incremental HV.
No sampled pair is treated as an independent experimental replicate.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import r1_offline as replay


def _analyze(task: tuple[str, str]) -> dict:
    path, results = map(Path, task)
    report = replay._read(path)
    summary = report["summary"]
    matches = list(results.glob(f"formal_panel_shard*/runs/{summary['key']}/{replay.ARM}"))
    assert len(matches) == 1
    run = matches[0]
    config = replay._read(run / "config.json")
    problem = replay.load_instance(run.parents[2] / "instances" / config["instance"])
    population = [
        replay.restore_candidate(c) for c in replay._read(run / "native_population.json.gz")
    ]
    archive = [replay.restore_candidate(c) for c in replay._read(run / "archive.json.gz")]
    snapshots = [json.loads(s) for s in (run / "checkpoints.jsonl").read_text().splitlines()]
    baseline = sorted(
        set(tuple(p) for p in snapshots[-1]["front"])
        | {c.evaluation.objectives for c in population}
    )
    final = [c.evaluation.objectives for c in archive]
    ref = tuple(summary["hv_reference"])
    hv = replay.hypervolume(baseline, ref)
    final_hv = replay.hypervolume(final, ref)
    trial_points = []
    rescued_points = []
    rows = []
    for row in report["rows"]:
        donor, recipient = population[row["donor"]], population[row["recipient"]]
        g, schedule = replay.mix_region(problem, donor, recipient, row["region"])
        exact = replay.evaluate(problem, g, schedule)
        assert exact.feasible
        point = exact.objectives
        delta = (replay.hypervolume(baseline + [point], ref) - hv) / hv
        final_delta = (replay.hypervolume(final + [point], ref) - final_hv) / final_hv
        assert replay.close(final_delta, row["final_hv_gain"], 1e-10)
        updated = dict(row, objectives=point, conservative_pre_hv_gain=delta)
        rows.append(updated)
        trial_points.append(point)
        if row["rescued"]:
            rescued_points.append(point)
    return dict(
        key=summary["key"],
        n=summary["n"],
        regions=summary["regions"],
        rows=rows,
        eligible=summary["eligible_triples"],
        total=summary["pair_regions"],
        nonreproducible=summary["nonreproducible_slots"],
        parent_error=summary["max_parent_start_error"],
        unique_parents=summary["unique_phenotypes"],
        conservative_pre_gain=(replay.hypervolume(baseline + trial_points, ref) - hv) / hv,
        final_gain=(replay.hypervolume(final + trial_points, ref) - final_hv) / final_hv,
        rescued_pre_gain=(replay.hypervolume(baseline + rescued_points, ref) - hv) / hv,
        rescued_final_gain=(replay.hypervolume(final + rescued_points, ref) - final_hv) / final_hv,
        normalization=summary["context_policy"],
    )


def _bootstrap_mean(values: list[float], seed: int) -> dict:
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(3000))
    return dict(mean=statistics.mean(values), ci95=[boot[75], boot[2924]])


def _summarize(group: list[dict], seed: int) -> dict:
    all_rows = [row for report in group for row in report["rows"]]

    def rates(field: str) -> list[float]:
        return [
            sum(bool(row[field]) for row in report["rows"]) / len(report["rows"])
            if report["rows"]
            else 0.0
            for report in group
        ]

    result: dict[str, Any] = dict(
        instances=len(group),
        trials=len(all_rows),
        eligible_rate=_bootstrap_mean([r["eligible"] / r["total"] for r in group], seed),
        rescue_rate=_bootstrap_mean(rates("rescued"), seed + 1),
        block_rate=_bootstrap_mean(rates("block_improves"), seed + 2),
        rescue_count=sum(row["rescued"] for row in all_rows),
        rescue_instances=sum(any(row["rescued"] for row in r["rows"]) for r in group),
        whole_rejected=sum(not row["whole_improves"] for row in all_rows),
        final_archive_acceptance=sum(row["enters_final_archive"] for row in all_rows),
        horizon_changes=sum(row["horizon_changed"] for row in all_rows),
        incorrect_naive_bill=sum(abs(row["naive_bill_error"]) > 1e-9 for row in all_rows),
        other_region_corrections=sum(
            abs(row["other_regions_bill_correction"]) > 1e-9 for row in all_rows
        ),
        max_naive_bill_error=max((abs(row["naive_bill_error"]) for row in all_rows), default=0),
        cost_median_ms={
            k: 1000 * statistics.median(row[k] for row in all_rows)
            for k in ("construction_seconds", "decode_seconds", "exact_seconds", "archive_seconds")
        },
    )
    for field in ("conservative_pre_gain", "final_gain", "rescued_pre_gain", "rescued_final_gain"):
        values = [r[field] for r in group]
        result[field] = dict(
            mean=statistics.mean(values),
            median=statistics.median(values),
            maximum=max(values),
            positive_instances=sum(v > 1e-9 for v in values),
        )
    return result


def main() -> None:
    """Write per-instance statistics, audit receipt and a cautious Chinese report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    args = parser.parse_args()
    files = sorted((args.panel / "runs").glob("*.json"))
    assert len(files) == 120
    manifest = replay._read(args.base_manifest)
    records = {row["key"]: row for row in manifest["instances"]}
    assert {p.stem.rsplit("_", 1)[0] for p in files} == set(records)
    for path in files:
        report = replay._read(path)
        expected = records[path.stem.rsplit("_", 1)[0]]["sha256"]
        assert report["summary"]["input_sha256"]["base_instance"] == expected
    args.out.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(_analyze, [(str(p), str(args.results)) for p in files]))
    groups = {
        str(n): _summarize([r for r in reports if r["n"] == n], 20261112 + n)
        for n in (50, 100, 200)
    }
    for group in groups.values():
        assert group["instances"] == 40
    result = dict(
        status="verified_offline_proxy_not_online_effect",
        by_size=groups,
        nonreproducible_slots=sum(r["nonreproducible"] for r in reports),
        unique_parent_audits=sum(r["unique_parents"] for r in reports),
        max_parent_start_error=max(r["parent_error"] for r in reports),
        max_mixed_start_error=max(row["max_start_error"] for r in reports for row in r["rows"]),
        total_trials=sum(len(r["rows"]) for r in reports),
        input_manifest_verified=True,
        material_passport=dict(
            origin="academic-research-suite",
            mode="validate",
            date="2026-11-12",
            version="r1-offline-v1",
        ),
    )
    replay._write(args.out / "analysis.json", result)
    replay._write(args.out / "instances.json", reports)
    flat = [
        dict(key=r["key"], n=r["n"], regions=r["regions"], **row)
        for r in reports
        for row in r["rows"]
    ]
    with (args.out / "trials_corrected.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        "# 区域块复用离线诊断",
        "",
        "状态：已完成离线正确性与机会诊断；未运行在线R1算法，未晋级工作版。",
        "",
        "## 数据与验证口径",
        "",
        "当前含GS4工作版、seed 16、冻结120实例（50/100/200各40）。使用回收包原始负载功率输入：beta=0.4、Prefill系数1、Decode系数0.25；逐例校验原始SHA，并核对相对基础面板仅增加功率模型字段。生产源码逐文件核对运行版本，Windows换行统一LF后哈希一致。",
        "",
        f"全部12000种群槽位核查，去重后核验{result['unique_parent_audits']}个输入表型；{result['total_trials']}个交换候选全部经SSGS重解码、exact可行性与目标复核、独立逐区间电费积分。",
        f"不可重解码槽位：{result['nonreproducible_slots']}；父代开始时刻最大误差：{result['max_parent_start_error']:.3g}秒；混合候选最大误差：{result['max_mixed_start_error']:.3g}秒。",
        "",
        "## 机会与实际增益",
        "",
        "每例在实际20槽位邻域中排除自身，枚举有向供体—接收者—区域，取同任务集合、非空且时序/机器有差异的块；固定随机种子均匀抽取最多96个。各候选独立、不链式、不改变原种群。表内救回比例以已抽取合格候选为分母，置信区间按实例重采样，不把解对当独立重复。",
        "",
        "|任务数|抽样交换|整解失败但区域成功|涉及实例|救回比例（实例均值及95%区间）|扩展最终档案HV的实例|",
        "|---|---:|---:|---:|---|---:|",
    ]
    for n, g in groups.items():
        rate = g["rescue_rate"]
        lo, hi = rate["ci95"]
        lines.append(
            f"|{n}|{g['trials']}|{g['rescue_count']}|{g['rescue_instances']}/40|{100 * rate['mean']:.2f}% [{100 * lo:.2f}%, {100 * hi:.2f}%]|{g['final_gain']['positive_instances']}/40|"
        )
    lines += [
        "",
        "精修前对照取最后一个精修前检查点前沿与最终种群的并集，排除检查点之后正常演化带来的已有价值。最终对照使用原完整档案。归一化参考由原有解固定，不随交换候选改变；HV表报同实例全部抽样候选的并集增益，不能逐候选相加，也不是相同墙钟下的在线算法提升。",
        "",
        "|任务数|精修前保守并集HV中位增益|最终档案并集HV中位增益|最终档案并集HV均值增益|",
        "|---|---:|---:|---:|",
    ]
    for n, g in groups.items():
        lines.append(
            f"|{n}|{100 * g['conservative_pre_gain']['median']:.6f}%|{100 * g['final_gain']['median']:.6f}%|{100 * g['final_gain']['mean']:.6f}%|"
        )
    lines += [
        "",
        "## 成本与全局时域耦合",
        "",
        "|任务数|构造中位ms|解码中位ms|exact中位ms|档案尝试中位ms|H变化次数|直接拼旧区域账单出错次数|未修改区域账单变化次数|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, g in groups.items():
        costs = g["cost_median_ms"]
        lines.append(
            f"|{n}|"
            + "|".join(f"{v:.3f}" for v in costs.values())
            + f"|{g['horizon_changes']}|{g['incorrect_naive_bill']}|{g['other_region_corrections']}|"
        )
    lines += [
        "",
        "成本为本机双进程离线诊断下的完整评价路径，不含独立复核开销；不是服务器吞吐基准，也没有实现增量加速。档案尝试使用独立复制的原始档案，包含重新构建身份缓存的成本，不能视为在线复用缓存的维护成本。原始逐例记录另有资格枚举耗时。",
        "",
        "## 解释边界与下一步",
        "",
        "1. 本轮支持或否定的是终态种群中的块复用机会，不是在线新子代的机会频率。历史理想点未完整存档，因此方向比较采用明确记录的候选感知离线上下文，不能声称精确重放历史接收决定。",
        "2. 单seed、已被探针使用的合成面板，仅作开发诊断；Bootstrap区间描述该面板实例间异质性，不提供跨seed稳定性或文献首创证据。未进行多方案显著性筛选，也不以区间替代独立确认。",
        "3. 所有候选都检查最终档案接收，包括标量不改善者。档案接受可能只有相同目标的不同表型，不能等同于HV增加。",
        "4. 区域块保持任务分配，可能加快分配收敛；后续在线试验需记录区域分配多样性。",
        "5. 若救回机会与前沿增益存在，下一步是只读在线子代观察或受限精确版R1开发对照；总预算保持50/100/200任务600/1200/2400秒含初始化。增量版必须单独做等价性及完整调用成本基准。当前结果不支持直接升级工作版。",
        "",
        "## 产物",
        "",
        "analysis.json、instances.json、trials_corrected.csv；原始回放在 ../panel/，其中pre_hv_gain为旧检查点代理，正文以本分析保守口径为准。代码、测试、版本与验收见上一级回执。",
        "",
    ]
    best = max(reports, key=lambda report: report["final_gain"])
    example_file = args.panel / "runs" / (best["key"] + ".json")
    examples = replay._read(example_file)["examples"]
    if examples:
        example = examples[0]
        ev = example["mixed"]["evaluation"]
        lines += [
            "## 一个具体案例（事后选取作解释，不代表平均表现）",
            "",
            f"实例：{best['key']}。供体槽位{example['donor']}，受体槽位{example['recipient']}，复制区域{example['region']}（均为零基索引）。",
            "",
            "|排程|总Flow（秒）|账单（CNY）|",
            "|---|---:|---:|",
            f"|原受体|{example['recipient_objectives'][0]:.3f}|{example['recipient_objectives'][1]:.6f}|",
            f"|整份供体|{example['donor_objectives'][0]:.3f}|{example['donor_objectives'][1]:.6f}|",
            f"|仅复制该区域|{ev['flow']:.3f}|{ev['tou'] + sum(ev['demand']):.6f}|",
            "",
            "该受体偏重Flow，整份供体虽然账单较低但Flow变差，未通过方向标量比较；只复制该区域后两目标都优于原受体。案例支持局部复用的机制解释，不能代替跨实例和等时间在线效果。",
            "",
        ]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
