"""Cross-check metrics and produce a readable diagnostic report and paired figures."""

import csv
import gzip
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("S4_ANALYSIS_ROOT", str(B / "preview")))
sys.path.insert(0, str(Path(os.environ.get("S4_SOURCE_ROOT", str(B.parents[3]))) / "src"))
import analyze_results as az  # noqa: E402

from geo_llm_scheduler.experiments.metrics import hypervolume  # noqa: E402

s = json.loads((B / "ANALYSIS_SUMMARY.json").read_text(encoding="utf8"))
rows = list(csv.DictReader((B / "per_run_analysis.csv").open(encoding="utf-8-sig")))
for r in rows:
    for k in [
        "hv",
        "hv_vs_a0_pct",
        "igd_plus",
        "min_flow",
        "min_bill",
        "exact",
        "archive_size",
        "overrun",
    ]:
        r[k] = float(r[k])
    r["n"] = int(r["n"])
arms = list(s["groups"]["all"])
by = {(r["key"], r["arm"]): r for r in rows}
keys = sorted({r["key"] for r in rows})
alts = {1.05: {a: [] for a in arms}, 1.2: {a: [] for a in arms}}
count = 0
for key in keys:
    fs = {}
    for a in arms:
        with gzip.open(ROOT / "runs" / key / a / "archive.json.gz", "rt", encoding="utf8") as f:
            ar = json.load(f)
        fs[a] = az.front(
            (x["evaluation"]["flow"], x["evaluation"]["tou"] + sum(x["evaluation"]["demand"]))
            for x in ar
        )
    pool = [p for f in fs.values() for p in f]
    lo = [min(p[j] for p in pool) for j in (0, 1)]
    sp = [max(p[j] for p in pool) - lo[j] or 1 for j in (0, 1)]
    ns = {a: [((x - lo[0]) / sp[0], (y - lo[1]) / sp[1]) for x, y in fs[a]] for a in arms}
    for a in arms:
        v = hypervolume(ns[a], (1.1, 1.1))
        assert math.isclose(v, by[key, a]["hv"], rel_tol=1e-12, abs_tol=1e-12)
        count += 1
        for ref in alts:
            base = hypervolume(ns["A0_FULL_S4"], (ref, ref))
            alts[ref][a].append(100 * (hypervolume(ns[a], (ref, ref)) / base - 1))
    # Draw only representative cases selected by median S4-vs-CMOEA-CD gap, one per size.
    group = [r for r in rows if r["n"] == by[key, "A0_FULL_S4"]["n"] and r["arm"] == "CMOEA_CD_PD"]
    median = st.median(r["hv_vs_a0_pct"] for r in group)
    chosen = min(group, key=lambda r: abs(r["hv_vs_a0_pct"] - median))["key"]
    if key == chosen:
        for title, subset, filename in [
            ("S4 and nine adapted baselines", ["A0_FULL_S4"] + arms[5:], "comparison"),
            ("S4 and four ablations", arms[:5], "ablation"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.tab20(np.linspace(0, 1, len(subset)))
            for i, a in enumerate(subset):
                f = fs[a]
                ax.step(
                    [x / 1000 for x, y in f],
                    [y for x, y in f],
                    where="post",
                    label=a,
                    lw=2.2 if i == 0 else 1.15,
                    color="#bd2643" if i == 0 else colors[i],
                )
            ax.set(
                xlabel="Flow (1000 s)",
                ylabel="Electricity bill (CNY)",
                title=f"{key} | {title} | seed 1 | lower-left better",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(B / "figures" / f"{filename}_{key}.png", dpi=160)
            plt.close(fig)
altout = {
    str(ref): {
        a: dict(mean=st.mean(v), wins=sum(x > 1e-8 for x in v), losses=sum(x < -1e-8 for x in v))
        for a, v in d.items()
    }
    for ref, d in alts.items()
}
(B / "METRIC_VERIFICATION.json").write_text(
    json.dumps(dict(independent_hv_checks=count, passed=True, references=altout), indent=2),
    encoding="utf8",
)
lines = [
    "# S4 120 实例、14 配置结果分析",
    "",
    "本批 1,680 次全部完成。此报告使用从已封存文件中独立提取并校验的前沿与摘要。完整原始包的下载及逐文件核验状态以同目录回收回执为准。",
    "",
    "本批固定 120 例（50／100／200 各 40），每例每方法 seed 1 一次，预算分别 600／1200／2400 秒，均含初始化。完整 S4、四项消融、五个既有增强基线、四个近期论文的 Python 问题适配。BILL4＋MOEA/D 未列入本次矩阵。",
    "",
    "## 指标与边界",
    "",
    "每个实例以全部 14 条最终前沿的共同目标范围归一化，参考点为 $(1.1,1.1)$，计算 HV。每一项百分比均先按同一实例相对 A0 完整 S4 计算，再对实例等权平均，负值表示该方法 HV 小于 S4。这不是 Flow 或电费百分比。IGD+ 使用 14 条前沿合并得到的非支配集作为参考，不是真实最优前沿。对重复目标点去重并移除严格支配点后计算。独立实现的 HV 对全部 1,680 次逐一复算一致。",
    "",
    "参考点变为 $(1.05,1.05)$、$(1.2,1.2)$ 的结果见 METRIC_VERIFICATION.json。四个消融对完整算法使用成对 Wilcoxon 检验并进行四项 Holm 校正；120 个案例来自 10 个有结构关联的实例族、每例只有一次随机运行，检验不能代替多种子稳定性验证，也不能认为这些经过机制探针的实例是完全未查看的留出集。",
    "",
    "## 汇总结果",
    "",
    "|方法|相对完整法平均 HV|中位数|该方法胜／负|平均排名|平均 IGD+|",
    "|---|---:|---:|---:|---:|---:|",
]
for a in arms:
    d = s["groups"]["all"][a]
    lines.append(
        f"|{a}|{d['mean']:+.2f}%|{d['median']:+.2f}%|{d['wins']}／{d['losses']}|{d['mean_rank']:.2f}|{d['igd']:.4f}|"
    )
lines += [
    "",
    "完整 S4 对九个外部方法各为 120 胜 0 负；这说的是当前面板、seed 1、统一指标和当前适配代码。四个近期方法经过 Python 编码与调度问题适配，不等于原文环境或作者原代码复现。五个旧基线同样为本项目增强实现。",
    "",
    "## 四项消融",
    "",
    "|比较|消融相对完整法平均／中位 HV|消融胜／负|Holm 校正后 p|说明|",
    "|---|---:|---:|---:|---|",
]
expl = {
    "A1_NO_SHIFT": "整体关闭外置右移，节省的 LS 时间仍可给 Q-LS",
    "A2_RANDOM_OPERATOR": "仅把在线 Q 选算子替换为可用动作均匀随机",
    "A3_RANDOM_INIT": "BILL4 改成 100 个随机合法初始个体",
    "A4_DIRECT_SHIFT": "LS 入场解直接右移并结束，省去算子轨迹和二次选解",
}
for a in arms[1:5]:
    d = s["groups"]["all"][a]
    p = s["ablation_wilcoxon"][a]["holm"]
    lines.append(
        f"|{a}|{d['mean']:+.2f}%／{d['median']:+.2f}%|{d['wins']}／{d['losses']}|{p:.4g}|{expl[a]}|"
    )
lines += [
    "",
    "去右移平均 HV 反而高 0.82%，71 胜 49 负，校正后 p≈0.0213；200 任务上为 28 胜 12 负。结果说明本批证据尚不支持 S4 的右移模块在最终 HV 上提供净收益。该消融释放时间可供 Q-LS，因此不能解释为右移候选永远无效。",
    "",
    "随机选算子平均高 0.33%，67 胜 53 负，校正后 p≈0.315。当前数据未显示在线 Q-learning 稳定优于随机，不能把没有显著差异当成两者完全等价。",
    "",
    "随机初始化平均降低 13.20%，107／120 例落后于完整法。与完整法比较，随机初始化有 93 例获得更低的最小 Flow，但只有 2 例获得更低的最小 Bill。该观察与 BILL4 偏向电费端有关的解释一致，但不单独证明因果机制。",
    "",
    "仅直接右移平均降低 6.28%，114／120 例落后。这比较了整个算子轨迹加选解与仅一次右移，不能把差值单独算成 Q 的贡献。",
    "",
    "## LS 与右移日志",
    "",
    "对全部 600 份完整法及四项消融的局部搜索日志进行汇总，统计按调用总量加权，而不是先算每个实例的比例再平均。结果存为 ls_stats.json。完整法 A0 共 153,070 次右移调用，51,723 次没有生成候选，占 33.79%；43,290 次记录了正 HV 增益，占 28.28%。因此右移确实能找到改进，只是这些即时收益不足以证明总预算下带来更好的最终前沿。",
    "",
    "A0 有 459,187 个局部算子步骤，A1 去掉右移后有 569,659 步，增加 24.06%。相应算子步骤累计计时分别约 25,580 秒与 31,947 秒。这个数据与“右移时间被其他 LS 使用”一致，但因两组搜索轨迹、调度及每步成本均不同，不能仅由相关数字断言因果。",
    "",
    "A4 直接右移有 2,696,488 条调用记录，其中 2,551,344 条没有候选，占 94.62%；有 144,140 条记录为正 HV 增益，占 5.35%。因此其低效与大量零候选相吻合，但日志本身不能证明这是落后的唯一原因。A0 的 Q 奖励 413,800／459,187 步为负，占 90.12%，这是本批记录的即时奖励分布，不代表每个被选动作对后续搜索都无价值。",
    "",
    "## 分规模",
    "",
    "|方法|50 任务平均变化|100 任务平均变化|200 任务平均变化|",
    "|---|---:|---:|---:|",
]
for a in arms:
    lines.append(
        "|"
        + a
        + "|"
        + "|".join(f"{s['groups'][str(n)][a]['mean']:+.2f}%" for n in [50, 100, 200])
        + "|"
    )
lines += [
    "",
    "## 运行耗时与工作量",
    "",
    "各方法评价次数不可直接等同迭代数；不同算法评价、档案和构造开销不同。下表是三个规模混合后的每运行平均，仅用于观察工作量。",
    "",
    "|方法|平均 exact 次数|平均最终 Archive 成员|最大超时秒数|",
    "|---|---:|---:|---:|",
]
for a in arms:
    r = [x for x in rows if x["arm"] == a]
    lines.append(
        f"|{a}|{st.mean(x['exact'] for x in r):,.0f}|{st.mean(x['archive_size'] for x in r):.1f}|{max(x['overrun'] for x in r):.3f}|"
    )
lines += [
    "",
    "## 后续判断",
    "",
    "对外比较的差距非常大，值得继续保留，但应在更换种子、验证适配实现和公平初始化差异后再用于论文级表述。右移与 Q 是当前主要消融问题，先根据日志诊断为什么它们没有形成稳定净收益。用户已另行记录 Archive 容量上限作为待参数实验的想法，本次没有改变它。",
    "",
    "三种规模各有一个基于 S4 与 CMOEA-CD 差距中位数选择的实例，分别提供外部比较和消融前沿图，见 figures。完整逐例数据见 per_run_analysis.csv，汇总和校正见 ANALYSIS_SUMMARY.json，源码见 analyze_results.py。",
]
report = "\n".join(lines) + "\n"
assert not any(x in report for x in ("\\(", "\\)", "\\[", "\\]"))
(B / "ANALYSIS_REPORT.md").write_text(report, encoding="utf8")
print(
    "verified",
    count,
    "reference 1.05/1.2",
    {k: {a: round(x["mean"], 3) for a, x in v.items() if a in arms[:5]} for k, v in altout.items()},
)
