"""Render complete ablation results and descriptive diagnostics."""

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__import__("sys").argv[1]).resolve() / "outputs/lsv2_ablation3_20261028"
A = R / "analysis"
F = R / "figures"
F.mkdir(exist_ok=True)
x = json.loads((A / "analysis.json").read_text())
s = json.loads((A / "supplement.json").read_text())
fronts = json.loads((A / "fronts.json").read_text())
ctrl = "Q1_NET_G9_B3L3"
arms = list(x["arms"])
names = {
    a: (
        "Full"
        if a == ctrl
        else "No " + a.split("_NO_")[1].replace("_", " + ").replace("P2M", "P2M")
    )
    for a in arms
}
cn = {
    a: (
        "完整 B3L3"
        if a == ctrl
        else "删 " + a.split("_NO_")[1].replace("_", "＋").replace("P2M", "P2_MIXED")
    )
    for a in arms
}
colors = dict(
    zip(
        arms,
        ["#20252b", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#7B3294"],
    )
)


def csvread(name):
    with (A / (name + ".csv")).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(F / (name + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(F / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 4.8))
ts = x["tests"]
d = np.array([t["mean"] for t in ts])
ci = np.array([t["ci95"] for t in ts])
ax.errorbar(
    d,
    np.arange(7),
    xerr=np.stack([d - ci[:, 0], ci[:, 1] - d]),
    fmt="o",
    color="#0072B2",
    capsize=4,
)
ax.set_yticks(range(7), [names[t["treat"]] for t in ts])
ax.axvline(0, color="gray", lw=1)
ax.set_xlabel("Paired normalized HV difference vs full pool (higher is better)")
ax.set_title(
    "16 instances, one seed | pointwise stratified-bootstrap 95% CI\nAll seven Holm-adjusted p = 1.000"
)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.2)
save(fig, "paired_hv")

parents = sorted(fronts, key=lambda p: (fronts[p]["n"], p))
by = {(r["parent"], r["arm"]): float(r["hv"]) for r in csvread("run_metrics")}
matrix = np.array([[by[p, a] - by[p, ctrl] for p in parents] for a in arms[1:]])
fig, ax = plt.subplots(figsize=(14, 5))
lim = max(abs(matrix.min()), abs(matrix.max()))
im = ax.imshow(matrix, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(16), [p.replace("st_", "") for p in parents], rotation=50, ha="right")
ax.set_yticks(range(7), [names[a] for a in arms[1:]])
for i in range(7):
    for j in range(16):
        ax.text(
            j,
            i,
            f"{matrix[i, j]:+.3f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if abs(matrix[i, j]) > lim * 0.55 else "black",
        )
fig.colorbar(im, ax=ax, label="HV difference vs full")
ax.set_title("Every instance shown; shared per-instance eight-arm reference")
save(fig, "instance_hv_heatmap")

for p in parents:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True, sharey=True)
    for ax, aa, title in [
        (axes[0], arms[:4], "Single removals"),
        (axes[1], [ctrl] + arms[4:], "Combined removals"),
    ]:
        for a in aa:
            q = np.array(fronts[p]["runs"][a])
            ax.plot(q[:, 0], q[:, 1], color=colors[a], label=names[a], lw=1.8 if a == ctrl else 1.3)
        ax.set_xlabel("Total flow time")
        ax.set_ylabel("Electricity bill")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(p + " | seed 16901 | lower-left is better")
    fig.tight_layout()
    save(fig, "pareto_" + p)

curves = csvread("anytime")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, n in zip(axes.flat, (20, 50, 100, 200)):
    for a in arms:
        z = [r for r in curves if r["arm"] == a and int(r["n"]) == n]
        fractions = sorted({float(r["fraction"]) for r in z})
        vals = [
            np.mean([float(r["hv"]) for r in z if float(r["fraction"]) == t]) for t in fractions
        ]
        ax.plot(fractions, vals, label=names[a], color=colors[a], lw=1.6)
    ax.set_title(f"n={n}, four instances")
    ax.set_xlabel("Fraction of total wall-clock budget")
    ax.set_ylabel("Mean normalized HV")
    ax.grid(alpha=0.2)
axes[0, 0].legend(fontsize=7, ncol=2)
fig.tight_layout()
save(fig, "anytime")

mainrows = [
    "| 配置 | 平均HV | 对完整池HV差 | 胜/负实例 | IGD+差（负更好） | Holm p |",
    "|---|---:|---:|---:|---:|---:|",
]
mainrows.append(f"| 完整 B3L3 | {x['arms'][ctrl]['metrics']['hv']:.6f} | — | — | — | — |")
for t in ts:
    a = t["treat"]
    mainrows.append(
        f"| {cn[a]} | {x['arms'][a]['metrics']['hv']:.6f} | {t['mean']:+.6f} | {t['positive']}/{t['negative']} | {t['igd_diff']:+.6f} | {t['p_holm']:.3f} |"
    )
size = ["| 配置 | 20 | 50 | 100 | 200 |", "|---|---:|---:|---:|---:|"]
for t in ts:
    size.append(
        "| "
        + cn[t["treat"]]
        + " | "
        + " | ".join(f"{t['per_size'][str(n)]:+.6f}" for n in (20, 50, 100, 200))
        + " |"
    )
action = [
    "| 配置 | LS调用数（16例合计） | 空调用率 | 空调用占步骤耗时 | 正局部HV率 | 负NET奖励率 | 平均代数 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for a in arms:
    z = s["actions"][a]
    action.append(
        f"| {cn[a]} | {z['calls']} | {z['empty_rate']:.2%} | {z['empty_time_share']:.2%} | {z['positive_hv_rate']:.2%} | {z['negative_rate']:.2%} | {x['arms'][a]['runtime']['generations']:.1f} |"
    )
a3 = [
    "| 配置 | A3调用 | 规则选中 | 大事件 | 捕获 | 召回 | 选中后命中率 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for a in arms:
    z = s["a3_frozen_rules"][a]
    a3.append(
        f"| {cn[a]} | {z['calls']} | {z['selected']} | {z['events']} | {z['hits']} | {z['recall']:.1%} | {z['precision']:.2%} |"
    )
intervals = [
    "| 配置 | HV差95%区间（未作同时校正） | 原始p | 等exact HV差 |",
    "|---|---:|---:|---:|",
]
for t in ts:
    intervals.append(
        f"| {cn[t['treat']]} | [{t['ci95'][0]:+.6f}, {t['ci95'][1]:+.6f}] | {t['p']:.6f} | {t['equal_exact_diff']:+.6f} |"
    )
history = csvread("historical_control_repeat")
hd = np.array([float(r["delta"]) for r in history])
finish = datetime.fromtimestamp(
    json.loads((R / "RECOVERY_RECEIPT.json").read_text())["package"]["finished"],
    timezone(timedelta(hours=8)),
).isoformat()
report = f"""# B3L3 三算子删除实验：回收与分析

## Material Passport

Verification Status: ANALYZED。本报告分析128条已完成服务器运行；独立重算指标、逐步重放Q更新并复核exact，未重新运行128条solver。来源为本地SHA验真结果 `results/`，冻结协议见工作树 `docs/reports/lsv2_ablation3_protocol_20261028.md`。开发筛选，不改默认算法。

## 结论

删P3是最有机制依据的精简候选；三者全删是本批平均HV最高的探索候选。七种处理均未建立显著优于完整池的证据，也未建立无损等价。A4、P2_MIXED单删平均下降，不能因负奖励多就认定可删。A3的大收益有早期线索，但该线索随动作池改变而变化，还不足以固定成只在某个时刻/步骤调用的策略。

128/128完成、0失败，结束时间 {finish}。2250文件逐SHA回收；128条从空Q开始、mask及终止规则审计通过；525967步TD重放、384个最终前沿点独立fresh exact通过。16组初始化指纹一致；独立重算256组指标及七项符号翻转/校正通过。服务器原件保留。

## 如何比较

七个删除组合与同批完整池对照；各16实例，20/50/100/200各4例，seed16901。Q1、NET、gamma0.9、B3L3、20%累计LS份额及300/600/1200/2400秒预算不变。

每实例用八臂最终非支配并集统一归一化，参考点(1.1,1.1)算HV；IGD+用同一并集。HV差为归一化面积的绝对差，不是百分比。七项预设主要比较使用实例配对符号翻转检验，统一Holm校正。区间为按规模分层的20000次实例bootstrap点区间；规模子组只有4例。符号翻转依赖零假设下差值符号可交换，单seed且开发集复用，统计仅作开发证据。动作日志的几十万行不当独立重复样本。

{chr(10).join(mainrows)}

全部删除比完整池平均增加0.008415；单删P3增加0.007691，差距很小。七项Holm p均为1，原始p也均大于0.14；没有哪组达到显著优于完整池。IGD+总体与HV方向相近。不能用平均排名宣布“三者都没有必要”。

{chr(10).join(intervals)}

等exact结果大体保留主要方向，但exact不是统一的运行成本单位，只能作辅助；主结论仍以等墙钟为准。参考点改为(1.2,1.2)的敏感性数值也完整导出。

## 哪些规模改善、哪些退步

下面均为相对同批完整池的HV差。

{chr(10).join(size)}

单删P3的增益主要在50和200；100并未改善。全部删除在100的均差最好，但20下降。删P3＋P2_MIXED的平均增益主要由50拉动，200下降；删P3＋A4亦在200明显下降。三因素平均主效应中，删P3约+0.009017，四规模均为正；删A4约−0.001387，删P2_MIXED约+0.001141。但这是跨其他删除条件平均的探索效应，不等同单删效果。两因素/三因素交互已导出，不能把各自单删效果简单相加。

![七项配对HV比较](figures/paired_hv.png)

![全部实例HV差](figures/instance_hv_heatmap.png)

## P3：空调用负担确认了，但它并非毫无价值

完整池P3共23975次调用，占全部步骤33.77%；21114次空调用，空率88.07%。P3空调用累计消耗517.75秒，占全部动作步骤计费时间15.29%（这里不是整个算法墙钟，也不包括全部frame管理开销）。P3总耗时685.57秒，约占动作步骤计费时间20.25%。

然而，2861次非空调用中2855次有正局部HV，比例99.79%。这是日志中被实际选择的状态上的表现，说明它在满足可生成候选的条件时有价值；不代表所有未被调用状态同样有效。

删P3后整体空调用率31.48%降为2.84%，空调用耗时占比17.53%降为2.99%。这是本批最明确的机制变化。但总步骤数70995降为60914，正HV步骤10556降为9418，并没有“删空动作以后动作次数必然更多”。较便宜的P3空步骤被较贵的其他动作替代。A7耗时575.92→700.31秒，A3为351.85→407.46秒，A4为239.15→457.16秒；资源没有自动全部流向最高价值动作。A4调用7550→13766，负NET比例97.82%，是后续需要关注的分配现象，但本轮没有随机控制器对照，不能独立归因于Q学习器。

{chr(10).join(action)}

表中调用/时间率为16run合并描述，平均代数为16run等权平均。各臂实际LS份额均约20.01%，删除动作并未释放一整块额外墙钟给MOEA/D。单删P3平均代数828.5→791.1；固定墙钟下进程速度和搜索路径也会影响代数，不能解释为实现故意削减。动作局部HV跨frame定义不同，累计值不等于最终全局HV，不能拿它直接代替前沿结论。负奖励率仍约92%–93%，说明删P3没有解决稀疏改进或成本扣除导致的负奖励占多数。

## A4 与 P2_MIXED 为什么不能按同一理由删除

完整池A4空率只有0.28%，不是类似P3的空动作；7550次中398次正局部HV、13个本run正奖励95分位尾部事件。单删后HV平均−0.004777，100规模均差−0.022836。观察不足以证明必要性，但不支持无损删除。

P2_MIXED空率0.48%，3946次中505次正局部HV，47个相对大正奖励事件；NET负奖励比例92.73%，仍有高价值尾部，不能用负奖励多数抹去。单删后平均HV−0.004628，200均差−0.014848。全部删除有正均差也不能反推单独删它有效，因为动作间有替代、顺序和搜索路径的相互作用。

## A3：封存条件在新日志中的表现

此处“规则”和“大事件阈值”都沿用开跑前封存的整实例留出结果，未用本批结果重新拟合。每个实例的规则在旧批排除该实例后拟合，通常是较早累计LS进度，少数是首frame。仍使用旧dev实例和seed，不把新调用或八个处理当独立确认。只检验已经调用A3时能否识别大收益，不能评价强制调用时会怎样。

{chr(10).join(a3)}

完整池规则选中324/2987=10.85%的A3调用，捕获6/9=66.7%的封存阈值大事件；命中率1.85%，相比全体A3事件率9/2987=0.30%有富集，但选中后仍有98.15%不是这种大事件。漏掉3次，不能当硬性禁用条件。

另用各run正奖励95分位作描述性标签（与上表封存训练阈值是不同标签）：完整池9个A3大事件中6个在首frame、6个在第三步，最大累计LS进度约3.06%。单删P3为8个，7个第三步，最大约4.53%。但全部删除为8个，第三步为0，最大累计LS进度52.35%，封存早期规则只捕获2/8=25%。删P3＋P2_MIXED也出现累计LS进度46.44%的事件。由此不能把A3限制成“仅早期”或“固定第三步”。

完整池首frame中，A3大事件6/84=7.14%，其他动作83/732=11.34%；早期机会并非A3独占。前沿段、前一步动作的分组统计见a3_conditions.csv，目前事件少且分散，没有足够证据支持一个稳定的单条件策略。现有日志缺少每个未选动作的同状态结果，也缺少可完整解释迁移/拥塞收益的候选层特征；条件线索不等同机制被验证。

## 为什么同seed还必须有同批完整对照

旧批与本批完整B3L3配置、Q参数和标定一致，按本批参考重新计分，本批对旧批平均HV差为{hd.mean():+.6f}，逐实例绝对差平均{np.abs(hd).mean():.6f}，最大{np.abs(hd).max():.6f}，本批仅4/16例更高。这种重复差异比本次平均删除效应还大。

固定墙钟与计时参与NET奖励，使相同seed不保证同一步骤数、同Q轨迹或相同前沿；共享服务器负载也可能参与，但本次没有资源控制对照，无法单独归因。主结论必须用同批完整池；历史对照仅作复现波动提醒，不能把本批的+0.008直接宣称超过上一批最好结果。保留同批对照并不能消除所有机器波动，单seed仍是局限。

## 当前可采用的判断与待验证项

当前已确认的是本面板上的空调用结构和测得的效果差，而不是默认算法升级。若继续筛选，单删P3最适合检验“削减无效调用”的单机制方向；全部删除可作为另一候选，但其优势尚未确认。A4/P2_MIXED暂不单独删作默认；A3保留探索能力，尤其不把晚期调用清零。本轮不额外启动任何实验。

统计误读检查11/11：分规模方向差异已报告；推断单位为实例；A3存在选择偏差；未按事后奖励筛选样本估因果；同时报告基准率、精度、召回；平均排名存在回归均值风险；128run全纳入无幸存者剔除；七项主比较统一Holm；交互/条件为探索且保留完整表；动作收益相关性不冒充策略因果；特征限决策前但仍无反事实验证。未发现数据验真失败，证据强度为开发筛选、非确认性结论。

## 文件索引

- `analysis/analysis.json`：主比较、区间、校正p、逐规模、逐实例、参考点敏感性及等exact。
- `analysis/run_metrics.csv`、`runtime.csv`、`run_diagnostics.csv`：128run完整指标。
- `analysis/factorial_per_instance.csv`：三因素主效应及交互。
- `analysis/action_profile.csv`：逐实例逐动作空率、奖励、时间、尾部。
- `analysis/a3_frozen_rule_per_run.csv`、`a3_events.csv`、`a3_conditions.csv`：A3条件与漏报。
- `figures/pareto_*.png/pdf`：全部16实例前沿；`anytime.png/pdf`：四规模进度曲线。
- `RECOVERY_RECEIPT.json`、`analysis/VALIDATION.json`：回收与数值交叉验证回执。
"""
(R / "REPORT.md").write_text(report, encoding="utf-8")
print(json.dumps(dict(report=str(R / "REPORT.md"), figures=len(list(F.glob("*.png"))))))
