"""Render Chinese interpretation and scientific plots for the verified width analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geo_llm_scheduler.experiments.macro_width import ARMS, read_json

COLORS = {"NATIVE6": "#424b57", "POOL3": "#c48a20", "POOL6": "#2673bd", "POOL10": "#b64949"}


def _csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _table(headers: list[str], rows: list[list]) -> str:
    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join("---" for _ in headers)
        + " |\n"
        + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
        + "\n"
    )


def _save(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(directory / (name + ".png"), dpi=160)
    fig.savefig(directory / (name + ".pdf"))
    plt.close(fig)


def generate(raw: Path, analysis: Path, out: Path, collection: Path, completion: Path) -> None:
    """Create data-linked tables, fixed-case front plots and a report without changing raw."""
    if out.resolve().is_relative_to(raw.resolve()):
        raise ValueError("Report output must stay outside immutable raw")
    out.mkdir(parents=True, exist_ok=False)
    figures = out / "figures"
    figures.mkdir()
    data = read_json(analysis / "analysis_summary.json")
    audit = read_json(analysis / "independent_audit.json")
    downloads = read_json(collection)
    execution = read_json(completion)
    if audit["state"] != "verified" or execution["state"] != "complete_verified":
        raise ValueError("Unverified analysis or experiment")
    runs = _csv(analysis / "run_metrics.csv")
    by_key = {r["key"]: r for r in runs}
    actions = [r for r in _csv(analysis / "probe_actions.csv") if r["arm"] == "POOL6"]
    costs = _csv(analysis / "operator_costs.csv")
    probes = [r for r in _csv(analysis / "probe_buckets.csv") if r["arm"] == "POOL6"]
    sensitivity = _csv(analysis / "sensitivity.csv")
    anytime = _csv(analysis / "anytime_metrics.csv")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = list(data["statistics"])
    for ax, metric, title in zip(
        axes, ("hv_diff", "igd_plus_gain"), ("Normalized HV difference", "IGD+ improvement")
    ):
        for i, name in enumerate(names):
            s = data["statistics"][name][metric]
            ax.plot(s["ci95"], [i, i], color="#777777", lw=2)
            ax.scatter(
                s["mean"],
                i,
                s=70,
                color=COLORS[name.split("-")[0]],
                marker="o" if s["p_holm_primary_six"] < 0.05 else "x",
            )
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(3), names)
        ax.set_title(title)
        ax.set_xlabel("40 independent bases; pointwise 95% bootstrap CI")
    _save(fig, figures, "primary_effects")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, jobs in zip(axes, (50, 100)):
        for arm in ARMS:
            rr = [r for r in anytime if r["arm"] == arm and int(r["jobs"]) == jobs]
            fractions = sorted({float(r["fraction"]) for r in rr})
            means = [
                np.mean([float(r["hv"]) for r in rr if float(r["fraction"]) == f])
                for f in fractions
            ]
            ax.plot(fractions, means, marker="o", color=COLORS[arm], label=arm)
        ax.set_title(f"{jobs} jobs / {1200 if jobs == 50 else 2400}s engine budget")
        ax.set_xlabel("Fraction of fixed engine budget")
        ax.set_ylabel("Mean normalized HV")
    axes[1].legend(fontsize=8)
    _save(fig, figures, "anytime_hv")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(8)
    ax.bar(
        x - 0.2,
        [100 * float(r["late_given_zero3_bucket_rate"]) for r in actions],
        width=0.4,
        label="4-6 help after zero at 3",
        color="#2673bd",
    )
    ax.bar(
        x + 0.2,
        [100 * float(r["late_given_zero6_bucket_rate"]) for r in actions],
        width=0.4,
        label="7-10 help after zero at 6",
        color="#c48a20",
    )
    ax.set_xticks(x, [f"A{i}" for i in range(1, 9)])
    ax.set_ylabel("Percent of eligible sampled buckets")
    ax.set_title("POOL6 trajectories: conditional late improvement (descriptive)")
    ax.legend()
    _save(fig, figures, "late_opportunities")
    cost_rows = []
    for action in range(1, 9):
        rr = [r for r in costs if r["arm"] == "POOL6" and int(r["action"]) == action]
        costs_ms = {
            field: 1000 * float(np.mean([float(r[field]) / float(r["calls"]) for r in rr]))
            for field in (
                "construction_seconds",
                "ssgs_seconds",
                "exact_seconds",
                "archive_seconds",
                "step_seconds",
            )
        }
        costs_ms["construction_wall_share"] = 100 * float(
            np.mean(
                [float(r["construction_seconds"]) / float(by_key[r["key"]]["elapsed"]) for r in rr]
            )
        )
        costs_ms["effective_evaluations"] = float(
            np.mean([float(r["evaluations"]) / float(r["calls"]) for r in rr])
        )
        cost_rows.append({"action": action, **costs_ms})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(8)
    for field, name, color in [
        ("construction_seconds", "Construction", "#2673bd"),
        ("ssgs_seconds", "SSGS", "#639a63"),
        ("exact_seconds", "Exact", "#c48a20"),
        ("archive_seconds", "Archive", "#8664a2"),
    ]:
        yy = np.array([r[field] for r in cost_rows])
        ax.bar(x, yy, bottom=bottom, label=name, color=color)
        bottom += yy
    ax.set_xticks(x, [f"A{i}" for i in range(1, 9)])
    ax.set_ylabel("Online ms/call, equal-run means")
    ax.set_title("Measured operator components; control/observer time shown separately in report")
    ax.legend()
    _save(fig, figures, "operator_costs")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bottom = np.zeros(4)
    for field, name, color in [
        ("construction_share", "Construction", "#2673bd"),
        ("ssgs_share", "SSGS", "#639a63"),
        ("exact_share", "Exact", "#c48a20"),
        ("archive_share", "Archive", "#8664a2"),
        ("residual_share", "Remaining wall", "#aaaaaa"),
    ]:
        yy = np.array([100 * data["arm_means"][arm][field] for arm in ARMS])
        ax.bar(range(4), yy, bottom=bottom, label=name, color=color)
        bottom += yy
    ax.set_xticks(range(4), ARMS)
    ax.set_ylabel("Percent of engine wall")
    ax.legend(ncol=3, fontsize=8)
    _save(fig, figures, "wall_components")
    cases = sorted({(int(r["jobs"]), int(r["base_seed"]), r["tariff"]) for r in runs})
    fronts = {}
    for key in cases:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arm in ARMS:
            r = next(
                r
                for r in runs
                if (int(r["jobs"]), int(r["base_seed"]), r["tariff"]) == key
                and r["arm"] == arm
                and int(r["algorithm_seed"]) == 1112
            )
            root = raw / ("formal_node" + r["node"])
            with (root / "runs" / r["key"] / "objectives.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                pts = sorted(
                    {
                        (float(p["flow_seconds"]), float(p["bill_cny"]))
                        for p in csv.DictReader(stream)
                    }
                )
            # Each run's objective envelope is displayed; lines do not assert feasible interpolation.
            best, nd = float("inf"), []
            for xx, yy in pts:
                if yy < best:
                    nd.append((xx, yy))
                    best = yy
            fronts[(key, arm)] = nd
            ax.plot(
                [p[0] for p in nd],
                [p[1] for p in nd],
                marker=".",
                ms=3,
                color=COLORS[arm],
                label=arm,
                lw=1,
            )
        ax.set_xlabel("Total flow time (s)")
        ax.set_ylabel("Electricity bill (CNY)")
        ax.set_title(f"{key[0]} jobs / base {key[1]} / {key[2]} / seed 1112")
        ax.legend(fontsize=8)
        _save(fig, figures, f"front_n{key[0]}_i{key[1]}_{key[2]}")
    chosen = [(50, 1525001, "H"), (50, 1525001, "T"), (100, 1535001, "H"), (100, 1535001, "T")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key in zip(axes.ravel(), chosen):
        for arm in ARMS:
            nd = fronts[(key, arm)]
            ax.plot(
                [p[0] for p in nd],
                [p[1] for p in nd],
                marker=".",
                ms=3,
                color=COLORS[arm],
                label=arm,
                lw=1,
            )
        ax.set_title(f"{key[0]} jobs / {key[2]} / base {key[1]}")
        ax.set_xlabel("Flow (s)")
        ax.set_ylabel("Bill (CNY)")
    axes[0, 0].legend(fontsize=8)
    _save(fig, figures, "fixed_case_fronts")
    primary_rows = []
    for name, values in data["statistics"].items():
        for metric, label in (("hv_diff", "HV差"), ("igd_plus_gain", "IGD+改善")):
            v = values[metric]
            primary_rows.append(
                [
                    name,
                    label,
                    f"{v['mean']:+.6f}",
                    f"[{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]",
                    f"{v['p_holm_primary_six']:.6f}",
                    f"{v['positive_bases']}/40",
                    "前者更好"
                    if v["p_holm_primary_six"] < 0.05 and v["mean"] > 0
                    else "前者更差"
                    if v["p_holm_primary_six"] < 0.05
                    else "未确认差异",
                ]
            )
    avg_rows = [
        [
            arm,
            f"{v['hv']:.6f}",
            f"{v['igd_plus']:.6f}",
            f"{v['unique_nd']:.2f}",
            f"{v['offspring_rate']:.3f}",
            f"{v['exact_rate']:.2f}",
        ]
        for arm, v in data["arm_means"].items()
    ]
    late_rows = [
        [
            f"A{r['group']}",
            r["zero3_buckets"],
            f"{100 * float(r['late_given_zero3_bucket_rate']):.2f}%",
            r["zero6_buckets"],
            f"{100 * float(r['late_given_zero6_bucket_rate']):.2f}%",
            f"{100 * float(r['base_visit_weighted_short_pool']):.2f}%",
        ]
        for r in actions
    ]
    c_rows = [
        [
            f"A{r['action']}",
            f"{r['construction_seconds']:.2f}",
            f"{r['ssgs_seconds']:.2f}",
            f"{r['exact_seconds']:.2f}",
            f"{r['step_seconds']:.2f}",
            f"{r['effective_evaluations']:.2f}",
            f"{r['construction_wall_share']:.2f}%",
        ]
        for r in cost_rows
    ]
    prefix_rows = [
        [
            f"A{k}",
            v["eligible_buckets"],
            f"{100 * v['difference_rate']:.2f}%",
            f"{100 * v['set_difference_rate']:.2f}%",
        ]
        for k, v in audit["by_action"].items()
    ]
    sens_rows = [
        [r["contrast"], r["metric"], r["group"], r["bases"], f"{float(r['mean']):+.6f}"]
        for r in sensitivity
        if r["metric"] in ("hv_diff", "igd_plus_gain")
    ]
    six = data["statistics"]["POOL6-NATIVE6"]
    seq_rate = np.mean([int(r["seq_missed_positive"]) for r in probes])
    seq_loss = sum(float(r["seq_missed_gain"]) for r in probes) / sum(
        float(r["gain10"]) for r in probes
    )
    text = "## Material Passport\n\n"
    text += _table(
        ["字段", "内容"],
        [
            ["资料范围", "完整正式640run/160四组块/40新base；不含192技术pilot"],
            ["原运行版本", data["source_commit"]],
            ["包canonical SHA256", data["source_hash"]],
            ["运维包装器版本", "6982fd2546abf1222b69fce01db417694991fc0a；原运行源码不改"],
            ["Verification Status", "ANALYZED；完整原始数据验真、指标交叉实现和统计重放通过"],
            ["fresh原实验重跑", "未进行；原worker最终独立exact证据由完整产物SHA绑定"],
            [
                "数据回收",
                f"{downloads['files']}文件/{downloads['raw_bytes']}字节；server/tar/local逐size-SHA一致，服务器原件保留",
            ],
            ["独立核查", "640 HV、32 IGD+、33配对统计、六主联合Holm与3840原生/共同池前缀对照"],
        ],
    )
    text += "\n# MacroSearch四组诊断：结果分析与下一步\n\n"
    text += "本轮最值得继续的发现是**候选生成规则与评价数量需要分开**：保持更大的生成池，但评价6个，比原生6更好；同池评价10个反而在本轮等时间整程中更差。不能把这句话缩成‘预算越大越好’，也不能说已经证明动态预算控制器有效。\n\n"
    text += f"POOL6相对NATIVE6，归一化HV平均增加{six['hv_diff']['mean']:+.6f}，IGD+平均改善{six['igd_plus_gain']['mean']:+.6f}，六项联合Holm分别为{six['hv_diff']['p_holm_primary_six']:.6f}/{six['igd_plus_gain']['p_holm_primary_six']:.6f}；两个指标各30/40base方向为正。逐base配对相对HV平均+{six['hv_pct']['mean']:.2f}%。默认main/F6并未晋级。\n\n"
    text += "## 1. 本轮比较的到底是什么\n\n"
    text += _table(
        ["组别", "生成", "线上exact评价上限"],
        [
            ["NATIVE6", "原生B=6生成规则", 6],
            ["POOL3", "B=10生成规则、完整构造成本计入", 3],
            ["POOL6", "同上", 6],
            ["POOL10", "同上", 10],
        ],
    )
    text += "四组都固定A3=tariff修正版、Q42、原A8、五步轨迹、Trigger、Polish、出生邻域替换；NATIVE6是本轮研究基线，不能直接当作未修正A3的旧main/F6。50/100任务各20base；原生H/T、算法seed1112/2212，同一base全部组和种子同host。每run1200/2400秒引擎预算，最终复验/导出另计。研究observer在四组同时开启且有实际wall成本，后续生产配置需独立确认。\n\n"
    text += "## 2. 在线整程主结果\n\n" + _table(
        ["组别", "平均HV↑", "平均IGD+↓", "unique ND", "offspring/s", "exact/s"], avg_rows
    )
    text += "每个原生实例使用全部四组/两seed最终Archive的共同经验非支配union作参照，ideal/nadir归一化；HV参照(1.1,1.1)。经验union不是true Pareto。归一化HV差与‘每base先求相对变化再平均’不同：0.814653到0.847817的总体均值比约+4.07%，逐base配对相对HV均值为+7.44%，不能混为同一个指标。\n\n"
    text += _table(
        ["对照", "指标（正=前者好）", "base均值", "未校正95% CI", "六主Holm p", "正向base", "判定"],
        primary_rows,
    )
    text += "40base独立，先平均H/T与两seed，分规模20000次bootstrap与100000次signflip，seed2026101209；六主联合Holm。主判定按预先指定的signflip/Holm；例如POOL3的HV percentile CI刚好不跨0，而signflip及Holm仍未通过，两种方法不同，不能挑有利的一项宣称差异或等价。其他27项统计只作诊断，不冒充额外主确认。\n\n![主效果](figures/primary_effects.png)\n\n"
    text += "POOL10−POOL6逐base相对HV均值−2.06%，offspring吞吐−9.60%，exact吞吐却+8.85%；更多评价没有转成更好的最终前沿。POOL3−POOL6的offspring吞吐+10.39%、exact吞吐−9.50%，但两项主质量没有证明更好，也没有确认等价。POOL6−NATIVE6的offspring吞吐+4.42%、exact吞吐+2.24%属于描述性，不能宣称速度提升已经确认或单独解释质量收益。\n\n![过程HV](figures/anytime_hv.png)\n\n"
    text += "## 3. 同为评价6个，为什么生成规则仍会不同\n\n"
    text += "原源码StructuralOperator对A1–A5截取前2B的候选库后随机抽B；A4/A5还受B影响的目标数量、位置采样及round-robin上限控制。原生6因此通常从最多12个候选里抽6，而POOL6先按10的规则生成（最多20库、抽10），再评价其中前6。A7代表池、A8尝试上限也依赖B；A6同源前缀可以一致，但较大的生成会继续消费算子RNG。此处是候选分布/生成与随机流联合因素，不是多评价了4个。\n\n"
    text += (
        "原生NATIVE6的3840个冻结调用已逐一与同源共同10池对照，480样本/动作；下表区分顺序差异与目标点集合差异，后者排除‘只调了顺序’的解释。目标集合相同仍不证明genotype、timing或RNG后态相同；也不能据此归因最终收益只来自某一个动作。\n\n"
        + _table(["动作", "样本", "有序目标前缀不同", "目标集合不同"], prefix_rows)
    )
    text += "**下一步应优先把‘生成库范围、抽样代表、返回数量、评价数量’拆成独立参数，并做匹配随机流对照。** 不要直接把B统一改10；也不要为省构造成本退回原生6，因候选分布可能一起退回。\n\n"
    text += "## 4. 候选后段有机会，但不能每次都投入\n\n"
    text += (
        "主要诊断使用POOL6的3840个动作×阶段reservoir样本，每动作480桶覆盖40base。每桶内部均匀抽一个调用；样本比例不是全量调用成功率，访问次数加权估计另存在probe_actions.csv，并先在run内再HT/seed/base平均。偏好/condition没有单独等额抽样，不能把组间差异直接解释为状态的因果效应。\n\n"
        + _table(
            [
                "动作",
                "前3零改善桶",
                "其后4–6有收益",
                "前6零改善桶",
                "其后7–10有收益",
                "访问加权候选不足10",
            ],
            late_rows,
        )
    )
    text += f"所有动作合并的样本条件比例约9.40%/7.49%，但动作差异明显：A1达到19.94%/14.50%，A8只有2.03%/1.29%。原SEQ在{100 * seq_rate:.2f}%样本桶中遗漏共同10池的正局部收益；各桶绝对增益加总中遗漏比例{100 * seq_loss:.2f}%，只是共享上下文的诊断汇总，不是最终HV损失百分比，也不是访问加权或独立检验。不能用‘前三个没改善’直接证明后段无价值，亦不能把离线有更好候选当作在线追加一定划算。\n\n![后段机会](figures/late_opportunities.png)\n\n"
    text += "A8访问加权约85.07%桶的共同池实际不足10；在线每call平均只评价2.65个，固定6本来已自动受实际可用候选限制。无候选不等于动作不存在其他全局邻居，也不能把未满预算再当作浪费了6次exact。全部3840主诊断桶均可还原可行候选位置；目标包络新增仅作当时诊断，不证明完整phenotype Archive的最终因果贡献。\n\n"
    text += "## 5. 成本与工程改动的优先级\n\n" + _table(
        [
            "动作",
            "构造ms/call",
            "SSGS ms/call",
            "exact ms/call",
            "完整step ms/call",
            "实际评价数/call",
            "构造占整程wall",
        ],
        c_rows,
    )
    text += "以上是在线实际轨迹各run均值，调用组成、机器负载与incumbent不同；不是纯同源benchmark。A3枚举/TOU代理排序约92ms/call，A7与A8构造约75/65ms/call；A1/A4/A5构造很便宜，但结构候选还需SSGS。对结构算子省掉构造并不自动省掉后续SSGS。共享准备数据、候选枚举/排序及timing候选构造应先做正确性与benchmark，再谈提速。\n\n"
    text += "POOL6整程平均构造10.95%、gateway SSGS14.87%、exact21.00%、Archive18.35%、剩余wall34.83%。Polish整体23.82%是重叠视角：包含其中exact/Archive，不能再相加；约53.39%的exact来自Polish。剩余wall包含状态/控制、诊断、去重等未分项开销，不能直接说全部是RL或observer。\n\n![算子成本](figures/operator_costs.png)\n\n![整程wall](figures/wall_components.png)\n\n"
    text += "## 6. 规模、电价与补跑敏感性\n\n"
    text += "16次补跑仍在原host、同输入/初始化/算法seed/单run预算，但执行时间和并行负载变化，不能声称影响为0。排除涉及补跑的10个完整base，保留30base（50任务10、100任务20）后主方向不反转；再用两规模各50%权重时POOL6−NATIVE6 HV差仍+.029877、IGD+改善+.016416。该敏感性为描述性，不能替代原40base主分析或单独认证补跑无偏。\n\n"
    text += (
        "两规模与H/T的主方向都一致，收益在H更大；子组样本有限，下面只报告均值，不挑子组另立确认。\n\n"
        + _table(["对照", "指标", "分组", "独立base", "均值"], sens_rows)
    )
    text += "## 7. 前沿图与数据边界\n\n![固定案例](figures/fixed_case_fronts.png)\n\n"
    text += "固定图为每规模第一个base、H/T、seed1112，不按胜负挑例；补充目录包含80原生实例的同seed四组前沿PNG/PDF。第二个seed参与全部统计，其原始前沿在raw。连线仅帮助查看离散点，不表示连线之间存在可行调度。‘解数更多’与‘分布/质量更好’不同：NATIVE6平均unique ND 90.06多于POOL6的86.46，但两项主质量整体较差。\n\n"
    text += "## 8. 建议下一步（本轮未启动）\n\n"
    text += "1. **优先确认生成器改动**：固定评价6，比较原生6、共同池6、匹配随机流的原生对照，以及将较大筛选库/代表规则拆到各动作的版本。用全新base、关闭研究observer的生产配置、相同总wall；检验本轮收益是否延续，不改状态/奖励一起掺入。\n2. **先做可等价的工程优化**：对A3代理排序/共享准备与A7/A8构造做同源候选身份、顺序、RNG后态、exact/Archive一致性和benchmark。若只省不可见的少量对象包装，收益可能很小；不得在缺benchmark时承诺整体加速。\n3. **再测试有条件追加评价**：以动作、偏好、已见前缀增益/可用候选及真实增量成本为少量特征，按base隔离训练/留出；保留随机继续抽查，并与成本匹配随机分配比较。尤其不能按前三个零收益硬停。只有独立预测/等时间整程证据支持后才实现预算控制器。\n4. **暂不增加大RL状态表或直接换复杂RL**：当前Q仍负责选动作，本轮没有验证状态扩维。候选池、算子选择和投入数量先各自归因；本轮也没有测试MacroSearch内部候选链或更深轨迹。\n\n"
    text += "## 9. 统计解释与完整性检查\n\n"
    checks = [
        ["Simpson", "检查50/100、H/T、补跑敏感性；方向未反转，不据子组宣布普适"],
        ["生态层级", "40base为单位；不将15360调用或160块当独立n"],
        ["选择偏差", "正式矩阵全部640保留；压力场景非现实总体的随机抽样"],
        ["collider", "完整矩阵为主；零改善条件样本只用于诊断，未作在线策略效应"],
        ["基率", "零3/零6分母明示；桶比例与访问加权估计分开"],
        ["均值回归", "全新40base，非按本轮极端胜负选例；不从旧批包络外推"],
        ["存活偏差", "没有丢run；16未运行者补齐，旧中断日志完整保留"],
        ["look-elsewhere", "六项预先主家族联合Holm；其余27项仅诊断"],
        ["分析自由度", "沿用冻结统计/seed；敏感性及prefix集合核查显式标为补充"],
        ["因果/相关", "线上对照支持本配置组合；动作收益、成本和中介来源不作单项因果认证"],
        ["反向因果", "线上预算先指定；但轨迹变化会改变后续状态/调用构成，不从末态反推最优投入"],
    ]
    text += _table(["11/11检查项", "处理与限制"], checks)
    text += "本轮原运行代码、数据与默认算法保持。完整统计已独立重放，原worker最终exact证据通过压缩原文与完整产物hash绑定，本轮未fresh重评所有最终解。代码/测试/CI、raw/报告交付的最终状态以复现回执和Notion版本记录为准；不把科学分析完成冒充已合并/创新认证。\n"
    for token in ("\\(", "\\)", "\\[", "\\]"):
        if token in text:
            raise ValueError("Old math delimiter in report")
    (out / "RESULTS_ANALYSIS.md").write_text(text, encoding="utf-8")
    (out / "report_metadata.json").write_text(
        json.dumps(
            {
                "source_commit": data["source_commit"],
                "source_hash": data["source_hash"],
                "figure_pairs": len(list(figures.glob("*.png"))),
                "all_math_delimiters_checked": True,
                "report_claims": "prespecified six-test results and clearly marked diagnostics",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Render the verified immutable-data analysis as a standalone report package."""
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("raw", "analysis", "out", "collection", "completion"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    generate(args.raw, args.analysis, args.out, args.collection, args.completion)


if __name__ == "__main__":
    main()
