"""Chinese evidence report for the authorized twelve-configuration screen."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]) / "outputs/lsv2_budget12_20261028"
A = ROOT / "analysis"
Q6 = "Q1_NET_G9_B6L5"
Q3 = "Q1_NET_G9_B3L3"
Q1 = "Q1_NET_G0_B1L1"
U6 = "UNIFORM_B6L5"
LABEL = {
    Q6: "Q9 B6L5",
    Q3: "Q9 B3L3",
    Q1: "Q0 B1L1",
    "Q1_NET_G0_B6L5": "Q0 B6L5",
    "Q1_NET_G0_B3L3": "Q0 B3L3",
    U6: "随机 B6L5",
    "UNIFORM_B3L3": "随机 B3L3",
    "UNIFORM_B1L1": "随机 B1L1",
    Q6 + "_NOA8": "Q9 B6L5 去A8",
    "UNIFORM_B6L5_NOA8": "随机 B6L5 去A8",
    "BILL_RANDOM": "历史 BILL4＋MOEA/D",
    "EXTH_SPEA2": "增强 SPEA2（本批）",
    "EXTH_MOEAD": "增强 MOEA/D（本批）",
    "EXTH_NSGAII": "增强 NSGA-II（历史）",
    "EXTH_RK_MOPSO": "增强 RK-MOPSO（历史）",
    "EXTH_EPS_IBEA": "增强 ε-IBEA（历史）",
}


def _rows(name: str) -> list[dict]:
    rows = list(csv.DictReader((A / name).open(encoding="utf-8-sig")))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def _table(head: list[str], rows: list[list]) -> str:
    return (
        "| "
        + " | ".join(head)
        + " |\n| "
        + " | ".join(["---"] * len(head))
        + " |\n"
        + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
        + "\n"
    )


def main() -> None:
    """Compose exact exported numbers with bounded interpretations and counterexamples."""
    z = json.loads((A / "analysis.json").read_text())
    m = _rows("run_metrics.csv")
    d = _rows("run_diagnostics.csv")
    rt = _rows("runtime.csv")
    b = _rows("baseline_metrics.csv")
    op = _rows("action_reward_diagnostics.csv")
    ph = _rows("phase_diagnostics.csv")
    by = {(r["parent"], r["arm"]): r for r in m}
    bb = {(r["parent"], r["arm"]): r for r in b}
    parents = sorted(
        {r["parent"] for r in m}, key=lambda p: (next(r["n"] for r in m if r["parent"] == p), p)
    )

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows]))

    def diag(arm, n=0):
        return [r for r in d if r["arm"] == arm and (not n or r["n"] == n)]

    report = """## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-28
- Verification Status: ANALYZED；数据传输、指标复算和Q更新已验证，未重新运行完整求解实验。
- Version Label: budget12_analysis_v1

# Q1不变的十二配置：完整回收与详细分析

**192/192全部完成，失败0，已完整回收到本地。** 北京时间00:47:16启动，02:42:58结束，约1小时55分42秒，早于08:00；未使用仅保留100GiB的放宽，未停止其他任务。

这轮结论是：**随机选择器B6L5本轮总体表现最好；Q9 B3L3是值得保留的大规模开发候选；B1L1不宜直接替代；去A8的效果依赖控制器，不能直接断言A8有害。** 当前数据仍未确认Q-learning相对匹配随机选择器的稳定独立增益。

## 1. 实际跑了什么、哪些可以直接比较

16个开发实例，20/50/100/200各4个；每配置每实例一个seed16901，共192次。各规模总预算300/600/1200/2400秒，包含初始化。Q9/Q0分别表示折扣0.9/0；Q1状态、NET奖励、学习率0.1和探索率0.2不变，均每次运行从空Q表开始。

| 组别 | 配置 | 数量 |
|---|---|---:|
| 完整池B6L5 | Q9、Q0、随机 | 3 |
| 完整池B3L3 | Q9、Q0、随机 | 3 |
| 完整池B1L1 | Q0、随机，全部terminal，不重复G9 | 2 |
| 去A8的B6L5 | Q9、随机 | 2 |
| 补齐外部方法 | 增强SPEA2、增强MOEA/D | 2 |

B是每次动作最多评价的完整候选预算，L是一条局部搜索轨迹最多动作数。B6L5、B3L3、B1L1的理论上限分别是30、9、1；实际可能少于上限。P3的一次完整候选内部仍可包含多个Prefill调整，B1不等于只移动一个操作。

内部10臂保留BILL4、累计20%局部搜索时间份额、Archive选起点、接受/回流规则。两外部臂沿用20启发式＋1个BILL单trial＋79随机初始化和每代20%旧Polish，与BILL4纯MOEA/D是不同方法。历史三个增强基线及BILL4纯MOEA/D另作参照，不计入192次。

当前12配置用同一实例输入、同一seed、同一预算；相同初始化策略内的种群与Archive指纹一致，历史基线输入和种群指纹也核对一致。但不同算法消耗随机流不同，墙钟终止与系统负载也会改变路径；同seed不能理解为所有内部随机决策相同，更不能消除跨批次混杂。

## 2. 评价方法与验收证据

主表按每实例本批12个最终前沿的非支配并集建立统一经验参考，归一化后HV参考点为(1.1,1.1)，IGD+使用同一经验前沿。HV越大越好，IGD+越小越好；HV不是省电比例，也不限制在1以下。所有均值按实例等权，各规模权重相同。历史比较另用12新配置＋4历史配置的共同经验参考，重新计算，不能拼接旧报告的指标。

- 完整回收3338文件，原始232742141字节，压缩包163533810字节；包及全部文件逐SHA256/大小匹配，服务器原件保留。
- 原runner验证器及每run完成封条通过，所有192次以time_budget终止；总实际solver时间约60.008小时，最大单run超时9.13秒。
- 运行结束时worker已对277727个最终种群/Archive成员完成独立exact核查，完成封条绑定该证据；本次另外对每个新旧run前沿抽查端点和中点，共768个fresh exact，均通过。不是本轮重新exact全量277727个解。
- 96个Q运行的412002次更新完整重放，奖励、可用动作、选择概率、bootstrap、TD target和最终Q表一致；run起始Q为空、L1 terminal、去A8掩码与状态退化均正确。
- 448组HV/IGD+数值及16个比较的均差、胜负、置换p值和Holm校正，用另一套计算路径独立核对通过。详见analysis/VALIDATION.json。

## 3. 十二配置的总体结果

"""
    ranking = sorted(z["arms"], key=lambda a: -z["arms"][a]["metrics"]["hv"])
    report += _table(
        ["配置", "平均HV ↑", "平均IGD+ ↓", "20规模HV", "50规模HV", "100规模HV", "200规模HV"],
        [
            [
                LABEL[a],
                f"{z['arms'][a]['metrics']['hv']:.5f}",
                f"{z['arms'][a]['metrics']['igd_plus']:.5f}",
            ]
            + [f"{z['arms'][a]['by_size'][str(n)]['hv']:.5f}" for n in (20, 50, 100, 200)]
            for a in ranking
        ],
    )
    report += """
随机B6L5同时具有最高平均HV和最低平均IGD+，因此不是某一个指标单独偏爱它。Q9 B3L3在100、200规模的平均HV为本批最高，但50规模偏弱；去A8的Q总体HV略高于Q9 B3L3，100规模却没有优势。不存在一版Q在四规模都稳定领先。

![全部内部配置相对随机B6L5的逐实例HV差](figures/hv_differences.png)

## 4. 预定的八项主比较：有信号，但没有通过多重校正的优势

以下差值均为“前者减后者”的归一化HV绝对差。以16实例为比较单位，枚举65536种配对符号翻转，双侧检验；八项主比较统一Holm。每个规模只有4实例，不把分层均值当显著性结论。

"""
    report += _table(
        ["比较", "HV均差", "胜/负", "原始p", "Holm p", "IGD+均差"],
        [
            [
                LABEL[t["treat"]] + " 对 " + LABEL[t["ctrl"]],
                f"{t['mean']:+.5f}",
                f"{t['positive']}/{t['negative']}",
                f"{t['p']:.5f}",
                f"{t['p_holm']:.5f}",
                f"{t['igd_diff']:+.5f}",
            ]
            for t in z["tests"][:8]
        ],
    )
    report += """
八项均未达到Holm后0.05。随机B1L1退步的原始p约0.020，但校正后约0.161；可以把它列为当前不优先的方案，不能称已严格证明它必然更差。Q9 B3L3胜12/16，也是开发信号。

B1L1的Q0对Q9 B6L5是整个配置比较；由于L1全部terminal，折扣本身不生效，但不能把这项称为只改变L的纯单因素实验。补充的匹配G0比较也保留在次要比较表。B与L同时变化，本轮不能区分收益来自宽度还是深度。

analysis.json保留按规模分层的20000次bootstrap区间，它与符号翻转检验并非互为反演；区间若不跨0，也不能代替预定的Holm检验宣布显著。开发实例已反复使用，p值仅作探索性支持，不代表对全新问题总体的确认。

## 5. B3L3为什么值得保留，B1L1为什么没有达到预期

你的原设想中，“短轨迹可以给更多起点做LS”确实发生了。这里的轨迹从Archive抽取起点，**不等于给同样数量的不同子代做LS，也不保证每次起点互不重复**。

"""
    rr = []
    for n in (100, 200):
        for a in (Q6, Q3, Q1, U6, "UNIFORM_B3L3", "UNIFORM_B1L1"):
            dd = diag(a, n)
            rrt = [r for r in rt if r["arm"] == a and r["n"] == n]
            rr.append(
                [
                    n,
                    LABEL[a],
                    f"{mean(dd, 'trajectories'):.1f}",
                    f"{mean(dd, 'steps'):.1f}",
                    f"{mean(dd, 'ls_exact'):.0f}",
                    f"{100 * (1 - mean(dd, 'zero_gain_share')):.1f}%",
                    f"{mean(rrt, 'generations'):.1f}",
                ]
            )
    report += _table(
        ["规模", "配置", "轨迹数", "动作数", "LS exact数", "正局部HV步比例", "等价代数"], rr
    )
    report += """
轨迹可能在总预算结束时刚计数但未执行完整动作，所以轨迹数与动作数/L可能有很小差异。等价代数为offspring数/100，不是都恰好整代结束。

**B3L3：** 100规模Q的轨迹数约553→1676，200规模约456→1409，均接近三倍；但代数只从853→885、855→852。主要变化是同样约20%的LS时间内重新分配起点、宽度和连续步数，并不是靠大幅增加MOEA/D代数取胜。平均质量提高与更广的起点覆盖相容，但本轮没有把起点、宽度、深度的贡献单独分开。

**B1L1：** 100规模Q轨迹数约10864，却只有约7483次LS exact；200规模约7546条轨迹、4087次LS exact。对照B6L5分别约12155和10219次LS exact。更多次“调用”没有带来更多完整候选，单步内找到改进的机会也减少。100规模正局部HV步比例从25.9%降至10.2%，200从35.0%降至18.2%。随机选择器B1L1同样下降，说明问题不能全部归因于Q学习。

这些是“更浅、更窄并不自动更好”的证据，而不是长程信用已被证明必需。B1L1同时改变候选宽度、轨迹深度、重新抽起点和frame刷新节奏，不能只按未来收益解释。

**重要反例与增益集中：** Q9 B3L3相对Q9 B6L5在100和200规模均胜3/4，但100的st_n100_d12下降0.04612，200的st_n200_d07下降0.00924。100的st_n100_d13单例增量0.10930，占该规模净增量约80.5%；200的st_n200_d14增量0.09299，占约79.8%。这是净增量占比，负差会影响分母，不是贡献归因。相对随机B6L5，Q9 B3L3在100胜3/4，在200只胜2/4。

## 6. 去A8：对Q改善、对随机下降，不宜直接删除

Q9去A8的HV均差+0.01976，胜12/16；随机去A8却为−0.02569，胜负8/8。Q去A8在100规模的均差仍为−0.00469，并非所有大实例受益。

A8并不是纯无效动作。在完整B6L5中，Q平均约3.0%的动作选A8；合并调用计数约12.4%有正局部HV、约47.0%无完整候选；A8约占所记录步骤耗时2.8%。随机选择器相应约4.0%的调用、11.3%正局部HV、3.2%的步骤耗时。因此直接删除它并非主要消除一个巨大的时间负担，还可能失去少数有价值的提案。

去A8同时改变动作集合、后续随机路径、bootstrap候选和状态维度。Q的已访问可用状态动作格中，不足5次的比例约52.2%→23.8%，学习覆盖更充分，可能是Q侧改善的解释之一；但随机侧下降说明不能把全部差异解释成“A8本身有害”。该交互目前只是描述线索，未做独立交互确认。

st_n50_d07上Q去A8由0.81519升至0.99891，但随机去A8由0.95621降至0.80280；st_n100_d09上随机去A8由0.82893降至0.57821。保留这些反例，不只展示去A8成功的图。

## 7. 状态少了以后，学习覆盖改善了，但质量没有自动改善

"""
    qarms = (Q6, Q3, Q1, Q6 + "_NOA8")
    report += _table(
        [
            "配置",
            "状态上限",
            "平均访问状态",
            "平均访问动作格",
            "已见可用格不足5次",
            "负奖励比例",
            "无即时HV但正TD目标",
        ],
        [
            [
                LABEL[a],
                limit,
                f"{z['arms'][a]['diagnostic']['q_states']:.2f}",
                f"{z['arms'][a]['diagnostic']['q_pairs']:.1f}",
                f"{100 * z['arms'][a]['diagnostic']['pairs_lt5_share']:.1f}%",
                f"{100 * z['arms'][a]['diagnostic']['q_reward_negative_share']:.1f}%",
                f"{100 * z['arms'][a]['diagnostic']['q_zero_gain_positive_target_share']:.1f}%",
            ]
            for a, limit in zip(qarms, (30, 18, 6, 15))
        ],
    )
    report += r"""
不足5次的分母是该run实际访问状态下出现过的可用动作集合，包含其中零访问动作；不是只对已访问格计算。状态上限不是神经网络输入维数；剩余步数/门控退化后，实际状态自然减少。

B1L1几乎访问所有6种状态，稀疏格大幅减少，质量却下降。因此**当前证据不支持“状态数量太少，所以必须补状态”这个推断**；也不能反过来证明这6种状态信息已足够。覆盖率和状态是否能够区分动作机会是两个问题。

Q9 B6L5约16.9%的全部动作在没有即时正HV时仍拿到正TD目标，Q9 B3L3约11.4%。代码确实会为未来价值提供信用，并非所有零即时收益动作都被一律当成无用；但正TD目标只是估计，不能当成这些动作真的实现长期收益的证据。匹配G0比较尚未确认未来项的稳定净价值。

## 8. 负奖励与P3偏好：本轮最具体的后续诊断线索

NET奖励为：

$$
r=\frac{\Delta HV-\rho\,t}{h}
$$

当改进不超过时间机会成本时，负奖励符合定义，不需要为了提高正奖励比例而人为发正分。但如果大量动作都负，便宜的空调用可能比稍贵但有潜在价值的调用更少扣分；策略可能偏向“少扣分”，这需要结合实际选择日志核查。

"""
    opp = []
    for a in (Q6, Q3, Q1):
        rr = [r for r in op if r["arm"] == a and r["operator"] == "P3"]
        early = [r for r in ph if r["arm"] == a and r["phase"] == 0]
        late = [r for r in ph if r["arm"] == a and r["phase"] == 3]
        opp.append(
            [
                LABEL[a],
                f"{100 * mean(rr, 'call_share'):.1f}%",
                f"{100 * mean(rr, 'empty_share'):.1f}%",
                f"{mean(rr, 'empty_seconds'):.4f}",
                f"{100 * mean(rr, 'greedy_share'):.1f}%",
                f"{100 * mean(early, 'p3_share'):.1f}% → {100 * mean(late, 'p3_share'):.1f}%",
            ]
        )
    report += _table(
        [
            "配置",
            "P3选择比例",
            "P3空候选比例",
            "空调用均秒",
            "选择时位于最大Q集合",
            "前1/4→后1/4的P3比例",
        ],
        opp,
    )
    report += """
本表是先算每run比例再等权平均，与上一节A8的合并调用比例不同。前后四分之一按累计LS时间划分，不是全程墙钟阶段。日志只能识别“被选动作当时属于最大Q集合”，不能区分该次实际走了探索分支还是利用分支。

Q9 B3L3选P3的比例从前段13.8%升到后段47.9%；总体约95.1%的P3选择发生在它属于当前最大Q集合时，说明这种偏好不是单靠均匀探索就能解释。Q0 B1L1还把约19.3%动作给了A4，但A4的每run平均正局部HV率约1.4%；相对之下A7选择比例只有约6.1%，在它实际访问的状态上正局部HV率约43.5%。

P3是冻结Decode、调整Prefill时刻以降低账单的算子；无完整候选不代表内部没计算，一次无候选也不能证明该算子在其他状态没有价值。P3 B3L3空调用每run平均约0.035秒；P3 B1L1约0.017秒。与成本扣分形式、后段选择偏好合起来看，**“低成本空调用在当前奖励下较容易被偏爱”是值得检验的解释**。但不同动作访问的状态由策略选择，直接比较命中率仍有选择偏差，不能据此直接改成固定A7或删除P3。

Q计算本身的记录耗时代理只占LS时间约0.5%—0.9%，没有证据表明Q表查找/更新吞掉了主要预算。优先诊断动作机会和奖励与终点质量的对齐，比先换DQN更贴近本次观察。这里是后续建议，本轮没有改奖励、加状态、屏蔽P3或启动新实验。

![累计LS时间内的选择变化](figures/learning_behavior.png)

## 9. Q控制器相对匹配随机选择器

下表另属明确标记的次要探索性比较族，共8项（6个Q对随机、2个匹配G0预算比较），单独Holm；analysis.json另给全部16项一起校正，结论也均未显著。

"""
    report += _table(
        ["比较", "HV均差", "胜/负", "次要族Holm p"],
        [
            [
                LABEL[t["treat"]] + " 对 " + LABEL[t["ctrl"]],
                f"{t['mean']:+.5f}",
                f"{t['positive']}/{t['negative']}",
                f"{t['p_holm']:.5f}",
            ]
            for t in z["tests"][8:]
        ],
    )
    report += """
上一轮Q1G9是开发筛选里的最佳候选，本轮完整B6L5却只在4/16实例胜匹配随机。这不与上一轮记录矛盾：上轮候选未通过多方案校正，本来也不是稳定优势结论；本轮种子/墙钟路径变化后，排名没有稳定重复。不能说Q已被否定，也不能再仅用“它胜了外部基线”论证Q控制器有效。

## 10. 与五个增强外部基线、BILL4纯MOEA/D比较

本表使用16种方法的共同经验参考，IGD+与主表略有不同。本批两增强基线和内部方法属于同批比较；其他四个基线来自历史批次，只做描述。初始化策略不同，比较的是完整方法包。

"""
    broadarms = (
        U6,
        Q3,
        Q6,
        Q6 + "_NOA8",
        "BILL_RANDOM",
        "EXTH_SPEA2",
        "EXTH_EPS_IBEA",
        "EXTH_NSGAII",
        "EXTH_MOEAD",
        "EXTH_RK_MOPSO",
    )
    report += _table(
        ["方法", "共同参考平均HV", "共同参考平均IGD+"],
        [
            [
                LABEL[a],
                f"{mean([r for r in b if r['arm'] == a], 'hv'):.5f}",
                f"{mean([r for r in b if r['arm'] == a], 'igd_plus'):.5f}",
            ]
            for a in broadarms
        ],
    )
    comps = []
    for q in (Q6, Q3, U6):
        for c in (
            "BILL_RANDOM",
            "EXTH_NSGAII",
            "EXTH_RK_MOPSO",
            "EXTH_EPS_IBEA",
            "EXTH_SPEA2",
            "EXTH_MOEAD",
        ):
            ds = [bb[p, q]["hv"] - bb[p, c]["hv"] for p in parents]
            comps.append(
                [
                    LABEL[q],
                    LABEL[c],
                    f"{np.mean(ds):+.5f}",
                    f"{sum(x > 1e-12 for x in ds)}/16",
                    "、".join(p for p, x in zip(parents, ds) if x < -1e-12) or "无",
                ]
            )
    report += _table(["本方法", "参照", "HV均差", "胜实例", "落后实例"], comps)
    report += """
新增SPEA2在五个增强外部基线中总体最好，但仍明显落后于LSV2主组；外部MOEA/D也未接近LSV2。随机B6L5对新SPEA2和新外部MOEA/D均胜16/16；这支持LSV2整体方案在本面板有优势，但再次说明这些优势不能单独归给Q学习。

Q9 B3L3对BILL4纯MOEA/D胜15/16，例外st_n100_d12；对增强NSGA-II/增强MOEA/D的例外都是st_n50_d07。个别实例或前沿区间可能交叉，不把HV胜负说成全前沿完全支配。

## 11. 逐实例主结果与前沿图

"""
    selected = (Q6, Q3, Q1, U6, Q6 + "_NOA8", "UNIFORM_B1L1")
    report += _table(
        ["实例"] + [LABEL[a] for a in selected],
        [[p] + [f"{by[p, a]['hv']:.5f}" for a in selected] for p in parents],
    )
    report += "\n每实例图分为Q预算、随机预算、A8消融、外部基线四个面板，避免16条曲线挤在一起；每条曲线是一条实际run，没有合并多次结果。\n\n"
    report += "![100规模四实例](figures/overview_n100.png)\n\n![200规模四实例](figures/overview_n200.png)\n\n"
    for p in parents:
        report += f"- [{p} 的完整四面板前沿图](figures/{p}.png)\n"
    report += """

## 12. 时间分配与敏感性检查

内部10臂实际LS份额约20.00%—20.29%，与累计份额规则一致；允许整条轨迹造成少量临时超额。资源没有在第一代全部耗完，也没有给第一代100个解逐个固定安排LS。全程只要累计份额低于目标就有机会进入新轨迹，不是每代独立20%配额。

统一参考点改为(1.2,1.2)后，八项主比较的均差方向均不变。在内部10臂每实例共同最小exact计数截面，均差方向也均不变。这两项只说明主要方向没有被所选HV参考点或终点exact总数轻易翻转；等exact不等CPU成本，也不是正式预定的主要终点。

![各规模随墙钟预算发展的HV](figures/anytime.png)

局部HV增益在算法动态frame内计算，不同frame的增益不能直接相加当成全程最终HV。算子命中率、空调用率、局部奖励和终点质量分别报告，不能拿其中一个替代另一个。

## 13. 这轮回答了什么，下一步怎样收窄

| 原问题 | 本轮证据 | 当前判断 |
|---|---|---|
| 短轨迹能否覆盖更多起点 | B3/L3约三倍轨迹，B1/L1十余倍 | 能，但次数不等于有效候选/独立起点 |
| B1L1是否更好 | Q和随机整体下降，100规模有明显反例 | 暂不优先，不直接替代 |
| B3L3是否有希望 | Q9胜原Q9 12/16，大规模均值最好 | 保留开发候选；增益集中，需重复确认 |
| 是否删除A8 | Q改善、随机下降，A8仍有正收益 | 不全局删除，保留成对消融 |
| 状态变少是否必须补充 | 覆盖提高但质量不单调提高 | 不能凭数量决定补状态 |
| Q的未来信用是否没生效 | 大量零即时HV步得到正TD目标，重放正确 | 信用机制已生效，净价值尚未确认 |
| 五外部方法是否追上 | 新补两方法仍落后LSV2主组 | 整体LS方案有优势，Q独立优势仍待证 |

建议保留**随机B6L5作为强对照，Q9 B3L3作为优先Q候选**；原Q9 B6L5及去A8版本保留必要参照，不把排名当成晋级。下一步先针对P3/低成本空动作偏好核查“什么状态确实有提案机会、同状态下成本扣分是否过度偏向空调用”，再决定是否改奖励或引入无需人工阈值的指标。继续遵守用户本轮Q1不变的要求，本次只分析、不实现这些改动。

短期不建议仅因为状态数少就补更多离散状态，也不建议在奖励与选择行为尚未解释清楚时直接换DQN。新的重复种子/实例与具体奖励或资格方案需要另行冻结，旧16开发实例不应再称全新留出。

## 14. 统计解释检查（11/11）

| 项目 | 核查结论 |
|---|---|
| Simpson总体/分层反向 | 已给四规模及逐实例，Q3在50下降而100/200提高，不能只报总体 |
| 生态谬误 | 实例是统计单位，不把步骤数当独立样本 |
| 筛选偏差 | 这是已筛过、已反复使用的开发面板，不推广到未知总体 |
| Collider偏差 | 未按成功/奖励筛选run；算子条件日志受策略选择，只作描述 |
| 基率忽略 | 同时报正收益、空候选、负奖励的分母与比例 |
| 回归均值 | Q1来自上一轮胜者筛选，未把本轮排名变化当机制证伪 |
| 幸存者偏差 | 192/192无剔除；历史仅16个匹配dev，不混旧holdout |
| 多重搜索 | 主8项Holm、次8项Holm，另给合并16项校正；外部只描述 |
| 分析自由度 | 原12配置与主8项来自冻结协议；P3/分层/等exact均标为诊断或次要 |
| 相关与因果 | 日志偏好不直接归因算子真实价值；外部方法包不归因Q |
| 反向因果 | 可能是搜索先变难再偏爱某算子，未把选择相关性当效果来源 |

bootstrap区间、置换p和Holm回答不同问题；非显著不等于两方法相等，未证明优势也不等于永久否定RL。

## 15. 交付与复现

完整原始数据在results/formal，回收证据RECOVERY_RECEIPT.json，分析数据在analysis/，21张图在figures/。分析脚本位于研究分支的research/lsv2_budget12/；按analyze.py、diagnose.py、plot.py、report.py、validate_analysis.py顺序运行，参数均为工作区根目录。分析不修改求解器源码或原始结果。运行源码670af6e保持独立冻结，分析版本见ANALYSIS_RECEIPT.json。

本批属于开发筛选，不新增PR/Release，不改默认算法。服务器原件保留；Notion只记录结论、状态与本地报告路径，详细哈希/测试/命令保存在回执。
"""
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(dict(report_characters=len(report), instances=len(parents))))


if __name__ == "__main__":
    main()
