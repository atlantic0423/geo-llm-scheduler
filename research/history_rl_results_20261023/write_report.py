"""Render the complete Chinese report from sealed tables, without new inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from geo_llm_scheduler.experiments.history_rl import ARMS


def report(data: Path, output: Path, destination: Path) -> None:
    """Explain all twelve arms, frozen families and practical limitations in Chinese."""

    def table(name: str) -> list[dict]:
        with (output / name).open(encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    summaries = {(int(r["n"]), r["arm"]): r for r in table("arm_summary.csv")}
    stats = json.loads((output / "statistics.json").read_text())
    supplementary = json.loads((output / "supplementary.json").read_text())
    extra = {(r["n"], r["arm"]): r for r in supplementary["summaries"]}
    acceptance = json.loads((data / "LOCAL_RECOVERY_ACCEPTANCE.json").read_text())

    def f(arm: str, name: str, n: int = 200) -> float:
        return float(summaries[n, arm][name])

    lines = [
        "# 历史动作与反馈状态12组：完整结果与机制分析",
        "",
        "## Material Passport",
        "",
        "- 状态：完整数据回收与冻结统计已完成；科学推断限于复用开发面板。",
        "- 来源：768次正式原始结果、原验收器、服务器/完整tar/解压文件逐SHA；技术pilot另列。",
        "- 运行源：1ec42415d0fc86f479c81ebb6e4b0601e63ece7f；本报告不改变默认算法。",
        "",
        "## 结论",
        "",
        "这套方法已经实际试验，历史状态和Q更新的实现核验通过，但尚未确认最终前沿质量的稳定提升。20项主检验和22项对普通BILL4 MOEA/D的次检验，全部未通过各自Holm校正。不能把均值最高的STOP2宣布为胜者，也不能把未显著解释成方法等价或无效。",
        "",
        f"更明确的描述性变化是局部搜索成本结构：200任务原Q42整步LS占{100 * f('STRICT_TWO_FOCUS', 'step_action_share'):.2f}%，上一动作9态为{100 * f('HISTORY_ACTION', 'step_action_share'):.2f}%，偏好历史75态为{100 * f('PREF_HISTORY', 'step_action_share'):.2f}%；STOP2和前沿STOP2分别为{100 * f('PREF_HISTORY_STOP2', 'step_action_share'):.2f}%、{100 * f('PREF_HISTORY_FRONTIER_STOP2', 'step_action_share'):.2f}%。这是状态与动作/调用政策改变后的实测组合，不是保持同一算法的纯加速，也不是每次run总墙钟缩短——所有正式run仍用相同总时间。",
        "",
        "当前不晋级75态或前沿奖励为默认。若继续研究，优先保留简洁状态和可审计的投入控制作为候选，研究进入/继续LS的净收益与中段覆盖；本批没有启动下一轮。",
        "",
        "## 数据与公平性",
        "",
        f"正式12组×原32输入×两seed16301/26301，共768次/64同host块；原16开发父，四规模20/50/100/200各4父。单run总墙钟300/600/1200/2400秒，初始化、状态、构造、exact、日志与后处理均计入生产预算，最终独立exact审计/导出另列。15:36:20开始，18:27:39结束，总2小时51分19秒。完整{acceptance['files']}文件/{acceptance['raw_bytes']}原始字节已回收，tar {acceptance['archive_size']}字节，SHA256 `{acceptance['archive_sha256']}`。",
        "",
        "原validator复核全部768次的九项产物size/SHA、输入/spec/配置/source/seed/host/初始化指纹、Q维度和预算均通过。原worker668168个最终存储成员独立exact证据由完整哈希绑定，本次没有fresh重算所有最终解。全部768步日志流的计数与终止完整性通过，全部Q表逐步更新回放一致；512个历史组run另核对START、上一动作/结果/奖励、next_state、二元信用及terminal零bootstrap。STOP2可核对连续未接受和终止标签，但日志没有每步完整Archive快照，未冒称仅凭日志独立复算了全部新目标点停止条件。",
        "",
        "统一使用本批每个输入所有12组与两seed的经验非支配union，按其ideal/nadir归一化，HV参考点1.1；这是经验包络，不是真实Pareto前沿。768个完整HV以不同方向积分交叉验证；每个IGD+在最多16个参考点上与独立标量实现交叉验证，完整指标采用分块NumPy计算。没有把抽样IGD+复核说成全参考点双实现复核。",
        "",
        "推断单位为16父：每父两变体、两seed的指标先平均，四规模等权；分层20000次bootstrap、100000次配对sign-flip，统计根seed2026102301，以contrast+metric的SHA派生子seed。65536种符号精确枚举仅为审计，Holm仍使用冻结的100000次Monte Carlo值。独立脚本已重建全部42项统计、两家族校正及父级聚合；每规模只有4父，规模拆分均为描述。",
        "",
        "## 12组总体结果",
        "",
        "下表为全部规模等权、父级汇总；相对HV列是每次配对比例再平均，不是两个总体均值相除。P100与外部Archive分开计量。",
        "",
        "|组|Archive HV|IGD+|对Q42配对HV均值变化|对普通底座配对HV均值变化|最终P100 HV|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        lines.append(
            f"|{arm}|{f(arm, 'hv', 0):.6f}|{f(arm, 'igd_plus', 0):.6f}|{f(arm, 'hv_vs_baseline_pct', 0):+.3f}%|{f(arm, 'hv_vs_bill_pct', 0):+.3f}%|{f(arm, 'pop_hv', 0):.6f}|"
        )
    lines += [
        "",
        "## 冻结主比较",
        "",
        "HV差为左减右；IGD+改善为右减左，两者正值均表示左侧更好。CI为边际95%bootstrap区间，不能代替联合Holm判断。",
        "",
        "|比较|指标|改善均值|边际95%CI|原p|Holm p|正向父/16|",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in stats["statistics"]:
        if row["family"] == "primary":
            lines.append(
                f"|{row['contrast']}|{row['metric']}|{row['mean']:+.6f}|[{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}]|{row['p']:.6f}|{row['p_holm']:.6f}|{row['positive']}|"
            )
    lines += [
        "",
        "PREF_HISTORY对同状态BANDIT的HV原p=0.012820，边际CI不含零，但联合20项校正后p=0.256397；IGD+也没有相同支持，因此不能宣称多步bootstrap胜出。其余主项Holm p均为1。22项对普通BILL底座的比较也均未通过，最小原p约0.214568；完整次家族在statistics.json中保留，未挑选最好组重新做未校正检验。",
        "",
        "## 四规模表现",
        "",
        "下表为Archive HV对同BILL4普通底座的配对相对变化，均为描述性结果。",
        "",
        "|组|20任务|50任务|100任务|200任务|",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        lines.append(
            "|"
            + arm
            + "|"
            + "|".join(f"{f(arm, 'hv_vs_bill_pct', n):+.3f}%" for n in (20, 50, 100, 200))
            + "|"
        )
    lines += [
        "",
        "历史状态在200任务下较原Q42的均值更好，但对普通底座的差距非常小；20任务STOP2的较高均值不能外推到100/200任务。相同组在不同规模方向并不完全一致，整体排序不是普遍优劣。",
        "",
        "## 成本、轨迹与覆盖",
        "",
        "200任务下的父级均值：",
        "",
        "|组|整步LS占比|已测LS子部件占比|外层代数|每轨迹步数|已访问状态/总行|Archive唯一ND|P100唯一ND|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        lines.append(
            f"|{arm}|{100 * f(arm, 'step_action_share'):.2f}%|{100 * f(arm, 'ls_measured_share'):.2f}%|{f(arm, 'generations'):.2f}|{f(arm, 'steps_per_trajectory'):.3f}|{f(arm, 'q_visited_states'):.2f}/{f(arm, 'q_states'):.0f}|{f(arm, 'archive_nd'):.2f}|{f(arm, 'population_nd'):.2f}|"
        )
    lines += [
        "",
        "整步LS与已测构造/SSGS/exact/Archive子部件相互重叠，不能相加。互斥分解图只用初始化、LS已测子部件、Polish、外层评价和剩余部分；剩余部分不是Q-learning耗时。原Q42的severity计算及支持工作没有独立完整计时，不能用差额给某个函数精确归因。历史状态专用编码计时只反映新路径，原组的零字段表示未单列观测，不能说原状态计算免费。",
        "",
        f"200任务PREF_HISTORY外层代数{f('PREF_HISTORY', 'generations'):.2f}，原Q42为{f('STRICT_TWO_FOCUS', 'generations'):.2f}，但普通底座仍为{f('BILL_PERMUTED', 'generations'):.2f}。腾出的LS时间没有自动转成更好前沿；额外外层机会、选择政策和不同搜索路径共同影响结果。",
        "",
        "### 状态信息增多之后，学习样本仍然稀疏",
        "",
        "|200任务组|已访问状态动作格/全部格|已访问格平均调用次数|已访问格中仅调用1次的比例|",
        "|---|---:|---:|---:|",
    ]
    for arm in (
        "STRICT_TWO_FOCUS",
        "HISTORY_ACTION",
        "HISTORY_RESULT",
        "PREF_HISTORY",
        "PREF_HISTORY_STOP2",
    ):
        e = extra[200, arm]
        lines.append(
            f"|{arm}|{e['state_action_visited']:.2f}/{int(f(arm, 'q_states')) * 8}|{e['mean_calls_per_visited_cell']:.2f}|{100 * e['singleton_visited_cell_fraction']:.2f}%|"
        )
    lines += [
        "",
        "9态提高了实际可见表格覆盖；再加结果、偏好后，状态行访问更广，但75态的已访问状态动作格约一半只选过一次。表格有更多信息，不等于每个动作价值学得更可靠。这是支持继续检查样本分配的机制观察，不能单独证明稀疏就是未显著的原因。",
        "",
        "### 提前停止",
        "",
        f"PREF_HISTORY_STOP2全部规模平均每轨迹{f('PREF_HISTORY_STOP2', 'steps_per_trajectory', 0):.3f}步，原固定5步；20任务{f('PREF_HISTORY_STOP2', 'steps_per_trajectory', 20):.3f}步，200任务{f('PREF_HISTORY_STOP2', 'steps_per_trajectory'):.3f}步。200任务约{100 * f('PREF_HISTORY_STOP2', 'early_stop_rate'):.2f}%轨迹触发连续两步无收益停止，整步LS从PREF的{100 * f('PREF_HISTORY', 'step_action_share'):.2f}%降至{100 * f('PREF_HISTORY_STOP2', 'step_action_share'):.2f}%。前沿STOP2为{f('PREF_HISTORY_FRONTIER_STOP2', 'steps_per_trajectory'):.3f}步。",
        "",
        "停止规则确实减少了后续调用，最终质量增量却没有通过校正；不能把未显著解释为无损早停。规则也可能截断需要前置铺垫的动作序列。",
        "",
        "### 前沿反馈与候选宽度",
        "",
        f"200任务原PREF零奖励比例{100 * f('PREF_HISTORY', 'zero_reward_rate'):.2f}%，前沿组{100 * f('PREF_HISTORY_FRONTIER', 'zero_reward_rate'):.2f}%，其中{100 * f('PREF_HISTORY_FRONTIER', 'area_success_rate'):.2f}%动作出现冻结坐标面积增加。局部接受比例为{100 * f('PREF_HISTORY', 'accepted_rate'):.2f}%和{100 * f('PREF_HISTORY_FRONTIER', 'accepted_rate'):.2f}%。奖励更常为正，最终HV/IGD+优势仍未确认；在线信用坐标来自初始化，与事后统一经验前沿坐标不同，边界之外的点可能拿不到面积信用。",
        "",
        f"PREF_HISTORY_B3在200任务的整步LS占比为{100 * f('PREF_HISTORY_B3', 'step_action_share'):.2f}%，候选exact均值{f('PREF_HISTORY_B3', 'mean_exact_per_step'):.2f}，局部接受率{100 * f('PREF_HISTORY_B3', 'accepted_rate'):.2f}%，B6对应{f('PREF_HISTORY', 'mean_exact_per_step'):.2f}与{100 * f('PREF_HISTORY', 'accepted_rate'):.2f}%。B3确实少做评价，但对B6与原Q42的B3均无确认质量收益；不能仅凭便宜就采用。prefix_metrics.csv只是在同一已生成候选批上的离线前缀观察，不是独立运行B3的反事实效果。",
        "",
        "### 动作衔接与两端",
        "",
        "200任务PREF中A6占已选步约6.9%，条件接受约1.3%；A1/A2/A5/A7的条件接受更高。不同动作面对的上下文不同，不能由这些条件比例直接删掉A6或宣布某动作有因果优势。transition_metrics.csv保留上一动作×结果×偏好→下一动作的完整计数/信用/时间，不能只看最常见的一条衔接就宣称学到了通用策略。",
        "",
        "200任务各组最低Bill与普通底座完全相同；PREF的最快Flow相对底座均值约改善0.068%，变化很小。事后限制实际保存点Flow不超过普通最快值1.1倍，PREF平均省费为−0.257%、ACTION为−0.102%、STOP2为−0.135%（负号表示更贵）；不能以总体HV的小幅均值增加宣称快速服务区已省费。该阈值是探索性描述，没有追加检验或与主家族混用。主要差别需要继续看中段覆盖。",
        "",
        "## 每组如何处理",
        "",
        "|组|本次判断与后续用途|",
        "|---|---|",
    ]
    decisions = [
        "相同初始化和替换顺序的必要净收益底线；本轮任何复杂组都未确认优于它。",
        "原Q42研究参考，200任务LS投入较大；保持对照，不因均值稍低宣布其被替代。",
        "匹配随机动作的必要对照；PREF未确认胜过它，不能声称已证明学习必需。",
        "最直接的论文迁移，9态且LS支持成本较低；质量无确认优势，作为简洁候选保留。",
        "反馈结果加入后均值略增，但对ACTION未确认；还不足以证明结果分类的增量。",
        "偏好避免不同目标方向混叠，但对RESULT无确认增量，且状态动作格更稀疏。",
        "同状态gamma=0是多步信用的必要对照；HV原p小但校正后失败，不宣布Q胜bandit。",
        "减少无收益后续调用的机制成立，质量保持或提升未获确认；只保留投入控制假设。",
        "信用从微小面积/秒惩罚改为有界成功信号，正奖更多；最终前沿增益仍未确认。",
        "组合减少调用，但不能将均值叠加称协同；200任务对普通均值几乎持平。",
        "原状态的小宽度对照，评价成本更低但最终质量未确认；不按成本单独晋级。",
        "同小宽度历史增量未确认；相对历史B6的HV仅2/16父正向，也未通过检验。",
    ]
    lines.extend(f"|{a}|{d}|" for a, d in zip(ARMS, decisions))
    lines += [
        "",
        "## 图表与复现",
        "",
        "- 01_quality_by_scale：全部12组四规模配对相对HV。",
        "- 02_cost_200：互斥成本分解，避免把重叠计时相加。",
        "- 03_paired_effects：16父级对Q42的描述性边际区间，其中部分并非冻结主比较，不追加检验，也不按边际区间宣称校正后显著。",
        "- 04_action_exposure_200：条件动作选择/时间/接受，不能作随机化动作因果解释。",
        "- 05_anytime：同输入经验坐标的中途前沿观察，阈值触发可能有延迟。",
        "- arm_01至arm_12_fixed_fronts：每规模预先按最小父编号、variant0/seed16301选固定案例，不挑最好看的一例。",
        "",
        "## 统计与研究完整性检查（11/11）",
        "",
        "|检查|处理|",
        "|---|---|",
        "|Simpson反转|保留四规模拆分；排序/方向异质性不强行归为同一规模规律。|",
        "|生态谬误|父级推断不外推为每次run或每个动作的效果。|",
        "|Berkson选择|复用开发面板，外部有效性有限；未称新留出。|",
        "|Collider控制|不按运行后的奖励/接受率筛选样本，也不对它们回归调整总效果。|",
        "|忽视基率|报告空批、无可行、零奖励与调用基数；没有分类器准确率推断。|",
        "|均值回归|全部32输入/12组保留，没有挑极差或极好父重复比较。|",
        "|幸存者偏差|正式768次完整，不排除失败或挑完整子集；技术pilot另列。|",
        "|多重搜索|20主/22次分别Holm，全部结果公开；未事后选最优组重检。|",
        "|分析分叉|主协议冻结；1.1倍Flow窗口、动作与前缀仅描述性，不包装成预注册胜负。|",
        "|相关与因果|固定配对政策比较与条件动作日志区分，不把接受率解释为动作因果价值。|",
        "|反向因果|奖励在动作完成后生成，历史输入只取前一步；Q回放核对无未来反馈泄漏。|",
        "",
        "## 交付边界",
        "",
        "768次计算、完整raw回收和冻结分析已完成；服务器原件与默认算法保留，未开下一轮。源码/报告/小型汇总走独立PR，完整raw、原运行bundle、技术pilot、分析bundle、图表和验真回执走GitHub Release。最终PR/CI及资产验证范围在REPRODUCTION.md和Notion09记录；不能把发布或测试通过当作算法有效性。",
        "",
    ]
    text = "\n".join(lines)
    assert not any(old in text for old in ("\\(", "\\)", "\\[", "\\]"))
    destination.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    report(args.data, args.output, args.destination)
