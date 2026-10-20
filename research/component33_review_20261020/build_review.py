"""Build a descriptive review of the sealed 33-arm study without running solvers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--results-dir", type=Path, required=True)
_parser.add_argument("--output", type=Path, required=True)
_parser.add_argument("--receipt-dir", type=Path, required=True)
_args = _parser.parse_args()
SOURCE = _args.results_dir
DOC = _args.output
OUT = _args.receipt_dir
OUT.mkdir(parents=True, exist_ok=True)
DOC.mkdir(parents=True, exist_ok=True)

# definition, interpretation, proposed handling; no post-hoc inferential tests.
NOTES = {
    "EXT_MOEAD": (
        "随机初始化普通MOEA/D",
        "纯随机合法MS/OS初始化；没有设计型初始化和模因搜索。",
        "它是初始化收益的锚点。与原混合初始化只改变初始化政策，因此可比较设计型初始化的整程净价值。不能用它单独评价MOEA/D框架的上限。",
        "保留随机条件作整体方法包对照，同时以同BILL初始化版本作机制研究底座。",
    ),
    "EXT_NSGAII": (
        "随机初始化NSGA-II",
        "rank/crowding及父子代精英选择的离散调度适配，纯随机初始化。",
        "整体HV高于随机MOEA/D，但本轮没有预注册五外部之间的胜负检验。完整版优于它是强初始化加搜索整包的结果，未分离搜索框架。",
        "若要比较主干，下一轮给NSGA-II与MOEA/D相同BILL4输入、种群和端到端预算。",
    ),
    "EXT_RK_MOPSO": (
        "随机键MOPSO适配",
        "随机键表示、原生领导库/网格；纯随机合法初始化。",
        "它在五外部中描述性HV最高，但IGD+不如NSGA-II/SPEA2，提示覆盖体积与到经验前沿距离的排序不同。不是双指标稳定冠军。",
        "保留为风格不同的基线；后续同强初始化时保留原生库与被动测量Archive的边界。",
    ),
    "EXT_SPEA2": (
        "随机初始化SPEA2",
        "strength-sum、密度与环境截断；纯随机初始化。",
        "平均HV与NSGA-II非常接近，平均IGD+略低。没有针对这两个外部方法的确认检验，不能把微小数值差异当优势。",
        "继续作有代表性的外部适配；效果讨论明确随机初始化条件。",
    ),
    "EXT_EPS_IBEA": (
        "随机初始化epsilon-IBEA",
        "归一化additive-epsilon指标、kappa=0.05的适配，纯随机初始化。",
        "完整版对它的两个指标通过主家族检验；这不等于指标选择框架无价值。旧13组曾在初始化阶段失败，不是IBEA迭代崩溃；本轮全部运行成功。",
        "保留基线，不因本轮表现而事后改弱参数或把旧初始化故障归给IBEA。",
    ),
    "MIXED_MOEAD": (
        "原混合初始化＋普通MOEA/D",
        "原100成员混合初始化，后续只有普通MOEA/D。",
        "相对随机MOEA/D两个指标均16/16父正向且通过组件家族检验；收益是在扣除初始化耗时之后的最终收益。它是所有混合底座单组件的正确参照。",
        "保留为必要消融，不用随机初始化外部替代这一同主干对照。",
    ),
    "BILL_MOEAD": (
        "BILL4初始化＋普通MOEA/D",
        "原十类各9个＋BILL10，BILL每个位置四选一；无局部搜索。",
        "相对原混合两个指标均16/16父正向并通过检验。它与完整版有相同的初始P100及Archive；最终平均HV反而更高，是当前最有用的简单研究底座。它不属于已证明的全域最优算法。",
        "优先作为后续简洁基准保留；先要求新模块在同初始化下超过它。",
    ),
    "ONLY_A1": (
        "仅A1：单阶段重分配",
        "混合初始化；逐子代强制调用A1，B1/L1，无Polish。",
        "HV下降通过校正，IGD+下降未通过。这里测到的是高频单动作政策连同其支持流程的代价，不是对所有A1调用机会的否定。",
        "停止逐子代强制A1的扩张；若保留，只在阶段资源/分配瓶颈明确时研究稀疏调用。",
    ),
    "ONLY_A2": (
        "仅A2：P/D路径重分配",
        "混合初始化；逐子代A2、B1/L1，无Polish。",
        "HV与IGD+均显著变差。路径评分、结构重建和调用支持开销挤压外层探索；现有数据不能分离哪一项是唯一原因。",
        "不作为默认高频动作；先诊断同P/同路径重复评分与合法候选质量，再考虑精确等价缓存或稀疏触发。",
    ),
    "ONLY_A3": (
        "仅A3：任务区域迁移",
        "混合初始化；使用tariff改进版A3，逐子代B1/L1。",
        "在单动作组中均值较好，但相对混合普通MOEA/D两指标仍下降，校正后未确认差异。A3曾在D8有信号，不代表本轮强制调用政策合算，也不构成跨批矛盾。",
        "保留问题特定候选价值，先限制调用频率；不能凭它是较好的单动作就断言应高频使用。",
    ),
    "ONLY_A4": (
        "仅A4：拥塞前插",
        "混合初始化；逐子代A4、B1/L1，无Polish。",
        "HV下降通过校正，IGD+未确认。它的显式构造计时很小，却仍使外层推进大降，说明不能只盯算子propose函数。",
        "先看拥塞机会、severity/诊断刷新及支持调用开销；不把低构造秒数当整体廉价。",
    ),
    "ONLY_A5": (
        "仅A5：P/D耦合前插",
        "混合初始化；逐子代A5、B1/L1，无Polish。",
        "HV在16父均下降且通过校正；IGD+没有通过家族检验。局部可改善当前标量不保证整程覆盖的净收益。",
        "暂不高频推广；与A4分别按真实等待机会研究，不能把耦合本身当已验证优势。",
    ),
    "ONLY_A6": (
        "仅A6：随机任务LNS",
        "混合初始化；逐子代A6、B1/L1，无Polish。",
        "两个指标下降均通过校正。随机破坏/重构没有在本轮端到端时间内补偿其机会成本。",
        "暂缓作为常规高频动作；将来若研究停滞重启，另设低频触发及匹配随机对照。",
    ),
    "ONLY_A7": (
        "仅A7：Active-Pack",
        "混合初始化；逐子代A7、B1/L1，时序动作且不做SSGS重建。",
        "HV下降通过校正。即使SSGS为零，候选构造、时间/资源机会分析和未单列的支持逻辑仍可很贵；200任务构造约占35%，剩余未细分约61%。",
        "优先做计时定位和重复临界时刻/资源查询的等价复用；先解决真实开销再谈增加调用。",
    ),
    "ONLY_A8": (
        "仅A8：Peak-Coalition",
        "混合初始化；逐子代A8、B1/L1，无Polish。",
        "单动作均值相对较好，但两指标均低于混合普通MOEA/D，校正后未确认差异。该政策没有证明净收益，未显著也不代表无效或等价。",
        "保留为有真实需量峰机会时的候选，后续应补demand压力留出和低频调用；本轮未覆盖该压力。",
    ),
    "POLISH_ONLY": (
        "仅Polish：逐子代右移精修",
        "混合初始化；每个子代做一次Polish，不选择RL动作或更新Q。",
        "两个指标显著变差。200任务约99%的生产时间在Polish内，几乎没有留下结构探索预算。它局部保证Flow不升并不保证等时间整程最优。",
        "明确不宜逐子代执行；优先比较稀疏、停滞或有明确P→D时差机会的调用，而非继续加密精修。",
    ),
    "RPERM_ONLY": (
        "仅随机邻域替换顺序",
        "混合普通MOEA/D只打乱邻域替换遍历顺序，关闭局部搜索。",
        "HV与IGD+均值改善，但Holm分别约0.516和0.0735，未确认。0.0735不能改口叫已显著，吞吐也没有明显崩塌。",
        "保留为低成本候选，别把它当核心创新或稳定增益；在简洁底座上再独立确认。",
    ),
    "POOL_RANDOM_B1_L1": (
        "随机动作池，B1、一步",
        "混合初始化；八动作均匀随机，每个子代一调用、一个exact候选上限，无Polish。",
        "相对混合普通MOEA/D，IGD+显著恶化，HV下降未通过校正。仅把动作凑成池子、随机调用并不能自动获得互补收益。",
        "作为动作选择研究的必要随机对照保留，不建议直接作为最终策略。",
    ),
    "POOL_RANDOM_B6_L1": (
        "随机动作池，B6、一步",
        "只把上一组候选exact上限由1改成6。",
        "相对B1平均HV与IGD+改善，但都没有通过层级家族校正。多看同源候选可能有价值，本轮还没有确认B6的净优势。",
        "保留宽度参照，下一轮在稀疏调用底座上比较；不要据本批宣布固定6已最优。",
    ),
    "POOL_Q_B6_L1": (
        "Q选择动作，B6、一步",
        "与随机B6一步相同，只有动作选择改为Q-learning。",
        "HV均值下降、IGD+也没有改善，校正后未确认。不能说Q已经优于随机；也不能把差异全归给某个状态定义，因为没有完整state×action收益轨迹。",
        "先增加被动的状态/动作/成本/收益日志，验证匹配关系；暂缓扩大状态数量。",
    ),
    "POOL_Q_B6_L5": (
        "Q选择动作，B6、五步",
        "相对Q一步延长为五步轨迹；强制调用，无Polish。",
        "HV略升而IGD+变差，两者均未确认。200任务外层仅约3.84代，说明五步是在大量消耗原本可用于新结构的资源。",
        "研究稀疏触发与浅轨迹，别将“更深”本身当优势；保留一步、随机及普通MOEA/D对照。",
    ),
    "POOL_Q_STRICT_B6_L5": (
        "严格触发的Q五步组件底座",
        "在上一组改为细方向严格匹配、零质量容差，仍混合初始化、无Polish。",
        "平均HV/IGD+改善、吞吐明显恢复，但两个校正检验均未通过。它是触发值得拆分的线索，不是严格门单独有效的确认结论。",
        "优先用既有2×2五组诊断分离触发政策与Polish，不重复开大矩阵。",
    ),
    "POOL_Q_SEVERITY_L5": (
        "severity动态预算组件底座",
        "混合初始化、强制触发/Q五步；换既有severity_v2预算。",
        "相对固定6五步两指标均值改善，但Holm均未通过；预算分配改动并未解除高频五步调用的吞吐瓶颈。",
        "动态预算暂缓作为主线，先把触发/轨迹总投入降下来，再测试预算是否有追加价值。",
    ),
    "POOL_Q_COVERAGE_L5": (
        "coverage动态预算组件底座",
        "混合初始化、强制触发/Q五步；换coverage_v2预算。",
        "相对固定6五步均值小幅改善但没有确认，200任务吞吐仍低。覆盖计算自身有计时，不能只看分配给算子的候选数。",
        "暂缓主推；保留为未来同等总开销的预算对照，不将它与完整DC组混为一组。",
    ),
    "FULL_CURRENT_I4": (
        "当前完整算法，BILL四选一",
        "BILL4＋tariff A3＋随机替换＋偏好/10%容差触发＋Q42/B6/五步＋轨迹末Polish。",
        "对五随机初始化外部十项主检验全通过，说明整个包有收益。但同样BILL初始化的简单MOEA/D均值更好，整包胜随机对照不能为Q、Polish等分别背书。",
        "保留作历史完整对照；后续用BILL普通底座逐项证明局部搜索净收益。",
    ),
    "FULL_STRICT_NO_POLISH_I4": (
        "完整组合：严格触发、关闭Polish",
        "相对当前完整同时收紧触发、关闭Polish，其他设置保留。",
        "对当前完整两个指标均显著改善；对BILL普通两个指标均未确认差异。其HV均值最高也不等于已证明最终冠军；两个变化的贡献仍待2×2拆分。",
        "保留为最有希望的简化候选，尚不晋级默认；先分析既有五组320次。",
    ),
    "FULL_CURRENT_I2": (
        "完整算法，BILL二选一",
        "当前完整版仅将初始化四选一改二选一；进化MacroSearch仍固定6。",
        "HV显著低于四选一，校正p约0.0468，接近阈值需谨慎；IGD+下降未确认。200任务省下约49秒初始化，没有换来明显外层代数增加。",
        "可保留极紧预算候选，但本轮不能称“基本无损加速”；当前质量参考仍用四选一。",
    ),
    "FULL_CURRENT_I1": (
        "完整算法，BILL单次构造",
        "当前完整版仅将BILL初始化改为单次；仍保留复杂搜索包。",
        "相对四选一HV和IGD+均显著恶化。200任务初始化少约75秒，外层仍约4代，节省的大部分预算没有转换成更多外层探索。",
        "不建议当前默认替代；先处理后续高成本搜索，再重新评估初始化强度的时间折中。",
    ),
    "FULL_M00_I4": (
        "完整包：原A3＋固定替换顺序",
        "BILL4完整包同时不启用tariff A3改动和随机替换。",
        "相对当前M11两指标均值较差，但七整包家族校正后未确认。这个比较一次移除两项，不能拆成每项的效应。",
        "保留旧版消融；如研究两因素交互，应完整2×2重新计算预注册对比，不能用均值相加。",
    ),
    "FULL_M10_I4": (
        "完整包：tariff A3＋固定替换顺序",
        "保留A3改进，关闭随机替换，其他完整包保持。",
        "比当前M11均值略好，但两指标均未确认。这不证明随机替换有害，也没有确认取消它更好。",
        "可作为简洁候选，不据小数点排名晋级；先解决Polish与高频调用的大头。",
    ),
    "FULL_M01_I4": (
        "完整包：原A3＋随机替换顺序",
        "关闭tariff A3改动、保留随机替换。",
        "相对当前M11两个指标均值略好，但校正后没有确认；不能由此推翻D8的A3信号，输入、初始化和背景机制不同。",
        "保留必要消融，不把跨批差异当成一个组件的普遍结论。",
    ),
    "FULL_DS_I4": (
        "完整包：severity动态预算",
        "当前BILL4完整包仅将固定6换为severity_v2，含原触发和Polish。",
        "两指标描述性改善，但历史整包家族Holm约0.229/0.187，均未确认。没有解决200任务仍约4代的总体瓶颈。",
        "暂缓作为下一步主方向；先在低频、轻量底座上衡量预算分配追加收益。",
    ),
    "FULL_DC_I4": (
        "完整包：coverage动态预算",
        "当前BILL4完整包仅换coverage_v2，保留触发和Polish。",
        "两指标均值改善但未通过校正，不能把它是几个完整版本中均值较好者解释为预算策略已成功。",
        "保留对照，暂不新增覆盖状态或参数网格；先分离调用次数、单次成本和净质量。",
    ),
}


def _sha(path: Path) -> str:
    """Compute a streaming digest of immutable evidence."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a plain Markdown table."""
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(r) + " |" for r in rows],
        ]
    )


def _build() -> None:
    """Reconcile measured nonoverlapping costs and write the requested review."""
    summary = json.loads((SOURCE / "summary_statistics.json").read_text(encoding="utf-8"))
    groups = {row["arm"]: row for row in summary["arms"]}
    assert len(groups) == len(NOTES) == 33 and set(groups) == set(NOTES)
    rows = list(
        csv.DictReader((SOURCE / "analysis/run_metrics.csv").open(encoding="utf-8", newline=""))
    )
    assert len(rows) == 1056
    raw = SOURCE / "raw/formal"
    costs = []
    matched_hashes = 0
    for row in rows:
        folder = raw / "runs" / row["key"] / row["arm"]
        s = json.loads((folder / "summary.json").read_text())
        diag = json.loads((folder / "diagnostics.json").read_text())
        marker = json.loads((folder / "complete.json").read_text())
        for name in ("summary.json", "diagnostics.json"):
            assert marker["files"][name]["sha256"] == _sha(folder / name)
            assert marker["files"][name]["size"] == (folder / name).stat().st_size
            matched_hashes += 1
        assert s["source"] == summary["canonical"] and s["counts"] == diag["counts"]
        t = diag["seconds"]
        origins = {k.split(":", 1)[1] for k in t if k.startswith("exact:")}
        init_origins = {x for x in origins if x.startswith(("initial", "bill_seed", "random"))}
        action_origins = {f"A{i}" for i in range(1, 9)}
        base_origins = origins - init_origins - action_origins - {"polish"}
        for field in ("ssgs", "exact", "archive"):
            assert math.isclose(
                t.get(field, 0),
                sum(v for k, v in t.items() if k.startswith(field + ":")),
                abs_tol=1e-7,
                rel_tol=1e-10,
            )
        parts = {
            "initialization": t.get("initialization", 0),
            "polish": t.get("polish", 0),
            "action_construction": sum(v for k, v in t.items() if k.startswith("construction:A")),
            "action_gateway": sum(
                t.get(f"{f}:{a}", 0) for f in ("ssgs", "exact", "archive") for a in action_origins
            ),
            "outer_gateway": sum(
                t.get(f"{f}:{a}", 0) for f in ("ssgs", "exact", "archive") for a in base_origins
            ),
            "budget_selection": t.get("budget", 0),
        }
        parts["unattributed"] = s["production_seconds"] - sum(parts.values())
        assert min(parts.values()) >= -1e-7
        assert math.isclose(sum(parts.values()), float(row["production_seconds"]), abs_tol=1e-6)
        costs.append(
            dict(
                key=row["key"],
                arm=row["arm"],
                n=int(row["n"]),
                base_seed=int(row["base_seed"]),
                variant=int(row["variant"]),
                production_seconds=s["production_seconds"],
                **{k + "_seconds": v for k, v in parts.items()},
                **{k + "_share": v / s["production_seconds"] for k, v in parts.items()},
            )
        )
    assert Counter(r["arm"] for r in rows) == Counter({a: 32 for a in groups})
    for arm, g in groups.items():
        selected = [r for r in rows if r["arm"] == arm]
        for field in ("hv", "igd_plus", "init_seconds", "polish_share", "generations_proxy"):
            assert math.isclose(mean(float(r[field]) for r in selected), g[field], abs_tol=1e-9)
    # This supplement preserves the original 70 tests, not a new test family.
    assert sum(len(v) for v in summary["statistics"].values()) == 70
    fields = list(costs[0])
    with (DOC / "cost_partition_runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(costs)
    averages = {}
    for arm in groups:
        averages[arm] = {}
        for n in (20, 50, 100, 200):
            selected = [r for r in costs if r["arm"] == arm and r["n"] == n]
            assert len(selected) == 8
            averages[arm][str(n)] = {
                k: mean(r[k] for r in selected)
                for k in fields
                if k.endswith(("_seconds", "_share"))
            }
    (DOC / "cost_partition_summary.json").write_text(
        json.dumps(averages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _figures(groups, averages)
    _report(summary, groups, averages)
    content = (DOC / "DIRECTION_REVIEW.md").read_text(encoding="utf-8")
    assert all(content.count(f"`{a}`") >= 2 for a in NOTES)
    assert not any(x in content for x in ("\\(", "\\)", "\\[", "\\]"))
    assert content.count("### ") >= 33
    receipt = dict(
        status="ANALYZED",
        formal_runs=1056,
        independent_parents=16,
        arms=33,
        current_read_summary_and_diagnostic_sha_bound=matched_hashes,
        original_summary_means_reproduced=33 * 5,
        original_frozen_tests_preserved=70,
        nonnegative_cost_partitions=1056,
        cost_partition_reconstruction=1056,
        new_confirmatory_tests=0,
        fresh_solver_runs=0,
        fresh_full_exact=False,
        source_commit=summary["source"],
        source_canonical=summary["canonical"],
        inputs={
            name: _sha(SOURCE / name)
            for name in [
                "summary_statistics.json",
                "RAW_MANIFEST.json",
                "analysis/run_metrics.csv",
                "analysis/action_costs.csv",
            ]
        },
        outputs={
            p.relative_to(DOC).as_posix(): dict(size=p.stat().st_size, sha256=_sha(p))
            for p in sorted(DOC.rglob("*"))
            if p.is_file()
        },
    )
    (OUT / "REVIEW_ACCEPTANCE.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in ("inputs", "outputs")}, ensure_ascii=False
        )
    )


def _figures(groups: dict, averages: dict) -> None:
    """Draw fixed-order scientific figures, avoiding overlapping timer categories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "pdf.fonttype": 42}
    )
    destination = DOC / "figures"
    destination.mkdir(exist_ok=True)
    labels = [
        "初始化整段",
        "Polish整段",
        "动作构造",
        "动作SSGS/exact/Archive",
        "外层SSGS/exact/Archive",
        "预算决策",
        "未细分余量",
    ]
    keys = [
        "initialization",
        "polish",
        "action_construction",
        "action_gateway",
        "outer_gateway",
        "budget_selection",
        "unattributed",
    ]
    colors = ["#5DA5DA", "#F15854", "#B276B2", "#FAA43A", "#60BD68", "#DECF3F", "#B8B8B8"]
    panels = [("FULL_CURRENT_I4", n) for n in (20, 50, 100, 200)] + [
        (a, 200)
        for a in ["BILL_MOEAD", "FULL_STRICT_NO_POLISH_I4", "POOL_Q_B6_L5", "ONLY_A3", "ONLY_A7"]
    ]
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    left = np.zeros(len(panels))
    for key, label, color in zip(keys, labels, colors, strict=True):
        values = np.array([averages[a][str(n)][key + "_share"] * 100 for a, n in panels])
        ax.barh(range(len(panels)), values, left=left, label=label, color=color)
        for i, v in enumerate(values):
            if v >= 9:
                ax.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", fontsize=8)
        left += values
    ax.set_yticks(range(len(panels)), [f"{a} / {n}任务" for a, n in panels], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("平均每run生产墙钟占比（互斥重分类，合计100%）")
    ax.set_title(
        "当前完整版及主要诊断组：耗时去了哪里？\n每行8个输入版本；灰色余量尚不能归因给Q或某个算子"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(destination / ("cost_partition." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    arms = list(groups)
    fig, axes = plt.subplots(1, 2, figsize=(14, 12), sharey=True)
    for ax, field, title in zip(
        axes, ["hv", "igd_plus"], ["HV（越大越好）", "IGD+（越小越好）"], strict=True
    ):
        bars = [
            "#D65F5F"
            if a == "FULL_CURRENT_I4"
            else "#3C918B"
            if a in ("BILL_MOEAD", "FULL_STRICT_NO_POLISH_I4")
            else "#7296B5"
            for a in arms
        ]
        ax.barh(range(33), [groups[a][field] for a in arms], color=bars)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(33), [f"{i:02d} {a}" for i, a in enumerate(arms, 1)], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.21)
    fig.suptitle(
        "全部33组逐项结果：16父等权均值，未绘制新的显著性判断\n开发面板/单seed；随机外部与强初始化内部的差距不是纯搜索框架效应"
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(destination / ("all_33_directions." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _report(summary: dict, groups: dict, averages: dict) -> None:
    """Write each arm's scope, reference, result, cost and actionable interpretation."""
    contrasts = {r["contrast"]: {} for fam in summary["statistics"].values() for r in fam}
    for family, records in summary["statistics"].items():
        for r in records:
            contrasts[r["contrast"]][r["metric"]] = dict(r, family=family)
    lines = [
        "# 33组实验逐项复盘：有效方向、耗时瓶颈与下一步",
        "",
        "2026-10-20。本文件面向研究决策，逐组解释已完成的33组/1056次结果；在既有冻结分析上新增互斥耗时重分类，没有重跑算法、增加显著性检验或修改默认算法。",
        "",
        "## 先看结论",
        "",
        "**当前最可靠的收益来自初始化；最需要收缩的是高频、深轨迹、带昂贵Polish的搜索政策。建议把BILL4＋普通MOEA/D当作研究比较底座，要求任何新增机制在同初始化、等总墙钟下提供额外收益。**",
        "",
        "1. 原混合、BILL4在相同普通MOEA/D下都取得校正后的整程正收益，且两个指标均16/16父正向。",
        "2. 当前完整包显著优于五随机初始化外部；这证明整个包有效，不能证明Q、五步轨迹或Polish分别有效。",
        "3. 严格触发＋无Polish显著优于当前完整包，但尚未确认优于BILL4普通MOEA/D；不能把未显著解释成等价。",
        "4. 200任务当前完整包的Polish占58.57%、初始化约4.50%；降低BILL构造强度没有解决外层约4代的问题。",
        "5. 新的成本拆分显示，200任务强制Q五步仍有57.15%时间未被现有细项计时覆盖，ONLY_A3约77.13%、ONLY_A7约60.70%。不能把它们直接叫作“Q表耗时”；应先细分诊断、状态/标量计算、集合构造、刷新和observer开销。",
        "6. 下一步先分析已经跑完的独立五组320次，再决定是否保留局部搜索；不要立即再开33组或重复五组实验。",
        "",
        "## 1. 数据口径与证据边界",
        "",
        "### Material Passport",
        "",
        _table(
            ["项目", "本次口径"],
            [
                [
                    "验证状态",
                    "ANALYZED；本批复算165项均值、2112个summary/diagnostics SHA绑定及1056项耗时恒等式；不是重新运行1056次算法。",
                ],
                [
                    "正式样本",
                    "16已有开发父；每规模4父；每父两压力版本、一个seed16101。33组各32次，共1056次。",
                ],
                [
                    "任务/拓扑",
                    "20/50/100/200任务；2/3/5地区、6/12/24机器；arrival12父、kv4父，无cheap-capacity/demand。",
                ],
                ["运行源", summary["source"]],
                ["运行canonical", summary["canonical"]],
                [
                    "原验收",
                    "1056/1056原validator；HV/IGD+各1056双实现；四家族70项冻结统计独立复算；最终874110成员exact证据绑定，非本轮fresh全量exact。",
                ],
                [
                    "运算预算",
                    "300/600/1200/2400秒，包含初始化；最终审计/导出单列。48单线程solver，正式7小时22分43秒。",
                ],
                [
                    "指标",
                    "每输入共同经验union归一化，HV参考点(1.1,1.1)，非真实Pareto。HV可大于1，不能跨批直接比较。",
                ],
                [
                    "统计单位",
                    "两个版本先父内平均，再四规模等权。20k分层bootstrap/100k配对sign-flip；70项按主10/组件24/层级22/整包14分别Holm。",
                ],
            ],
        ),
        "",
        "下文HV、IGD+为16父等权均值；“200任务代数”为每100个外层子代一代的代理，可含未完成一代，不等于不同算法执行了相同内部工作。每组200任务计时为8个输入版本均值。未出现新的留出集、多个随机种子稳定性或框架创新确认。",
        "",
        "## 2. 全部33组的结果总表",
        "",
        _table(
            [
                "序号",
                "组别",
                "HV↑",
                "IGD+↓",
                "200任务外层代数",
                "200任务初始化秒",
                "200任务Polish占比",
            ],
            [
                [
                    str(i),
                    f"`{a}`",
                    f"{g['hv']:.6f}",
                    f"{g['igd_plus']:.6f}",
                    f"{g['per_scale']['200']['generations_proxy']:.2f}",
                    f"{g['per_scale']['200']['init_seconds']:.2f}",
                    f"{100 * g['per_scale']['200']['polish_share']:.2f}%",
                ]
                for i, (a, g) in enumerate(groups.items(), 1)
            ],
        ),
        "",
        "![33组的HV与IGD+均值](figures/all_33_directions.png)",
        "",
        "### 四个任务规模的逐组结果",
        "",
        "单元格为HV↑ / IGD+↓，各规模4父、两个版本先均值。全部规模一并列出；本表只描述，不新增规模内检验，不因某规模均值好就宣布局部冠军。不同规模的输入和预算不同，跨列差异不是任务数的单独因果效应。",
        "",
        _table(
            ["组别", "20任务", "50任务", "100任务", "200任务"],
            [
                [f"`{a}`"]
                + [
                    f"{g['per_scale'][str(n)]['hv']:.4f} / {g['per_scale'][str(n)]['igd_plus']:.4f}"
                    for n in (20, 50, 100, 200)
                ]
                for a, g in groups.items()
            ],
        ),
        "",
        "## 3. 逐组解释：这组回答了什么、结果能说明什么",
        "",
        "ONLY_A组全部采用混合初始化、逐子代强制调用、B1/L1、无Polish；它们测的是这种调用政策，不是算子在理想状态下的潜力。POOL组件组使用混合初始化，完整包组使用BILL，不能把两种底座直接当纯单变量差异。",
        "",
    ]
    for i, (arm, g) in enumerate(groups.items(), 1):
        title, definition, interpretation, treatment = NOTES[arm]
        x = g["per_scale"]["200"]
        c = averages[arm]["200"]
        lines += [
            f"### {i:02d}. {title}｜`{arm}`",
            "",
            f"**设置与问题：** {definition}",
            "",
            f"**观测结果：** HV={g['hv']:.6f}，IGD+={g['igd_plus']:.6f}；200任务约{x['generations_proxy']:.2f}代，初始化{x['init_seconds']:.2f}秒。",
            "",
            "**统计参照：** " + _stat_text(arm, contrasts),
            "",
            f"**成本线索：** 200任务初始化整段{100 * c['initialization_share']:.2f}%、Polish整段{100 * c['polish_share']:.2f}%、动作构造{100 * c['action_construction_share']:.2f}%、动作评价三项{100 * c['action_gateway_share']:.2f}%、外层评价三项{100 * c['outer_gateway_share']:.2f}%、未细分余量{100 * c['unattributed_share']:.2f}%；预算决策余项见成本表。这里各项互斥，未细分不是已定位的单模块。",
            "",
            "**如何理解：** " + interpretation,
            "",
            "**下一步处理：** " + treatment,
            "",
        ]
    lines += _cost_section(groups, averages)
    lines += _next_section()
    lines += [
        "",
        "## 6. 不能从本轮得出的结论",
        "",
        "- 不把1056次当1056个独立样本；本轮只有16开发父与一个算法seed，尚未覆盖便宜地区容量和需量费压力。",
        "- 不把同输入同seed当作不同算法消耗了相同的后续随机序列；组件政策改变也会带来随机流后果。",
        "- 不把FULL对五随机外部的优势说成Q或MOEA/D框架优势；不把均值第一或未显著当冠军、等价或无效证明。",
        "- 不把动作attempts当完整exact次数，不把Archive暂时插入数当最终贡献，也不把未接受候选直接称作零价值。",
        "- 不把互斥耗时余量直接称作“Q学习耗时”；不将本轮组件强制调用的负效应解释成所有状态下动作无用。",
        "- 不把关闭Polish与收紧触发的组合效果分别归因；不把代数增加当质量必然提高。",
        "- 不跨批比较HV数值大小，不混旧13组、辅助18组和本33组为一个统计样本。",
        "- 本轮没有新统计假设或新增显著性家族；已按原报告检查辛普森/生态/选择/碰撞点/基率/回归均值/幸存者/多重搜寻/分叉路径/因果归因/反向因果11类问题。成本与吞吐的关联仍不是单一根因证明。",
        "",
        "## 7. 数据、复现与配套文件",
        "",
        "- [原完整分析与全部70项统计](../component33_20261020/RESULTS_ANALYSIS.md)、[原复现说明](../component33_20261020/REPRODUCTION.md)。",
        "- [逐run互斥成本明细](cost_partition_runs.csv)、[全部33组×四规模成本汇总](cost_partition_summary.json)。",
        "- [原33组GitHub归档](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/component33-analysis-20261020-b148c4a)，以已发布v2报告/包为准；本报告不覆盖既有Release资产。",
        "- 构建脚本随本报告提交为`build_review.py`，需要原完整结果的raw/formal、summary_statistics.json、analysis/run_metrics.csv和analysis/action_costs.csv；入口参数见本目录REPRODUCTION.md。",
        "- 本补充报告只新增解释、互斥成本汇总和图表。原运行代码/输入/预算/默认算法保持；本轮没有连接服务器、启动新研究实验或创建定时。仓库门禁中的mini夹具/CLI smoke属于工程检查，不计入正式研究样本。",
        "",
    ]
    (DOC / "DIRECTION_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


def _stat_text(arm: str, contrasts: dict) -> str:
    """Show only the matching, already frozen contrast and adjusted p-values."""
    names = [k for k, v in contrasts.items() if v["hv"]["left"] == arm]
    if arm.startswith("EXT_"):
        names = ["FULL_CURRENT_I4-" + arm]
    result = []
    for name in names:
        hv, igd = contrasts[name]["hv"], contrasts[name]["igd_plus"]
        flag_h = "通过" if hv["p_holm"] < 0.05 else "未通过"
        flag_i = "通过" if igd["p_holm"] < 0.05 else "未通过"
        result.append(
            f"{hv['left']} 对 {hv['right']}（{hv['family']}家族），HV改善{hv['mean']:+.6f}、p_Holm={hv['p_holm']:.6f}（{flag_h}）；IGD+改善{igd['mean']:+.6f}、p_Holm={igd['p_holm']:.6f}（{flag_i}）。改善负号表示左方法更差。"
        )
    return " ".join(result)


def _cost_section(groups: dict, averages: dict) -> list[str]:
    """Explain nesting, report disjoint shares, and limit bottleneck attribution."""
    polish_rows = []
    for n in (20, 50, 100, 200):
        observations = [
            json.loads(p.read_text())
            for p in sorted(
                (SOURCE / "raw/formal/runs").glob(f"*_n{n}_*/FULL_CURRENT_I4/diagnostics.json")
            )
        ]
        assert len(observations) == 8
        calls = mean(d["counts"]["polish_calls"] for d in observations)
        duration = mean(d["seconds"]["polish"] for d in observations)
        polish_rows.append(
            [
                str(n),
                f"{calls:.2f}",
                f"{duration:.2f}",
                f"{duration / calls:.4f}",
                f"{100 * mean(d['seconds'].get('exact:polish', 0) / d['seconds']['polish'] for d in observations):.2f}%",
                f"{mean(d['counts']['polish_exact'] for d in observations):.2f}",
            ]
        )
    action_rows = []
    with (SOURCE / "analysis/action_costs.csv").open(encoding="utf-8", newline="") as stream:
        action_data = list(csv.DictReader(stream))
    for action in range(1, 9):
        selected = [
            r
            for r in action_data
            if r["arm"] == "FULL_CURRENT_I4" and r["n"] == "200" and r["action"] == f"A{action}"
        ]
        assert len(selected) == 8
        action_rows.append(
            [
                f"A{action}",
                *[
                    f"{mean(float(r[k]) for r in selected):.2f}"
                    for k in (
                        "selections",
                        "construction_seconds",
                        "ssgs_seconds",
                        "exact_seconds",
                        "archive_seconds",
                        "exact",
                    )
                ],
            ]
        )
    lines = [
        "",
        "## 4. 耗时分析：先消除重复计时，再判断优先级",
        "",
        "### 4.1 原计时为什么不能直接相加",
        "",
        "原gateway的全局exact/SSGS/Archive计时包含初始化和Polish内的对应调用；Polish整段又包含它自己的exact/Archive。把这些全部堆起来会重复计算。本次从每run的origin标签重分类：初始化整段、Polish整段、动作构造、动作来源SSGS/exact/Archive、其余外层来源SSGS/exact/Archive、预算决策，最后以生产总时间减去前六类得到未细分余量。1056次的所有余量非负，分项逐run重构生产总时间。",
        "",
        "“动作评价三项”不再含初始化或Polish；“外层评价三项”也不含动作调用。初始化整段采用gateway.seconds.initialization，包含其内部事件/测量工作，因而可略大于原summary中构造器的initialization_seconds。余量是剩余生产墙钟，可能含状态/severity、诊断刷新、归一化/标量、身份集合、遗传/替换、observer、Python调度等；没有细项证据前不能强行命名。单线程墙钟也会反映同机负载/等待，不是硬件CPU指令级剖析。",
        "",
        "### 4.2 当前完整算法，规模越大Polish越占主导",
        "",
        _table(
            [
                "任务数",
                "初始化%",
                "Polish%",
                "动作构造%",
                "动作评价三项%",
                "外层评价三项%",
                "预算%",
                "未细分%",
                "外层代数",
            ],
            [
                [
                    str(n),
                    *[
                        f"{100 * averages['FULL_CURRENT_I4'][str(n)][k + '_share']:.2f}"
                        for k in [
                            "initialization",
                            "polish",
                            "action_construction",
                            "action_gateway",
                            "outer_gateway",
                            "budget_selection",
                            "unattributed",
                        ]
                    ],
                    f"{groups['FULL_CURRENT_I4']['per_scale'][str(n)]['generations_proxy']:.2f}",
                ]
                for n in (20, 50, 100, 200)
            ],
        ),
        "",
        "![互斥成本构成](figures/cost_partition.png)",
        "",
        "200任务Polish约1400秒量级，初始化约108秒。把初始化彻底变免费，单纯按当前4.50%占比推算的整体加速上界也只有约1.05倍；这不是可执行优化承诺。I2/I1虽省时间，却失去初始化质量，外层仍约4代，说明它们没有切中当前大头。",
        "",
        "同理，Polish占58.57%并不意味着关闭它后质量无损；保持其他工作完全不变的理想吞吐上界约2.41倍。严格无Polish的外层代数相对当前约42.7倍，显然还包含严格门大幅减少进入局部搜索的作用，不能把42.7倍全部记给删除Polish。",
        "",
        "### 4.3 所有33组的200任务互斥成本",
        "",
        "每行8个输入版本均值，单位百分比；四舍五入可使行和差0.01。完整四规模结果见JSON。",
        "",
        _table(
            ["组别", "初始化", "Polish", "动作构造", "动作评价", "外层评价", "预算", "未细分"],
            [
                [
                    f"`{a}`",
                    *[
                        f"{100 * averages[a]['200'][k + '_share']:.2f}"
                        for k in [
                            "initialization",
                            "polish",
                            "action_construction",
                            "action_gateway",
                            "outer_gateway",
                            "budget_selection",
                            "unattributed",
                        ]
                    ],
                ]
                for a in groups
            ],
        ),
        "",
        "### 4.4 Polish内部与八个动作：具体贵在哪里",
        "",
        "下面是当前完整算法的Polish内部，不能再次叠加到前面的互斥分项中。调用均耗时定义为每run平均Polish总秒数除以平均调用数；不是单次调用耗时分布或P95。",
        "",
        _table(
            [
                "任务数",
                "调用数/run",
                "Polish秒/run",
                "秒/调用（均值之比）",
                "exact占Polish内部",
                "Polish完整exact数/run",
            ],
            polish_rows,
        ),
        "",
        "**200任务的Polish，约96.3%的内部时间在完整exact评价。** 所以不应笼统称为“枚举位置本身太慢”。它每run约134.6次调用、4.55万次完整exact，单次调用总耗时均值之比约10.49秒；大规模时少量调用也会吃掉大部分预算。除稀疏调用外，后续可诊断exact内资源可行性、电价/固定窗口计费及重复数据查询的成本；这只是定位建议，任何增量/缓存实现必须与完整exact交叉验证，不能以近似账单偷换真值。",
        "",
        "当前完整版200任务的八个动作（各8run均值）：",
        "",
        _table(
            ["动作", "选择数", "构造秒", "SSGS秒", "exact秒", "Archive秒", "完整exact数"],
            action_rows,
        ),
        "",
        "A7构造约77.3秒，A3约23.2秒，A8约22.8秒，是当前包中较大的显式动作构造投入；这些数字不包括未单列的支持时间。A7/A8是时序动作，SSGS为0也不意味着廉价。动作选择次数和exact数不同；完整exact数也不能当最终有用新结构数。",
        "",
        "### 4.5 三个真正值得优先定位的计算问题",
        "",
        "**第一，Polish调用频率。** 当前完整在200任务约59%，POLISH_ONLY约99%。应先回答在多少子代上、什么机会下执行才划算，而不是仅追求精修候选变多。普通BILL底座已把93%左右生产时间用于外层SSGS/exact/Archive，能提供一个明确的机会成本参照。",
        "",
        "**第二，模因搜索支持流程的未细分时间。** POOL_Q_B6_L5约57.15%、ONLY_A3约77.13%、ONLY_A7约60.70%的生产时间仍在余量中。A4/A5显式构造非常快也仍很慢，说明“算子propose慢”不是充分解释。源码上可检查trajectory内severity、接受后的refresh_diagnostics、归一化标量和Archive身份集合、事件observer等，但这些是待测位置，不是已经被本轮计时证明的根因。建议只加被动计时，不改RNG/候选/规则；先拿到能闭合到总墙钟的调用树。",
        "",
        "**第三，已定位的候选构造成本。** ONLY_A7构造约35%，ONLY_A3约15%；当前完整版内部A7构造累计也较高。A2/A3的attempts包括大量轻量评分，不能按attempt数当成完整exact次数。若做等价优化，优先重复资源/临界时刻/路径与位置评分查询；每项必须用候选、RNG、目标/可行性和完整工作量benchmark复核。",
        "",
        "### 4.6 P100保持能力是另一个问题，不要和计时混为一谈",
        "",
        f"当前完整初始P100 HV={groups['FULL_CURRENT_I4']['initial_p100_hv']:.6f}、最终P100={groups['FULL_CURRENT_I4']['population_hv']:.6f}，但最终Archive={groups['FULL_CURRENT_I4']['hv']:.6f}。原分析有10/16父最终P100 HV低于初始P100，BILL普通5/16、严格无Polish6/16。被动Archive保住发现过的点，不代表下一代父种群也保住端点和方向覆盖。",
        "",
        "这提示要记录端点丢失、每个权向量所持解、重复基因和方向空洞；还不能据此认定某一个替换上限或归一化公式有bug。Archive与P100的目标本来不同，P100 HV下降也不是每个子问题标量都恶化的证明。",
    ]
    return lines


def _next_section() -> list[str]:
    """Provide a staged recommendation, without silently authorizing new runs."""
    return [
        "",
        "## 5. 下一步怎么做：按决策价值排序",
        "",
        "### 第一步：先分析已经完成的五组320次，零重复计算",
        "",
        "既有独立批次c656731包含16全新父、两变体、两seed19101/19201、四种压力；全I4初始化，五组为普通MOEA/D、原触发+Polish、严格触发+Polish、原触发无Polish、严格触发无Polish。现有总览已记录320计算完成；本轮没有回收/验收或分析该批，不把状态记录当质量结论。",
        "",
        "它恰好补上本33组缺少的完整2×2：分别看触发主效应、Polish主效应和交互，再和BILL普通比较。按其已经冻结的父级/多重校正规则分析，不改seed、不拼接本33组，也不重复开480次。",
        "",
        _table(
            ["得到什么证据", "后续选择（建议）"],
            [
                [
                    "关闭Polish单独改善，而strict追加价值不清楚",
                    "先简化Polish政策，保留原触发参照；不要把组合收益全给strict。",
                ],
                [
                    "strict有收益而Polish只在strict下有用",
                    "研究机会门与少量精修的交互，保留完整2×2。",
                ],
                [
                    "局部搜索最好的组仍没有确认超过BILL普通",
                    "用BILL普通作后续质量底座；暂缓扩大Q状态/深轨迹，转向保持与低成本构造。",
                ],
                [
                    "局部搜索出现稳定追加收益",
                    "再研究成本敏感的稀疏调用；其状态/动作设计建立在真实收益分布上。",
                ],
            ],
        ),
        "",
        "以上是决策路径，不是以未显著宣布等价；如果区间仍宽，应报告不确定性，再决定是否需要新的确认样本。",
        "",
        "### 第二步：做轻量、可闭合的被动成本诊断",
        "",
        "只在隔离诊断版本记录前态→动作→构造→完整评价→Archive/接受→诊断刷新→Q更新→Polish，以及observer时间；记录逐调用分布、P50/P95、重复机会和复用率。增加计时的自身开销要单列，与未插桩参考做固定工作量差分。这个步骤不增加Q状态、不改动作预算或研究语义。",
        "",
        "成功标准：大多数余量能被可解释计时闭合；若使用缓存，再保持每个候选/SSGS结果/exact目标与可行性/Q/RNG一致，并证明同工作量wall减少。不要先把57%余量全部归给Q，然后盲目换强化学习模型。",
        "",
        "### 第三步：在同初始化下比较搜索框架",
        "",
        "普通MOEA/D、NSGA-II至少加入共同BILL4初始化；预算包含构造成本，初始基因/Archive处理与被动评价规则明确。其余三外部可随后补。先回答搜索框架追加效果，保留本轮五随机初始化作为整个方法包的外部条件，不覆盖既有结果。",
        "",
        "### 第四步：只保留一条轻量搜索与一条种群保持候选",
        "",
        "**轻量搜索候选：** 在全新留出上比较普通底座、匹配随机稀疏调用、简单机会/停滞门、少量成本敏感Q；先一步或短轨迹。动作状态从“能否有合法机会、预期改善/成本、方向覆盖缺口”取少量可检验特征，不一次扩成更大的稀疏Q表。",
        "",
        "**种群保持候选：** 记录快端/费用端/中间方向丢失，再独立测试代表保留或更新政策。目标是保住后续可繁殖的好结构与方向覆盖，不以盲目增加ND解数量替代HV、IGD+和覆盖率。",
        "",
        "这两条都尚未实施。不要把触发、Q状态、reward、算子、替换同时修改后只做“原版/新版”比较；否则仍然无法定位收益。",
        "",
        "### 本轮后的取舍清单",
        "",
        _table(
            ["内容", "建议状态", "理由"],
            [
                [
                    "BILL4初始化",
                    "优先保留的研究底座",
                    "同主干计入成本后有清楚整程正收益；本轮16父双指标全部正。",
                ],
                [
                    "普通MOEA/D",
                    "必须保留的强对照",
                    "同BILL时非常有竞争力，复杂模块须跨过这一门槛。",
                ],
                [
                    "严格触发＋无Polish",
                    "优先候选，待独立五组拆分",
                    "优于当前包，但未确认超越BILL普通。",
                ],
                [
                    "逐子代Polish/强制深轨迹",
                    "停止作为默认扩张方向",
                    "机会成本大且同初始化等时间未体现优势。",
                ],
                [
                    "A3/A8问题特定动作",
                    "保留有机会时的候选",
                    "不能把高频调用负结果当所有状态下无用；A8还缺demand留出。",
                ],
                ["随机替换", "低成本备选，不作主贡献", "本轮追加收益未通过校正。"],
                [
                    "Q扩状态、两种动态预算",
                    "暂缓扩大",
                    "现有动作/状态匹配与成本证据不足，动态预算追加收益未确认。",
                ],
                [
                    "A7及模因支持流程等价加速",
                    "优先定位，不提前称已优化",
                    "显式构造与未细分余量均大；必须先保证语义一致。",
                ],
                [
                    "P100端点/方向保持",
                    "独立研究候选",
                    "主种群可能丢已找到的好结构，Archive指标不能遮住问题。",
                ],
            ],
        ),
        "",
        "本报告提出建议，不更改当前默认，不自动启动实验。",
    ]


if __name__ == "__main__":
    _build()
