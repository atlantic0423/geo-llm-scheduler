"""Publication export of all input fronts plus predeclared quality and cost views."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ARMS = ("BILL_MOEAD", "MAIN_Q", "LLM_FLASH")
COLORS = {"BILL_MOEAD": "#777777", "MAIN_Q": "#2573B8", "LLM_FLASH": "#CE4E38"}
LABELS = {"BILL_MOEAD": "BILL+MOEA/D", "MAIN_Q": "主算法 Q", "LLM_FLASH": "LLM Flash"}
plt.rcParams.update(
    {
        "font.family": "Microsoft YaHei",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def table(root: Path, name: str) -> list[dict]:
    """Load one complete exported scalar table."""
    with (root / (name + ".csv")).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, path: Path) -> None:
    """Save matching PNG/PDF artifacts with fixed canvas and no creation timestamp."""
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def plot(root: Path) -> dict:
    """Draw all32 inputs, shared fast windows and descriptive mechanism summaries."""
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fronts = json.loads((root / "normalized_fronts.json").read_text())
    refs = json.loads((root / "references.json").read_text())
    runs = table(root, "run_metrics")
    summaries = json.loads((root / "arm_summary.json").read_text())
    parents = table(root, "parent_metrics")
    actions = table(root, "action_metrics")
    focuses = table(root, "focus_metrics")
    policies = table(root, "policies")
    count = 0
    for refid, reference in refs.items():
        items = [r for r in runs if fronts[r["key"] + "|" + r["arm"]]["reference"] == refid]
        fast = min(p[0] for p in reference["front"])
        for zoom in (False, True):
            fig, ax = plt.subplots(figsize=(7.6, 5))
            for arm in ARMS:
                for index, row in enumerate(
                    sorted((r for r in items if r["arm"] == arm), key=lambda r: r["seed"])
                ):
                    values = (
                        np.asarray(fronts[row["key"] + "|" + arm]["archive"]) * reference["span"]
                        + reference["ideal"]
                    )
                    if zoom:
                        values = values[values[:, 0] <= 1.25 * fast]
                    if len(values):
                        ax.plot(
                            values[:, 0] / 1000,
                            values[:, 1],
                            color=COLORS[arm],
                            linestyle="-" if index == 0 else "--",
                            marker="o" if index == 0 else "x",
                            markersize=2.8,
                            linewidth=0.9,
                            alpha=0.8,
                        )
            first = items[0]
            ax.set(
                title=f"{first['n']}任务｜{first['regions']}地区/{first['machines']}机器｜{first['pressure']}｜v{first['variant']}"
                + ("｜共同快端窗口" if zoom else ""),
                xlabel="总 Flow（千秒）",
                ylabel="总电费（CNY）",
            )
            handles = [Line2D([0], [0], color=COLORS[a], label=LABELS[a]) for a in ARMS]
            handles.extend(
                [
                    Line2D([0], [0], color="black", linestyle="-", label="seed 16401"),
                    Line2D([0], [0], color="black", linestyle="--", label="seed 26401"),
                ]
            )
            ax.legend(handles=handles, fontsize=8, ncol=2)
            ax.grid(alpha=0.18)
            fig.text(
                0.1,
                0.01,
                "点为实际可行调度；连线只辅助观察，不表示可行插值。"
                + (" 仅显示Flow≤共同最快值1.25倍的实际点。" if zoom else ""),
                fontsize=8,
            )
            save(fig, out / (refid + ("_fast" if zoom else "_full")))
            count += 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, label in zip(axes, ("hv", "igd_plus"), ("HV（越高越好）", "IGD+（越低越好）")):
        for index, arm in enumerate(ARMS):
            values = [r for r in summaries if r["n"] and r["arm"] == arm]
            ax.plot(
                [r["n"] for r in values],
                [r[metric] for r in values],
                marker="o",
                color=COLORS[arm],
                label=LABELS[arm],
            )
        ax.set(xlabel="任务数量", ylabel=label, xticks=[20, 50, 100, 200])
        ax.grid(alpha=0.18)
    axes[0].legend()
    fig.suptitle("三组最终前沿｜同一算例共同经验参考；按原始父算例汇总")
    fig.text(0.1, 0.01, "不同规模采用不同共同参考，只在相同规模内比较三组。", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out / "quality_by_size")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric in zip(axes, ("hv", "igd_plus")):
        for arm, offset in [("MAIN_Q", -0.07), ("BILL_MOEAD", 0.07)]:
            values = []
            for parent in sorted({r["base_seed"] for r in parents}):
                llm = next(
                    r for r in parents if r["base_seed"] == parent and r["arm"] == "LLM_FLASH"
                )
                base = next(r for r in parents if r["base_seed"] == parent and r["arm"] == arm)
                values.append(
                    (
                        int(llm["n"]),
                        (float(llm[metric]) - float(base[metric])) * (1 if metric == "hv" else -1),
                    )
                )
            xs = [(20, 50, 100, 200).index(n) + offset for n, _ in values]
            ax.scatter(
                xs,
                [v for _, v in values],
                label="LLM − " + LABELS[arm],
                alpha=0.8,
                color=COLORS[arm],
                s=34,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(
            xticks=range(4),
            xticklabels=[20, 50, 100, 200],
            xlabel="任务数量",
            ylabel=metric + " 改善（正值有利于LLM）",
        )
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=8)
    fig.suptitle("16个父算例的配对差异｜每点先平均两变体与两seed")
    fig.tight_layout()
    save(fig, out / "paired_parent_effects")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    components = [
        ("initialization_share", "初始化", "#9B9FA3"),
        ("outer_evaluation_share", "外层评价/Archive", "#4C78A8"),
        ("ls_measured_share", "LS组件计时", "#F28E2B"),
        ("polish_share", "Polish", "#59A14F"),
        ("llm_wait_share", "LLM等待", "#B279A2"),
        ("llm_features_share", "LLM特征", "#E9B44C"),
        ("other_share", "其他/支持开销", "#D6D6D6"),
    ]
    bars = [
        next(r for r in summaries if r["n"] == n and r["arm"] == a)
        for n in (20, 50, 100, 200)
        for a in ARMS
    ]
    bottom = np.zeros(len(bars))
    for key, label, color in components:
        heights = np.array([100 * r[key] for r in bars])
        ax.bar(range(len(bars)), heights, bottom=bottom, color=color, label=label, width=0.8)
        bottom += heights
    ax.set(
        xticks=range(12),
        xticklabels=[
            f"{r['n']}\n" + {"BILL_MOEAD": "BILL", "MAIN_Q": "Q", "LLM_FLASH": "LLM"}[r["arm"]]
            for r in bars
        ],
        ylabel="生产墙钟占比（%）",
        ylim=(0, 102),
        title="互不重复的已计时组成｜LS整步wall另行报告，不叠加",
    )
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    save(fig, out / "time_breakdown")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, value, title in zip(axes, ("calls", "action_s"), ("算子调用份额", "算子整步耗时份额")):
        matrix = []
        for arm in ("MAIN_Q", "LLM_FLASH"):
            totals = np.array(
                [
                    sum(
                        float(r[value])
                        for r in actions
                        if r["arm"] == arm and int(r["action"]) == a
                    )
                    for a in range(1, 9)
                ]
            )
            matrix.append(totals / totals.sum() * 100)
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(max(x) for x in matrix))
        for i in range(2):
            for j in range(8):
                ax.text(
                    j, i, f"{matrix[i][j]:.1f}", ha="center", va="center", fontsize=8, color="black"
                )
        ax.set(
            xticks=range(8),
            xticklabels=[f"A{i}" for i in range(1, 9)],
            yticks=range(2),
            yticklabels=["主算法Q", "LLM Flash"],
            title=title + "（%）",
        )
        fig.colorbar(image, ax=ax, shrink=0.55)
    fig.tight_layout()
    save(fig, out / "operator_exposure")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, group in zip(axes, ("Flow", "Balanced", "Electricity")):
        matrix = np.array(
            [
                [
                    np.mean(
                        [
                            float(r[f"A{a}"])
                            for r in policies
                            if r["group"] == group and int(r["update_index"]) == u
                        ]
                    )
                    for a in range(1, 9)
                ]
                for u in (0, 1, 2)
            ]
        )
        ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=30)
        ax.set(
            xticks=range(8),
            xticklabels=range(1, 9),
            yticks=range(3),
            yticklabels=["初次", "1/3", "2/3"],
            title=group,
            xlabel="动作编号",
        )
    fig.suptitle("有效模型响应的平均原始动作权重｜非实际曝光加权；概率另含10%探索")
    fig.tight_layout()
    save(fig, out / "policy_weight_updates")
    fig, ax = plt.subplots(figsize=(9, 4))
    names = sorted({r["requested_focus"] for r in focuses if int(r["action"]) in (3, 6)})
    x = np.arange(len(names))
    for offset, action in [(-0.18, 3), (0.18, 6)]:
        counts = np.array(
            [
                sum(
                    int(r["calls"])
                    for r in focuses
                    if int(r["action"]) == action and r["requested_focus"] == name
                )
                for name in names
            ]
        )
        ax.bar(x + offset, 100 * counts / max(1, counts.sum()), width=0.36, label=f"A{action}")
    ax.set(
        xticks=x,
        xticklabels=names,
        ylabel="实际A3/A6调用份额（%）",
        title="重点策略实际进入A3/A6的分布",
    )
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend()
    save(fig, out / "applied_focus")
    anytime = table(root, "anytime_metrics")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, n in zip(axes.flat, (20, 50, 100, 200)):
        for arm in ARMS:
            points = []
            targets = sorted(
                {
                    float(r["target"])
                    for r in anytime
                    if r["target"] != "initialization" and int(r["n"]) == n
                }
            )
            for target in targets:
                values = [
                    r
                    for r in anytime
                    if int(r["n"]) == n
                    and r["arm"] == arm
                    and r["target"] != "initialization"
                    and float(r["target"]) == target
                ]
                points.append(
                    (
                        np.mean([float(r["fraction"]) for r in values]),
                        np.mean([float(r["hv"]) for r in values]),
                    )
                )
            ax.plot(*zip(*points), color=COLORS[arm], label=LABELS[arm], marker="o", markersize=3)
        ax.set(title=f"{n}任务", xlabel="实际生产时间 / 预算", ylabel="共同参考HV")
        ax.grid(alpha=0.18)
    axes[0, 0].legend()
    fig.suptitle("Anytime描述性均值｜记录发生在安全边界，横轴采用实际时间")
    fig.tight_layout()
    save(fig, out / "anytime")
    return dict(frontier_pairs=count, overview_pairs=7, total_pairs=count + 7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.root)))
