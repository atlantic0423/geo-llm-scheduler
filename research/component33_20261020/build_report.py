"""Rebuild fixed scientific panels and the complete prespecified statistical appendix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _table(headers: list[str], rows: list[list]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, r)) + " |" for r in rows),
        ]
    )


def _figure(fig: plt.Figure, folder: Path, name: str) -> None:
    fig.savefig(folder / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(
        folder / (name + ".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _build(results: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    figures = results / "figures"
    figures.mkdir(exist_ok=True)
    analysis = json.loads((results / "analysis/analysis.json").read_text())
    arms = {a["arm"]: a for a in analysis["arms"]}
    names, sizes = list(arms), [20, 50, 100, 200]
    costs = list(csv.DictReader((results / "analysis/action_costs.csv").open()))
    parents = list(csv.DictReader((results / "analysis/parent_metrics.csv").open()))
    anytime = list(csv.DictReader((results / "analysis/anytime_metrics.csv").open()))
    fronts = json.loads((results / "analysis/normalized_fronts.json").read_text())
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.18})
    fig, ax = plt.subplots(figsize=(8.5, 12))
    data = np.array([[arms[a]["per_scale"][str(n)]["hv"] for n in sizes] for a in names])
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1.21, aspect="auto")
    ax.set(
        xticks=range(4),
        xticklabels=sizes,
        yticks=range(33),
        yticklabels=names,
        xlabel="Jobs (4 independent parents per scale)",
        title="All 33 arms: normalized final Archive HV",
    )
    ax.grid(False)
    for i in range(33):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{data[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if data[i, j] > 0.75 else "black",
            )
    fig.colorbar(im, ax=ax, label="HV; reference (1.1,1.1)", fraction=0.04)
    _figure(fig, figures, "01_all_arms_hv")
    fig, axes = plt.subplots(2, 2, figsize=(19, 13))
    for ax, (family, tests) in zip(axes.flat, analysis["statistics"].items()):
        tests = [t for t in tests if t["metric"] == "hv"]
        for i, t in enumerate(tests):
            ax.plot(t["ci95"], [i, i], color="0.4")
            ax.scatter(t["mean"], i, color="#2171b5" if t["p_holm"] < 0.05 else "#969696")
        ax.axvline(0, color="black", lw=0.7)
        labels = [
            t["contrast"]
            .replace("FULL_CURRENT_I4", "CURRENT")
            .replace("FULL_STRICT_NO_POLISH_I4", "STRICT_NO_POLISH")
            for t in tests
        ]
        ax.set(
            yticks=range(len(tests)),
            yticklabels=labels,
            xlabel="Paired HV improvement; marginal 95% bootstrap CI",
            title=f"{family}: blue = family Holm p < .05",
        )
        ax.invert_yaxis()
    fig.tight_layout()
    _figure(fig, figures, "02_frozen_hv_contrasts")
    selected = [
        "EXT_MOEAD",
        "MIXED_MOEAD",
        "BILL_MOEAD",
        "FULL_CURRENT_I4",
        "FULL_STRICT_NO_POLISH_I4",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for arm in selected:
        v = arms[arm]["per_scale"]
        label = arm.replace("FULL_", "")
        fields = [("hv", 1), ("generations_proxy", 1), ("polish_share", 100), ("init_share", 100)]
        for ax, (field, factor) in zip(axes.flat, fields):
            ax.plot(sizes, [factor * v[str(n)][field] for n in sizes], "o-", label=label)
    for ax, title in zip(
        axes.flat,
        [
            "Final Archive HV",
            "Offspring / P100 (log)",
            "Polish inclusive share (%)",
            "Charged initialization share (%)",
        ],
    ):
        ax.set(title=title, xlabel="Jobs", xticks=sizes)
    axes[0, 1].set_yscale("log")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    _figure(fig, figures, "03_quality_cost_and_search")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    targets = ["initialization", 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    for ax, n in zip(axes.flat, sizes):
        for arm in selected:
            xs, ys = [], []
            budget = {20: 300, 50: 600, 100: 1200, 200: 2400}[n]
            for target in targets:
                subset = [
                    s
                    for s in anytime
                    if s["arm"] == arm
                    and int(s["n"]) == n
                    and (
                        s["target"] == "initialization"
                        if target == "initialization"
                        else s["target"] != "initialization"
                        and abs(float(s["target"]) / budget - target) < 1e-8
                    )
                ]
                assert len(subset) == 8
                xs.append(np.mean([float(s["fraction"]) for s in subset]))
                ys.append(np.mean([float(s["hv"]) for s in subset]))
            ax.plot(xs, ys, "o-", markersize=3, label=arm.replace("FULL_", ""))
        ax.set(
            title=f"{n} jobs",
            xlabel="Actual elapsed / requested production budget",
            ylabel="Archive HV",
        )
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    _figure(fig, figures, "04_anytime")
    for n in sizes:
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
        keys = sorted(k for k in fronts if k.startswith(f"r0_n{n}_"))
        for ax, key in zip(axes.flat, keys):
            for arm in selected + ["EXT_NSGAII", "EXT_EPS_IBEA"]:
                ps = np.asarray(fronts[key][arm])
                ax.plot(ps[:, 0], ps[:, 1], ".-", ms=2, lw=0.8, label=arm.replace("FULL_", ""))
            ax.set(
                title=key.split(f"n{n}_")[1].replace("_s16101", ""),
                xlim=(0, 1.1),
                ylim=(0, 1.1),
                xlabel="Normalized Flow",
                ylabel="Normalized Bill",
            )
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
        fig.suptitle(f"{n} jobs: all eight input variants; HV reference window only")
        fig.tight_layout(rect=(0, 0.10, 1, 0.95))
        _figure(fig, figures, f"05_fronts_{n}")
    summary = {k: v for k, v in analysis.items() if k != "references"}
    (destination / "summary_statistics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    text = [
        "## 全部33组汇总",
        "",
        "每父两版本均值后四规模等权。代数代理为offspring/100。",
        "",
        _table(
            ["组", "HV↑", "IGD+↓", "IGD↓", "初始Archive HV", "后续HV增量", "外层代数代理"],
            [
                [
                    a,
                    *[
                        f"{v[k]:.6f}"
                        for k in [
                            "hv",
                            "igd_plus",
                            "igd",
                            "initial_hv",
                            "hv_gain_after_init",
                            "generations_proxy",
                        ]
                    ],
                ]
                for a, v in arms.items()
            ],
        ),
        "",
        "## 当前完整版的动作成本",
        "",
        "每run均值。Archive插入是当时插入次数，不能当作最终独立贡献或标量接受。分项时间不能与包含它们的Polish或全局exact重复相加。",
        "",
        _table(
            [
                "动作",
                "选择次数",
                "构造秒",
                "SSGS秒",
                "exact秒",
                "Archive秒",
                "exact候选",
                "当时Archive插入",
            ],
            [
                [
                    f"A{i}",
                    *[
                        f"{np.mean([float(c[k]) for c in costs if c['arm'] == 'FULL_CURRENT_I4' and c['action'] == f'A{i}']):.3f}"
                        for k in [
                            "selections",
                            "construction_seconds",
                            "ssgs_seconds",
                            "exact_seconds",
                            "archive_seconds",
                            "exact",
                            "insertions",
                        ]
                    ],
                ]
                for i in range(1, 9)
            ],
        ),
        "",
        "## 完整70项冻结检验",
        "",
        "正号表示左方法改善：HV为左减右，IGD+为右减左；每行16独立父。CI是边际95%区间，p_Holm按本小节完整家族校正。",
    ]
    for family, tests in analysis["statistics"].items():
        text += [
            "",
            f"### {family}（{len(tests)}项）",
            "",
            _table(
                ["左−右", "指标", "改善均值", "95%CI", "p原始", "p_Holm", "正/负父"],
                [
                    [
                        t["contrast"],
                        t["metric"],
                        f"{t['mean']:+.6f}",
                        f"[{t['ci95'][0]:+.6f}, {t['ci95'][1]:+.6f}]",
                        f"{t['p_signflip']:.6f}",
                        f"{t['p_holm']:.6f}",
                        f"{t['positive']}/{t['negative']}",
                    ]
                    for t in tests
                ],
            ),
        ]
    text += [
        "",
        "## 全部16父的HV对照",
        "",
        "描述性展示，不按结果挑选子组。表内两个版本先父内平均。",
        "",
        _table(
            [
                "父seed",
                "任务",
                "拓扑/压力",
                "随机MOEA/D",
                "混合MOEA/D",
                "BILL普通",
                "当前完整",
                "严格无Polish",
            ],
            [
                [
                    p,
                    next(r["n"] for r in parents if r["base_seed"] == p),
                    next(
                        f"R{r['regions']}M{r['machines']}/{r['pressure']}"
                        for r in parents
                        if r["base_seed"] == p
                    ),
                    *[
                        f"{float(next(r['hv'] for r in parents if r['base_seed'] == p and r['arm'] == a)):.6f}"
                        for a in selected
                    ],
                ]
                for p in sorted({r["base_seed"] for r in parents})
            ],
        ),
        "",
    ]
    appendix = "\n".join(text)
    (destination / "STATISTICAL_APPENDIX.md").write_text(appendix, encoding="utf-8")
    report = destination / "RESULTS_ANALYSIS.md"
    if report.exists():
        prefix = report.read_text(encoding="utf-8").split("<!-- GENERATED APPENDIX -->")[0]
        report.write_text(prefix + "<!-- GENERATED APPENDIX -->\n\n" + appendix, encoding="utf-8")
    print(json.dumps(dict(figures=len(list(figures.glob("*.png"))))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    _build(args.results, args.destination)
