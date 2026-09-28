"""Reproduce paired endpoint, convergence and state/action diagnostics from frozen runs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from geo_llm_scheduler.domain.models import Genotype, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.validation import file_hash, validate_result
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.io.loaders import load_instance

LABELS = {
    "V1": "普通MOEA/D",
    "V2": "NSGA-II",
    "V3": "完整算法·固定6",
    "V4": "屏蔽·固定6",
    "V5": "严重度预算",
    "V6": "屏蔽＋严重度预算",
    "M0": "无屏蔽",
    "M1": "屏蔽A6",
    "M2": "屏蔽A4/A5",
    "M3": "屏蔽A4/A5/A6",
    "B1": "预算边界1/2",
    "B2": "预算边界0.75/1.5",
    "B3": "预算边界0.5/1.25",
}
CONDITIONS = ("常态", "资源", "KV迁移", "区域", "分时电价", "需量", "可压缩")
FIELDS = (
    "calls",
    "opportunities",
    "positive",
    "no_candidate",
    "accepted",
    "reward_sum",
    "budget_sum",
    "effective_sum",
    "seconds",
    "explore",
    "budget3",
    "budget6",
    "budget10",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a UTF-8 table with deterministic field order."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_stats(values: list[float], seed: int = 20260928) -> dict:
    """Summarize ten independent instance differences, stratifying bootstrap 5+5."""
    import numpy as np

    x = np.asarray(values, dtype=float)
    if len(x) != 10:
        raise ValueError("Validation contrasts require exactly ten instances")
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 5, size=(20000, 5))
    b = rng.integers(5, 10, size=(20000, 5))
    boot = np.concatenate((x[a], x[b]), axis=1).mean(axis=1)
    observed = abs(float(x.mean()))
    null = [abs(float(np.mean(x * signs))) for signs in itertools.product((-1, 1), repeat=10)]
    return {
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_raw": sum(v >= observed - 1e-14 for v in null) / len(null),
        "wins": int((x > 1e-12).sum()),
        "ties": int((abs(x) <= 1e-12).sum()),
        "losses": int((x < -1e-12).sum()),
        "mixed_mean": float(x[:5].mean()),
        "price_difference_mean": float(x[5:].mean()),
    }


def holm(values: list[float]) -> list[float]:
    """Apply step-down Holm correction across the declared family of contrasts."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    prior = 0.0
    for rank, index in enumerate(order):
        prior = max(prior, min(1.0, (len(values) - rank) * values[index]))
        result[index] = prior
    return result


def scan_run(args: tuple[str, str]) -> dict:
    """Verify a run and stream its trace into auditable bounded-size summaries."""
    root, name = Path(args[0]), args[1]
    spec = JobSpec(**json.loads((root / "specs" / name).read_text(encoding="utf-8")))
    folder = root / "runs" / spec.key
    valid, reason = validate_result(folder, spec)
    if not valid:
        raise ValueError(f"{spec.key}: {reason}")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    problem = load_instance(root / "instances" / Path(spec.instance_path).name)
    archive = json.loads((folder / "archive.json").read_text(encoding="utf-8"))
    sample_indices = sorted({0, len(archive) // 2, len(archive) - 1})
    for index in sample_indices:
        saved = archive[index]
        genotype = Genotype(tuple(saved["genotype"]["ms"]), tuple(saved["genotype"]["os"]))
        result = evaluate(problem, genotype, Schedule(tuple(saved["schedule"]["starts"])))
        original = saved["evaluation"]
        if not result.feasible or not math.isclose(result.flow, original["flow"], abs_tol=1e-7):
            raise ValueError("Archived schedule exact validation failed")
        if not math.isclose(result.bill, original["tou"] + sum(original["demand"]), abs_tol=1e-7):
            raise ValueError("Archived bill exact recomputation failed")
    stats = [[[0.0] * len(FIELDS) for _ in range(8)] for _ in range(42)]
    progress = [[[0.0] * 4 for _ in range(8)] for _ in range(8)]
    checkpoints = []
    last = None
    at60 = None
    seen = defaultdict(set)
    for line in (folder / "trace.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        generation = row["generation"] + 1
        subproblem = row["subproblem"]
        if subproblem in seen[generation]:
            raise ValueError(f"Duplicate generation/subproblem: {spec.key}")
        seen[generation].add(subproblem)
        if row["elapsed"] <= 60:
            at60 = row
        for step in row["steps"]:
            s, a = step["state"], step["action"] - 1
            available = step["available_actions"]
            if a + 1 not in available:
                raise ValueError("Selected a masked action")
            for action in available:
                stats[s][action - 1][1] += 1
            vals = (
                1,
                0,
                int(step["reward"] > 0),
                int(step["feasible"] == 0),
                int(step["accepted"]),
                step["reward"],
                step["budget"],
                step["effective"],
                step["seconds"],
                int(step["explore"]),
                int(step["budget"] == 3),
                int(step["budget"] == 6),
                int(step["budget"] == 10),
            )
            for index, value in enumerate(vals):
                stats[s][a][index] += value
            bucket = min(7, (generation - 1) // 25)
            for index, value in enumerate(
                (1, step["reward"] > 0, step["reward"], step["effective"])
            ):
                progress[bucket][a][index] += value
        if generation in (25, 50, 100, 150, 200) and len(seen[generation]) == 100:
            checkpoints.append(
                {
                    "generation": generation,
                    "elapsed": row["elapsed"],
                    "points": row["archive_objectives"],
                }
            )
        last = row
    if set(seen) != set(range(1, 201)) or any(v != set(range(100)) for v in seen.values()):
        raise ValueError("Incomplete 200-generation trace")
    if at60 is None or last is None or last["elapsed"] < 60:
        raise ValueError("Cannot form a common 60-second observed checkpoint")
    checkpoints.append(
        {"generation": -1, "elapsed": at60["elapsed"], "points": at60["archive_objectives"]}
    )
    qtable = json.loads((folder / "qtable.json").read_text(encoding="utf-8"))
    if qtable["selections"] != [[cell[0] for cell in state] for state in stats]:
        raise ValueError("Q selection counts do not match trace")
    points = []
    with (folder / "objectives.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            points.append((float(row["flow_seconds"]), float(row["bill_cny"])))
    return {
        "key": spec.key,
        "scope": spec.stage,
        "arm": spec.arm,
        "instance": spec.instance_seed,
        "algorithm_seed": spec.algorithm_seed,
        "instance_hash": spec.instance_hash,
        "initial_hash": spec.initial_hash,
        "source_commit": spec.source_commit,
        "summary": summary,
        "stats": stats,
        "progress": progress,
        "checkpoints": checkpoints,
        "points": points,
        "trace_rows": sum(len(v) for v in seen.values()),
        "exact_samples": len(sample_indices),
    }


def nondominated(points: list) -> list[tuple[float, float]]:
    """Return the minimizing two-dimensional empirical front."""
    result = []
    best = math.inf
    for x, y in sorted(set(map(tuple, points))):
        if y < best:
            result.append((x, y))
            best = y
    return result


def evaluate_group(group: list[dict], convergence: bool = False) -> tuple[list[dict], dict]:
    """Use one common scale per instance and a pooled empirical terminal front."""
    import numpy as np

    all_points = [p for run in group for p in run["points"]]
    if convergence:
        all_points += [p for run in group for c in run["checkpoints"] for p in c["points"]]
    array = np.asarray(all_points)
    ideal = array.min(axis=0)
    scale = np.maximum(array.max(axis=0) - ideal, 1e-12)

    def normalize(points: list) -> list[tuple[float, float]]:
        return [tuple(p) for p in ((np.asarray(points) - ideal) / scale).tolist()]

    front = nondominated([p for run in group for p in normalize(run["points"])])
    rows = []
    for run in group:
        checkpoints = run["checkpoints"] if convergence else [{"points": run["points"]}]
        for checkpoint in checkpoints:
            points = normalize(checkpoint["points"])
            row = {k: run[k] for k in ("key", "scope", "arm", "instance", "algorithm_seed")}
            row.update(hv=hypervolume(points, (1.1, 1.1)), igd_plus=igd_plus(points, front))
            if convergence:
                row.update(generation=checkpoint["generation"], elapsed=checkpoint["elapsed"])
            else:
                summary = run["summary"]
                row.update(
                    elapsed=summary["elapsed"],
                    exact=summary["counts"].get("exact", 0),
                    archive_size=summary["archive_size"],
                    archive_peak=summary["archive_peak_size"],
                    archive_seconds=summary["timings"].get("archive", 0),
                    trigger_rate=summary["trigger_rate"],
                    trigger_success_rate=summary["trigger_success_rate"],
                )
            rows.append(row)
    return rows, {
        "ideal": ideal.tolist(),
        "scale": scale.tolist(),
        "reference": [1.1, 1.1],
        "empirical_front_size": len(front),
    }


def contrasts(rows: list[dict]) -> list[dict]:
    """Compute planned paired contrasts at the instance, not seed, level."""
    import numpy as np

    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["instance"])].append(row)
    comparisons = [
        ("V3-V1", {"V3": 1, "V1": -1}),
        ("V3-V2", {"V3": 1, "V2": -1}),
        ("V4-V3", {"V4": 1, "V3": -1}),
        ("V5-V3", {"V5": 1, "V3": -1}),
        ("V6-V3", {"V6": 1, "V3": -1}),
        ("V6-V5", {"V6": 1, "V5": -1}),
        ("V6-V4", {"V6": 1, "V4": -1}),
        ("交互", {"V6": 1, "V5": -1, "V4": -1, "V3": 1}),
    ]
    results = []
    for name, coefficients in comparisons:
        for metric in ("hv", "igd_plus"):
            direction = 1 if metric == "hv" else -1
            values = [
                direction
                * sum(
                    c * np.mean([r[metric] for r in groups[(arm, i)]])
                    for arm, c in coefficients.items()
                )
                for i in range(111, 121)
            ]
            results.append({"contrast": name, "metric": metric, **paired_stats(values)})
    for row, p in zip(results, holm([r["p_raw"] for r in results])):
        row["p_holm"] = p
    return results


def save_figure(fig: Figure, folder: Path, name: str) -> None:
    """Save matching publication-resolution PNG and vector PDF artifacts."""
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(runs: list[dict], endpoints: list[dict], curves: list[dict], output: Path) -> None:
    """Draw terminal comparisons, convergence, costs and sample-aware heatmaps."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    folder = output / "figures"
    arms = [f"V{i}" for i in range(1, 7)]
    for scope, chosen in (
        ("development", ["M0", "M1", "M2", "M3", "B1", "B2", "B3"]),
        ("validation", arms),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        for ax, metric, label in zip(
            axes, ("hv", "igd_plus"), ("超体积（越大越好）", "IGD+（越小越好）")
        ):
            for i in sorted({r["instance"] for r in endpoints if r["scope"] == scope}):
                vals = [
                    np.mean([r[metric] for r in endpoints if r["instance"] == i and r["arm"] == a])
                    for a in chosen
                ]
                ax.plot(range(len(chosen)), vals, "o-", alpha=0.55, label=f"实例{i}")
            ax.set_xticks(range(len(chosen)), [f"{a}\n{LABELS[a]}" for a in chosen], fontsize=8)
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
        axes[-1].legend(fontsize=7, ncol=2)
        fig.suptitle(
            "开发集筛选（每点为3种子均值）"
            if scope == "development"
            else "独立验证（每点为5种子均值）"
        )
        save_figure(fig, folder, f"{scope}_quality")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for a in arms:
        data = [r for r in curves if r["arm"] == a and r["generation"] > 0]
        means = [
            (
                g,
                np.mean([r["hv"] for r in data if r["generation"] == g]),
                np.mean([r["elapsed"] for r in data if r["generation"] == g]),
            )
            for g in (25, 50, 100, 150, 200)
        ]
        axes[0].plot([v[0] for v in means], [v[1] for v in means], "o-", label=LABELS[a])
        axes[1].plot([v[2] for v in means], [v[1] for v in means], "o-", label=LABELS[a])
    for ax in axes:
        ax.set_ylabel("收敛图专用共同尺度下的平均超体积")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_xlabel("完成代数")
    axes[1].set_xlabel("对应检查点的平均实际秒数（非等时间检验）")
    save_figure(fig, folder, "convergence")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, key, label in zip(
        axes, ("elapsed", "exact", "archive_seconds"), ("耗时/秒", "精确评价次数", "档案维护/秒")
    ):
        vals = [np.mean([r[key] for r in endpoints if r["arm"] == a]) for a in arms]
        ax.bar(arms, vals)
        ax.set_ylabel(label)
        for x, y in enumerate(vals):
            ax.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.15)
    save_figure(fig, folder, "costs")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, instances, title in zip(
        axes, (range(111, 116), range(116, 121)), ("综合随机实例", "区域电价差异实例")
    ):
        for arm in arms:
            vals = [
                np.mean([r["hv"] for r in endpoints if r["instance"] == i and r["arm"] == arm])
                for i in instances
            ]
            ax.plot(list(instances), vals, "o-", label=LABELS[arm])
        ax.set(title=title, xlabel="实例种子", ylabel="200代超体积")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    save_figure(fig, folder, "validation_subgroups")
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for ax, instance in zip(axes.flat, range(111, 121)):
        for arm in arms:
            front = nondominated(
                [
                    p
                    for r in runs
                    if r["instance"] == instance and r["arm"] == arm
                    for p in r["points"]
                ]
            )
            ax.plot([p[0] for p in front], [p[1] for p in front], label=arm)
        ax.set(title=f"实例{instance}", xlabel="总流动时间/秒", ylabel="电费/元")
        ax.legend(fontsize=6, ncol=3)
    fig.suptitle("每组5条独立运行合并后的经验前沿（不是单次运行性能）")
    save_figure(fig, folder, "pareto_envelopes")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    all_stat = {a: np.sum([r["stats"] for r in runs if r["arm"] == a], axis=0) for a in arms[2:]}
    for arm, stat in all_stat.items():
        cell = stat[9]
        count = cell[:, 0]
        for ax, values, label in zip(
            axes,
            (
                count / max(1, count.sum()),
                np.divide(cell[:, 2], count, out=np.zeros(8), where=count > 0),
                np.divide(cell[:, 3], count, out=np.zeros(8), where=count > 0),
            ),
            ("选择比例", "正改善率", "无可行候选率"),
        ):
            if arm in ("V4", "V6"):
                values[3:6] = np.nan
            ax.plot(range(1, 9), values, "o-", label=f"{arm}，状态9调用{int(count.sum())}")
            ax.set_xticks(range(1, 9), [f"A{i}" for i in range(1, 9)])
            ax.set_ylabel(label)
            ax.legend(fontsize=7)
    fig.suptitle("被屏蔽的目标状态9；V4/V6的A4–A6为结构性不可选，曲线留空")
    save_figure(fig, folder, "state9_mask_effect")
    global_reward_max = max(
        1.0,
        max(
            float(
                np.divide(s[:, :, 5], s[:, :, 0], out=np.zeros((42, 8)), where=s[:, :, 0] > 0).max()
            )
            for s in all_stat.values()
        ),
    )
    for arm in ("V3", "V4", "V5", "V6"):
        data = [r for r in runs if r["arm"] == arm]
        stat = np.sum([r["stats"] for r in data], axis=0)
        calls = stat[:, :, 0]
        state_n = calls.sum(axis=1)
        metrics = {
            "selection": (
                np.divide(
                    calls, state_n[:, None], out=np.zeros_like(calls), where=state_n[:, None] > 0
                ),
                "状态内选择概率",
                0,
                1,
            ),
            "success": (
                np.divide(stat[:, :, 2], calls, out=np.zeros_like(calls), where=calls > 0),
                "正改善率",
                0,
                1,
            ),
            "reward": (
                np.divide(stat[:, :, 5], calls, out=np.zeros_like(calls), where=calls > 0),
                "平均奖励",
                -1,
                1,
            ),
            "no_candidate": (
                np.divide(stat[:, :, 3], calls, out=np.zeros_like(calls), where=calls > 0),
                "无可行候选率",
                0,
                1,
            ),
        }
        for name, (values, title, lower, upper) in metrics.items():
            if name == "reward":
                upper = global_reward_max
            fig, axes = plt.subplots(1, 3, figsize=(15, 8), constrained_layout=True)
            for p, (ax, pref) in enumerate(zip(axes, ("流动时间偏好", "均衡偏好", "电费偏好"))):
                sl = slice(p * 14, (p + 1) * 14)
                im = ax.imshow(
                    np.ma.masked_where(calls[sl] < 30, values[sl]),
                    vmin=lower,
                    vmax=upper,
                    cmap="RdYlBu_r" if name == "reward" else "viridis",
                    aspect="auto",
                )
                ax.set_xticks(range(8), [f"A{i}" for i in range(1, 9)])
                ax.set_yticks(
                    range(14),
                    [
                        f"{s}: {CONDITIONS[(s % 14) // 2]}·{'停滞' if s % 2 else '改善'}"
                        for s in range(p * 14, (p + 1) * 14)
                    ],
                    fontsize=8,
                )
                ax.set_title(pref)
                for s in range(14):
                    for a in range(8):
                        k = s + p * 14
                        excluded = k == 9 and arm in ("V4", "V6") and a in (3, 4, 5)
                        n = int(calls[k, a])
                        label = (
                            "屏蔽"
                            if excluded
                            else f"n={n}"
                            if n < 30
                            else f"{values[k, a]:.2f}\n{n}"
                        )
                        ax.text(
                            a,
                            s,
                            label,
                            ha="center",
                            va="center",
                            fontsize=5.5,
                            color="black" if n < 30 else "white",
                        )
            fig.colorbar(im, ax=axes, shrink=0.7)
            fig.suptitle(
                f"{arm} {LABELS[arm]}：{title}\n灰格为调用少于30；数字下行为调用数；屏蔽格为结构性不可选"
            )
            save_figure(fig, folder, f"{arm}_{name}")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
        axes[0].bar(range(42), state_n)
        axes[0].set(
            xlabel="状态编号",
            ylabel="调用次数",
            title=f"{arm}：状态覆盖 {int((state_n > 0).sum())}/42",
        )
        prog = np.sum([r["progress"] for r in data], axis=0)[:, :, 0]
        prop = np.divide(
            prog,
            prog.sum(axis=1)[:, None],
            out=np.zeros_like(prog),
            where=prog.sum(axis=1)[:, None] > 0,
        )
        for a in range(8):
            axes[1].plot(np.arange(1, 9) * 25, prop[:, a], "o-", label=f"A{a + 1}")
        axes[1].legend(ncol=4, fontsize=8)
        axes[1].set(xlabel="每25代区间终点", ylabel="动作选择比例", title="算子使用随搜索进程变化")
        save_figure(fig, folder, f"{arm}_coverage_progress")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, chosen in zip(axes, (["M0", "B1", "B2", "B3"], ["V3", "V4", "V5", "V6"])):
        bottom = np.zeros(len(chosen))
        for index, budget in enumerate((3, 6, 10)):
            counts = [
                np.sum(
                    [np.asarray(r["stats"])[:, :, 10 + index].sum() for r in runs if r["arm"] == a]
                )
                for a in chosen
            ]
            total = [
                np.sum([np.asarray(r["stats"])[:, :, 0].sum() for r in runs if r["arm"] == a])
                for a in chosen
            ]
            ratios = np.divide(counts, total)
            ax.bar(chosen, ratios, bottom=bottom, label=f"预算{budget}")
            bottom += ratios
        ax.set_ylabel("按调用数累计的请求预算比例")
        ax.legend()
    save_figure(fig, folder, "budget_activation")


def main() -> None:
    """Analyze the complete frozen matrix without modifying recovered raw files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()
    root, output = args.root, args.output
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    spec_files = sorted((root / "specs").glob("*.json"))
    if len(spec_files) != 363:
        raise ValueError("Expected 363 frozen specifications")
    todo = [(str(root), p.name) for p in spec_files if not (cache / p.name).exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, run in enumerate(pool.map(scan_run, todo), 1):
            (cache / f"{run['key']}.json").write_text(json.dumps(run), encoding="utf-8")
            if i % 20 == 0:
                print(f"Validated and aggregated {i}/{len(todo)}", flush=True)
    if args.scan_only:
        print("All 363 runs validated; analysis cache complete", flush=True)
        return
    import numpy as np

    runs = [json.loads((cache / p.name).read_text(encoding="utf-8")) for p in spec_files]
    paired = defaultdict(set)
    for run in runs:
        paired[(run["scope"], run["instance"], run["algorithm_seed"])].add(
            (run["instance_hash"], run["initial_hash"])
        )
    if any(len(v) != 1 for v in paired.values()):
        raise ValueError("Paired initial conditions differ")
    endpoints, curves, references = [], [], {}
    for scope in ("development", "validation"):
        for instance in sorted({r["instance"] for r in runs if r["scope"] == scope}):
            group = [r for r in runs if r["scope"] == scope and r["instance"] == instance]
            rows, ref = evaluate_group(group)
            endpoints.extend(rows)
            references[f"{scope}_{instance}_terminal"] = ref
            rows, ref = evaluate_group(group, True)
            curves.extend(rows)
            references[f"{scope}_{instance}_convergence"] = ref
    # Independently recomputed endpoints must reproduce the pipeline's metrics.
    for scope in ("development", "validation"):
        old = {
            r["key"]: r for r in csv.DictReader((root / "analysis" / f"{scope}_metrics.csv").open())
        }
        for row in endpoints:
            if row["scope"] == scope:
                for metric in ("hv", "igd_plus"):
                    if not math.isclose(row[metric], float(old[row["key"]][metric]), abs_tol=1e-10):
                        raise ValueError(f"Metric mismatch {row['key']} {metric}")
    data = output / "data"
    write_csv(data / "run_metrics.csv", endpoints)
    write_csv(data / "convergence.csv", curves)
    validation = [r for r in endpoints if r["scope"] == "validation"]
    tests = contrasts(validation)
    write_csv(data / "paired_contrasts.csv", tests)
    write_csv(
        data / "time60_contrasts.csv",
        contrasts([r for r in curves if r["scope"] == "validation" and r["generation"] == -1]),
    )
    state_rows, action_rows, overview = [], [], []
    for arm in sorted({r["arm"] for r in runs}):
        group = [r for r in runs if r["arm"] == arm]
        stats = np.sum([r["stats"] for r in group], axis=0)
        for s in range(42):
            state_n = stats[s, :, 0].sum()
            for a in range(8):
                row = {
                    "arm": arm,
                    "state": s,
                    "action": a + 1,
                    "state_calls": state_n,
                    **dict(zip(FIELDS, stats[s, a].tolist())),
                }
                row["selection_probability"] = stats[s, a, 0] / state_n if state_n else ""
                row["selection_given_available"] = (
                    stats[s, a, 0] / stats[s, a, 1] if stats[s, a, 1] else ""
                )
                state_rows.append(row)
        for a in range(8):
            action_rows.append(
                {"arm": arm, "action": a + 1, **dict(zip(FIELDS, stats[:, a].sum(axis=0).tolist()))}
            )
        rows = [r for r in endpoints if r["arm"] == arm]
        total = stats[:, :, 0].sum()
        overview.append(
            {
                "arm": arm,
                "runs": len(group),
                **{
                    k: float(np.mean([r[k] for r in rows]))
                    for k in (
                        "hv",
                        "igd_plus",
                        "elapsed",
                        "exact",
                        "archive_size",
                        "archive_peak",
                        "archive_seconds",
                        "trigger_rate",
                    )
                },
                "state_coverage": int((stats[:, :, 0].sum(axis=1) > 0).sum()),
                "calls": int(total),
                "state9_calls": int(stats[9, :, 0].sum()),
                "positive_rate": stats[:, :, 2].sum() / total if total else 0,
                "no_candidate_rate": stats[:, :, 3].sum() / total if total else 0,
                "mean_requested_budget": stats[:, :, 6].sum() / total if total else 0,
                "mean_effective_budget": stats[:, :, 7].sum() / total if total else 0,
            }
        )
    write_csv(data / "state_action.csv", state_rows)
    write_csv(data / "action.csv", action_rows)
    write_csv(data / "arm_summary.csv", overview)
    (data / "reference_definitions.json").write_text(
        json.dumps(references, indent=2), encoding="utf-8"
    )
    audit = {
        "status": "ANALYZED",
        "runs": len(runs),
        "development": 63,
        "validation": 300,
        "trace_rows": sum(r["trace_rows"] for r in runs),
        "generations_per_run": 200,
        "paired_cells": len(paired),
        "paired_hashes_match": True,
        "qtable_selections_match_trace": True,
        "archived_schedules_exact_rechecked": sum(r["exact_samples"] for r in runs),
        "terminal_metrics_reproduced": True,
        "source_commits": sorted({r["source_commit"] for r in runs}),
        "spec_hashes": {p.name: file_hash(p) for p in spec_files},
        "notes": "Artifact and independent aggregation verification; no stochastic rerun.",
    }
    (data / "validation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    make_figures(runs, endpoints, curves, output)
    print(
        json.dumps(
            {
                "runs": len(runs),
                "trace_rows": audit["trace_rows"],
                "figures": len(list((output / "figures").glob("*.png"))),
            }
        )
    )


if __name__ == "__main__":
    main()
