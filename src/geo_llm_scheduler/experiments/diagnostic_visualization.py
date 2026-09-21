"""Main-only D01/D02 behavioral summaries, heatmaps, and comparison reports."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from geo_llm_scheduler.rl.state import CONDITION_LABELS, PREFERENCE_LABELS, PROGRESS_LABELS, decode

ACTIONS = tuple(f"A{i}" for i in range(1, 9))
LOW_SAMPLE = 10
SPARSE_STATE = 30


@dataclass
class BehaviorSummary:
    """Counts and reward sums pooled before any state/action ratios are computed."""

    visits: np.ndarray
    selections: np.ndarray
    positives: np.ndarray
    reward_sums: np.ndarray
    condition_counts: Counter[str]
    scenario_conditions: dict[str, Counter[str]]
    progress_counts: np.ndarray
    progress_totals: np.ndarray
    total_steps: int

    @property
    def selection_probability(self) -> np.ndarray:
        """Return pooled $P(a|s)=sum N(s,a)/sum N(s)$ without per-run averaging."""
        return self.selections / np.maximum(self.visits[:, None], 1)

    @property
    def success_rate(self) -> np.ndarray:
        """Return pooled positive-reward rate per state/action cell."""
        return np.divide(
            self.positives,
            self.selections,
            out=np.full((42, 8), np.nan, dtype=float),
            where=self.selections > 0,
        )

    @property
    def mean_reward(self) -> np.ndarray:
        """Return reward sum divided by pooled selection count per cell."""
        return np.divide(
            self.reward_sums,
            self.selections,
            out=np.full((42, 8), np.nan, dtype=float),
            where=self.selections > 0,
        )


def read_phase_steps(output_root: str | Path, phase: str) -> list[dict[str, str]]:
    """Read only one phase's complete per-run step tables."""
    rows: list[dict[str, str]] = []
    for path in sorted((Path(output_root) / phase / "runs").glob("*/rl_steps.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def summarize_steps(rows: list[dict[str, str]]) -> BehaviorSummary:
    """Pool raw counts and rewards across runs before deriving any ratios."""
    visits = np.zeros(42, dtype=int)
    selections = np.zeros((42, 8), dtype=int)
    positives = np.zeros((42, 8), dtype=int)
    reward_sums = np.zeros((42, 8), dtype=float)
    condition_counts: Counter[str] = Counter()
    scenario_conditions: dict[str, Counter[str]] = {}
    progress_counts = np.zeros((10, 8), dtype=int)
    progress_totals = np.zeros(10, dtype=int)
    for row in rows:
        state = int(row["state_id"])
        action = int(row["selected_action"][1:]) - 1
        reward = float(row["reward"])
        condition = row["dominant_condition"]
        scenario = row["scenario"]
        visits[state] += 1
        selections[state, action] += 1
        positives[state, action] += int(reward > 0)
        reward_sums[state, action] += reward
        condition_counts[condition] += 1
        scenario_conditions.setdefault(scenario, Counter())[condition] += 1
        bucket = min(9, max(0, int(float(row["normalized_progress"]) * 10)))
        progress_counts[bucket, action] += 1
        progress_totals[bucket] += 1
    return BehaviorSummary(
        visits,
        selections,
        positives,
        reward_sums,
        condition_counts,
        scenario_conditions,
        progress_counts,
        progress_totals,
        len(rows),
    )


def _state_labels() -> list[str]:
    return [
        f"{condition[:5]}-{progress[0]}"
        for condition in CONDITION_LABELS
        for progress in PROGRESS_LABELS
    ]


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _annotate_low_sample(axis: plt.Axes, row: int, column: int) -> None:
    axis.add_patch(
        Rectangle(
            (column - 0.5, row - 0.5),
            1,
            1,
            fill=False,
            hatch="///",
            edgecolor="lightgray",
            linewidth=0.0,
        )
    )


def _panel_heatmap(
    values: np.ndarray,
    samples: np.ndarray,
    title: str,
    directory: Path,
    stem: str,
    color_label: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    percent: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 10), sharey=True, constrained_layout=True)
    finite = values[np.isfinite(values)]
    low = float(np.min(finite)) if finite.size else 0.0
    high = float(np.max(finite)) if finite.size else 1.0
    shown = np.nan_to_num(values, nan=low)
    image = None
    for preference, axis in enumerate(axes):
        image = axis.imshow(
            shown[preference * 14 : (preference + 1) * 14],
            aspect="auto",
            cmap="viridis",
            vmin=low if vmin is None else vmin,
            vmax=high if vmax is None else vmax,
        )
        axis.set_title(PREFERENCE_LABELS[preference])
        axis.set_xticks(range(8), ACTIONS)
        axis.set_yticks(range(14), _state_labels())
        axis.set_xlabel("Action")
        for row in range(14):
            for column in range(8):
                state = preference * 14 + row
                count = int(samples[state, column])
                value = values[state, column]
                rendered = (
                    "—"
                    if not math.isfinite(float(value))
                    else (f"{100 * value:.0f}%" if percent else f"{value:.3g}")
                )
                axis.text(
                    column,
                    row,
                    f"{rendered}\nn={count}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white",
                )
                if count < LOW_SAMPLE:
                    _annotate_low_sample(axis, row, column)
    axes[0].set_ylabel("Condition-progress state (I=Improving, S=Stagnating)")
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.75, label=color_label)
    fig.suptitle(f"{title}\nHatched cells have n < {LOW_SAMPLE}", fontsize=15)
    _save(fig, directory, stem)


def create_dataset_figures(
    summary: BehaviorSummary,
    directory: str | Path,
    label: str,
) -> list[Path]:
    """Create six 300-dpi PNG/PDF diagnostic figure pairs for one dataset."""
    destination = Path(directory)
    fig, axes = plt.subplots(1, 3, figsize=(18, 9), sharey=True, constrained_layout=True)
    for preference, axis in enumerate(axes):
        block = summary.visits[preference * 14 : (preference + 1) * 14]
        colors = ["#b8b8b8" if value < SPARSE_STATE else "#3267a8" for value in block]
        axis.barh(range(14), block, color=colors)
        axis.set_title(PREFERENCE_LABELS[preference])
        axis.set_yticks(range(14), _state_labels())
        axis.invert_yaxis()
        axis.set_xlabel("Visits")
        for index, value in enumerate(block):
            axis.text(value, index, f" {value}", va="center", fontsize=7)
    axes[0].set_ylabel("Condition-progress state")
    fig.suptitle(f"{label}: 42-state visit frequency (gray: visits < {SPARSE_STATE})")
    _save(fig, destination, "state_visit_frequency")

    _panel_heatmap(
        summary.selection_probability,
        summary.selections,
        f"{label}: State × action pooled selection probability",
        destination,
        "state_action_selection",
        "P(a|s)",
        vmin=0,
        vmax=max(0.25, float(np.max(summary.selection_probability))),
        percent=True,
    )
    _panel_heatmap(
        summary.success_rate,
        summary.selections,
        f"{label}: State × action positive-improvement rate",
        destination,
        "state_action_positive_improvement",
        "Positive reward rate",
        vmin=0,
        vmax=1,
        percent=True,
    )
    _panel_heatmap(
        summary.mean_reward,
        summary.selections,
        f"{label}: State × action mean reward",
        destination,
        "state_action_mean_reward",
        "Mean reward",
    )

    scenarios = sorted(summary.scenario_conditions)
    incidence = np.zeros((len(scenarios), len(CONDITION_LABELS)), dtype=float)
    scenario_totals = {
        scenario: sum(summary.scenario_conditions[scenario].values()) for scenario in scenarios
    }
    for row, scenario in enumerate(scenarios):
        for column, condition in enumerate(CONDITION_LABELS):
            incidence[row, column] = summary.scenario_conditions[scenario][condition] / max(
                1, scenario_totals[scenario]
            )
    fig, axis = plt.subplots(figsize=(12, max(3.5, len(scenarios))), constrained_layout=True)
    image = axis.imshow(incidence, cmap="magma", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(7), CONDITION_LABELS, rotation=35, ha="right")
    axis.set_yticks(range(len(scenarios)), scenarios)
    for row, scenario in enumerate(scenarios):
        for column, condition in enumerate(CONDITION_LABELS):
            count = summary.scenario_conditions[scenario][condition]
            axis.text(
                column,
                row,
                f"{100 * incidence[row, column]:.1f}%\nn={count}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )
    fig.colorbar(image, ax=axis, label="Incidence")
    axis.set_title(f"{label}: DominantCondition incidence")
    _save(fig, destination, "dominant_condition_incidence")

    fig, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    centers = (np.arange(10) + 0.5) / 10
    for action in range(8):
        usage = summary.progress_counts[:, action] / np.maximum(summary.progress_totals, 1)
        axis.plot(centers, usage, marker="o", linewidth=1.5, label=ACTIONS[action])
    axis.set_xlabel("Normalized search progress")
    axis.set_ylabel("Selection share within progress bin")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend(ncol=4)
    axis.set_title(f"{label}: A1–A8 usage over search progress")
    _save(fig, destination, "action_usage_over_progress")
    return sorted(destination.glob("*.png")) + sorted(destination.glob("*.pdf"))


def _comparison_heatmap(
    controlled: BehaviorSummary,
    mixed: BehaviorSummary,
    values: tuple[np.ndarray, np.ndarray],
    samples: tuple[np.ndarray, np.ndarray],
    title: str,
    destination: Path,
    stem: str,
    *,
    percent: bool,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 18), sharey=True, constrained_layout=True)
    finite = np.concatenate([value[np.isfinite(value)] for value in values])
    low, high = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 1.0)
    if percent:
        low, high = 0.0, max(0.25, high)
    image = None
    for dataset_index, (summary, matrix, counts, dataset_label) in enumerate(
        zip((controlled, mixed), values, samples, ("D01 controlled", "D02 mixed"))
    ):
        del summary
        shown = np.nan_to_num(matrix, nan=low)
        for preference, axis in enumerate(axes[dataset_index]):
            image = axis.imshow(
                shown[preference * 14 : (preference + 1) * 14],
                cmap="viridis",
                vmin=low,
                vmax=high,
                aspect="auto",
            )
            axis.set_title(f"{dataset_label} — {PREFERENCE_LABELS[preference]}")
            axis.set_xticks(range(8), ACTIONS)
            axis.set_yticks(range(14), _state_labels())
            for row in range(14):
                for column in range(8):
                    state = preference * 14 + row
                    count = int(counts[state, column])
                    value = matrix[state, column]
                    rendered = (
                        "—"
                        if not math.isfinite(float(value))
                        else (f"{100 * value:.0f}%" if percent else f"{value:.2g}")
                    )
                    axis.text(
                        column,
                        row,
                        f"{rendered}\nn={count}",
                        ha="center",
                        va="center",
                        fontsize=5,
                        color="white",
                    )
                    if count < LOW_SAMPLE:
                        _annotate_low_sample(axis, row, column)
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.65)
    fig.suptitle(f"{title}\nCounts remain separate; hatched cells have n < {LOW_SAMPLE}")
    _save(fig, destination, stem)


def create_comparison_figures(
    controlled: BehaviorSummary,
    mixed: BehaviorSummary,
    directory: str | Path,
) -> list[Path]:
    """Create side-by-side figures without pooling D01 and D02 counts."""
    destination = Path(directory)
    fig, axes = plt.subplots(1, 2, figsize=(17, 9), sharey=True, constrained_layout=True)
    for axis, summary, label in zip(axes, (controlled, mixed), ("D01 controlled", "D02 mixed")):
        axis.barh(range(42), summary.visits, color="#3267a8")
        axis.set_title(f"{label}: {int(np.count_nonzero(summary.visits))}/42 states")
        axis.set_xlabel("Visits")
        axis.set_ylabel("State ID")
    fig.suptitle("Controlled vs Mixed: state coverage (counts kept separate)")
    _save(fig, destination, "state_coverage_comparison")

    fig, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    x = np.arange(7)
    width = 0.38
    controlled_rates = [
        controlled.condition_counts[condition] / max(1, controlled.total_steps)
        for condition in CONDITION_LABELS
    ]
    mixed_rates = [
        mixed.condition_counts[condition] / max(1, mixed.total_steps)
        for condition in CONDITION_LABELS
    ]
    axis.bar(x - width / 2, controlled_rates, width, label="D01 controlled")
    axis.bar(x + width / 2, mixed_rates, width, label="D02 mixed")
    axis.set_xticks(x, CONDITION_LABELS, rotation=30, ha="right")
    axis.set_ylabel("Incidence")
    axis.legend()
    axis.set_title("Controlled vs Mixed: DominantCondition distribution")
    _save(fig, destination, "condition_distribution_comparison")

    _comparison_heatmap(
        controlled,
        mixed,
        (controlled.selection_probability, mixed.selection_probability),
        (controlled.selections, mixed.selections),
        "Controlled vs Mixed: State × Action selection",
        destination,
        "state_action_selection_comparison",
        percent=True,
    )
    _comparison_heatmap(
        controlled,
        mixed,
        (controlled.success_rate, mixed.success_rate),
        (controlled.selections, mixed.selections),
        "Controlled vs Mixed: State × Action positive improvement",
        destination,
        "state_action_success_comparison",
        percent=True,
    )
    _comparison_heatmap(
        controlled,
        mixed,
        (controlled.mean_reward, mixed.mean_reward),
        (controlled.selections, mixed.selections),
        "Controlled vs Mixed: State × Action mean reward",
        destination,
        "state_action_reward_comparison",
        percent=False,
    )
    return sorted(destination.glob("*.png")) + sorted(destination.glob("*.pdf"))


def _describe_states(states: list[int]) -> str:
    if not states:
        return "无"
    return ", ".join(f"{state}({'/'.join(decode(state))})" for state in states)


def _correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float | None:
    if int(np.count_nonzero(mask)) < 2:
        return None
    left, right = first[mask], second[mask]
    if float(np.std(left)) == 0 or float(np.std(right)) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def write_reports(
    controlled: BehaviorSummary,
    mixed: BehaviorSummary,
    output_root: str | Path,
    report_root: str | Path,
) -> tuple[Path, Path]:
    """Write preliminary D01 and independent D02 comparison reports."""
    destination = Path(report_root)
    destination.mkdir(parents=True, exist_ok=True)
    d01_path = destination / "qlearning_diagnostic_heatmaps.md"
    d02_path = destination / "qlearning_mixed_validation.md"
    controlled_covered = [state for state in range(42) if controlled.visits[state] > 0]
    mixed_covered = [state for state in range(42) if mixed.visits[state] > 0]
    controlled_sparse = [
        state for state in range(42) if 0 < controlled.visits[state] < SPARSE_STATE
    ]
    mixed_sparse = [state for state in range(42) if 0 < mixed.visits[state] < SPARSE_STATE]
    controlled_unvisited = [state for state in range(42) if controlled.visits[state] == 0]
    mixed_unvisited = [state for state in range(42) if mixed.visits[state] == 0]

    run_summaries = []
    for path in sorted((Path(output_root) / "main" / "runs").glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_summaries.append((float(payload["elapsed"]), int(payload["archive_size"])))
    large_archives = [item for item in run_summaries if item[1] >= 1000]
    archive_max = max((item[1] for item in run_summaries), default=0)
    archive_correlation = _correlation(
        np.array([item[0] for item in run_summaries]),
        np.array([item[1] for item in run_summaries]),
        np.ones(len(run_summaries), dtype=bool),
    )

    d01_actions = [
        ACTIONS[action] for action in range(8) if int(np.sum(controlled.selections[:, action])) > 0
    ]
    target_map = {
        "D1_resource": "Resource",
        "D2_kv": "KV",
        "D3_region": "Region",
        "D4_tou": "TOU",
        "D5_demand": "Demand",
        "D6_compressible": "Compressible",
    }
    activation = []
    for scenario, condition in target_map.items():
        counts = controlled.scenario_conditions.get(scenario, Counter())
        activation.append(
            f"- {scenario} → {condition}: {100 * counts[condition] / max(1, sum(counts.values())):.1f}%"
        )

    d01_report = f"""# Q-learning Diagnostic Heatmaps

状态：**D01 diagnostic / preliminary behavioral visualization**。D01 是 controlled diagnostic scenarios；结果不构成正式论文性能结论。

## 数据与计算

- 完整主诊断：63/63 runs，7 scenarios × 3 instance seeds × 3 algorithm seeds，每条 30 generations。
- Scale、pilot、stress 与 D02 mixed 数据均未进入 D01 aggregation。
- Selection probability 在全部 D01 runs 上先累计计数，再计算：

$$
P(a\\mid s)=\\frac{{\\sum_{{runs}}N_{{run}}(s,a)}}{{\\sum_{{runs}}N_{{run}}(s)}}.
$$

- Heatmap 显示 cell 样本量；$n<{LOW_SAMPLE}$ 使用斜线标记。

## 图表

![D01 state visits](figures/d01/state_visit_frequency.png)

![D01 selection](figures/d01/state_action_selection.png)

![D01 success](figures/d01/state_action_positive_improvement.png)

![D01 reward](figures/d01/state_action_mean_reward.png)

![D01 conditions](figures/d01/dominant_condition_incidence.png)

![D01 progress](figures/d01/action_usage_over_progress.png)

每张图同时保存同名 PDF；PNG 为 300 dpi。

## 诊断摘要

- State coverage：{len(controlled_covered)}/42。
- 稀疏状态：{_describe_states(controlled_sparse)}。
- 未访问状态：{_describe_states(controlled_unvisited)}。
- A1–A8 实际调用：{", ".join(d01_actions)}。
- 总 RL steps：{controlled.total_steps:,}。

{chr(10).join(activation)}

- Archive final size ≥1,000：{len(large_archives)}/63 runs；最大值 {archive_max}。
- Archive size 与 elapsed correlation：{archive_correlation if archive_correlation is not None else "不可计算"}。

Archive 第一版不裁剪，接近或超过 1,000 members 是 dominance check、内存与序列化风险信号；本报告只记录，不自动修改算法。

## Controlled vs Mixed validation

D02 是独立 mixed random validation instances，不是 D01 的第八个场景。两套 count 从未合并。对比结果见 [qlearning_mixed_validation.md](qlearning_mixed_validation.md)。
"""
    d01_path.write_text(d01_report, encoding="utf-8")

    common_states = [
        state
        for state in range(42)
        if controlled.visits[state] >= SPARSE_STATE and mixed.visits[state] >= SPARSE_STATE
    ]
    top_matches = sum(
        int(np.argmax(controlled.selections[state]) == np.argmax(mixed.selections[state]))
        for state in common_states
    )
    selection_mask = np.repeat(
        np.array(
            [
                controlled.visits[state] >= SPARSE_STATE and mixed.visits[state] >= SPARSE_STATE
                for state in range(42)
            ]
        )[:, None],
        8,
        axis=1,
    )
    success_mask = (controlled.selections >= LOW_SAMPLE) & (mixed.selections >= LOW_SAMPLE)
    selection_corr = _correlation(
        controlled.selection_probability, mixed.selection_probability, selection_mask
    )
    success_corr = _correlation(controlled.success_rate, mixed.success_rate, success_mask)
    reward_corr = _correlation(controlled.mean_reward, mixed.mean_reward, success_mask)
    replicated = bool(
        common_states
        and top_matches / len(common_states) >= 0.5
        and selection_corr is not None
        and selection_corr > 0.3
    )
    mixed_actions = [
        ACTIONS[action] for action in range(8) if int(np.sum(mixed.selections[:, action])) > 0
    ]
    mixed_conditions = ", ".join(
        f"{condition}={100 * mixed.condition_counts[condition] / max(1, mixed.total_steps):.1f}%"
        for condition in CONDITION_LABELS
    )
    conclusion = (
        "D01 中的 state-dependent action preference 得到 mixed-instance 行为复核。"
        if replicated
        else "D01 与 D02 的 action preference 存在明显差异，当前不宣称 mixed-instance 行为复核。"
    )
    d02_report = f"""# Q-learning Mixed-instance Validation

状态：**D02 mixed random validation / preliminary behavioral visualization**。D02 使用冻结后不再校准的普通综合随机实例；不构成正式论文性能结论。

## 数据边界

- D02：5 instance seeds × 3 algorithm seeds = 15/15 runs，每条 30 generations。
- D01 是 controlled diagnostic scenarios；D02 是 mixed random validation instances。
- 两套数据只并列比较，不合并 count，不生成总热力图。
- D02 selection probability 同样先累计 count，再计算 $P(a|s)$；低样本 cell 显示 $n$ 并加斜线。

## D02 图表

![D02 state visits](figures/d02/state_visit_frequency.png)

![D02 selection](figures/d02/state_action_selection.png)

![D02 success](figures/d02/state_action_positive_improvement.png)

![D02 reward](figures/d02/state_action_mean_reward.png)

![D02 conditions](figures/d02/dominant_condition_incidence.png)

![D02 progress](figures/d02/action_usage_over_progress.png)

## Controlled vs Mixed 图表

![Coverage comparison](figures/comparison/state_coverage_comparison.png)

![Condition comparison](figures/comparison/condition_distribution_comparison.png)

![Selection comparison](figures/comparison/state_action_selection_comparison.png)

![Success comparison](figures/comparison/state_action_success_comparison.png)

![Reward comparison](figures/comparison/state_action_reward_comparison.png)

## 结果摘要

- D02 state coverage：{len(mixed_covered)}/42。
- D02 稀疏状态：{_describe_states(mixed_sparse)}。
- D02 未访问状态：{_describe_states(mixed_unvisited)}。
- D02 A1–A8 实际调用：{", ".join(mixed_actions)}。
- D02 DominantCondition：{mixed_conditions}。
- D01/D02 共同充分采样状态：{len(common_states)}；top-selected action 一致：{top_matches}/{len(common_states)}。
- Selection profile correlation：{selection_corr if selection_corr is not None else "不可计算"}。
- Positive-improvement correlation：{success_corr if success_corr is not None else "不可计算"}。
- Mean reward correlation：{reward_corr if reward_corr is not None else "不可计算"}。

{conclusion}

该结论只描述本轮合成诊断行为。无论一致或不一致，都不据此修改 mixed generator、42-state、Q-learning、A1–A8、Trigger 或 MacroSearch。
"""
    d02_path.write_text(d02_report, encoding="utf-8")
    return d01_path, d02_path
