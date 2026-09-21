# D01 规模校准报告

状态：Stage S 已完成。基线 commit 为 `5fa4d971a476a992b6d93446d72146b4bfaa2502`，运行日期为 2026-09-21。算法配置固定为 population 100、3 generations、Q-learning、preference trigger、固定预算 6 和 5 个 RL steps；仅 Jobs 数量变化。

## 实测结果

| Jobs | 终止 | Wall-clock | RL steps | Trigger hits | Exact | SSGS | Archive final/peak | Trace |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 50 | generations | 54.42 s | 360 | 72 | 3,100 | 1,900 | 9 / 21 | 495,327 B |
| 100 | generations | 207.06 s | 400 | 80 | 6,110 | 2,254 | 165 / 165 | 1,294,770 B |
| 200 | time limit | 182.99 s | 95 | 19 | 3,521 | 722 | 7 / 7 | 119,612 B |

200 Jobs 使用 180 秒软停止，只完成 82 个 generation-subproblem 事件，overshoot 为 2.99 秒，因此不能视为完整 3 代结果。100 Jobs 完整运行但超过任务书建议的 180 秒阈值，且 Archive 已增长到 165。50 Jobs 完整运行、产生 360 个 Q-learning transition，并留有足够余量进行 63 条重复运行。

## 决策

Round 1 选择 50 Jobs。该选择以可重复运行和完整固定代数回放为优先，并不表示 50 Jobs 是算法可扩展性的上限。100/200 Jobs 的结果保留为 profiling 证据，不与正式主诊断矩阵混合。

50 Jobs 的主要已计时模块为 exact evaluator 10.91 秒、SSGS 6.59 秒、A7 construction 5.61 秒；RightShiftPolish 为 2.82 秒。模块计时存在嵌套，不能简单相加为 wall-clock。

原始数据位于 gitignored 的 `outputs/diagnostics/scale/`，直接聚合表为 `outputs/diagnostics/aggregate/scale_calibration.csv` 和 `runtime_breakdown.csv`。
