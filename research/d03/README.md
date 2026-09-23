# D03 v2 正式诊断结果

本目录公开 D03 v2 的**正式**聚合数据、图表和核验材料。逐图解释与下一步实验计划见 [D03 逐图分析报告](../../docs/reports/d03_results_figure_analysis.md)。图中的 D03-I（定向覆盖组）与 D03-II（综合动态组）各自计算，不能合并成一个热力图。`data/reference/condition_all_suites.csv` 仅供 S01 的 D01/D02 背景对照使用。

| 数据组 | 正式运行 | 每条代数 | 局部动作记录 | 状态覆盖 |
|---|---:|---:|---:|---:|
| D03-I | 42/42 | 50 | 257,010 | 42/42 |
| D03-II | 15/15 | 50 | 95,050 | 18/42 |

正式运行采用冻结源码提交 [`2526ad1`](https://github.com/atlantic0423/geo-llm-scheduler/commit/2526ad1afc425b4d6ff847055eebd4785744acc8)。仓库当前 `main` 在 D03 结束后更新过 A4/A5；本包**不以当前算子行为重新解释或重算历史轨迹**。预校准 51 条、每条 15 代，仅用于预先选择实例参数，与 57 条正式运行分开。参数、种子、冻结选择和完成状态见 [frozen_selection.json](data/frozen_selection.json)、[pipeline_state.json](data/pipeline_state.json)、[D03-I 实例清单](data/manifests/type_i_evaluation.csv)与 [D03-II 实例清单](data/manifests/type_ii_evaluation.csv)。

`figures/type_i/`、`figures/type_ii/` 是正式聚合图；`figures/state_action_cn/` 是三面板中文 42 状态图；`figures/analysis/` 是基于同一批正式数据的二次诊断图。每张图均有 PNG 与 PDF；源表位于 `data/aggregate/` 和 `data/analysis/`。原始流水线产生的 `condition_transition_matrix` 图没有进入本发布包，因为它的相邻记录可能跨子问题；同一轨迹的可核验切换改用 `S04_within_trajectory`。`D01/D02` 原始数据、D03 预校准记录和额外的 matched-operator probe 均不参与 D03 正式聚合。

完整正式运行的 trace、summary、qtable、diagnostic、config、population 等原始产物与实例 JSON 以 [GitHub Release `d03-v2-results-2026-09-24`](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/d03-v2-results-2026-09-24) 附件保存。Release 分为 D03-I、D03-II、预校准三个 ZIP；其大小和 SHA-256 见 [附件清单](data/release_assets.json)。ZIP 内的 JSON 绝对本机路径被改为仓库相对路径以便公开与跨机读取；原始文件逐项 SHA-256 列在 ZIP 内的 `original_source_sha256.tsv`，本地原件未被改写。图表和数据文件的 SHA-256 见 [checksums.sha256](checksums.sha256)。

验收命令 `python research/d03/verify_publication.py` 核对 57/57、两组动作记录总数、正式实例清单、41 对 PNG/PDF、报告链接、MathJax 分隔符和 134 个数据/图文件校验和。此批次完整工程门禁：`scripts/validate_all.py` **153 passed**，整体覆盖率 **92.58%**、核心 **97.08%**；确定性回放、wheel build 和 CLI smoke 均通过。发布后以 GitHub Actions 结果为最终 CI 记录。

本目录是诊断数据归档，不是 Q-learning 优于随机/固定算子策略的性能证明。所有因果解释与后续实验需求以 [逐图报告](../../docs/reports/d03_results_figure_analysis.md) 的证据边界为准。
