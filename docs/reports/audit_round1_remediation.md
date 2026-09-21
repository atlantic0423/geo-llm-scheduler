# GitHub 源码审计 Round 1 修复报告

日期：2026-09-21。

本批次处理审计确认的实现与实验记录问题，不改变 exact evaluator、SSGS、A1–A8 move 语义、Q-learning terminal 语义或 External Archive 的 phenotype retention。

## 修复内容

- Coverage-aware 先按 Flow/费用容差生成确定性的 objective-space representatives，再计算 warm-up 和 reference-direction density；相同目标的不同 phenotype 不重复增加 coverage。
- A4/A5 改为 lazy bounded round-robin。A5 对合法 paired-position rank 使用有界无放回采样，不再物化完整位置对集合；日志分开记录 raw moves、qualifying pool 和 selected proposals。
- summary 增加 git_commit、git_branch、git_dirty；Git 不可用时明确写 null。source/config/instance hash 保留。
- Archive 的历史成功插入次数改名为 archive_insertions，增加 archive_peak_size、archive_final_by_origin 和每批 archive_net_retained，避免把临时插入解释为最终贡献。
- 归一化 Tchebycheff 比较改用独立无量纲 scalar tolerance。
- MacroSearchResult 使用 evaluated 和 feasible_scored，score 与 feasible candidate 显式成对。
- soft wall-clock stop 增加 termination_reason、time_budget_seconds 和 time_overshoot_seconds；停止边界仍为子问题边界。

## 验收证据

- `scripts/validate_all.py`：ruff lint/format 通过，mypy 51 个源文件通过，117 tests 通过。
- 整体覆盖率 96.84%，核心覆盖率 97.08%。
- 新增反例覆盖相同 objective 多 phenotype、Archive 插入顺序敏感性、A4/A5 大位置空间有界构造、独立 scalar tolerance、MacroSearch score 对齐、Git 不可用和 soft-stop overshoot。

## 保留的研究边界

- External Archive 第一版仍不裁剪；新增 peak/final 指标用于大规模 profiling 后决定是否引入 size control。
- 仓库可见性和 LICENSE 属于项目负责人的发布决策，本批次不修改。
- wall-clock 仍为可审计的 soft stop，不在 RL trajectory 中间强制中断。
