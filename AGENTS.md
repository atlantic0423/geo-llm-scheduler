# 项目开发约定

每次回答前恢复当前项目进展；主要开发任务开始前重读 Notion 项目总览、当前完整算法框架、数学模型 v8、Coding Contract v1、相关当前决策与任务。当前算法框架、数学模型、Coding Contract 与用户确认的补充约定共同约束实现。

- GitHub `atlantic0423/geo-llm-scheduler` 的 `main` 是源码、测试、配置、文档与 CI 的代码唯一权威；Notion 是研究规格、决策、任务、实验与同步记录的权威。
- 每个有意义的批次按“读取规格 → 修改 → 本地完整验收 → commit → push → GitHub Actions 通过 → 将 commit SHA/PR/CI/测试同步 Notion 09”闭环执行。
- 首次 baseline 后，常规开发使用 `codex/<topic>` 或 `research/<topic>` 分支和 PR；`main` 始终保持可运行。PR 说明包含目的、规格映射、文件、测试结果与 Notion 关联。

- Genotype.MS 是 assignment 唯一权威；Schedule 只拥有 timing、事件缓存和 diagnostics。
- Structural change 必须使 phenotype/evaluation 失效并 SSGS rebuild。
- Timing-only 不调用 SSGS；冻结未选 operation；accepted A7/A8/Polish 后全量刷新 resource-wait diagnostics，绝不改变 timing。
- A7 与 Compressible 共用有限 activity/resource critical-time generator；不增加 instance planning upper bound；原 batch horizon 限制只属于 A8。
- 所有 complete+feasible+exact candidate 独立尝试 Archive 更新，包括未被标量接受者及 Polish 全部候选。
- MacroSearch 同源、无候选链；完整 exact evaluations 才占预算；相同候选集合顺序不改变 best-of-B。
- Exact evaluator 是真值。优化必须保留 reference、等价性回归与前后 benchmark。
- Public API 必须有 type hints、docstring 和测试；随机过程接收显式 RNG，禁止全局随机状态。
- 每模块执行 lint/type、unit、golden、边界/反例和回归；失败修正后重跑。未通过不得标记完成。
- 每个有意义的代码、测试、配置、文档或目录变化，在同批次结束前同步 Notion 09，记录 GitHub branch、commit SHA、PR/CI、规格映射与真实验收结果。完整 ZIP 只用于 milestone、release、论文投稿/返修或可复现包等关键 checkpoint；日常版本以 Git commit 为准。
- 未定义且改变研究语义的问题单列待确认；继续无依赖模块。研究层变化同步当前框架和研究决策，不把 experimental 写成已验证。
- Markdown 数学只使用美元符号行内或独立双美元展示格式，不使用旧式反斜杠公式分隔符。
- 本地质量门 scripts/validate_all.py 同时执行格式、lint、mypy、全回归和覆盖率阈值；结果写 outputs/coverage.json。
- 研究语义待确认项及关闭记录保存在 docs/reports/open_questions.md；不得用测试通过代替规格确认。
- 每条 Q-learning 轨迹最后一步 terminal=True，未来价值为零；其他步正常 bootstrap。共享 Q-table 不随轨迹重置；Polish 在 episode 外，不参与学习更新。
- 禁止提交 secrets、凭据、私密数据、机器绝对路径、大型输出或原始 Notion 导出；不得以 Notion ZIP 代替 Git 历史，不得降低门禁换取 CI 通过，也不得把 smoke 通过表述成研究效果成立。

同步入口：https://app.notion.com/p/3e28878b80198186a626f3cf77b4a8dd
