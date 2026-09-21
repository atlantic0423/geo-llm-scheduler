# Geo-distributed LLM Prefill–Decode Scheduling

静态离线、多区域 LLM 两阶段调度研究实现。每个 Job 先 Prefill 后 Decode，同 Region，可跨实例传输 KV；实例允许满足 Compute/VRAM 累计容量的并发。双目标为总 Flow Time 与连续 TOU + 固定 900 秒 Demand 费用。

已提供 exact evaluator、事件驱动 SSGS、十类初始化、plain MOEA/D、A1–A8、RightShiftPolish、共享 42×8 Q-table、五种预算策略及实验入口。示例配置只验证运行，不能用 smoke 结果推断研究有效性。当前验收状态见 [验收矩阵](docs/acceptance_matrix.md)。

## 安装与运行

Python ≥3.11，在仓库根目录执行：

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python scripts/validate_all.py
.venv/Scripts/python -m geo_llm_scheduler.cli.validate_instance --instance examples/smoke.json
.venv/Scripts/python -m geo_llm_scheduler.cli.run --config configs/smoke.yaml --seed 1
.venv/Scripts/python -m geo_llm_scheduler.cli.run --config configs/experiments/plain_moead.yaml --seed 1
.venv/Scripts/python -m geo_llm_scheduler.cli.run --config configs/experiments/full_method.yaml --seed 1
.venv/Scripts/python scripts/reproduce.py
.venv/Scripts/python scripts/run_experiment.py --configs configs/experiments/plain_moead.yaml configs/experiments/full_method.yaml --seeds 1 2 3
```

Linux/macOS 用 python3 创建环境，把 .venv/Scripts/python 替换为 .venv/bin/python。requirements-lock.txt 记录本次 Python 3.13 环境；跨版本优先按 pyproject.toml 解析依赖。

结果默认写入 outputs：config、summary、population、Archive、Q-table、逐步 trace 和 objectives.csv。summary 同时记录 Git commit/branch/dirty、内容hash、soft-stop overshoot、Archive insertions/peak/final来源；重复输出路径会覆盖同次运行，正式实验用 --output 指定独立目录。HV/IGD+ 需提供统一参考点/前沿，默认留空。

## 研究文档

- [问题模型与输入](docs/problem_model.md)、[实现契约](docs/coding_contract.md)
- [架构](docs/architecture.md)、[算法](docs/algorithm.md)、[算子](docs/operator_reference.md)
- [配置](docs/configuration.md)、[实验协议](docs/experiment_protocol.md)、[复现](docs/reproducibility.md)
- [规格映射](docs/traceability_matrix.md)、[开发指南](docs/developer_guide.md)
- [规格理解报告](docs/specifications/understanding.md)、[severity 公式](docs/specifications/state_severity.md)
- [D01 规模校准](docs/reports/diagnostic_scale_calibration.md)、[D01 行为诊断](docs/reports/diagnostic_round1.md)

## D01 行为诊断

诊断实例由独立 instance seed 生成，算法随机流使用独立 algorithm seed。正式矩阵采用 50 Jobs、7 个场景、3 个实例 seed、3 个算法 seed和每条 30 代：

```powershell
.venv/Scripts/python scripts/run_diagnostics.py --stage generate --jobs 50 --instance-seeds 1 2 3 --generations 30
.venv/Scripts/python scripts/run_diagnostics.py --stage main --config configs/diagnostics/main/base.yaml --jobs 50 --instance-seeds 1 2 3 --algorithm-seeds 101 202 303 --generations 30 --resume
.venv/Scripts/python scripts/run_diagnostics.py --stage aggregate --instances instances/diagnostic --output outputs/diagnostics
```

每条 run 使用独立目录，已有完整 `summary.json` 时 `--resume` 会跳过该 run，非空的不完整目录会明确报错。Raw outputs 位于 gitignored 的 `outputs/diagnostics/`；聚合 CSV 可直接用于 state-action heatmap 和 profiling。

权威源为 Notion 当前数学模型、算法框架、Coding Contract 与用户确认的补充。按 [AGENTS.md](AGENTS.md) 同步 [Notion 09](https://app.notion.com/p/3e28878b80198186a626f3cf77b4a8dd)。A8 为有界启发式，未找到改善不等于证明不存在。第一版采用全量精确计算，未实现增量 evaluator 或断点续跑。

权威链接：[当前算法](https://app.notion.com/p/3e08878b80198191b9e8cb69c1b4ea91)、[数学模型 v8](https://app.notion.com/p/3e08878b801981dea577c81a3ef73eb0)、[Coding Contract](https://app.notion.com/p/3e28878b801981dd804ec9517608986d)。参数状态见配置文档；working / experimental 尚未做研究有效性验证。

源码在 src/geo_llm_scheduler 下按 domain → scheduling/evaluation → moead/operators/macrosearch/rl → engine → experiments/cli 分层，archive 独立。配置、测试、文档、结果分别在 configs、tests、docs、outputs。主流程：十类初始化 → SSGS/exact → MOEA/D 子代 → Trigger → 共享 Q-learning 与同源 MacroSearch → 一次 Polish → 邻域替换；每个可行 exact 结果独立尝试 Archive。

## Development Workflow

GitHub [`atlantic0423/geo-llm-scheduler`](https://github.com/atlantic0423/geo-llm-scheduler) 的 `main` 是代码、测试、配置、文档和 CI 的权威版本。首次 baseline 后，常规修改从 `main` 创建 `codex/<topic>` 或 `research/<topic>` 分支，经 Pull Request、CI 和 review 后合并。不要直接在 `main` 上长期开发。

每个重要批次依次完成：读取 Notion 当前规格、修改与测试、commit/push、等待 GitHub Actions 通过、将 commit SHA/PR/CI 与规格映射同步到 Notion 09。详细约束见 [AGENTS.md](AGENTS.md)。

## Testing and CI

本地统一门禁：

```powershell
.venv/Scripts/python scripts/validate_all.py
.venv/Scripts/python scripts/reproduce.py
.venv/Scripts/python -m pip wheel . --no-deps --wheel-dir dist
```

`validate_all.py` 执行 ruff lint/format、mypy、全部 unit/golden/integration/regression tests，并强制整体覆盖率不低于 80%、核心模块不低于 90%。GitHub Actions 在 Python 3.11 和 3.13 上执行相同门禁、确定性回放、wheel 构建和最小 CLI smoke，不依赖 Notion token、私有数据或本地路径。

## Reproducibility

固定实例、展开后的配置、master seed、源码 commit SHA 和 Python 环境共同标识一次运行。固定代数用于精确路径回放；wall-clock 停止会受机器负载影响。正式实验应保存 `summary.json`、trace、Archive、Q-table、输入和配置，并在方法间使用统一 HV/IGD+ 参考。

## Research Specification and Versioning

Notion 继续维护数学模型、当前算法框架、Coding Contract、研究决策、实验和任务；GitHub `main` 维护这些规格的当前代码实现。研究语义变化应先同步当前算法框架和决策，再修改实现与测试。Notion 09 日常记录 commit SHA、PR、CI 与测试结果；完整 ZIP 仅用于 milestone、release、论文 checkpoint 或可复现包。历史 Phase 0–9 ZIP 保留。

仓库未附开源许可证；公开可见不等于授予使用、修改或再分发许可，许可证由项目负责人另行决定。
