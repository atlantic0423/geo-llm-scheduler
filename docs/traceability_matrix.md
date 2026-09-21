# 规格 → 代码 → 测试

路径均相对仓库。任务书、Notion当前框架/模型/Contract与用户六项补充共同构成规格。

| 机制/来源 | 实现 | 独立验证 |
|---|---|---|
| Contract 不可变对象/MS权威/单位 | domain/models.py, io/loaders.py, validation.py | unit/test_domain.py |
| Contract容差/身份/RNG | utils/numeric.py, rng.py | unit/test_domain.py |
| 模型release/KV/容量/半开区间 | evaluation/exact.py | golden/test_evaluator.py, test_capacity_edges.py |
| active union/TOU/尾窗/全Region尾部 | evaluation/exact.py | golden/test_evaluator.py |
| event-driven earliest/双资源诊断 | scheduling/resources.py, ssgs.py | golden/test_ssgs.py |
| 初始化十类与variation | initialization/generators.py, moead/variation.py | integration/test_baseline.py |
| dynamic ideal/population maximum/lambda_j | moead/core.py | golden/test_moead.py |
| Archive phenotype-aware/独立分发 I1 | archive/pareto.py, engine/evaluation.py | golden/test_moead.py, integration/test_baseline.py |
| A1–A6结构边界/重建 I3 | operators/structural.py | integration/test_macrosearch.py, golden/test_insertions.py |
| MacroSearch同源/预算/顺序 I2 | macrosearch/search.py | integration/test_macrosearch.py |
| A7有限关键点/正压缩/无A8horizon | scheduling/timing.py, operators/active_pack.py | golden/test_ssgs.py, test_timing_operators.py |
| A8并列峰/支撑组/中点/rollback/gate | operators/peak_coalition.py | golden/test_a8_adversarial.py, test_timing_operators.py |
| Polish冻结D/严格降费/独立Archive | operators/right_shift.py | golden/test_timing_operators.py |
| accepted timing全量刷新 I3 | scheduling/ssgs.py, engine/trajectory.py | integration/test_accepted_timing.py |
| 六severity唯一公式 | diagnostics/severity.py, workload.py | golden/test_severity.py |
| Trigger方向/quality | engine/trigger.py, rl/state.py | golden/test_trigger.py |
| 42×8共享Q/reward/Bandit | rl/controller.py, engine/trajectory.py | unit/test_rl.py, integration/test_full_engine.py |
| 五预算/短轨迹/回放 | macrosearch/budget.py, engine/run.py | integration/test_full_engine.py |
| 版本化artifact/HV/IGD+ | experiments/runner.py, metrics.py | integration/test_experiments.py |
| 配置非法边界 | config.py | unit/test_config.py |
| Markdown/安装/依赖 | pyproject.toml, docs, scripts | unit/test_scaffold.py |

Q最后一步边界的规格确认状态见reports/open_questions.md。验收状态与未完成项见acceptance_matrix.md；不将覆盖率视为数学证明。
