# Acceptance Matrix

| Phase | Gate | 状态与证据 |
|---|---|---|
| 0 | 理解报告/结构/AGENTS/安装/docs | 通过，phase0报告；最终文档本批次补全 |
| 1 | 不可变数据/MS唯一/单位/身份/RNG | 通过，unit/test_domain.py |
| 2 | exact feasibility/Flow/TOU/Demand | 通过，evaluator/capacity golden |
| 3 | earliest/event/diagnostics | 通过，SSGS golden |
| 4 | 初始化/normalization/Archive/MOEA-D | 通过，moead golden与baseline integration |
| 5 | A1–A6/同源/预算/去重 | 通过，macrosearch与insertions |
| 6 | A7/A8/Polish timing和反例 | 通过，timing_operators/a8_adversarial/accepted_timing |
| 7 | Trigger/42×8/reward/budgets/trajectory | 通过；用户确认最后一步terminal，test_terminal.py验证边界及Polish隔离 |
| 8 | full/CLI/消融/日志/回放 | 通过，23个实验/消融配置实际运行 |
| 9 | 核心90%/整体80%/profiling/docs/ZIP | 通过：109 tests，整体96.56%、核心96.66%；最终报告见reports/final_acceptance.md |

## 跨模块强制不变量

I1：所有complete+feasible+exact candidate独立尝试Archive，包含初始化、普通子代、全部MacroSearch与Polish。
I2：相同候选集合评价顺序不改变best-of-B，context一致，tie按稳定身份。
I3：structural必须rebuild，timing-only冻结未选operation，刷新diagnostics不得改timing。

每阶段已有规格映射、golden/反例、全回归三类证据；精确测试名称见traceability_matrix。全部阶段报告保留原始结果，新批次不得用旧结果代替新验证。最终是否完成以实际命令和待确认项为准。
