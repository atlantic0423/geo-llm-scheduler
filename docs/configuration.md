# 配置

当前 `configs/default.yaml` 是 `working-default/v1` profile，选择完整 B2 + P1 + N7 + GS4；通过主 CLI 或 `experiments.runner.run_config` 加载。它需要显式实例、冻结校准文件和新输出目录，详见 [当前默认算法](default_algorithm.md)。相对路径按工作目录解析。

下表是历史核心 `Config` 的字段和参数，原 YAML 已保存为 `configs/legacy_core.yaml`；`load_config` 仅加载这种核心配置，未知字段报错。它不负责组装完整工作版。显式 smoke 和旧 experiments 配置继续用于历史复现。

| 参数 | 默认 | 状态 |
|---|---|---|
| population/neighborhood/neighbor_probability/replacement_cap | 100/20/.90/2 | working |
| crossover_probability/mutation_probability | .90/.20 | working |
| mutation_weights | .4/.4/.2 | working |
| alpha/gamma/epsilon_start/epsilon_end | .30/.70/.30/.05 | working |
| rl_steps/stagnation_threshold | 5/5 | working |
| severity_thresholds | 资源 .13、KV .05、区域 .60、分时电价 .43、需量 .33、可压缩 .16 | v2 working；[选择依据](reports/severity_thresholds_v2.md) |
| budgets/fixed_budget | 3,6,10/6 | working |
| A8 singleton/member/position/attempt-multiplier | 2/8/6/2 | working |
| a6_destroy_ratio | .10，至少2且不超过job数 | working |
| trigger_delta | .10 | experimental |
| initialization_perturbation/initialization_attempts | .10/10000 | experimental engineering |
| random_attempt_multiplier | 20 | experimental engineering |
| generations/seconds/seed | 代数/可选时间软上限/主种子 | run-specific |

method=plain/full；controller=qlearning/bandit/random。polish可关闭，enabled_operators支持leave-one-out。
budget_policy=fixed/static/random/severity/coverage。static_budgets顺序A1–A8默认3,6,6,3,3,10,6,10，向量为experimental。
severity比率≤1/≤2/>2对应低/中/高，A4/A5按Flow/Balanced/Electricity对应高/中/低；A6停滞用高否则低。coverage先按 Flow/费用容差对 Archive objective vector 去重；不同 phenotype 的相同目标只计一个 coverage representative。representative 数不足population时用中，否则方向计数/平均密度<.5用高、≤1.5用中、其他低。两方案独立实验。
trigger_mode=preference/strict/always/fixed，fixed_ls_probability=.5。用户确认的三个severity公式为experimental implementation convention，其数学定义唯一。未做敏感性实验前不得称working值最优。
