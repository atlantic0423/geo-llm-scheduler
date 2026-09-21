# 配置

YAML 覆盖 Config，未知字段报错。全部字段/default见 src/geo_llm_scheduler/config.py。configs/default.yaml是working参数；smoke和experiments为小规模示例。相对实例路径按工作目录解析，从仓库根运行。

| 参数 | 默认 | 状态 |
|---|---|---|
| population/neighborhood/neighbor_probability/replacement_cap | 100/20/.90/2 | working |
| crossover_probability/mutation_probability | .90/.20 | working |
| mutation_weights | .4/.4/.2 | working |
| alpha/gamma/epsilon_start/epsilon_end | .30/.70/.30/.05 | working |
| rl_steps/stagnation_threshold | 5/5 | working |
| severity_thresholds | 六个.20 | working |
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
