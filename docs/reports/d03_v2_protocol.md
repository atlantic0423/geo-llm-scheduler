# D03 DominantCondition Balanced Diagnostic Protocol

D03 是受控诊断实验，与 D01 controlled diagnostic 和 D02 mixed validation 分开保存、分开聚合。它不代表真实业务状态分布，也不用于宣称 Q-learning 优于 Random 或 Bandit。

## 冻结语义

- 42-state、六项 severity、阈值、Normal 判定和相对阈值 argmax 规则保持不变。
- A1–A8、Trigger、Q-learning、MacroSearch、`L_RL` 和 Fixed Equal budget $B=6$ 保持不变。
- 每个 independent run 使用独立 Q-table；run 内共享。
- D03 只增加只读日志、实例生成、校准、聚合、图和报告。

## 运行矩阵

Calibration 与 Frozen Evaluation 使用不相交的 instance seeds。Calibration 数据不进入正式图表。

- Type-I calibration：7 conditions × 3 profiles × 2 instance seeds × 1 algorithm seed，15 generations。
- Type-II calibration：3 profiles × 3 instance seeds × 1 algorithm seed，15 generations。
- Type-I frozen evaluation：7 conditions × 2 unseen instance seeds × 3 algorithm seeds，50 generations，共 42 runs。
- Type-II frozen evaluation：5 unseen instance seeds × 3 algorithm seeds，50 generations，共 15 runs。

Type-I 为每个 condition 选择 target share 最大的预定义非极端 profile。Type-II 依次按 `effective_distinct_10`、distinct conditions 和 transition count 选择 profile。并列使用预定义 profile 顺序稳定裁决。选择结果写入冻结 manifest，并绑定源码 commit/hash；正式结果不会反向修改 generator。

## Severity-scale 诊断

在运行 D03 calibration 前，先从 D02 raw RL-step 数据回算六项：

$$
R_d = \frac{D_d}{\theta_d}.
$$

分别保存原始 $D_d$、$R_d$ 的分位数、$P(R_d>1)$、$P(d=\arg\max R)$、$P(d=\arg\max R\mid R_d>1)$、pairwise win rate 和 top1-top2 margin。Generation 1 单列，并同时输出 1–10、11–20、21–30、31–40、41–50。D03 额外保存 generation-0 initial-population snapshot，但该 snapshot 不进入 RL visit 或 Q update。

## 停止边界

若预定义的 moderate profile 仍无法提供足量样本，流水线如实冻结和报告失败模式，不扩大到明显失真的输入范围。正式 evaluation 输出不会用于回调 generator。
