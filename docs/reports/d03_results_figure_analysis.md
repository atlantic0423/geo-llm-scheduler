# D03 正式结果逐图分析与下一步实验计划

**版本与用途：**2026-09-24，D03 v2 诊断性结果。本文与 [GitHub D03 数据目录](../../research/d03/README.md) 一一对应；每一项均链接实际提交的 PNG、可放大的 PDF 和依据表。它解释**观察到什么、当前公式和参数为何可能产生该现象、证据到哪里为止**。这些图不构成论文中的算法性能优越性结论。

## 1. 先读实验与统计口径

D03-I（定向覆盖组）为七类目标实例族 × 两个正式实例种子 × 三个算法种子，**42/42** 条；D03-II（综合动态组）为五个独立实例 × 三个算法种子，**15/15** 条。均为 50 个请求、三区域各三台同构服务实例、种群 100、完整算法、表格 Q-learning、每次局部调用固定候选预算 6、每条局部轨迹 5 步、每条 50 代。D03-I 有 257,010 条动作记录，D03-II 有 95,050 条。51 条 15 代预校准运行只负责**提前冻结 profile**，不参与以下正式百分比。依据：[完成状态](../../research/d03/data/pipeline_state.json)、[冻结选择与种子](../../research/d03/data/frozen_selection.json)、[D03-I 清单](../../research/d03/data/manifests/type_i_evaluation.csv)、[D03-II 清单](../../research/d03/data/manifests/type_ii_evaluation.csv)。

严重程度为六项 $D_k$，冻结阈值均为 $\theta_k=0.20$；相对分数 $R_k=D_k/\theta_k$。全部 $R_k\leq1$ 才标为 Normal，否则取最大相对分数对应的主导条件。由于六个 $D_k$ 的公式和实例分布不同，**相同阈值并不保证条件频率相同**。42 个状态为偏好（流转/均衡/电费）× 七个主导条件 × 进展（改善/停滞）。学习率 0.30、折扣率 0.70，探索概率由 0.30 线性降至 0.05。问题与定义见[状态公式](../specifications/state_severity.md)、[当前完整框架](https://app.notion.com/p/3e08878b80198191b9e8cb69c1b4ea91)、[D03 任务记录](https://app.notion.com/p/3e38878b8019819fad44d9d2c6fc5482)。

选择热力图必须先对**本组全部正式运行累计调用数**，再求比例：

$$
P(a\mid s)=\frac{\sum_r N_r(s,a)}{\sum_r\sum_b N_r(s,b)},\qquad
p_+(s,a)=\frac{\sum_r N_r^+(s,a)}{\sum_r N_r(s,a)}.
$$

其中 $p_+$ 是**标量奖励为正的调用比例**，并非双目标同时改善率；平均奖励也是按调用次数加权，单位不是人民币。无候选时奖励为 $-1$。图中小于 30 次的格灰显；即便超过 30，同一实例的许多步骤仍相关，不能当作独立样本。D03-I 和 D03-II **绝不拼接计数**。后续 A4/A5 的正等待选择修改不属于本轮轨迹；本轮冻结源码为 [`2526ad1`](https://github.com/atlantic0423/geo-llm-scheduler/commit/2526ad1afc425b4d6ff847055eebd4785744acc8)。

## 2. 汇报时先讲的六个结论

1. **执行完成不等于设计目标达成。** 定向组访问 42/42 状态，但目标为 Normal 和 Compressible 的实例族分别仅有 0% 和 0.55% 目标标签；综合组仅访问 18/42 状态。
2. **综合组的主导条件几乎是分时电价/区域。** TOU 占 82.29%，Region 占 17.15%；不是 Demand 不严重：Demand 的相对分数中位数 1.428、97.2% 越阈值，却常在最大值竞争中输给 TOU。
3. **可压缩目标失配有两道门。** 在 35,465 条对应记录中，只有 15.06% 的 Compressible 分数先越线；越线的 5,342 条里只有 195 条赢得主导。主要对手是 Demand 和 Region，Resource 在这组从未越线。
4. **动作效果有状态差异，但“选最多、正改善最多、奖励最大”并不等价。** D03-I 的 33 个较充分支持状态中，这三个排名完全一致的机会有限；D03-II 的 12 个高样本状态都最常选 A7，但其中一半平均奖励最大的是 A2。
5. **A7 是广泛有效的动作，不是 Compressible 标签专属动作。** 在 Compressible 分数未过线时仍有大量正改善；A5/A8 低收益则需先区分无候选与有候选但标量不改善。旧版 A4/A5 行为仅作历史观察。
6. **不能据此宣称 Q-learning 优于对照。** 动作后期集中与探索率退火、触发选择和状态组成同时变化；同实例跨三个算法种子的最终贪心动作完全一致只占 D03-I 的 8/91、D03-II 的 4/39 个合格实例—状态组合。

## 3. 正式聚合图：D03-I 定向覆盖组（12 张）

下列 `I-01` 至 `I-12` 均来自 **42 条正式运行**，PNG/PDF 是同一张图的两种格式。

### I-01｜条件总体分布

[图](../../research/d03/figures/type_i/condition_distribution.png) · [PDF](../../research/d03/figures/type_i/condition_distribution.pdf) · [数据](../../research/d03/data/analysis/condition.csv)。Region 占 **46.64%**、Demand 17.65%、Resource 13.41%、TOU 13.00%，KV 仅 1.76%。七种条件都出现，数量却极不均匀；这是定向实例族混合后的**受 Trigger 筛选的动作记录分布**，不能推断原始实例或整个种群同样分布。它说明“42 状态可到达”，不说明“七类样本足量且均衡”。

### I-02｜条件随代变化（原始计数）

[图](../../research/d03/figures/type_i/condition_by_generation.png) · [PDF](../../research/d03/figures/type_i/condition_by_generation.pdf) · [逐代比例表](../../research/d03/data/analysis/generation.csv)。第 1 代 Region 约 **19.6%**，第 50 代约 **52.3%**；Compressible 则由约 23.1% 降至 2.7%。原图纵轴为计数，局部搜索触发次数随代改变，面积高低不能直接等同概率变化；上述百分比从同一代的计数分母另算。增长可与区域分数竞争和搜索选择共同解释，不能单凭图归因于 Q-learning。

### I-03｜改善状态的覆盖

[图](../../research/d03/figures/type_i/state_coverage_improving.png) · [PDF](../../research/d03/figures/type_i/state_coverage_improving.pdf) · [状态表](../../research/d03/data/analysis/state.csv)。横向拆出偏好和七类条件后，图显示覆盖主要由 Region、Demand 等高频条件支撑；例如 Electricity–Normal–Improving 仅 **8 次**、Flow–KV–Improving 仅 **28 次**。颜色或“有访问”不能替代独立实例数；这两格不足以判断其最佳动作。

### I-04｜停滞状态的覆盖

[图](../../research/d03/figures/type_i/state_coverage_stagnating.png) · [PDF](../../research/d03/figures/type_i/state_coverage_stagnating.pdf) · [状态表](../../research/d03/data/analysis/state.csv)。Electricity–Normal–Stagnating **5 次**，Flow–KV–Stagnating **13 次**；加上上一张的两格，D03-I 有四个状态总访问不足 30 次。停滞状态并非缺席，但某些标签几乎只来自一条运行，不能据“全覆盖”声称后期策略已充分学习。

### I-05｜六项相对严重程度分布

[图](../../research/d03/figures/type_i/severity_ratio_distribution.png) · [PDF](../../research/d03/figures/type_i/severity_ratio_distribution.pdf) · [分布表](../../research/d03/data/aggregate/type_i/severity_scale_summary.csv)。这张总体箱线图混合七个不同生成族，Resource、KV、TOU、Compressible 的总体中位数接近零并不表示对应实例族内永远不严重；必须回到[分族表](../../research/d03/data/analysis/d03_mechanism_family_severity.csv)。Region 总体中位数约 **1.691**，Demand 约 **1.216**，所以两者有大量机会争夺主导；六项公式的自然尺度并未因阈值同为 0.20 而自动一致。

### I-06｜相对严重程度随代变化

[图](../../research/d03/figures/type_i/severity_ratio_by_generation.png) · [PDF](../../research/d03/figures/type_i/severity_ratio_by_generation.pdf) · [逐代分数](../../research/d03/data/aggregate/type_i/severity_by_generation.csv)。它区分“条件占比变了”与“各项相对分数本身变了”；I-02 的 Region 占比上升不能仅凭频数判为阈值效果。由于每代仍是受触发门筛选的不同调度集合，曲线不等于固定同一解的纵向因果变化，需结合各实例族单独复核。

### I-07｜严重程度两两胜率

[图](../../research/d03/figures/type_i/severity_pairwise_win_rate.png) · [PDF](../../research/d03/figures/type_i/severity_pairwise_win_rate.pdf) · [两两计数](../../research/d03/data/aggregate/type_i/severity_argmax_competition.csv)。图回答“越过阈值以后，与另一项相比谁更大”，帮助解释 Region/Demand 抢走 Compressible 标签。它是**六项分数的两两竞争**，不是动作 A1–A8 的胜率，也不是七类标签占比；并列及多对比较的口径不能直接相加。

### I-08｜最大与次大分数差

[图](../../research/d03/figures/type_i/severity_top_margin.png) · [PDF](../../research/d03/figures/type_i/severity_top_margin.pdf) · [差值表](../../research/d03/data/aggregate/type_i/severity_margin_summary.csv)。总体中位差约 **1.054**；许多记录的主导者并非微弱领先，所以简单修改同分破局规则不会让各类自然均衡。不同实例族差异很大，整体中位差不能代表每个条件边界；后续应在近边界状态上研究真实转移机会。

### I-09｜主导条件 × 算子选择

[图](../../research/d03/figures/type_i/condition_action_selection.png) · [PDF](../../research/d03/figures/type_i/condition_action_selection.pdf) · [数据](../../research/d03/data/analysis/condition_action.csv)。这张图合并偏好与进展，能快速找到 A7 的广泛使用及某些条件下 A2/A5 的变化；它**不是完整 42 状态策略图**。例如 KV 中 A2 被调用 **841 次**，但 KV 条件仅由少数独立实例支撑。选择频繁可能源于学习、探索率、候选适用性或被 Trigger 采到的调度组成，不能单独读成因果最优。

### I-10｜主导条件 × 算子正改善率

[图](../../research/d03/figures/type_i/condition_action_success.png) · [PDF](../../research/d03/figures/type_i/condition_action_success.pdf) · [数据](../../research/d03/data/analysis/condition_action.csv)。KV 中 A2 的 **841 次**调用约 **42.1%** 正改善；Resource 中 A5 约 **22.2%**；Region 中 A3 约 **3.1%**。标签与同名算子不必一一对应：A3 改区域不均衡的代理排序，接受条件却是流转时间—电费标量；Region 集中不保证搬区被接受。每格仍需看调用数与独立实例数。

### I-11｜主导条件 × 算子平均奖励

[图](../../research/d03/figures/type_i/condition_action_reward.png) · [PDF](../../research/d03/figures/type_i/condition_action_reward.pdf) · [数据](../../research/d03/data/analysis/condition_action.csv)。它把无候选 $-1$ 与正改善幅度纳入平均，补足“成功率不等于收益”。KV 中 A2 的观测平均奖励约 **2.923**，A7 约 **2.510**；但动作不是在同一起点随机分配，这两数不是 A2 对 A7 的公平因果估计。应与 I-09、I-10 三图并排看。

### I-12｜总体动作选择熵

[图](../../research/d03/figures/type_i/policy_entropy.png) · [PDF](../../research/d03/figures/type_i/policy_entropy.pdf) · [逐代动作表](../../research/d03/data/analysis/action_progress.csv)。图按同代**所有状态的总体动作比例**求熵，可描述后期动作分布集中；它没有先按状态求策略熵。后期 A7 占比上升的同时，探索率按预设由 0.30 降至 0.05，状态组成也变化，因此熵下降不能单独证明学到了稳定状态策略。

## 4. 正式聚合图：D03-II 综合动态组（13 张）

以下各图只使用 **15 条正式运行**；其生成器并未指定要让七类条件均匀出现。

### II-01｜条件总体分布

[图](../../research/d03/figures/type_ii/condition_distribution.png) · [PDF](../../research/d03/figures/type_ii/condition_distribution.pdf) · [数据](../../research/d03/data/analysis/condition.csv)。95,050 次局部动作中 TOU **78,215 次（82.29%）**、Region **16,298 次（17.15%）**、Demand **532 次（0.56%）**、Compressible **5 次**；Normal、Resource、KV 未出现。综合随机实例容许这些机制存在，但最大分数规则、生成范围和 Trigger 观察窗口没有理由保证其成为标签。

### II-02｜条件随代变化（原始计数）

[图](../../research/d03/figures/type_ii/condition_by_generation.png) · [PDF](../../research/d03/figures/type_ii/condition_by_generation.pdf) · [逐代比例表](../../research/d03/data/analysis/generation.csv)。第 1 代 TOU 约 **65.9%**、Demand 11.7%；第 10 代 TOU 约 **85.8%**、Demand 0.6%；第 50 代 TOU 约 **83.7%**。这是选中局部搜索后的记录，原图计数受每代触发次数影响；用比例表才能说明结构改变。Demand 衰退与 TOU 竞争增强相容，但不能由此确定是哪个模块造成。

### II-03｜改善状态覆盖

[图](../../research/d03/figures/type_ii/state_coverage_improving.png) · [PDF](../../research/d03/figures/type_ii/state_coverage_improving.pdf) · [状态表](../../research/d03/data/analysis/state.csv)。综合组只覆盖 **18/42** 状态，改善面板主要落在 TOU 与 Region；Flow–Compressible–Improving 仅 **5 次**。50 代并没有自动填满状态空间，稀缺条件下动作颜色不可外推至所有综合实例。

### II-04｜停滞状态覆盖

[图](../../research/d03/figures/type_ii/state_coverage_stagnating.png) · [PDF](../../research/d03/figures/type_ii/state_coverage_stagnating.pdf) · [状态表](../../research/d03/data/analysis/state.csv)。Electricity–Demand–Stagnating 仅 **2 次**；主导分布的失衡同时延伸到进展维度。状态总访问数与独立实例数要一起看，不能用大量 TOU 反复访问补偿 Demand 的缺口。

### II-05｜六项相对严重程度分布

[图](../../research/d03/figures/type_ii/severity_ratio_distribution.png) · [PDF](../../research/d03/figures/type_ii/severity_ratio_distribution.pdf) · [分布表](../../research/d03/data/aggregate/type_ii/severity_scale_summary.csv)。TOU 相对分数中位数 **2.103**、100% 越线，Demand 中位数 **1.428**、97.2% 越线，但 Demand 只获 0.56% 的主导标签；Resource、KV、Compressible 中位数分别约 **0.127、0.103、0.561**。因此主因不是“只有 TOU 严重”，而是**TOU 相对分数普遍更高**。同一区域的分时电价表 `0.42→0.72→0.36→0.66→0.48 元/千瓦时` 经 min-max 归一化后为约 `0.167,1,0,0.833,0.333`；长操作跨价段会持续产生非零活动能量加权分数。

### II-06｜相对严重程度随代变化

[图](../../research/d03/figures/type_ii/severity_ratio_by_generation.png) · [PDF](../../research/d03/figures/type_ii/severity_ratio_by_generation.pdf) · [逐代分数](../../research/d03/data/aggregate/type_ii/severity_by_generation.csv) · [初始种群](../../research/d03/data/aggregate/type_ii/initial_population_severity.csv)。TOU 高分在**初始种群**已存在：1,500 个快照里 TOU 中位数约 **2.112**、全部越线，初始分数最大约 63.9%。故 82.29% 的后续占比不能全归因于 Q-learning；初始种群与 Trigger 后的局部轨迹不是同一采样总体，也不能简单把差额归因于学习。

### II-07｜严重程度两两胜率

[图](../../research/d03/figures/type_ii/severity_pairwise_win_rate.png) · [PDF](../../research/d03/figures/type_ii/severity_pairwise_win_rate.pdf) · [两两计数](../../research/d03/data/aggregate/type_ii/severity_argmax_competition.csv)。TOU 相对分数在约 **99.36%** 的记录中高于 Demand；这解释了 Demand 即使越阈值也常未获标签。两两比较针对同一批被触发记录，不代表改变 Demand 费率后就能翻转标签：在当前 Demand 严重程度的归一化定义下，全部区域费率同比例缩放会约掉。

### II-08｜最大与次大分数差

[图](../../research/d03/figures/type_ii/severity_top_margin.png) · [PDF](../../research/d03/figures/type_ii/severity_top_margin.pdf) · [差值表](../../research/d03/data/aggregate/type_ii/severity_margin_summary.csv)。最高与次高相对分数差的中位数约 **0.528**，仅约 **4.70%** 的记录差值不超过 0.1。TOU 占优通常不是破同分的偶然结果；一次算子接受也不易跨越较宽的主导边界。要检验“状态切换困难”应按同一轨迹连接行动前记录，见 S04。

### II-09｜主导条件 × 算子选择

[图](../../research/d03/figures/type_ii/condition_action_selection.png) · [PDF](../../research/d03/figures/type_ii/condition_action_selection.pdf) · [数据](../../research/d03/data/analysis/condition_action.csv)。A7 在 TOU 下的选择比例约 **40.61%**，在 Region 下约 **40.66%**，两行几乎相同。这支持“广泛选用 A7”，**不支持**“已经学会明显区分 TOU 与 Region 的 A7 使用策略”。其它条件样本过少，不宜读取深浅差异。

### II-10｜主导条件 × 算子正改善率

[图](../../research/d03/figures/type_ii/condition_action_success.png) · [PDF](../../research/d03/figures/type_ii/condition_action_success.pdf) · [数据](../../research/d03/data/analysis/condition_action.csv)。A7 在本组总体正改善率约 **36.39%**，明显高于 A5 的 **2.48%**；但按标签合并偏好和进展，不能据此宣布某一个 42 状态的最优动作。A5 的 **2,421 次**调用中约 **78.89%** 无可行候选，是低正改善的重要直接背景。

### II-11｜主导条件 × 算子平均奖励

[图](../../research/d03/figures/type_ii/condition_action_reward.png) · [PDF](../../research/d03/figures/type_ii/condition_action_reward.pdf) · [数据](../../research/d03/data/analysis/condition_action.csv)。总体 A7 平均奖励约 **0.970**，A2 为 **1.087**；A7 成功更频繁并不必然产生更大平均标量收益。A5 总体平均奖励约 **−0.708**，与大量 $-1$ 无候选返回相容，不能直接解释为“找到的候选普遍损害目标”。

### II-12｜总体动作选择熵

[图](../../research/d03/figures/type_ii/policy_entropy.png) · [PDF](../../research/d03/figures/type_ii/policy_entropy.pdf) · [逐代动作表](../../research/d03/data/analysis/action_progress.csv)。后期动作频率趋于集中，A7 从前 10 代约 **20.1%** 升至后 10 代约 **53.9%**。这张图仍是**总体动作熵**，不是每个状态的条件熵；探索率退火与 TOU/Region 占比变化都会改变总体熵。实际最后十代 A7 多数调用来自贪心分支，但仍需公平对照验证是否有净收益。

### II-13｜代表性条件时间线

[图](../../research/d03/figures/type_ii/representative_condition_timelines.png) · [PDF](../../research/d03/figures/type_ii/representative_condition_timelines.pdf) · [选择规则](../../research/d03/data/aggregate/type_ii/representative_timeline_selection.csv) · [全部运行转移表](../../research/d03/data/aggregate/type_ii/run_transition_summary.csv)。时间线说明个别运行确有局部条件更替，但一条被选中的例子不能代表 15 条。按“每 run 某条件至少 10 次”为有效出现，D03-II 每 run 通常只有 **2–3** 类有效条件；15 条里仅 **5 条**的“每代众数条件”发生变化。此处众数变化并不等于同一个解的真实状态转移。

## 5. 42 状态 × 八算子：必须成组三图读（6 张）

这六张中文图每张分成流转偏好、均衡偏好、电费偏好三个 **14×8** 面板，格内标调用数，少于 30 次灰显。[逐格数据](../../research/d03/data/analysis/state_action.csv) 和[状态访问/实例支持](../../research/d03/data/analysis/state.csv)是核对入口。D03-I 的 336 格中 314 格曾访问、270 格达到 30 次；D03-II 只有 130 格曾访问、102 格达到 30 次。以下“最高”均为**被选择后的描述性统计**，不是同起点公平试验。

### H-01｜D03-I 状态—动作选择比例

[图](../../research/d03/figures/state_action_cn/d03_i_selection_42state_cn.png) · [PDF](../../research/d03/figures/state_action_cn/d03_i_selection_42state_cn.pdf)。较充分的 33 个状态中，最常被选的是 A7 **21 个**、A2 **6 个**、A1 **5 个**、A5 **1 个**。例如状态 6（流转 × 区域 × 改善）A7 被选 **11,809/30,305≈39.0%**；状态 30（电费 × 资源 × 改善）A5 被选约 **22.6%**。这说明状态输入与选择频率有关，但条件名称不应当机械映射到同名动作；本组很多条件只来自两个正式实例。

### H-02｜D03-I 状态—动作正改善率

[图](../../research/d03/figures/state_action_cn/d03_i_success_42state_cn.png) · [PDF](../../research/d03/figures/state_action_cn/d03_i_success_42state_cn.pdf)。状态 30 中 A5 的 **1,242 次**调用正改善 **32.5%**、无候选约 **0.2%**，与 II 组的 A5 总体差别很大；状态 18（均衡 × KV × 改善）A2 正改善约 **52.5%**，却仅两个独立实例。33 个较充分状态里，最常选动作同时是最高正改善动作的只有 **20 个**，说明行为偏好与即时成功并不完全一致。

### H-03｜D03-I 状态—动作平均奖励

[图](../../research/d03/figures/state_action_cn/d03_i_reward_42state_cn.png) · [PDF](../../research/d03/figures/state_action_cn/d03_i_reward_42state_cn.pdf)。状态 20（均衡 × 区域 × 改善）A2 平均奖励约 **1.16**，A7 正改善率更高约 **25.0%**；状态 22（均衡 × TOU × 改善）A7 正改善更频繁，但 A1 平均奖励约 **2.12**、高于 A7 的 **1.29**。最常选动作同时是最高平均奖励动作的只有 **18/33**；奖励含幅度及无候选惩罚，还受未来价值和探索影响，不需与当前成功率逐格同序。

### H-04｜D03-II 状态—动作选择比例

[图](../../research/d03/figures/state_action_cn/d03_ii_selection_42state_cn.png) · [PDF](../../research/d03/figures/state_action_cn/d03_ii_selection_42state_cn.pdf)。可比较的 **12 个**高样本状态全部最常选 A7，且全是 TOU 或 Region 的偏好 × 进展组合。状态 8（流转 × TOU × 改善）A7 占 **36.6%**，状态 22（均衡 × TOU × 改善）占 **45.2%**，状态 20（均衡 × Region × 改善）仍占 **49.0%**。它更像通用动作占优与少数条件重复访问，不能推广为七类状态的成熟策略。

### H-05｜D03-II 状态—动作正改善率

[图](../../research/d03/figures/state_action_cn/d03_ii_success_42state_cn.png) · [PDF](../../research/d03/figures/state_action_cn/d03_ii_success_42state_cn.pdf)。在上述 12 个高样本状态，A7 也全部有最高观测正改善率；状态 8 的 A7 约 **43.4%**，A2 约 **28.2%**。但稀疏的 Demand/Compressible 格灰显，无法推断其效果。A7 的优势是被当前控制器选择后记录的条件统计，下一步必须在**相同父解快照**上同时评价多个算子才能排除选择偏差。

### H-06｜D03-II 状态—动作平均奖励

[图](../../research/d03/figures/state_action_cn/d03_ii_reward_42state_cn.png) · [PDF](../../research/d03/figures/state_action_cn/d03_ii_reward_42state_cn.pdf)。12 个高样本状态中，A7 仅 **6 个**奖励最大，另 **6 个**为 A2。状态 8 中 A7 平均奖励约 **0.82**，A2 约 **1.20**；状态 36（电费 × TOU × 改善）A7 则约 **2.11**、高于 A2 的 **1.94**。这正是偏好维度可能发挥作用的线索，但 Q 值还含未来收益，跨动作观察均值又有选择偏差，不能据此直接改策略表。

## 6. 二次核验与机制图（10 张）

### S01｜D01、D02、D03-I、D03-II 条件占比并列

[图](../../research/d03/figures/analysis/S01_condition_shares.png) · [PDF](../../research/d03/figures/analysis/S01_condition_shares.pdf) · [四套参考表](../../research/d03/data/reference/condition_all_suites.csv)。每个面板有**独立分母**；D01/D02 只作背景，不与 D03 计数混合。D01 的 Region 53.99%，D02 的 TOU 82.19%，D03-I 分散到七类但 Region 46.64%，D03-II TOU 82.29%。延长到 50 代和定向造实例是不同干预：前者没有使综合组自然均衡，后者虽拓展可达性但带来实例族结构；不能直接比较四组算法性能。

### S02｜定向实例族 × 实际主导条件

[图](../../research/d03/figures/analysis/S02_target_activation.png) · [PDF](../../research/d03/figures/analysis/S02_target_activation.pdf) · [分族表](../../research/d03/data/analysis/scenario_condition.csv)。对角线依次为 Normal **0%**、Resource **100%**、KV **16.51%**、Region **99.10%**、TOU **81.38%**、Demand **59.84%**、Compressible **0.55%**。Resource/Region 纯得几乎不切换；Normal 与 Compressible 失配。KV 组全部请求有正迁移延迟仍有约一半局部记录 KV 分数为零，说明调度可选择同实例 P/D；“参数可用”不等于编码真的产生跨实例迁移。

### S03｜按代归一化的条件占比

[图](../../research/d03/figures/analysis/S03_generation_shares.png) · [PDF](../../research/d03/figures/analysis/S03_generation_shares.pdf) · [逐代表](../../research/d03/data/analysis/generation.csv)。这张图修正 I-02/II-02 的计数分母：D03-I 的 Region 后期上升，D03-II 的 TOU 从第 1 代约 65.9% 上升到第 10 代约 85.8%。它描述**被触发的局部记录组成**，不等于每个解在 50 代内从 Demand 迁到 TOU，也不能把比例变化独立归因于学习。

### S04｜真正同轨迹相邻步骤的条件切换

[图](../../research/d03/figures/analysis/S04_within_trajectory.png) · [PDF](../../research/d03/figures/analysis/S04_within_trajectory.pdf) · [相邻记录表](../../research/d03/data/analysis/within_trajectory.csv) · [接受与切换](../../research/d03/data/analysis/d03_mechanism_transition_by_acceptance.csv)。只连接同 run、同代、同子问题且轨迹步号相邻的行动前状态。D03-I **1,100/205,608=0.535%** 改标签；D03-II **222/76,040=0.292%**。D03-II 未接受移动的 **55,737 对**没有切换，接受的 **20,303 对**仅 **222 对（1.09%）**切换；接受后不换标签的主导分差中位数约 0.494，换标签时约 0.114。未改标签不等于未改善；图也不包括末步未连接的后继状态。

### S05｜偏好 × Trigger 触发漏斗

[图](../../research/d03/figures/analysis/S05_trigger_funnel.png) · [PDF](../../research/d03/figures/analysis/S05_trigger_funnel.pdf) · [触发表](../../research/d03/data/analysis/trigger_summary.csv)。D03-I 流转/均衡/电费偏好触发率约 **37.72%/19.06%/16.88%**，D03-II 约 **28.12%/33.14%/15.09%**。因此热力图描述“通过 Trigger 的调度”而非种群均匀样本；状态覆盖的缺口既可来自生成/分数，也可能来自触发筛选，下一步需同时记录所有候选的触发前状态。

### M01｜Compressible：阈值与竞争两阶段失败

[图](../../research/d03/figures/analysis/D03_compressible_competition.png) · [PDF](../../research/d03/figures/analysis/D03_compressible_competition.pdf) · [逐条复算摘要](../../research/d03/data/analysis/d03_compressible_competition.json) · [分族分数](../../research/d03/data/analysis/d03_mechanism_family_severity.csv)。Compressible 家族 35,465 条记录中，其中位相对分数 **0.589**，仅 **5,342（15.06%）** 越过 1；越线后 Demand 赢 **3,715**、Region 赢 **1,432**、Compressible 仅赢 **195（3.65%）**。Resource 在本组始终没有越线，不能说它“抢走”可压缩标签。生成器短 Prefill、长 Decode、释放散至 10,800 秒，而分母是全部操作时长；这会压低总体压缩比例，同时分散长时段可产生相对突出的需量峰—均差。两机制贡献尚未被独立对照拆分。

### M02｜D03-I 全局动作结果面板

[图](../../research/d03/figures/analysis/d03_type_i_global_operator_dashboard.png) · [PDF](../../research/d03/figures/analysis/d03_type_i_global_operator_dashboard.pdf) · [动作表](../../research/d03/data/analysis/action.csv)。A7 总体调用 **67,157 次**（26.13%），正改善 **17,145 次**（25.53%）；A8 仅调用 **9,249 次**（3.60%），正改善 **794 次**（8.58%），但 A8 每次调用的平均标量下降约 **2.355 千分之一**，高于 A7 的 **1.384**；每分钟则约 **1.599** 与 **0.705**。故“低成功率”不等于“没有价值”，仍要区分候选稀少、单次幅度和时间成本。全局平均混合七个生成族，也会掩盖 A5 在资源状态的适用性；A8 的有界搜索及 900 秒峰窗使其更依赖具体实例结构，不能仅归因于控制器选错。

### M03｜D03-II 全局动作结果面板

[图](../../research/d03/figures/analysis/d03_type_ii_global_operator_dashboard.png) · [PDF](../../research/d03/figures/analysis/d03_type_ii_global_operator_dashboard.pdf) · [动作表](../../research/d03/data/analysis/action.csv)。A7 调用 **38,470 次**（40.47%），正改善 **14,001 次**（36.39%）；A5 仅 **2,421 次**（2.55%）且正改善 **60 次**（2.48%），其中约 **78.89%** 无可行候选。A8 调用 **5,053 次**、正改善 **733 次**（14.51%），约 **27.4%** 无候选，但每次调用平均标量下降约 **3.535 千分之一**、每分钟约 **4.233**，均高于 A7 的 **1.602** 与 **1.497**。A5 旧版需成对找到合法前插位置，A8 受原结束时刻及有限枚举约束，因此必须拆分“无动作”和“有动作但不改善”。D03 后的 A4/A5 修改不能回填为本图效果。

### M04｜状态 6：各动作每次调用的标量改善

[图](../../research/d03/figures/analysis/d03i_state6_mean_scalar_gain_per_call.png) · [PDF](../../research/d03/figures/analysis/d03i_state6_mean_scalar_gain_per_call.pdf) · [逐格计数](../../research/d03/data/analysis/state_action.csv) · [原始轨迹归档](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/d03-v2-results-2026-09-24)。状态 6 为流转偏好 × Region × Improving；A7 被选 **11,809 次**、正改善约 **27.5%**，A3 约 **3,041 次**、正改善 **4.5%**。按**全部调用含零改善**计算，A7 平均标量下降 **0.555 千分之一**，A6 **0.855**，A8 **0.762**；图把幅度补进成功率比较，说明“最常成功”与“单次平均降幅最大”不是一回事。不同动作面对的原始父解并未配对。

### M05｜状态 6：单位时间的标量改善

[图](../../research/d03/figures/analysis/d03i_state6_scalar_gain_per_minute.png) · [PDF](../../research/d03/figures/analysis/d03i_state6_scalar_gain_per_minute.pdf) · [原始轨迹归档](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/d03-v2-results-2026-09-24) · [运行来源](../../research/d03/data/analysis/provenance.csv)。把同一状态各动作的累积标量降幅除以包含选择、预算、构造、评价和状态更新的累积耗时，A7 约 **0.262 千分之一/分钟**、A6 **0.558**、A8 **0.466**。A7 选择最多但在这个状态并非按每次调用或每分钟收益最高；仅凭全局 A7 优势会隐藏局部效率差异。这个耗时仍是实际执行环境下的观测值，公平比较需同父解、同预算、同硬件负载重跑。

## 7. 下一步计划：每项针对哪张图的哪种不确定性

| 优先级 | 新实验或核验 | 保持什么不变、改变什么 | 能回答的问题与判据 |
|---|---|---|---|
| 1 | **同父解算子反事实** | 冻结 D03 的若干父解快照、实例、随机流、精确评价和单次预算；每个父解分别尝试 A1–A8，记录候选产生、可行、正标量改善、Archive 更新、运行秒数 | 对 H-01～H-06、M02～M05：分离“控制器爱选”与“同起点动作真有效”；按状态和实例做配对差/置信区间，不用不同动作自选样本均值作因果证据。 |
| 2 | **Q-learning 对照与等预算复现** | 同实例/种子、相同总精确评价数和墙钟限制，比较 Q-learning、Random、Bandit、固定动作混合；独立新的 evaluation seeds | 对 I-12、II-12、H 图：回答学习控制器是否带来流转—电费 Pareto 改善，而非仅发生 A7 集中；按实例报告超体积、Archive 质量和运行时间。 |
| 3 | **状态尺度与标签竞争敏感性** | 原 D03 保留作基线；另建版本，对六项 $D_k$ 的尺度、阈值和主导选择做预注册网格，先在冻结调度上离线重标，再用**新独立 seeds**完整重训 | 对 II-05～II-08、M01：区分“低于阈值”与“越线但输给别人”；比较标签稳定性、条件切换、动作区分度与优化结果。不以七类频率均匀作为唯一目标，不能把离线重标当重训结果。 |
| 4 | **定向生成器外部验证** | 明确 Normal、KV、Compressible 三族的参数范围及不同 profile 是否真的不同；在 calibration seeds 上冻结后用新的 evaluation seeds 验证 | 对 S02、I-03/04：分别检验目标条件占比、独立实例支持、非纯粹性、时间内转移，不覆盖本轮失败记录。KV 同时测编码实际跨实例率，Compressible 同时测活动区间压缩候选及原始分母。 |
| 5 | **Trigger 前后采样与轨迹身份** | 不改当前算法决策，增加候选进入 Trigger 前状态、通过/未通过标记、真实 `next_state` 和轨迹 ID 日志 | 对 S04/S05：分离总体分布、Trigger 选择偏差与同一解真实转移；原跨子问题 `condition_transition_matrix` 不再作转移证据。 |
| 6 | **A4/A5/A8 适用域与峰窗核验** | 分别记录正等待工序数、合法插入位置、可行候选数、被拒标量原因；A8 再记录峰窗集合、900 秒均值和原结束时刻约束 | 对 I/II-10/11、M02/M03：区分无候选与质量差。历史 D03 与现行 A4/A5 需**新版本独立重跑**，不能直接把前后热力图归于单一代码改动。 |
| 7 | **Archive 与耗时 profiling** | 选 Archive 峰值小/大且同规模的运行，用采样 profiler 拆解 exact evaluator、去重、Archive 更新、候选枚举 | 对 M02/M03：D03-I 的 Archive 峰值最高 **2,930**，五条 run 至少达 1,000；先证实瓶颈在哪里，再决定是否优化。性能改动须以 exact evaluator 等价回归保障语义。 |

**汇报边界：**D03 证明的是诊断运行完整、若干状态可到达、条件和动作行为不均衡以及具体失配位置。它没有随机化动作分配，也没有等预算控制器对照；所有机制解释中，公式可推出的数学性质与日志可直接观察的数值应和“仍需上述实验检验的因果贡献”分开讲。
