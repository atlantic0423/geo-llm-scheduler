# 算法流程

## 初始化与 MOEA/D

FCFS/SPT/LPT/Random OS 各配 MinLoad/Random，加 KV-aware、Energy-aware，共十类初始化近似均分配额。扰动后 genotype 去重；缺额由随机方法补足，超过明确尝试上限报错。全部个体 SSGS/exact/Archive，随机打乱后对应均匀权重。

每代随机遍历子问题；按概率选择邻域或全种群的两个父代。job-linked crossover 保持 P/D assignment 来源一致；mutation 为 OS swap、同 Region 单实例变更、Job 整体 Region/path 变更。

$$
\widetilde f_m(x)=\frac{f_m(x)-z_m^*}{\max(z_m^{max}-z_m^*,\epsilon_{norm})},\qquad
g_i(x)=\max_m\lambda_m^i|\widetilde f_m(x)|.
$$

ideal 由所有可行 exact 更新，maximum 只取当前 population。MacroSearch 固定开始时 maximum，全部 raw objectives 完成后共享比较上下文。邻居替换使用邻居自身权重，最多 replacement_cap 个；后续阶段重算 population maximum。

## Trigger、状态与轨迹

默认 Trigger 检查 objective 的 reciprocal-weight 方向所属 preference 与子问题一致，再检查相对 scalar 质量容忍度。近零向量跳过角度门；strict/always/fixed 为消融。

State = 3 Preference × 7 Dominant × 2 Stagnation = 42。Dominant 为 Normal 或六种 severity；最大 severity/threshold 超过1才激活，平局按固定顺序。Preference 按子问题索引三等分。stagnation 按当前子问题处理后是否严格改善更新。

每 run 一个共享 42×8 Q-table。epsilon 为探索概率，按代数或时间进度线性衰减。Random 不学习，Bandit gamma=0。每条轨迹最后一步使用 terminal，未来价值为零；其他步骤从 next state bootstrap。terminal 不重置共享 Q-table。RightShiftPolish 在 episode 外，不贡献 reward 或未来价值。

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma(1-d)\max_{a'}Q(s',a')-Q(s,a)\right],
\qquad d=\mathbf1(\ell=L_{RL}-1).
$$

每步从当前 incumbent 独立构造同源候选，最多 B 个不同完整 exact，统一上下文选 best。严格 scalar 改善才接受；timing-only 接受后刷新全部 diagnostics，提取 next state。

$$
r=\begin{cases}
-1,&\text{无可行候选},\\
0,&\text{无改善},\\
\min(10,100(g_i(x)-g_i(y))/(g_i(x)+\epsilon)),&\text{改善}.
\end{cases}
$$

L_RL 步后一次 Polish，不产生 reward/next state，再邻域替换。所有 raw exact candidate 独立尝试 Archive。

## 停止

generations 为代数上限，seconds 为可选 wall-clock 软上限，包含初始化。每个子问题前检查，正在运行的候选/轨迹不被中断，所以可能超出一个轨迹时间。固定代数用于 replay；正式等时间比较报告实际 elapsed 和超时幅度。
