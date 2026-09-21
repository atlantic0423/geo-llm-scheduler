# State severity 唯一实现约定

状态：用户确认的 experimental implementation convention；不代表研究有效性已验证。时间为秒，单位遵守 Coding Contract。所有统计来自当前完整可行 schedule，assignment 只读 Genotype.MS。

## Resource 与主动推迟

SSGS 记录资源等待。任何 accepted timing-only move 后全量重新计算：冻结其他操作，移除当前操作，在其当前前序 EST 后找到最早资源可行开始时刻，不改变实际 schedule。

$$
w_o^{res}=S_o^{min}-EST(o),\qquad w_o^{intent}=S_o-S_o^{min}.
$$

$$
D_{Resource}=\min(1,\sum_o w_o^{res}/(\sum_o p_o+\epsilon)).
$$

A4/A5 的 Flow-wait 仍使用当前开始时刻减 EST，不与资源等待混淆。

## KV

$$
D_{KV}=\frac{\sum_i\tau_i^{KV}\mathbf1(m_i^P\ne m_i^D)}{F_{flow}+\epsilon}.
$$

## Region

令区域资源 workload 为操作 duration 与对应 demand 乘积之和，容量为该区域所有实例容量之和。

$$
L_r=\max\left(\frac{\sum_{o\in r}p_oq_o^C}{\sum_{m\in r}Q_m^C},
\frac{\sum_{o\in r}p_oq_o^V}{\sum_{m\in r}Q_m^V}\right).
$$

$$
D_{Region}=\begin{cases}0,&\max_r L_r=0,\\
(\max_r L_r-\min_r L_r)/\max_r L_r,&\text{otherwise}.\end{cases}
$$

## TOU

只在当前 batch 的非零长度区间内取 tariff min/max；active union 记为 $A_m$。常价区域归一化值为零。

$$
\widehat c_r(t)=\begin{cases}0,&c_r^{max}=c_r^{min},\\
(c_r(t)-c_r^{min})/(c_r^{max}-c_r^{min}),&\text{otherwise}.\end{cases}
$$

$$
D_{TOU}=\frac{\sum_m\int_{A_m}P_m^{active}\widehat c_{r(m)}(t)\,dt}
{\sum_mP_m^{active}|A_m|}.
$$

无活动能量时取零；不重复累计重叠操作活动能量。

## Demand

覆盖当前 batch 的所有固定窗口均参与均值，包含零负载窗口；末窗分母仍为 900 秒。

$$
d_r=\begin{cases}0,&P_r^*=0,\\
(P_r^*-\operatorname{mean}_b\bar P_{r,b})/P_r^*,&\text{otherwise},\end{cases}
\qquad D_{Demand}=\frac{\sum_r\kappa_rd_r}{\sum_r\kappa_r}.
$$

全部费率为零时整体取零。

## Compressible

与 A7 共用当前 activity/resource 关键时刻生成器，冻结其他操作，只保留合法移动。无 planning upper bound，也不施加 A8 horizon。

$$
g_o=\max_{t\in K_o}\max(0,|A_m(x)|-|A_m(y_{o,t})|),\qquad
D_{Compress}=\frac{\sum_og_o}{\sum_op_o+\epsilon}.
$$

空候选集的最大值取零。Region/TOU/Demand/Compressible 范围为 $[0,1]$。空 batch 的所有 severity 均为零。正容量、非负费率、有限参数与 tariff 覆盖由输入验证。

六个独立阈值工作值为 0.20，全部不超过阈值时为 Normal，否则选最大 severity/threshold，等值按 Resource、KV、Region、TOU、Demand、Compressible 顺序确定。

