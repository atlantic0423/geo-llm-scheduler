# R1 区域增量评价：条件可分结构与 exact 对账

状态：研究路径，软件默认算法不变；在线 R1 尚未实现。实现见 `src/geo_llm_scheduler/evaluation/regional.py`，对账脚本 `scripts/reconcile_r1_incremental.py`。

## 结构

当前模型下，每个 Job 的 Prefill 与 Decode 位于同一区域，实例只属于一个区域，批次结束时刻 $H$ 为最后一个 Decode 完成时刻。对一个可行完整排程，记区域 $r$ 的 Job 集为 $J_r$，其最后完成时刻为 $E_r$，则 $H=\max_r E_r$，且

$$
F=\sum_r F_r,\qquad F_r=\sum_{j\in J_r}\left(C^{D}_j-a_j\right),
$$

$$
\mathrm{TOU}=\sum_r\left(T_r^{\mathrm{core}}+\frac{P_r^{\mathrm{idle}}}{3600}\int_{E_r}^{H}\pi_r(t)\,dt\right),
$$

其中 $T_r^{\mathrm{core}}$ 只依赖区域 $r$ 在 $[0,E_r)$ 上的功率，$P_r^{\mathrm{idle}}$ 为区域内实例空闲功率之和。固定 900 秒窗口 $k$ 的平均功率为核心部分加空闲尾段 $P_r^{\mathrm{idle}}\,|[E_r,H)\cap[900k,900(k+1))|/900$；需量费为区域费率乘全部窗口最大值。尾段窗口值只可能是边界窗口的核心值加部分空闲、整窗空闲 $P_r^{\mathrm{idle}}$ 或不超过它的末窗，因此峰值可由缓存的前缀最大值在常数时间内得到。

结论：给定 $H$ 时两个目标按区域可加；区域间只通过标量 $H$ 耦合，修正项是闭式尾段积分。这是条件可分，不是无条件区域独立；若引入跨区域共享链路、全局功率约束或开关机决策，结论需要重新证明。

## 实现

- `RegionLedger`：每个解每个区域一份，保存 $J_r$、$F_r$、$E_r$、$P_r^{\mathrm{idle}}$、核心 TOU、核心窗口和前缀峰值。核心项由 exact 评价器自身的功率分段计算，窗口逐比特等于 exact 的对应窗口。
- `evaluate_with_ledgers`：与 `exact.evaluate` 共用校验和分段，在同一遍内返回相同的 `EvaluationResult` 与账本；测试断言二者完全相等。
- `splice` + `combine`：同任务集合的区域块换入后，不解码直接预测目标。预测只用于筛选；可行性、标量接受与档案更新仍以完整 SSGS 解码和 exact 为准。

## 对账协议

输入为 `outputs/r1_live_20261112` 回收的 120 次 seed 17 观察记录。对每个采样子代、接收者和相同非空任务集合的区域，同时计算账本预测与参考路径（拼基因型、SSGS 解码、exact），比较 Flow、TOU、各区域需量、$H$、方向标量、方向改善、救回及当时档案 HV 正增量判定，并分别计时。供体另测 exact 与融合评价耗时，作为每个子代建账本的边际成本。结果写入本地 `outputs/r1_incremental_20261113/`。

不声称：在线净收益、等墙钟 HV 改善或首创性。
