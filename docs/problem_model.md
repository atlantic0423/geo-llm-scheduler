# 问题模型与输入

Job 的 profile 静态、确定且同阶段同构，不随实例改变；实例 phases 控制阶段可用性。操作编号 $2i$ 为 Prefill，$2i+1$ 为 Decode；实例和 Region 索引均从零开始。

$$
S_i^P\ge r_i,\qquad S_i^D\ge C_i^P+\tau_i^{KV}\mathbf1(m_i^P\ne m_i^D),\qquad C_o=S_o+p_o.
$$

P/D 必须同 Region。每个实例上每个事件区间的 Compute 需求总和不超过 1，VRAM 增量 working-set 总和不超过可调度容量。半开区间 $[S_o,C_o)$ 在 completion 时立即释放资源。

$$
F=\sum_i(C_i^D-r_i),\qquad T_{\mathrm{end}}=\max_i C_i^D.
$$

实例 active union 为 $A_m$；同实例多个并发操作只产生一次 active power。

$$
P_m(t)=\begin{cases}
P_m^{active},&t\in A_m,\\
P_m^{idle},&0\le t<T_{\mathrm{end}},\ t\notin A_m,\\
0,&t\ge T_{\mathrm{end}}.
\end{cases}
$$

$$
C_{\mathrm{TOU}}=\frac1{3600}\sum_r\int_0^{T_{\mathrm{end}}}c_r(t)\sum_{m\in r}P_m(t)\,dt.
$$

$$
\bar P_{r,b}=\frac1{900}\int_{900b}^{900(b+1)}\sum_{m\in r}P_m(t)\,dt,\qquad
C_{\mathrm{dem}}=\sum_r\kappa_r\max_b\bar P_{r,b}.
$$

费用目标为两者之和。尾窗仍除以900，全部并列峰可识别。global-last Decode 提前可能改变所有 Region 尾部。空 batch 目标与 severity 为零；搜索需足够不同 genotype 填满种群。

## JSON v1

完整例子见 [smoke.json](../examples/smoke.json)。顶层 schema_version=1、units、jobs、regions、instances。Job 包含 name/release/prefill/decode/kv_delay；profile 包含 duration/compute/vram；Region 包含 name/tariffs/demand_rate；tariff 包含 start/end/price；实例包含 name/region/vram/idle_kw/active_kw/phases。

loader 支持 s/ms/h、kW/W；尽管字段名为 idle_kw/active_kw，原始值仍按 units.power 转换。货币统一使用人民币，JSON 中以 ISO 4217 代码 `CNY` 表示；费用率始终为 CNY/kWh、CNY/kW。其他货币、当地时区需在外部按明确汇率/时区预处理。`save_instance` 输出 s/kW/GB/CNY。

数值需有限，容量为正，需求/价格/费率非负；tariff 从零连续覆盖。若实际 batch 超出 tariff，exact 报错，不隐含延长价格或加调度 horizon。用户需提供足够长的价格时间轴。
