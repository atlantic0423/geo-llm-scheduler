# D02 Mixed-instance Validation Protocol

状态：**预注册式实现协议**。D02 用于在普通综合随机实例上复核 D01 中观察到的 Q-learning state-dependent action behavior。D02 不是 D01 的第八个 scenario；两套数据只并列比较，不合并 aggregation。

## 固定运行矩阵

- Jobs：50。
- Regions：3；每个 Region 3 台同规格 Serving Instance。
- Population：100。
- Method：full。
- Controller：qlearning。
- Budget policy：fixed；fixed budget：6。
- Generations：30。
- Instance seeds：11、22、33、44、55。
- Algorithm seeds：101、202、303。
- 总计：15 runs。

算法机制、42-state、severity threshold、Q-learning 参数、A1–A8、MacroSearch、Trigger 与 $L_{RL}$ 均保持不变。

## Mixed generator

生成器不指定目标 DominantCondition。以下范围在观察任何 D02 状态分布前冻结，完整结构也写入 `instances/diagnostic_mixed/manifest.csv` 的 `parameters_json`：

- Release：80% 从 0–5400 秒的 60 秒网格均匀抽取；20% 从 900、2700、4500 秒附近的 $\{-120,-60,0,60,120\}$ 秒偏移抽取，形成轻度 burst mixture。
- Prefill duration：60、120、240、360 秒。
- Decode duration：180、300、600、900 秒。
- Compute：0.15、0.30、0.45、0.60。
- VRAM：2、4、6 GB。
- KV migration delay：0、30、60、120 秒。
- 每个 Region 的三台实例均为 VRAM 16 GB、idle 10 kW、active 100 kW。
- 三个 Region 使用相同中等 TOU：0–900 秒 0.42 CNY/kWh、900–1800 秒 0.72、1800–2700 秒 0.35、2700–3600 秒 0.62，此后 0.48，并延伸到保守 schedule horizon 之后。
- Demand rate：每个 Region 5 CNY/kW。

每个实例必须通过 `validate_problem()`。生成后不根据 DominantCondition incidence 调整参数；mixed 状态分布不均衡属于实验观察。

## 数据隔离与统计

D01 raw outputs 位于 `outputs/diagnostics/main/`，D02 位于 `outputs/diagnostics/mixed/`。聚合目录分别为 `outputs/diagnostics/aggregate/d01/` 与 `outputs/diagnostics/aggregate/d02/`。

State × Action selection 先跨同一数据集全部完整 runs 累计计数，再计算：

$$
P(a\mid s)=\frac{\sum_{runs}N_{run}(s,a)}{\sum_{runs}N_{run}(s)}.
$$

不简单平均各 run 百分比。低于 10 个样本的 cell 显示样本量并加斜线；State 图拆为 Flow、Balanced、Electricity 三个 14×8 panel。D01 与 D02 的 coverage、condition、selection、success 和 reward 只做 side-by-side 比较。

## 恢复与完整性

`scripts/run_diagnostic_night_pipeline.py` 按完整 run identity 检查 `summary.json`、`trace.jsonl`、`qtable.json`、`diagnostic.json`、`rl_steps.csv` 与 `rl_steps.jsonl`。只有 identity、30 generations、generation-limit termination 和全部非空可解析 artifacts 同时满足时才跳过。失败或截断目录保留到 incomplete archive 后重跑，不删除已完成 run。

最终图与报告均标记为 diagnostic / preliminary behavioral visualization，不作为正式论文性能结论。
