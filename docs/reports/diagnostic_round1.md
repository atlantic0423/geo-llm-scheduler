# D01 较大规模算法行为诊断

状态：Stage S、Stage P 和 generator-only calibration 已完成；Stage M 的 63-run 主矩阵待运行。本文档在主矩阵完成后补入最终 state/action/runtime/Archive 结论，目前只记录冻结的方法与主运行前证据。

## 固定设计

- 规模：50 Jobs，理由见 [规模校准](diagnostic_scale_calibration.md)。
- 主矩阵：7 scenario families × 3 instance seeds × 3 algorithm seeds，共 63 runs。
- 每条 run：30 generations、population 100、neighborhood 20、Q-learning、preference trigger、固定预算 6、5 个 RL steps、启用 A1–A8 与 RightShiftPolish。
- Instance seeds：1、2、3；algorithm seeds：101、202、303。
- D5 映射：seed 1 为 `single_peak`，seed 2 为 `multi_tied_peak`，seed 3 为 `mixed_peak`。
- 所有单位为秒、kW、kWh、CNY。场景只改变实例生成参数，不改变 StateExtractor、severity 阈值、Trigger、reward、MacroSearch 或算子语义。

## 最终场景参数

| 场景 | 关键结构 |
|---|---|
| D0 Balanced | release 0–3600 s；低 Compute/VRAM；平坦 0.45 CNY/kWh |
| D1 Resource | 75% 在 0–300 s 释放；Compute 0.45/0.60/0.75；VRAM 4/6/8 |
| D2 KV | KV 600/900/1200 s；短 P/D；低资源压力 |
| D3 Region | 区域恒定电价 0.25/0.75/1.25 CNY/kWh；低资源压力 |
| D4 TOU | 900 s 边界的 0.25/1.25/0.20/0.45 CNY/kWh；长操作 |
| D5 Demand | 三种峰形；Demand rate 20 CNY/kW；低到中资源压力 |
| D6 Compressible | release 0–7200 s；长 Decode；低资源压力；平坦费用 |

完整参数、seed、variant、实例路径和 SHA-256 以 `instances/diagnostic/manifest.csv` 为准。

## Pilot 与校准

初始 pilot 使用 7 × 1 × 1、每条 5 代。随后进行了两轮仅修改 generator 参数的校准，没有修改算法或阈值。最终 pilot 的目标 condition 结果如下：

| 场景 | 目标 condition | Dominant incidence | 目标 continuous severity mean | 结论 |
|---|---|---:|---:|---|
| D1 | Resource | 100.0% | 1.000 | 达标 |
| D2 | KV | 14.1% | 0.159 | 达标 |
| D3 | Region | 99.3% | 0.893 | 达标 |
| D4 | TOU | 72.5% | 0.425 | 达标 |
| D5 | Demand | 69.8% | 0.483 | 达标 |
| D6 | Compressible | 22.1% | 0.289 | 达标 |

D0 未形成 Normal 主导，而是在 Region 与 Demand 之间分布。它仍保留为无高资源、无 KV、无 TOU 的比较基线；不再为获得更“好看”的 Normal 覆盖继续调参。D5 相对 D0 的 Demand incidence 提高 16.6 个百分点；D6 相对 D0 的 Compressible incidence 提高 22.1 个百分点。D6 pilot 记录到 9,314 个正压缩 move、859 个涉及 operation；D5 同时出现 singleton、coalition、repair success 和 strict peak reduction。A1–A8 在 pilot 中均被选择和精确评价。

## Instrumentation 与产物

每个 RL step 直接保存 run/scenario/seed、generation/subproblem/progress、解码后的三维状态标签、六项 continuous severity、动作选择方式、预算、construction/proposal/exact/feasible/accepted/reward、目标变化、Archive 变化和计时。A7 另存正压缩 operation/move 与 $G_{pack}$；A8 另存目标 Region、原峰值、并列峰数量、singleton/coalition、repair success 和 strict peak reduction。

主矩阵完成后由聚合器生成 `run_summary.csv`、`state_visits.csv`、三个 state-action 矩阵、`operator_diagnostics.csv`、`a7_a8_diagnostics.csv`、`scenario_condition_incidence.csv`、`runtime_breakdown.csv` 和 `archive_diagnostics.csv`。这些文件是后续热力图的直接输入。

## 待主矩阵回答

42 states 和 336 个 state-action cells 的实际覆盖、长期稀疏状态、各动作成功率和 reward、最耗时三个模块、Archive 增长风险、问题分级与正式实验 blocker 将在 63 runs 完成后填写。当前不根据单 seed pilot 作算法优劣结论。
