# D7：状态、算子、MacroSearch 与替换单因素筛选

版本：2026-10-09，待执行。用户授权以效果为先，直接双 CPU 开跑，不设置定时跟进。默认算法不晋级，D5/D6/E15/E16 不续跑。本批是筛选，不是独立确认或新颖性结论。

## 研究问题与冻结八组

当前 42 状态只保留最大相对阈值超限条件；D03 的条件访问高度偏斜。A3 区域迁移排序主要看负载，没有直接看电价。增加状态不能弥补动作本身没有相应变化。本批分别检验状态压缩、有限共存信息、候选预算、A3 排序及 MOEA/D 替换。

| 组 | 唯一干预 | 待验证问题 |
| --- | --- | --- |
| BASE | 原 F6：42 状态、固定 B6、birth 替换、原 A8 | 同批基准 |
| STATE6 | Preference × Progress，6 状态；删去条件标签 | 原条件是否增加学习负担而没有收益 |
| STATE84 | 原 42 状态 × 至少两项超阈值标志 | 丢失条件共存是否影响选算子 |
| B3 | 固定预算 3 | 更便宜的构造是否增加搜索吞吐和质量 |
| B10 | 固定预算 10 | 原 B6 是否投入不足 |
| SEQ | 既有 3→6→10 顺序预算 | 提前结束是否真的带来等时间收益 |
| A3_TOU | 仅 A3 排序先比较电价迁移代理 | 算子与 TOU 条件的动作不匹配是否关键 |
| RPERM | 既有随机邻域替换顺序 | 框架替换顺序是否限制收益 |

所有组均保留原 shared Q、五步轨迹、reward、触发、Polish、SSGS、exact evaluator、Archive、标量接受、初始化与 RNG manager。STATE6/84 有意改变 Q 值共享关系，不宣称各组逐步随机流相同；相同主 seed 和初始群体用于配对。A8 使用原 LEGACY，无 W/GW。SEQ 是原有政策，不重新启动 E15。

STATE84 的共存位在六个原 severity/threshold 比值中至少两个严格大于 1 时为 1；阈值与 dominant tie-break 不变。编码为原状态乘 2 加共存位；最多 84×8 Q 项。STATE6 编码为 Preference 乘 2 加 Stagnating。

A3_TOU 在当前操作开始时间和原 duration 上，对目标机器 active-minus-idle 功率与其区域 TOU 分段积分求代理费用。比较同一 job 两操作迁移的代理增量，随后按原负载/KV/索引键打破平局。操作×机器代理每次 A3 构造缓存一次，计入构造墙钟。代理忽略区间并集重叠、迁移后的新开始时间与 demand，因此不是真实边际 Bill。全量修复与 exact 接受仍是唯一 ground truth；原 2B 候选池和 B 抽样额度不变。

## 固定矩阵与预算

正式独立 base 共 32：50 jobs 的 1205001–1205016 与 100 jobs 的 1215001–1215016。每 base 同时 H/T，算法 seed 1101/2202，八组，1024 run / 128 八组块。base 位置模 2 分配服务器，同 base 的 tariff/seed 和每块八组均在同一 host。每节点 512 run / 64 块。

50/100 jobs 每 run 算法墙钟分别 900/1800 秒；两节点各 24 solver，总 384 solver-hour，理想预算约 8 小时，真实 ETA 加上 overshoot、校验导出与调度。八组采用 8 条偶数阶 Williams 序列，每个位置和一阶前序均衡；块顺序按 seed 20261009 固定打乱，再长块优先。先预跑技术验收，不能依据预跑质量改变矩阵。

技术 pilot：50 jobs 的 1225001–1225006、100 jobs 的 1235001–1235006，seed 6606/7707，八组；384 run / 48 块，30/60 秒。与正式及旧数据分离。仅检查正确性、吞吐、资源与运行入口，不用于正式统计、算法选择或追加样本。

## 观察与预先分析规则

主指标：等时间 normalized HV。每个 base/tariff 的八组和两 seed 共同经验非支配并集定义参考前沿，归一化按并集 ideal/nadir；轴跨度不大于 1e-12 时除数固定为 1，否则为跨度，参考点 (1.1,1.1)。近似集不为距离指标裁剪，HV 仅纳入两个坐标均不大于 1.1 的点；不再除以参考矩形面积。32 base 为独立单位，先在 base 内平均 H/T 与两 seed，再做配对分析；禁止把 1024 run 当独立样本。七个 variant−BASE 的 HV 属同一个主检验家族，双侧 100000 次 sign-flip 与 Holm；20000 次 base cluster bootstrap 95% CI，随机 seed 2026100907。

IGD+ 的七项比较作为次级家族单独 Holm；普通 IGD、双向 coverage、各目标极值、anytime 与 seed/规模/tariff 子组只作补充。不能用事后有利子组或子组显著替代整体主家族，也不能把不显著当等价。候选方向即使通过筛选，也需新独立样本确认，不直接晋级默认。

每 run 仅存有界 state×action 汇总、每算子 calls/attempts/effective/feasible/accepted/reward/seconds、construction_seconds/exact_evaluations/proposals，预算×有效候选直方图、Q/访问/更新/选择、五个 anytime 截面。保留总吞吐与构造成本。状态访问随算法轨迹变化，线上条件相关性不是同状态反事实；若需证明状态与动作匹配，后续应冻结同源状态池，对所有动作和预算做独立 counterfactual 回放并留出 base 验证，本批不冒充该因果检验。

## 验收、续跑与交付边界

新独立运行源码 commit 和 canonical source hash 冻结；同一版本通过本地 full gates、默认 replay、GitHub 双 Python CI 与双节点技术 pilot 后才启动正式。每 worker 最后独立 exact 复验最终 population 与 Archive，八项产物 SHA、输入/初始/config/source/seed/host 验真后事务发布完成标记。原 D5/P2 validator 继续验证。资源守卫、互斥锁、stop.request、failure 与 quarantine 保留；不自动重试算法失败、不声称内部搜索 checkpoint。

完成项以完整块/完整产物可移植；中断 run 重做同一任务，跨机器部分块一起重做，原尝试留 quarantine。计算结束仍须实际 worker 0、supervisor 退出、exit complete、全部 1024/128 原 validator 与 hash 通过，才可验收。

本轮交付止于技术验收和真实开跑，发布启动证据/源码/矩阵/完整 pilot 并记录预计结束。用户要求不设置定时任务；正式结果未产生，质量统计与完成交付留待后续明确处理。
