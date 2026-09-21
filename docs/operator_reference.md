# 算子参考

A1–A6 structural，A7/A8 timing-only。接口 operators/base.py；执行 macrosearch/search.py。一次 invocation 的候选全部来自同一 incumbent。

| Action | 改动边界 | 候选依据 |
|---|---|---|
| A1 Phase-Reassign | 单 operation MS，同 Region | resource-wait 降序与目标 workload |
| A2 PD-Path-Reassign | 一个 Job 同 Region P/D 路径 | Resource/KV 条件，搬移后负载/KV |
| A3 Job-Region-Relocate | 一个 Job 整体 Region/path | 假设搬移后 Region imbalance、路径负载、KV |
| A4 Flow-Congestion-Insert | 单 operation OS 前插 | 正 Flow-wait，保持 P-before-D，round-robin |
| A5 PD-Coupled-Insert | 同 Job 两次 occurrence 前插 | 两侧等待较小值，合法 paired insertion |
| A6 Random-Job-LNS | 部分 Job MS/OS | 比例默认 .1，至少2且不超过N，随机合法重插 |
| A7 Active-Pack | 单 operation start | 正 active-union gain，合法有限关键时刻 |
| A8 Peak-Coalition | 一个 Region 的少量 starts | 原并列峰 singleton-first 与支撑组 coalition |

A1–A5 从按规则生成的2B pool无放回抽取B；A6直接生成不同候选并限制尝试数。A4/A5 使用 lazy bounded round-robin，达到2B不同候选后立即停止；A5 从合法位置对空间做有界无放回采样，不物化完整二次位置对集合。日志分别记录 raw moves considered、2B qualifying pool 和最终 proposals。结构候选均 rebuild。

## A7

与 Compressible 共用 scheduling/timing.py。从当前实例 starts/ends 和减 duration 点，加 release/precedence 边界生成有限位置，无 planning upper bound，也不施加 A8 horizon。Prefill 受冻结 Decode 限制。保留合法且正压缩候选。

每 operation 提取最大 gain 和较小 Flow 增量代表；按 gain最大/Flow增量最小做非支配排序、crowding；主代表优先形成2B pool。零跨度 crowding 贡献零，并列 seeded。代理不替代 exact。

## A8

按需量贡献加权选择正费率且有活动 workload 的 Region，检查全部原并列峰。由 elementary segments 的 active sets 计算移除潜力。先尝试少量 singleton（2）；未发现严格削峰时构造累计正边际 coalition（最多8），允许小贡献累计。

整体移除成员，非成员冻结。P-before-D、合法窗窄者优先单路径重插；关键点包括资源/TOU/900秒边界、原成员事件、减duration、端点及相邻中点。每成员最多抽6个位置，取第一个可行，失败整体 rollback，无 beam/helper/barrier。

不超过原 batch horizon。repair后验证目标 Region 全部窗口最大功率严格下降，不能只迁移峰。最多2B条recipe，失败/重复/未削峰不占 exact。合格不同候选直接 exact，不足不凑满。T_end提前必须重算全Region尾部。搜索有界，不提供完备性保证。

## RightShiftPolish

整条trajectory后一次，不属于action，不占B。枚举所有可右移 Prefill 的资源/TOU/Demand关键点，Decode及其他操作冻结。所有资源可行候选exact并尝试Archive；只执行一个严格降低总费用的移动，费用相同优先较小右移距离。Flow不变，分别记录候选时刻数/exact数/接受数/时间。
