# D4 两项离线原型：证书真实门控成本与全窗口有限重插（v1）

用户于 2026-10-07 明确授权两项同时在当前两 CPU 容器运行。研究分支独立，默认 MOEA/D/A8/Q/Archive 不变，不组合两个 treatment，不重启或调整 D3。此处冻结的是实验语义，不是正式算法采用决定。

## 来源与组别

复用已验真的 D4/cfeb369 全部 32 base、192 来源、1728 panel、10368 来源调用；它们已被分析，全部视为开发集，不把原 split 字段冒充新留出。50/100 jobs、H/T、三个算法 seed、阶段和偏好均保留，包括重复与空调用。

证书独立 campaign：每来源调用四个新配对调用 seed，每 seed 四个新鲜 operator invocation 时序重复。四组 LEGACY 原 A8；REFERENCE 新独立 recipe 流、无门控；GUARD_ALL 全非空 recipe；GUARD_SINGLE 仅成员数为一。每次调用懒建一次局部全窗口缓存，随后复用；无跨调用全局缓存。报告实际 CPU 与墙钟成本及单次分布，预先随机化臂顺序，audit 不计入 construction timing。REFERENCE 与两门控必须 proposal、attempt budget、最终父 RNG 及成员序列完全一致；原共享 RNG 的 LEGACY 作为随机流改变的独立对照，不强求与新流相同。

位置独立 campaign：每来源调用八个新 seed、每 seed 一次调用；REFERENCE 原有限 first-feasible 修复（新流），RANDOM_SCORED 对同一最多六个随机位置逐一判资源与打分，仍选首个可行；PEAK_RANK 相同采样额度，选全窗口 partial 最大功率最低的位置，稳定随机顺序破同值。操作数最多八、两次 singleton、最多六个 recipe/B=3 等原约束保留，不增加 beam/restart。随机 scored 控制必须与 REFERENCE 输出相同；PEAK_RANK 的后续资源路径可能改变，匹配额度不冒充匹配 CPU 时间。

## 正确性与边界

construction 计时关闭 trace；计时结束另执行同 seed 带 trace 的完整复验，二者 proposal/budget/父 RNG 必须一致，防止诊断产物开销被混入真实门控成本。时序重复之外的审计调用不计为独立样本。

证书保留原 D4 active-union、idle、固定 900 秒尾除数、未选原尾工序固定 horizon 假设及保守浮点余量。前置否决的 counterfactual 完整修复由同 seed 的 REFERENCE 提供；每个非空 recipe 的证书用独立非选工序积分核对。所有完成的重插及 proposals fresh exact 验证可行、未选 timing 不变、不延长原 horizon。任何 false deny、源码/输入/产物 hash 不一致或 RNG 变化作为实际失败保留，不自动算法重试。

partial 全窗口评分始终积分到原全局 horizon，包括暂未插回的 Decode 之后的 idle，不使用缺工序的临时 horizon。未插回工序不预测未来贡献，所以只是构造启发式；最终仍经过原全窗口 peak gate，再按完整 exact Bill/Flow 观察。该评分与完整 exact、独立残余积分及尾窗/遮蔽反例测试交叉验证。

每个配对比较用各臂候选池联合临时 ideal、原 panel maximum 与 weight；保留零增益来源。按 32 base 等权聚类总结，其他层分组描述，不把 recipe/repeat 当独立 base。不以 deny 率或部分功率下降直接宣称创新、HV、在线净收益。共享物理 CPU/D3 重叠负载影响墙钟，CPU 成本也需谨慎；门控正净收益仍需空闲时序复验和新 base 在线等时间消融。

## 执行、恢复与交付

先完整本地质量门、默认 deterministic replay、PR 双 Python CI；再源与输入 SHA 核对、两机短预跑、实际 RSS/耗时校准与原 validator。预跑完整 raw/清单/源码 bundle/复现说明发布并回读后，再正式启动。门控两个 worker 在 node0；位置四个 worker 在 node1；D3 原并发保留。不按可见 128 线程盲目满载。

任务/unit 原子 checkpoint、SHA sidecar、完整 key marker、同 key 同 host、OS 排他锁与停止派发。进程中断复用已验真单位，从未完成单位重做；不声称恢复内部函数状态。未提交产物移 quarantine；实际失败停止新任务，其他在途安全收尾；stop.request 保留直至用户明确续跑。每个 key 2 小时 watchdog；心跳保存实际进程/RSS/进度。资源不足停止新派发，绝不触碰 D3。

正式两项预计由目标 pilot 校准；这是来源调度离线回放，计算规模远小于重跑完整在线 MOEA/D，不能套用 D4 原来源的 16–20 小时。完成后完整大小/SHA 清单、raw 及复现包走 Release，分析报告走独立 PR，Notion 04/05/06/09 与总览同步实际状态。开跑仅说明执行已启动。
