# 初始化第二步：固定基因的时序空间与结构对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-11
- Verification Status: PLANNED（效果未验证；正式启动以运维验收为准）
- Version Label: initialization_step2_v1

用户授权第二步直接启动，确认进入正式计算后返回 ETA 并结束。本轮不设置定时任务、不运行第三步、不改变默认初始化或已有批次。

## 研究问题与冻结对照

第一步混合初始化优于随机和单启发式，但 native-T 费用端仍有缺口。第二步从相同原混合初始个体出发，分辨有界 timing-only 与 assignment/order 改动的诊断收益，不预设整程改善。

| 组 | 每个原个体追加 exact 上限 | 候选语义 |
|---|---:|---|
| RAW | 0 | 原 SSGS 基因、时序与 exact 全量回放 |
| TIME16 | 16 | 原 RightShiftPolish 枚举的有界前缀 |
| TIME64 | 64 | 同一枚举的更长前缀 |
| STRUCT16 | 16 | 同源 A3 电价排序区域迁移与 A5 P/D 联合前移交替候选 |
| STRUCT64 | 64 | 同一最大结构池的更长前缀 |

所有候选均从同一个冻结原个体出发，不连锁搜索。TIME 使用原 Polish 枚举顺序和关键时刻，至多执行一次 Prefill 右移，MS/OS、Decode 和其他工序不变；不存在预算内改善不证明其他时序改善不存在。STRUCT 两组各构造最多32个 A3 和32个 A5 候选，使用相同逐个体派生 RNG 和最大池，交替去重后取16/64前缀。池不足不补重复，记录实际 exact 次数、枚举/构造与 SSGS 耗时；这只是匹配追加 exact 上限，不冒充等时间或完全相同实现成本。

每个原个体保留一个代表：要求可行、Flow 不增加、Bill 严格降低，否则保留原解；Bill 最小，容差内按 Flow、稳定身份打破并列。TIME 的 Flow 必须严格数值不变。完整所有可行 exact 候选进入 explored Archive；主比较只用保留的100成员种群前沿，额外候选 Archive 单列诊断。不使用MOEA/D权重选择，不对扩大候选数的好处作免费比较。

## 输入、样本与版本

正式沿用第一步的36个诊断 base、72个 native H/T 输入、50/100/200任务、六压力场景，使用预先指定的两个初始化 seed11001/11002，每个 P100。共144同主机五组块、720种群、14,400冻结原个体、72,000保留代表；追加 exact 上限2,304,000（另加基线回放、独立复验和 Archive 最终复验）。两节点保持第一步相同host，逐byte复制输入/父种群并绑定原完成marker、source/input/population hash。

这是第一步诊断集合上的机制比较，不是新留出效果确认。独立技术pilot复用另外18个技术base、36 native输入、seed11901，180种群，3,600父个体；不混正式统计。pilot只查看正确性、时间、RSS、磁盘和吞吐以校准 ETA，不根据目标改善筛样或调预算。正式两seed在pilot前固定。

原初始化source e57d385保持冻结；新预处理模块和入口在独立研究分支。runner文件SHA与全部module canonical、Git commit、实例、原个体、spec/manifest等共同冻结。原第一步、D8、默认算法和历史源不修改。

## 工程、恢复与资源

原个体在worker中fresh SSGS与exact回放一致；每个追加候选、最终代表和explored前沿均独立exact复验。测试覆盖截断、前缀/RNG回放、Flow/基因/Decode冻结、原Polish未截断等价、结构/时序计数、错误exact与产物篡改拒绝。

各host最多24solver，总48；启动前实读服务器台账、身份、cgroup资源与全部实验进程。原6GiB/物理20GiB守卫保留；个人目录占用单列，不能用物理free冒充用户额度。只对已完成本批结果建议释放洁净文件缓存，字节/mtime不变，原件不删除。每个原个体写进度心跳，独占supervisor锁；停派发后完成当前块安全收尾。失败保留证据、两个shard停新派发，不自动算法重试。完成组可用完整marker续跑；未完成组保留原件，另行人工恢复，不声称内部搜索checkpoint。

全144块/720组、产物hash、input/parent/source/seed/host、最终独立exact、真实worker/supervisor0及exit complete通过才验收。不按目录数或预计时点宣布完成。原结果与历史记录不覆盖。

## 统计与解释边界

36 base为独立单位，先平均 H/T 和两 seed 的配对差。按50/100/200分层20,000 cluster bootstrap、100,000 sign flip，统计 seed2026101102。每个native输入全部正式五组×两seed的保留代表构成共同经验唯一ND union，ideal/nadir归一化，退化轴span1，HV参考点(1.1,1.1)，IGD+不裁剪；它不是真实Pareto前沿。

主家族：TIME16/TIME64/STRUCT16/STRUCT64各对RAW的HV和IGD+共8项联合Holm。次家族：同预算TIME−STRUCT两对、TIME64−TIME16与STRUCT64−STRUCT16两对，各HV/IGD+共8项另行Holm。报告所有预定比较，不把种群或候选当独立样本，不把未显著当等价。

补充指标：Bill/TOU/各区域Demand、Flow、ND和双向覆盖、两端差距、accepted比例、实际exact/ssgs/构造/刷新diagnostics/独立复验成本、预算截断与池不足。场景×规模每格仅2base，以描述为主，不事后筛选或追加到显著。时序前缀有原枚举位置偏倚、结构对照仅两个已定义邻域；阴性结果不能推广为不存在可行改进。

后续计入初始化成本的整程等总wall与新留出确认另行安排。本轮不执行进化、RL、MOEA/D邻域替换、方向选择、配额优化或A3×新初始化整程交互。
