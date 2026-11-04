# LSV2：P3外置快速右移研究臂

状态：本地实现验收通过、等待远端验收和启动，开发筛选；不晋级默认。用户确认将P3移出Q动作池，作为独立快速右移；原Q内区域mask设计已替代。

## 语义

Q1 NET gamma0.9、B3L3、每run从空Q开始。八个处理臂均移除P3的选择与bootstrap资格，Q轨迹末步先terminal结算，再按当前终点解归一化Flow三分区判断是否独立精修。精修不占L3的一步，不增加Q更新，不将收益或耗时算进最后一个Q动作；gap取种反馈只看Q轨迹增益。

Q轨迹和一次精修作为一个调度单位，共享原20%累计局部改进份额，完整成本只计一次；下一次调用等待份额恢复。不得在轨迹末端再检查瞬时份额，否则正常轨迹会消耗刚积累的小额余额，造成精修永久饥饿。总墙钟截止仍在精修开始及每个完整候选exact前检查；一次构造不可中断，允许记录边界小幅超额，不增加预算或另分配20%。

保留P3的增量功率/账单检查及有限关键时刻生成，但只允许Prefill开始时间增加，Decode、MS/OS和Flow不变；完整候选经过统一gateway exact，每个可行完整候选均尝试Archive。所有新入档时序候选刷新等待诊断并按既有回流规则进入MOEA/D种群。一次精修最多生成3个候选调度，不反复迭代到收敛；独立随机流`lsv2:fast_rightshift`。

内部任务池：只计正右移余量任务，数量为E；部分池为 $\min(E,\max(20,\lceil0.2n\rceil))$，全部池为E。成功移动上限保持 $\max(5,\lceil0.1n\rceil)$；全池不等于必扫完全部，记录eligible、targets、每候选scanned/moves及上限命中。移动方向与外置角色同时改变，不能把总差异全归因于移出Q。

## 矩阵与入口

10配置：原完整B3L3、删P3无精修，以及all/flow_end/middle/bill_end四区域×partial/all两种任务池。区域仅决定LS终点精修资格，不对每代全种群额外扫描。

开发面板100、200各4例，历史seed16901，1200/2400秒含初始化，共80run。原始输入与旧标定按SHA复用，完整池/删P3同批重跑。源码及每配置、实例、种子、入口密封入manifest；8路物理核16–23，其他批次不动，220GiB可用内存/磁盘派发保留量，单worker3GiB地址上限，批次MemoryMax32GiB。

- `scripts/run_lsv2_fast_rightshift.py`：显式单run入口，独立目录、空Q、源/输入/标定hash、独立精修日志和产物校验。
- `scripts/run_lsv2_fast_rightshift_campaign.py`：prepare/preflight/validate/worker/dispatch，复用已有初始化指纹与独立exact预检、逐run封存、失败停止派发及可续跑完整run。
- 每run包含`rightshift_steps.json.gz`、`lsv2_steps.json.gz`、Q前后表、初始/最终种群与档案、anytime、summary、heartbeat和SHA complete标记。精修状态和步数进入心跳。

## 验收与解释边界

专项：右移方向/Flow/exact、部分/全池和n20等价、无右移余量反例、原P3冻结参考与RNG逐项一致、Q terminal与信用隔离、三分区边界、截止、共享份额、匹配Uniform路径、单run落盘与矩阵输入标定密封。

固定候选评分测时对比增量与完整exact，仅代表评分内核，不能称整算法提速倍数。上线另在100/200进行压缩技术烟测，记录初始化及实际精修调用，再启动正式矩阵。

2026-10-28本地验收：1353 passed、11 skipped；整体覆盖率89.34%，规定核心覆盖率97.18%；ruff、格式、mypy均通过。专项与相关回归56项通过。完整日志、覆盖率、固定评分基准和最终部署回执存于工作区`outputs/lsv2_fast_rightshift_impl_20261028/`。本条不代表远端已开跑。

主结论使用等总墙钟HV/IGD+及前沿，精修空率与耗时作为机制指标。只删除Q中的P3不能保证精修不再空，也不能保证比随机更优。当前只验证工程正确性，最终效果待80run完成后回收分析。
