# 两种Prefill右移策略及工程复用验收

## Material Passport

- 任务：用户授权的Polish降本策略实现与固定33组实例比较。
- 实现基础：PR72的e6e67440ec9e59629f786ed09e2c477a86a4ef9c；新策略独立研究分支，不修改旧运行源或默认算法。
- 证据：32项新增专项、component33原始输入及初始排程的24对固定工作量检查；本地源绑定全门禁775通过/1跳过，301.62秒，overall86.64%、core97.32%；GitHub CI和服务器状态以最终交付回执为准。
- 边界：固定解工程测时已经完成；正式等总墙钟实验未完成，不将局部加速或单解降费当作整程提升。

## 策略定义

| 组别 | 调用方式 | 每次处理范围 |
|---|---|---|
| REFERENCE_REUSE | 每个进入LS的子代，Q轨迹结束后 | 原Polish：所有合法Prefill/原关键位置，返回一个最佳单移动；同一exact复用层 |
| TWO_FOCUS | 每个进入LS的子代，Q轨迹结束后 | TOU暴露、需量暴露各选一个不同Prefill，顺序最多接受两个移动 |
| SPARSE_SCAN | LS类条件5%、非LS类条件10% | 按逆序遍历所有Prefill，可连续接受多个降费移动 |
| NO_POLISH | 无 | 原LS及所有其他设置保留 |
| BILL_MOEAD | 无LS | 同一BILL4初始化的普通MOEA/D |

四内部组统一A3地区电价策略、随机邻域替换遍历、固定MacroSearch6、五步Q轨迹和Q42，触发采用preference及10%质量容差。Polish位于Q轨迹之外，不参与Q奖励或bootstrap。所有组P100，原90+10混合BILL四选一，实际初始化计入总墙钟。

TWO_FOCUS的TOU排序分数是该Prefill独占的机器active增量功率乘电价积分；需量排序分数是它对所属地区所有并列峰窗的独占active增量平均功率贡献乘需量费率。并发重叠工序不能重复领取整台机器功率。TOU选第一名，再从其余有slack的Prefill中选需量第一名；并列operation id小者优先，只有一个时只选一个。两个id在调用开始时冻结，处理第二个时使用第一移动后的实际排程重检。分数仅为优先级代理，不能保证被选Prefill存在可行降费位置；没有收益时不额外扫描第三个来偷偷扩大预算。

SPARSE_SCAN的抽样分母分别为LS与非LS子代，使用独立显式polish_sampling随机流。若LS比例为$p$，预期总调用比例是$0.05p+0.10(1-p)$，不是15%。小样本实际比例会随机波动，记录两类分母、抽样次数和实际调用次数。一次扫描按调用开始时的Decode开始时刻、Prefill开始时刻、operation id降序；无slack跳过，每个Prefill枚举原有限关键位置，接受一个严格总Bill下降的最佳位置后更新当前排程，再处理下一个。

两候选只改Prefill timing，冻结Decode、MS/OS，Flow严格保持。所有可行完整exact候选继续单独尝试原Archive，包括未被返回的候选。每次接受后使用更新资源视图；最终刷新resource-wait diagnostics，不改变已接受时序。本轮保留原边界位置集，不叠加新窗口平衡点。

## 复用层及正确性

PrefillEvaluationContext完整exact核验不可变父解一次，复用未变化地区的TOU原始加法项、需量值与固定900秒窗口。只重算移动地区的原功率分段/TOU项及窗口，按原region/segment/tariff顺序逐项累计，避免增量相减或前缀反导数改动浮点顺序。Decode和horizon不变，Flow直接复用已核验的原值。

容量检查不只检查被移工序：受影响机器的每个工序均按原reference求和顺序重验，以覆盖资源容差附近的浮点差异；P到D约束也重验。非法单移动回到完整reference，跨problem/genotype、同时移动多个工序或移动Decode则拒绝使用该context。当前排程不变时，全扫描共享一个context；接受移动后才失效重建。

每个完整语义候选仍通过统一gateway更新计数、ideal、Archive与observer。父解核验另计polish_parent_audits及polish_context成本，耗时也包含在总Polish/生产墙钟中。polish_context是Polish的嵌套子项，不能与总Polish时间相加。默认调用未启用复用参数时仍走原exact。

专项覆盖四规模/四类合成边界的移动目标逐浮点完全相等、参考Polish所有候选和Archive一致、两新策略复用前后连续移动序列一致、多个并列峰/尾窗/价段边界、容量容差/非法父/context绑定、条件抽样与独立随机流、默认engine空回调一致，以及always/preference/strict三种触发下原轨迹Polish与新位置参考回调的所有候选、Q表、动作选择、随机流和trigger-success一致。campaign另验原输入SHA、64块/两seed/320矩阵、原文件损坏拒绝和分离pilot。

## 固定工作量测时

使用component33首个拓扑的四规模原输入，每输入取原FULL_CURRENT_I4初始种群第0与90号，共8固定解；各对照复用开关，合计24对，包含6442个单侧候选exact。候选序列/目标、返回解、Archive和接受次数逐对完全相等。执行顺序按预先固定规则交替；单次本地Python3.13墙钟，非重复稳健估计、非服务器并发ETA。

| 策略 | 相同工作完整exact累计秒 | 复用累计秒 | 观测节时 |
|---|---:|---:|---:|
| 原单最佳Polish | 33.5541 | 9.5242 | 71.62% |
| 两重点Prefill | 1.4885 | 0.6554 | 55.97% |
| 连续全Prefill扫描 | 35.9324 | 11.1612 | 68.94% |

200任务原Polish两个固定解分别4.8374→0.7807秒、24.7364→7.2867秒。两个重点策略在其中一个固定解检查14个位置但没有可行候选，另一个评价48个候选并接受一次；这提示聚焦可能漏掉可行机会，不能仅凭速度宣布策略有效。连续扫描在两个固定解分别接受16与9个移动；这也不证明它在等总墙钟优于原Polish，正式实验需要同时比较吞吐与前沿。

完整benchmark.json和原日志存于本批输出/Release，不将原始用户数据、大输出或机器绝对路径提交代码仓库。

另以6任务、P100、完整一代跑通五组流水线，五组初始population/Archive hash一致、每组100子代和全部714个最终存储成员fresh完整exact验证通过；原Polish/两重点各19次LS后调用、稀疏8次调用、无Polish/普通MOEA-D为0。这只是技术小例，不计正式样本。

首次未显式绑定子进程源码的完整门禁有773通过/1跳过、两项旧mini集成观察超时。实查普通子进程导入了主checkout，随后以PYTHONPATH=src绑定新worktree重新执行原全部门禁，得到775通过/1跳过；未放宽原65秒观察上限或覆盖率门槛。原失败日志保留。D03、明确源绑定确定性回放、wheel及CLI辅助验收也已完成。

## 固定实验面板与后续验收

当前及下一轮完整实验前统一用component33的原始32输入/hash：16开发父、两变体、20/50/100/200各8输入，原2/3/5地区、6/12/24机器拓扑。原面板实际为三个arrival拓扑和一个KV拓扑，不称覆盖cheap/demand压力，也不称新留出。旧16101加算法seed26101，64配对块/320新run。重复seed不增加独立父数，旧1056次不合并统计。

300/600/1200/2400秒总生产wall包含初始化、LS、Polish、context与observer。所有技术pilot单列。完整运行须通过各run八产物与source/input/spec/config/hash、每块初始化一致/同host、最终完整exact证据、真实worker0和exit complete；失败不自动重试，保留stop/quarantine，续跑仅完整run级，不声称内部搜索checkpoint。

统计按16父为独立单位、父内版本/seed均值、四规模等权；主8项为两新候选各对原Polish与无Polish的HV/IGD+，联合Holm，次8项为各内部对BILL普通。20k分层bootstrap、100k配对signflip、seed2026102011。经验共同ND union不是真实Pareto。初始化/搜索/Polish成本及实际代数与P100/Archive结果分开报告，未显著不等价。

## 长期工程约定

用户明确要求今后实现尽可能复用不可变绑定的资源、账单和输入索引，避免重复构造与全局重算；同时保留完整exact差分、边界检查和前后benchmark。改变候选范围、调用频率、Archive或随机流属于研究改动，不得冒充纯工程加速。此要求及固定component33面板已写入Notion采用决策、维护协议、项目总览和当前框架。
