# 初始化加速与构造强度并行实验

## 研究问题与冻结范围

在原BILL10中，保留原十类来源各9个成员，再增加10个费用引导成员，最终P100。比较REF4（原扫描、四选一）、FAST4（日历索引、同工件Prefill查询缓存、电价积分索引缓存，四选一）、FAST2及FAST1。所有臂逐run从头构造原P100，成本完整计入；没有MOEA/D迭代、RL、Polish、路径预筛或种群规模改变。默认参数仍调用原路径；二选一仅显式加速研究入口允许。

数学模型v8、Coding Contract及原始6d3d0e6初始化语义保持。时间秒、功率kW、费用CNY；半开资源区间、全部900秒固定需量窗、尾窗900秒分母、前缀active union和idle成本均不改变。

## 工程等价与质量的区别

资源日历按operation编号维护顺序，包括未提交的Prefill。保持compute/VRAM求和次序、完成事件排序、同地区P/D路径顺序、归一化及5%限制候选表和RNG调用不变。同一工件的P最早时刻不依赖尚未提交的D，局部缓存仅在该工件内有效。每次构造有独立电价索引，跳过零重叠项后使用原fsum公式，缓存相同区间；不采用可能改变浮点结果的积分原函数相减。不跨run共享缓存或种子库。

REF4与FAST4必须逐基因、开始时刻、评价、最终P100/Archive、全部40trial评分观察及RNG后态SHA严格相等；同平台全部正式配对再次核验。每个trial原独立exact复核、最终100成员原SSGS/exact复验都保留。Python跨版本仅遵守原数值容差，不声称逐bit等价。

FAST2/1改变每个slot的独立构造数、去重历史和后续随机流位置；因此比较整个构造策略，不能当相同候选集合的纯选择消融。10个slot权重仍按0/.1/.25/.5循环，名额3/3/2/2；完整trial数40/20/10，重复构造尝试上限100保持。Archive主分析仅来自最终P100，不混入未选trial。

## 矩阵与执行

正式32个全新独立base：20/50/100/200任务，各12或24实例、固定3地区，mixed/burst/vram_tight/demand_high四场景；每base两seed15101/15201，共64块、256run。每个base独立生成，不将不同压力场景当单因素配对因果样本。全部地区电价异质、周期4800秒；不再使用旧H/T或500任务。

pilot为独立四base（20/200任务、12/24实例、mixed）、每base一seed15001、共16run。pilot只决定资源/时间，不纳入正式质量统计。正式四臂每块同host同CPU顺序执行，64块采用Williams四行均衡位置和前序；块间最多24路，选不同物理核。规模轮转派发避免先做完小规模才开始大规模。

每child地址空间上限4GiB，24child合计96GiB，低于用户200GB预算；此为保守地址空间限制，不声称已设置聚合200GB cgroup。额外检查主机可用内存大于16GiB、NVMe剩余大于10GiB才派发。缓存随child退出释放。不清全局cache，不干预其他进程。单run两小时技术保护，不是搜索预算。算法失败停止新派发，已运行臂安全完成，不自动重试。

manifest绑定输入、spec、source、seed、host；complete绑定四个最终产物的size/SHA，原独立exact证据由这些产物绑定。完整块还要通过REF4/FAST4严格等价验收。续跑先验证完整产物再跳过；未完整尝试移入quarantine，失败须人工审查，stop.request不自动移除。使用排他flock，禁止重复supervisor；这是run级续跑，不是内部搜索checkpoint。退出complete需全部256run/64块验真且无活跃child。

## 预注册分析

生产wall、CPU分别报告（原P100、引导bank、选择），独立exact审计和导出另列。主成本比较为FAST4/REF4的配对生产CPU与wall比例，32base为独立单位、先平均各seed的对数时间比，再报告几何比例、20k base bootstrap区间；同一机器负载下的实测而非跨服务器速度结论。描述cache数量及峰值RSS。

质量比较FAST2−FAST4与FAST1−FAST4，以各base两seed全部最终P100共同经验非支配union作为参考；按该base共同目标min/max归一化，HV参考点(1.1,1.1)，IGD+参考为共同union。HV差和IGD+改善方向均为正表示较好；四项主检验联合Holm，32base等权、两seed先平均；20k bootstrap、100k paired signflip、分析seed2026101517。普通IGD、ND数、最低Flow/Bill、Flow受限费用端与固定方向覆盖只作描述诊断，不另挑显著子组。未显著不等价；经验参考不是真实Pareto，初始质量不代表最终等墙钟进化效果。正式统计完成前不采用任何新默认。

## 材料清单与完成条件

- 原参考：6d3d0e662a72c2ff0a3ded9ad7bf4d0ccf3c97c2；本批独立Git分支，准确source/runner摘要写入manifest。
- 生成器：initial_bill_speed.speed_problem，base/算法/bank seed全部显式，输入JSON冻结；无外部私有数据。
- 产物：input/spec/manifest、各臂config/summary/population/archive/complete、进度与日志、退出状态、quarantine历史。
- 先过reference/边界/反例/复现/篡改测试、完整本地门禁与准确head CI及目标专项/技术pilot，再正式启动。
- 后续全量回收逐SHA、原始结果Release、独立分析报告PR/CI。启动验收与质量分析分开；本轮没有创建自动跟进。
