# MOEA/D P1 机制诊断执行协议

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent。
- Origin Mode：run。
- Origin Date：2026-10-01。
- Verification Status：执行记录以 campaign 的真实质量门、P0 和 launch gate 为准。
- Version Label：P1 diagnostics v1。
- 研究计划：[完整计划](https://app.notion.com/p/3ec8878b801981a18b21d9a1608be691)。
- 本批授权：用户要求启动 P1，允许使用两台现有服务器；后续 P2/P3 不自动启动。

## 算法与观测

F6 保持全算法、固定 B6、5 步 Q 轨迹、轨迹后 Polish；M0 为普通 MOEA/D。
population=100，neighborhood=20，replacement cap=2，默认工作参数与 E15 F6 一致。
时间预算为 50 jobs 600 秒、100 jobs 1200 秒；generation 上限 1,000,000。
Observer 只观测，不改变 variation、traversal、Trigger、Q、算子和预算随机流。
日志/检查点 I/O 计入 engine 时间；离线诊断不反馈给在线搜索。

`run(..., retain_trace=False, observer=...)` 使用独立 offspring 计数；不保留逐子代
完整 archive 历史，不裁剪真实 archive。默认 retain_trace=True 保留兼容行为。
NSGA-II 的历史 trace 也可关闭，仅用于 P0 的 N0 复现和未来独立对照。
A7 过滤阶段被抽为同一公共选择函数，随机调用次序、raw generator 和原筛选保持不变。

## 矩阵和采样

开发集 50 jobs 种子 810001–810012、100 jobs 820001–820012。
每个基础实例有 H/T，算法种子 1101/2202/3303/4404/5505；F6/M0 共 480 条。
每个 base/seed 的 H 初始化池冻结后供两个 tariff 和两个 arm 共用。
按 base/seed 配对块平衡分给两节点，每节点 240 条，所有组均在两节点出现。

在预算 10%、50%、90% 后首个完整 offspring 边界保存当前 population、archive、
normalization、Q、stagnation 和已创建 RNG 流状态。检查点用于恢复反事实输入，
不宣称可从代中断点精确续跑原在线轨迹；完整 run 续跑仅复用校验通过的完成项。
每个 Preference 在 distinct phenotype 中均匀选一个，再从未选 phenotype 选一个。
跨层重复记缺失，不按是否能改善补样。保存条件选择概率和该表型所在索引概率。

## H1–H4 产物

- H1：同 MS/OS 的 earliest SSGS 重解码；Flow/TOU/Demand/Bill 和固定 context scalar。
  F6 样本从原/重解码两起点以相同具名流执行 A7/A8/A7/A8/A7，每步 B6，最后 Polish。
- H2：在线最终 offspring 用当时 replacement context 审计全部 own-weight 改善集合；
  原邻域、随机顺序、改善顺序、关联方向邻域、cap 分开保存。
  共同候选池另在 checkpoint 的冻结 context 审计，明确其反事实含义。
- H3：每表型 A7 的全部 raw moves 与代表/pool/selected 映射；共同池最多 10 exact，
  从其余 raw moves 中无放回均匀抽最多 40 个做额外 exact。
  这是抽样 raw regret 证据，不能称穷尽 raw pool 的最优 oracle。
- H4：共同 B10 序列的 3/6/10 前缀，每前缀只更新已见 ideal，独立复制 archive；
  固定描述性 HV 范围由整个共同池离线生成，不参与 online/scalar 决策。
  各池保存构造、exact、archive 成本；自然 B3/B6/B10 仍属于后续 P2。

A6/A8 的 attempts、proposals、construction diagnostics 随在线 step 和离线共同池保存。
候选 exact 事件保存 identity、origin、目标分项和完成时间；完整解保存在检查点和诊断池。
不宣称每个在线 exact 的完整基因/时间表都已保存。

最多 5760 个正式采样表型；计划反事实 exact 上界 869760。
额外 checkpoint exact 真值校验最多 5760 次，Polish、初始化控制与资源 pilot 单独计数。
诊断按检查点逐个处理，压缩文件流式写盘；内存不随历史 offspring 数累积。

## P0 与资源门

8 个独立 pilot bases（710001–710004 / 720001–720004），H、seed6606，
F6/M0/N0 固定 200 代各重复两次，共 48 条，比较完整 exact/decision 顺序摘要、
population、archive 身份、Q/访问/更新/动作和计数；与旧 3c59ada 子样本交叉核验。
seed7707 用于独立并发和长时 RSS 预跑，普通算法与 F6 至少覆盖 P1 的 1200 秒。

每节点按 1/2/4/8 逐级预跑，分别测单节点和双节点同时负载。
稳态 working set 门为 70%，78% 为暂停派发上限；计划实现更保守地在 70% 停派发。
所有底层数值库线程为 1；8 是本次程序允许的最高单节点并发，最终以 pilot 冻结。
OOM counter 增加、worker 失败或磁盘不足 20 GiB 时停止新派发，保留在途与失败现场。
无自动重试，不删除 E15 stop.request，不覆盖 E16 worktree。
实际到期未知时使用较早 2026-10-04 迁出目标；不启动无法在前 24 小时前收尾的波次。

## 启动与验真

`scripts/run_p1_campaign.py prepare` 冻结 source/commit、实例、初始化、完整 config 和矩阵。
`batch --phase P0 --workers N` 运行固定代数验收；RSS/THR 阶段独立保存。
P1 需要 `ops/launch_gate.json`，含真实 acceptance 和 manifest SHA256，不能只凭文档启动。
实际参数与绝对服务器路径由 campaign manifest/ops 启动记录给出。
worker 先写 sampling_complete，再执行离线诊断；complete 校验全部产物的大小与 SHA256。
失败不冒充完成，缺失不当作零收益。P1 完成后再聚合 base 层报告，不据表型数扩大样本量。

代码经本地质量门、PR/CI/合并后冻结部署；完整结果在采集完成后回收、Release 校验并同步 Notion09。
本协议本身没有研究效果结论。
