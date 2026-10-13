# MacroSearch 生成规则与调用随机流四组确认

本协议冻结于 2026-10-13；研究选项未合并、未采用为默认算法。用户授权正式运行后返回预计用时并结束会话，不设置自动跟进。

## 问题与干预

上一批等时间实验中 POOL6 优于 NATIVE6，但按 10 的规则生成同时改变代表、抽样与随机流消费。本批固定精确评价上限为 6，用四组区分生成规则的收益是否依赖跨调用随机流耦合。

| 组 | 生成预算 | 评价上限 | 算子随机流 |
|---|---:|---:|---|
| NATIVE_SHARED | 原生 6 | 6 | 原具名共享流 |
| POOL_SHARED | 10 | 6 | 原具名共享流 |
| NATIVE_CALL | 原生 6 | 6 | 每次调用隔离子流 |
| POOL_CALL | 10 | 6 | 每次调用隔离子流 |

CALL 在每次 A1–A8 调用前从该算子的原具名父流取一个 128 位 seed，创建显式子流；子流内部消费不影响下次调用的父流。SHARED 不取额外随机数，保持原默认调用行为。不同臂的轨迹和调用次数可能不同；匹配的是给定算子的第几次调用对应的父 seed 消费规则，不保证跨臂候选、调用时间或所有随机事件相同。

生成 6/10 对 A1–A5 的 top-2B 池、代表和抽样，对其他算子的构造参数仍有联动。该因子是生成规则包，并非纯候选库大小。候选均从同一步冻结 incumbent 构造，不形成候选链；独立完整 exact 才消耗评价预算，所有可行 exact 候选尝试更新 Archive。A8 仍可能少于六个候选，不能将评价上限当实际数量。

四组均保持修正 A3 tariff、Q42、原 A8、原五步轨迹、Trigger、Polish、Archive、出生邻域顺序、固定 100 成员初始化及其他 F6 配置。关闭重诊断 reservoir 和离线 probe，保留相同的轻量心跳、anytime 及结果导出。初始化计入规则沿用原引擎秒预算；这批不研究新初始化，不跨批直接比较数值。

## 正式矩阵

40 全新 base，50/100 任务各 20；六场景 mixed、burst、kv_high、resource_high、region_asymmetric、demand_high 按固定索引轮转。native H/T 输入分别构造；每个版本 seed 1313/2313；四臂共 640 run、160 四组块。正式 base 为 1585001–1585020 与 1595001–1595020。每次引擎预算分别 1200/2400 秒，代数上限只避免提前结束；合计 320 solver-hour。

当前两节点按固定位置分配 14/26 base，拟 16/32 worker；同 base 的全部 H/T、两个 seed 和四臂均在同一主机。每个四组块用四阶 Williams 顺序，位置与直接前序均衡。输入、初始化、配置、源码、seed、host 和顺序全部冻结并哈希，先长实例。运行后不加样、删臂、按效果选择算例或改每 run 预算。

技术 pilot 使用另 12 base：1605001–1605006 与 1615001–1615006，算法 seed 6813/7813，H/T、四臂，共 192 run；30/60 秒。只用于资源、导出与正确性校准，不纳入正式统计或用于选择收益更好的参数。

## 运行与完成证据

正式派发必须在本地完整门、准确 head 的双 Python CI、两目标相关测试、192 pilot 完整验真和退出、640 正式输入独立检查通过后开始。启动至少两次核对实际 supervisor/worker 命令、PID、fresh 心跳和同 PID exact 正增长。16/32 是上限；1.5 GiB 内存预留、2 GiB 个人配额预留、256 MiB child 上限、nice 10 与完整结果导出均保留。两个容器共享宿主 CPU，不按独占总核数承诺完成时间。

个人配额只对本批原子心跳临时文件的 ENOENT 重读四次；持续不确定关闭派发门，永久或无关错误不忽略。十二小时运维窗口预留安全收尾；它不是预计完成时间或搜索预算。用户新暂停优先，真实失败保留证据并停止两 shard 新派发，不自动重试算法错误。任务级断点重用完整产物；中断 attempt quarantine 后重跑，不能声称内部搜索状态恢复。旧运行根、暂停标记和其他项目进程不动。

全部 640/160 的九项 worker 产物及 complete marker、配置/输入/初始化/source 哈希、Q42、seed、同 host、原 validator、真实 worker/supervisor 零、exit/pipeline complete 才能验收。原 worker 最终独立 exact 证据由产物哈希绑定；后续没有 fresh 全量 exact 时必须如实表述。

## 预注册统计

40 base 为独立单位，每个 base 对 H/T 与两个 seed 求均值；按 50/100 任务分层的配对 20,000 次 cluster bootstrap、100,000 次 sign-flip，显式 seed 2026101310。各 native 算例采用全部四臂、两 seed 的共同经验 ND union 作归一化与 IGD+ 参考；不是已知真实 Pareto 前沿。归一化 HV 越大越好、IGD+ 越小越好，定义 IGD+ 改善为对照减处理。

主家族六项联合 Holm：POOL_SHARED 相对 NATIVE_SHARED、POOL_CALL 相对 NATIVE_CALL，以及两个生成效应的差中差，各检查 HV 与 IGD+。CALL 下保留收益支持其不完全依赖跨调用父随机消费；差中差不显著不能证明没有 RNG 交互。

次级家族四项 Holm：NATIVE_CALL 相对 NATIVE_SHARED、POOL_CALL 相对 POOL_SHARED，各 HV/IGD+。其余 IGD、覆盖、anytime、exact/秒、后代/秒、构造/SSGS/exact/Archive 分解为描述性诊断；不将单候选代理收益替代最终质量。没有观察性后验挑子组作确认，不混上一批 40 base 或 pilot。收益是否值得实现后续候选构造优化，留待完整结果再决定。

## 复现入口

`scripts/run_macro_factor_campaign.py prepare` 生成冻结矩阵，`run` 在显式 root/worker/deadline 下派发；`geo_llm_scheduler.experiments.macro_factor_worker` 产生完整压缩原始证据。源码默认 `macro_rng_policy=shared`、`macro_generation_budget=0`、`macro_diagnostics=False`，研究四组由 `macro_factor.arm_config` 显式选取。结果回收和分析是后续独立任务，本轮仅关闭实现、预跑、启动交付。
