# 外部随机初始化与组件拆分：首轮实施及预算方案

本批为独立研究入口和正式前协议。规划脚本只打印 JSON；正式 runner 独立负责输入封存、初始化预检、技术 pilot、派发与验收。实际部署和启动事实记录在独立运维回执，不以本协议文字代替。已有历史结果保留，不计作新矩阵的完成结果。

## 比较口径

五个外部方法使用纯随机合法 MS/OS 初始化，不调用我们设计的十类混合或 BILL。每个任务均匀抽取一条合法、同地区的 P/D 机器路径；重复任务 OS 随机打乱。只按基因型去重，不按费用、负载、目标值或可行性评分挑优。SSGS/exact 评价、初始 Archive 和全部初始化成本均计入总墙钟。相同输入和 seed 的五个外部方法获得相同初始基因。

内部单组件比较统一以原十类混合 P100 的普通 MOEA/D 为参考。BILL 单列：保持 90 个原混合成员，加入十个四选一 BILL 成员；100 原始候选及 40 追加候选均实际评价并尝试进入测量 Archive。其效果包含计算成本和未入选候选的 Archive 贡献，不能只归因于最终十个成员。

完整算法 `FULL_CURRENT_I4` 必须重跑。其 Config 与既有 `comparison13.arm_config("M11_I4", seed, seconds)` 逐字段相等：BILL10 加速四选一、A1–A8、tariff A3、随机替换顺序、Q42、固定 MacroSearch 6、五步轨迹、原 preference/10% 触发门和轨迹末 Polish。没有加入 SEQ、STATE84 或辅助 walk。另一完整组合是待验证候选，不能冒充当前完整算法。

不同初始化的外部比较衡量整个算法包；同初始化组件比较用于机制诊断。二者不能互相替代。

## 26 组清单

| 编号 | 入口名 | 定义 / 比较用途 |
| --- | --- | --- |
| 1–5 | EXT_MOEAD、EXT_NSGAII、EXT_RK_MOPSO、EXT_SPEA2、EXT_EPS_IBEA | 五个外部离散调度适配，随机 P100，无本方法局部搜索 |
| 6 | MIXED_MOEAD | 原十类混合 P100 + 普通 MOEA/D，内部共同参考 |
| 7 | BILL_MOEAD | 第 6 组仅增加 BILL10 四选一初始化 |
| 8–15 | ONLY_A1 … ONLY_A8 | 第 6 组分别仅增加一个动作，每子代一调用、候选上限 1、无 Polish |
| 16 | POLISH_ONLY | 第 6 组仅增加一次 Polish，不选择 RL 动作、不更新 Q |
| 17 | RPERM_ONLY | 第 6 组仅随机化邻域替换顺序，关闭局部搜索 |
| 18 | POOL_RANDOM_B1_L1 | 八动作均匀随机、一调用、候选上限 1 |
| 19 | POOL_RANDOM_B6_L1 | 第 18 组仅将候选上限改为 6 |
| 20 | POOL_Q_B6_L1 | 第 19 组仅换为 Q-learning 选动作 |
| 21 | POOL_Q_B6_L5 | 第 20 组仅将轨迹延长为五步 |
| 22 | POOL_Q_STRICT_B6_L5 | 第 21 组改为严格方向、零质量容差触发 |
| 23–24 | POOL_Q_SEVERITY_L5、POOL_Q_COVERAGE_L5 | 第 21 组分别换为既有两种动态预算 |
| 25 | FULL_CURRENT_I4 | 当前完整算法，按上述完整定义重新运行 |
| 26 | FULL_STRICT_NO_POLISH_I4 | 完整组合候选：严格触发、关闭 Polish，其余保留 |

组件不是全部可以孤立存在。Q-learning 必须有动作池，MacroSearch 宽度必须有动作，五步轨迹必须有单步参照。因此第 18–24 组采用逐层对照，不能把各项增量相加或宣称交互已经分离。两种动态预算在此共同底座上诊断，不等于重跑历史 DS/DC 完整算法。

## 基础方法与参数来源边界

本批没有为降低外部方法表现而修改交叉、变异、选择压力、种群或预算。遗传方法保留既有、共同的离散 MS/OS 交叉概率 0.9、每子代变异概率 0.2；这是调度适配参数，不能写成原连续论文的逐变量多项式变异。所有值随 Config 导出，算法名保持适配标识。

原始文献定义基础机制，不意味着其连续实验设置可以直接套到本问题：

- MOEA/D：Zhang & Li (2007)，[原论文](https://tiezhongyu2005.github.io/resources/popularization/MOEA-D_2007.pdf)，DOI 10.1109/TEVC.2007.892759。当前实现使用归一化 Tchebycheff、P100/T20、邻域概率 0.9、最多两次替换，属于明确的研究适配，不宣称逐行复现 2007 全部设置。
- NSGA-II：Deb et al. (2002)，[原论文](https://dynamics.org/~altenber/UH_ICS/EC_REFS/MULTI_OBJ/DebPratapAgarwalMeyarivan.pdf)，DOI 10.1109/4235.996017。保留 rank/crowding、二元锦标赛和父子代精英环境选择；变异编码使用共同调度适配。
- SPEA2：Zitzler, Laumanns & Thiele (2001)，[原技术报告](https://sop.tik.ee.ethz.ch/publicationlistfiles/zlt2001a.pdf)。保留 strength-sum、近邻密度和逐次距离截断，原生交配库与被动测量 Archive 分开。
- epsilon-IBEA：Zitzler & Künzli (2004)，[原文出版记录](https://doi.org/10.1007/978-3-540-30217-9_84)。当前实现为归一化 additive-epsilon、kappa=0.05、逐次删除并更新贡献，不称为所有 IBEA 指标变体。
- RK-MOPSO：Coello & Lechuga (2002)，[作者上传的原论文](https://www.researchgate.net/publication/3949342_MOPSO_A_proposal_for_multiple_objective_particle_swarm_optimization)，DOI 10.1109/CEC.2002.1004388。当前调度适配使用随机键、惯性 0.4、加速系数 1.5/1.5、速度界 0.2、原生库 P100、10×10 网格、随机单键重置概率 0.2。这些不是最早论文的逐项复现：原论文速度式系数、原生库及网格实验设置不同。不得只写“原版 MOPSO”隐藏上述差异。本批保留该适配的全部既有设置，没有按新结果挑差参数。

如后续要求严格的特定原论文配置，应另建可追踪 profile 并验证，不能把当前适配重命名后声称已经完成。

## 八小时左右的首轮资源计划

用户允许约 500GB 项目内存上限。前次只读资源核验：219 节点约 755GiB 物理内存、743GiB 可用、48 物理核 / 96 线程。该快照不是当前内存承诺；部署前须再核验。500GB 是上限，不是必须占满的目标，实际 cgroup 配置和资源快照以启动回执为准。

建议 48 个单线程 solver，优先每物理核一个。不能只因内存充足直接提高到 96 路，后者会改变单位墙钟有效计算量。与其他用户进程隔离，不结束其任务、不清全局 cache。

正式提案为 20 个已有独立父算例：20/50/100/200 任务各五个；每父两个已有压力变体、先一个共同算法 seed 16101。共 40 输入版本 × 26 组 = 1040 次全新运行。覆盖 2/3/5 地区，6/12/24 台机器；按 20→50→100→200 轮转。选择依据仅为规模、拓扑及已有输入身份，不依据旧结果好坏。

五种父类型：R5M12/arrival、R3M6/arrival、R3M24/arrival、R3M12/kv、R2M12/kv。原 key、生成 seed 与版本对应见规划 JSON。需要从已归档原输入复制并逐 SHA 验证，不能仅凭同名或重新生成即声称同输入。cheap-capacity/demand 未纳入本轮；一 seed 和复用开发实例也限制泛化，后续留出确认单列。

每次生产预算依次 300/600/1200/2400 秒，包含实际初始化。总计 325 solver-hours，48 路理想下界约 6 小时 46 分。逐 run 派发模型在每次额外预留 60 秒时约 7 小时 34 分；这不是实测，也不包含源验收、技术预跑和完整回收分析。预计正式计算约 7.5–8 小时，正式前须用独立技术预跑校准超时、初始化及导出成本。不得把整个 26 臂大规模块串到单个 slot；那会超过夜间窗口。

另列四父 / 208 次储备，当前不进入 1040 次主矩阵。不得看表现后加样。若窗口不够，应在启动前缩减为完整、平衡的父算例队列，不能事后只报告完成或表现好的方法。

## 正式执行和分析仍须完成的门槛

- 独立新 campaign，冻结 source/输入/初始化规则/Config/seed/主矩阵；逐 run 互斥、意外中断证据及安全收尾。失败不自动改算法重试，旧失败历史不覆盖。
- 所有 40 输入与两种内部初始化预检；随机初始化独立 exact 校验；独立技术 pilot 不混正式。
- 正式每 run 保存完整配置、初始种群、最终种群/Archive、Q 表、anytime、计时/次数/结构诊断；最终全部保留解独立 exact，产物大小/SHA 绑定验收。不能靠目录数验收。
- 分析以 20 个父算例为独立单位，两个版本先父内平均；同一父所有比较同 host、同计算策略。一个 seed 只能支撑首轮筛选，不能声称随机稳定性已确认。
- 主家族为完整算法第 25 组分别对五外部方法的 HV 与 IGD+，十项共同 Holm。
- 组件家族固定十二对、二十四项 Holm：MIXED 对 EXT_MOEAD，BILL 对 MIXED，八个 ONLY_A 分别对 MIXED，POLISH_ONLY 和 RPERM_ONLY 分别对 MIXED。
- 次级家族固定十一对、二十二项 Holm：第 18 组对 MIXED，第 19 对 18、第 20 对 19、第 21 对 20、第 22/23/24 分别对 21；第 26 组分别对 25、EXT_MOEAD、MIXED、BILL。不得依据结果增减对比。
- 每父两个版本先平均，二十父按任务规模分层 bootstrap 20,000 次；配对 sign-flip 100,000 次，固定 seed 2026101908。双侧检验，保留负效应和所有未显著结果，不把未显著当等价。归一化以每个输入本轮全部方法的共同经验 union 的 ideal/nadir 为准，退化维 span 取 1，HV 参考点为归一化的 (1.1, 1.1)。
- 经验前沿使用同输入全部本轮方法的共同非支配 union，不是真实 Pareto 前沿；旧批只可另作历史观察，不混成新条件下的参照基线。

## 复现入口与实现边界

`python scripts/plan_component_random.py --workers 48 --overhead-seconds 60` 输出可审计的方案、完整 Config、输入身份和时间模型。它不会启动实验。

`experiments.component_random.run_arm` 是新比较入口，负责选择三种初始化政策和运行引擎。`engine.run` 只增加一个可选轨迹回调供纯 Polish 诊断；省略时原候选、Q、计数和 RNG 行为由差分测试约束。旧 comparison13/mechanism5/default 的运行入口和冻结历史不改变。

`scripts/run_component_random.py prepare --root NEW_ROOT --historical-root OLD_FORMAL` 复制并核验旧输入，拒绝覆盖已有 campaign；可使用 `--pilot` 生成另列的四输入、104 次短预算技术队列。`preflight --root ROOT --key KEY` 对随机、原混合、BILL 三种策略的每个生产候选 fresh decode/exact 审计，绑定正式初始 P100 和 Archive。`run --root ROOT --parallel 48` 在同一 host 逐 run 调度；`validate --root ROOT` 逐产物/配置/输入/source/初始化预检与最终 exact 证据验收。

派发保留每 run 锁、supervisor 锁、真实孤儿 worker 检查、失败停派发、stop.request 安全收尾以及不完整尝试 quarantine。优先每物理核一个 slot，单 worker 地址空间 2GiB；聚合 cgroup 限制在部署时设置。计划约 7.5–8 小时，9 小时窗口仅为保守停止准入保护，不是自动延长样本或新增组别的授权。各 run 全部独立构造初始化；技术预检和 pilot 不充当正式运行。

源码/测试通过不代表已部署或科研效果成立。最终是否启动、准确 T0、实际 PID/心跳/资源与 ETA 必须读取运维回执；不在本协议预填完成状态。

本地实现验收：16 专项通过；最终完整门禁 673 passed / 1 skipped、520.22 秒，ruff lint/format 与 113 源文件 mypy 通过，overall 86.22%、core 97.28%。新增 runner、原输入字节复制/封存、三种初始化独立 decode/exact、每规模轮转、原完整配置逐字段不变和默认轨迹差分均有测试。更早的中间门禁因新增正式 runner 主动中止，其日志保留；不把它计为完成的全套验收。准确 head CI 和服务器技术校准见独立回执。
