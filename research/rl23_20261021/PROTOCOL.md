# RL23：严格门下局部搜索净收益筛选协议

用户授权：完成原清单23组实现，并在219服务器启动后结束本轮。目标是相同总墙钟的前沿收益，不以代数、奖励或接受率代替质量。

## 冻结矩阵

使用component33原32输入及其SHA，16开发父、两价格变体，20/50/100/200任务各8份。seed16101/26101，23组、64配对块、1472run。无500任务，不生成新面板，不称留出。每run300/600/1200/2400秒，包含从头BILL四选一90+10/P100初始化、observer、搜索及精修。最终独立exact复验/序列化单列。按20→50→100→200轮转，块内23臂旋转并反向，64块只达到近似平衡，不冒称完整Williams设计。

1—6为BILL普通出生替换、BILL普通随机替换、严格原Polish、严格无Polish、严格两重点、严格低频全扫。7—23为时间惩罚、带下限收益率、前沿奖励及时间惩罚、持续时间折扣、TIME42、TIME84、ACTIVE42、匹配随机、Bandit、POOLED6、L1、L3、STOP2、B1、B3、必要条件屏蔽、时间惩罚加ACTIVE42。

所有模因组明确strict细方向、trigger_delta=0；第5组为原质量奖励/Q42/B6/L5研究基准。第6组仍LS后5%、非LS10%的独立Polish抽样，严格门只控制LS。Polish始终在episode外，不向最后一个Q动作错误分配精修信用。普通随机替换组仍method=plain，无LS。

## 工作参数与校准

技术pilot只取原面板首四输入，四规模各一份、单seed、全部23臂92run，预算60/90/150/300秒。pilot不混正式。各规模仅使用STRICT_TWO_FOCUS与MATCHED_RANDOM记录的完整action_seconds中位数作为共同时间参考，成本是计时而非质量/终点筛选，不按动作分别归一化。每规模至少8个调用；不足则停止冻结，补技术采样，不静默任取常数。参考必须正且有限，pilot原始日志、调用数及中位数封存，写入正式manifest后重封SHA，不改源。所有正式奖励组共享该规模参考。

时间惩罚系数固定0.25；收益率时间下限为参考时间的0.1；Archive HV奖励放大100并截断10，原不可行−1分支保留后扣成本。Archive在线HV只使用初始化完整Archive的ideal与初始P100的maximum形成固定context，参考(1.1,1.1)，参考外矩形贡献0，重复目标不增加面积。不读事后全体结果union。每次HV支持成本计入动作时间。

TIME42将原停滞位替换为实际预算进度是否达到50%；TIME84保留原42并加该位，不是D8复合84。ACTIVE42比较当前两个加权绝对归一化Tchebycheff项，某项大于另一项1.1倍时标记该目标受限，否则Balanced；零零并列Balanced。权重、严格门、接受与replacement不改。

STOP2最多五步，连续两步既未被原当前方向标量接受，也无存活Archive新目标向量则终止；目标向量按完整exact数值精确去重（不是phenotype计数），此数值口径冻结，不再事后改变。最后实际步骤terminal，future=0。时长折扣组保留原即时质量奖励，只在非末步使用gamma的归一化时长次幂，不宣称等于最终HV最优。

必要条件屏蔽组仅在全部operation无大于原TOL.time的Flow-wait时排除A4/A5；A8等其余动作保持。该必要条件来自原A4/A5生成器，非粗condition禁用。掩码缓存只绑定同一不可变problem、candidate与enabled actions，任一改变失效，仅持有一个parent；不新增候选探测，不全局缓存解。

## 工程与观察

原engine/trajectory默认和旧冻结研究文件不改；新研究通过既有trajectory_override/offspring_transform接入。仅允许plain显式permuted的Config验证扩展，默认仍birth。基准轨迹的候选、Archive、Q与RNG固定工作量差分须一致。

状态复用既有severity，当前step的NormalizationContext复用；原Polish工程复用层共用于三种精修。只有前沿奖励组计算在线HV；只有STOP2保存目标集合判断早停。同解必要条件缓存已有差分/失效测试与固定工作量微benchmark，后者只代表支持计算，不外推整程加速。

每run九项冻结产物：config、summary、initial、population、archive、qtable、anytime、diagnostics、steps.jsonl.gz。逐步记录候选1—B的objectives/可行性、冻结比较context、权重、原/新reward、动作完整耗时/参考、状态/动作、terminal、构造/预算成本、Archive留存；不存整份历史排程副本。最终P100与Archive全部独立完整exact复验，marker绑定九产物大小/SHA及输入/spec/source/host。

## 统计与质量判断

离线指标用同输入所有新臂两seed最终Archive共同经验ND union，HV参考(1.1,1.1)、IGD+；不是true Pareto。每父先平均变体/seed，四规模等权，16独立父；20k分层bootstrap、100k双侧配对sign-flip，统计根seed2026102101。每规模4父主要描述，不另挑规模作主要显著性声明。

预先列四个Holm家族：7—22各对5的HV/IGD+共32项；3/4/5/6内六对Polish比较共12项；3/4/5/6各对2共8项；23各对7/13共4项。1—2作替换诊断，附效应与区间，不借其单独显著性宣称机制。报告边际CI并区分同时推断，未显著不等价。

Archive/P100分开报告，检查中段覆盖与两端；分解构造、SSGS、exact、Archive、状态、Q、Polish互斥成本与未归类部分。单位时间局部成功仅解释，最终相对普通底座的等总wall质量决定候选价值。不得将23组重复run冒充1472个独立样本。

## 运维

独立新root/source/pilot96/formal；原03db6ab/e6e6744以及旧结果保持。96单线程worker绑定逻辑CPU、nice10、每child2GiB地址空间，用户cgroup MemoryMax400GB/High360GB；至少保留200GB要求按220GiB派发门保守实现（含20GiB余量），不干预其他用户任务。

八小时从正式supervisor启动计。96路理论总工时下界4h47m30，固定队列纯预算5h10m，附180秒/run约5h58m；实际ETA须看目标pilot与负载。失败立即停新派发并安全收尾，不自动算法重试；用户stop立即停止新增。全run锁、campaign锁、SHA完整验真，意外中断只有无失败并排除存活worker/新暂停才按原冻结入口续跑，未完成attempt移入quarantine，不宣称内部搜索恢复。

用户仅要求确认正式运行后返回，无Codex定时，无自动扩大矩阵或延长八小时。完整回收与质量分析以后按用户实际指令执行。
