# DeepSeek Flash三组等时间开发实验

## Material Passport

日期：2026-10-24。类型：用户已授权的隔离实现与实验协议。目标是先构建有效LLM适配版，不以新颖性为首要目标。继承1ec42415的已验收引擎和原component33输入；默认及历史运行源不修改。实现/验收/启动状态以本批真实回执为准，本文不预先宣称已启动或质量改善。

## 研究处理与对照

- LLM_FLASH：BILL4/P100、严格细方向门/零质量容差、B6/L5、两重点Polish；LLM替代Q动作控制，增加有限A3/A6 focus和轨迹早停。
- MAIN_Q：原STRICT_TWO_FOCUS参照路径，Q42/B6/L5、严格门、两重点Polish、既有A3 tariff和permuted替换。
- BILL_MOEAD：原BILL_PERMUTED普通MOEA/D，无LS/Polish，BILL4/P100与permuted替换匹配。

不增加静态LLM臂或其它算法臂。这是三组整包开发比较，不可将引导候选生成和早停收益单独归因语言推理。

## 实例、随机数和生产预算

仅使用component33原manifest绑定的32输入、16开发父、两变体、20/50/100/200四规模。原始字节及SHA不变，正式seed16401/26401，共64配对块×3臂=192run。原面板已反复使用，只作开发筛选，不称新留出或新增独立父。

生产预算按规模300/600/1200/2400秒，初始化、在线特征/日志、API限流排队/等待、解析和搜索全部计入。最终独立exact审计及导出单列。LLM显式启用安全边界复查：observer模型等待耗尽预算后不再产生新子代；原主算法和普通对照保留既有截止检查及原子子代收尾规则。轨迹和完整candidate仍沿原原子步骤收尾，报告实际预算超出量。该复查是默认关闭的独立API选项，不改变历史控制路径。

原命名随机流保持；LLM动作选择另用llm_action流。A3原合法move空间/2B候选列表及A6原合法repair保持；focus改变优先级或destroy抽样，属于研究策略变化。每个MacroSearch候选同源，全部完整可行exact候选保留Archive机会。

## 全局prompt与动态输入

llm_global_prompt.py逐次发送完整问题说明：静态P→D、MS/OS决策、同Region、Compute/VRAM累积容量、KV、非抢占、两目标、idle/active区间并集、TOU积分、固定900秒Demand及尾窗/并列峰、真实细方向接受和模型权限。

动态输入为真实实例/价格、P100与Archive、三偏好组代表及等待/费用/覆盖、全部八动作曝光与成本/收益、上一有效策略和剩余预算。按Job累加两阶段等待再求均值；覆盖按去重ND目标关联，区别于槽位。代表点与诊断解标明实际来源。模型看不到未来共同union、最终IGD+或未选动作反事实。

反馈包含总/平均每call墙钟、占所有动作时间比例、占观测生产区间比例、局部接受/秒、Archive新目标成功call/秒、支配移除成员/秒、标量改善/秒和在线HV代理增量/秒。不同状态下的曝光有偏，不能当动作因果排名；便宜或接受率高都不等于前沿更好。SSGS/exact计时与动作wall重叠，禁止相加作互斥成本。

在线HV代理仅使用初始化冻结ideal/maximum及参考(1.1,1.1)，排除超出参考框的点；是局部诊断尺度，不是真实前沿或最终共同参考HV，不用于修改候选接受。

## 策略、模型与成本守卫

- 提供商官方地址https://api.deepseek.com；模型deepseek-flash，非思考模式，JSON输出，max_tokens2048，不发送temperature。保存API返回模型标识；若仅返回别名，不冒称不可变服务快照已被锁定。
- 使用search_policy_v2/search_policy_response_v2；三组恰好各一份、八动作权重0～100且和为正，未知字段/重复key/NaN/Inf/布尔权重/错配ID拒绝。
- 支持七种focus：FLOW_WAIT、RESOURCE_WAIT、KV_DELAY、REGION_LOAD、TOU_EXPOSURE、DEMAND_PEAK、MIXED。v0.2草案的ACTIVE_UNION未直接实现，v2将其从允许输出删除；模型可提高A7概率。
- 动作概率为归一权重混合0.10均匀探索；A6评分归一后混合0.20均匀抽样，无放回destroy，约10%且至少2Job。
- A3 TOU复用已有稳定费用差桶，REGION_LOAD切回负载排序；其它focus主键优先、原排序作次键。A1/A2/A4/A5/A7/A8生成器保持。
- 每run最多3次请求：初始化、约总预算1/3和2/3安全边界；错过检查点合并，不补队列。
- 单次请求硬截止为min(20秒,总预算2%,剩余生产时间,剩余等待额度)；累计等待额度min(60秒,总预算5%)。排队、HTTP子进程启动与等待共同消耗额度。最多16个服务请求并发，求解器最多96worker。
- 请求由独立子进程执行，超时杀死并回收子进程，避免socket/DNS阻塞越过硬截止；无重试，无自动换模型。
- 首次失败均匀随机原生成器/L5；后续失败保留最近合法策略。每run至少获得1次有效策略才作为完整LLM处理验收；失败原件保留，不自动重跑。
- 连续无局部接受且无Archive新目标事件达到阈值1/2/3/5时停止当前LS，仍最多L5；同目标不同phenotype不算新目标贡献。所有候选逐次插入事件独立计数。

2026-10-24真实开发接口probe在原20Job输入上获得完整有效三组响应：模型返回deepseek-flash，模型等待约2.72秒、摘要约0.012秒，输入16027/输出471tokens。它只确认接口与单样本成本，不是多并发尾延迟、成功率或质量证据。

## 验收与服务器启动

先通过本地完整门禁、确定性回放、D03、wheel/CLI及准确head双Python CI，再正式派发。模型解析/硬截止/回退/窗口分母、真实细方向接受、全部候选Archive机会、timing诊断刷新、focus合法性与原MIXED随机流均有专门测试。

技术pilot仅原四规模各1输入、单seed、三组共12run，每run300秒；不是额外正式样本。对68个正式/技术初始化键独立SSGS/exact验证，校验共同初始化指纹；每个技术和正式run结束后全量独立exact核对最终P100与Archive。

219/neu01-2288H-V6、独立`<project_nvme>/llm_flash_20261024_<commit>`，实际入口仅记私有运维台账，不连接旧到期容器。最多96单CPU绑定worker，先物理核再SMT；保守220GiB MemAvailable/磁盘派发门，用户内存保留要求至少200GB。每worker地址空间2GiB，批次MemoryHigh360GB/MemoryMax400GB；启动前按整波保守内存核对。凭据由本机加密存储/远端owner-only tmpfs环境加载，不进Git、manifest、prompt或日志。

## 分析预先约定

最终Archive共同经验参考下的HV/IGD+为主；LLM对MAIN_Q、BILL_MOEAD两个配对比较×两指标，共4主检验单独Holm。先按父聚合变体和两seed，16父为独立单位；按规模分层20k bootstrap、100k配对signflip，固定seed2026102401。不是新父确认；不以某个最好seed选优。

次级描述anytime、Flow/Bill端与中段覆盖、P100、实际生产时间超额、初始化/LS/Polish/API/特征成本、有效策略率、回退和实际动作/focus分布。经验union不称true Pareto。模型理由不作效果判据；显著性未通过不等价。
