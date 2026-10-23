# D1 第三步：新实例真实结构路径确认协议

Material Passport: academic-research-suite / experiment-agent; Mode: design + run + validate; Date: 2026-10-04; Version: D1_realpath_v1; Verification Status: IMPLEMENTATION_VALIDATED; Scientific Status: UNVERIFIED.

用户已授权在本地实现、验收并开始第三步。当前采用的 F6、真实算子、目标与可行性语义不改；本实验观察 F6 原轨迹后在离线目标上执行三组策略。E16、旧 P2/D1/E15 结果、服务器与第四步完整在线比较不属于本批次。

本地实现验收：Python3.13.5，294 tests通过，ruff/format/mypy通过，overall coverage85.67%、core96.86%；24次既有控制回放及10,100次exact核对通过，最终源码再次独立对比冻结3ec2355的plain/full/NSGA-II，三种决策签名均一致。首轮完整回归2个mini campaign失败源于复用venv子进程默认加载旧editable工作树；显式PYTHONPATH绑定本次src后2项与完整门均通过，不将首次失败隐藏。工程通过不等于第三步机制已获支持；仍需独立资源预跑和正式campaign。

## 研究问题与矩阵

第二步终态内部留出有条件延续的机制信号，但目标来自人工等比例 OS/INSTANCE/REGION 扰动。本步检验新基础实例、早中晚阶段、真实交叉/变异和 A1–A6 实际评价路径的适用性，并测量自然机会频率。

50 jobs 使用基础种子 830001–830008，100 jobs 使用 840001–840008；与已有开发、P2、E15 基础种子分开。每基础实例配 H/T 电价、算法种子 1101/2202/3303，合计 16 bases、96 次 F6 运行。生成器沿用 `generate_e15_pair`，H/T 共用初始化基因型池。

50/100 jobs 来源预算分别 600/1200 秒，合计 24 worker-hours。早中晚窗口为预算的 [0,0.2)、[0.4,0.6)、[0.8,1.0)，围绕 10%/50%/90%。窗口外仍统计自然调用分母。最多六个来源 worker、两个离线 worker，资源预跑后可下调，不可无记录上调。

开发资源预跑仅使用 810001/820001、H/T、6606/7707，合计8次，预算为30/60秒。只验正确性、资源与存储，不用预跑质量筛选规则。

## 来源与真实事件

普通 reproduction 保留现有邻域/全局父代池和 RNG；来源预先规定为 `rng.sample(pool, 2)` 返回的第一个父代 a，另一个父代索引 b 只作溯源。不会在交叉后找最相似或效果最好的父代。A1–A6 来源为该 MacroSearch step 冻结 incumbent。只观察去重和预算截断后真正交给 evaluator 的结构提案，不观察被丢弃构造尝试；记录是在评价前发生。

记录 actual MS/OS、来源实际时序、调用子问题索引/权重、调用时 ideal 与当前人口 maximum、路径、generation/step、时间窗口。variation 的 context 来自子代评价前，随后共同 SSGS base 更新 ideal；MacroSearch 保存原 batch context。A7/A8/polish 属于时序路径，不作为本步结构目标。

每个 run/phase/caller-preference 使用独立 RNG 的 Algorithm R 均匀 reservoir，最多16条，共9个桶、144条/运行。所有真实提案（包括 unchanged、空等待、最终未被算法接受的提案）都参与抽样和分母；不按可行性、目标或 gate 结果挑选。保存每个桶完整调用数、采样数和包含概率。最多保留32个 source 的正等待缓存；不逐子代复制 archive。

自然 gate 频率在全部真实调用上计数：actual MS 相同、OS 改变且 exact 诊断有正 intentional residual wait。未改变和窗口外/超时原子收尾调用也记录。observer 用时进入来源总墙钟预算，单独报告；有界观测保持相同代数/同 exact cap 的算法决策序列，但墙钟内吞吐会受观测影响。因此频率只代表本次被观测 F6，不能声称未经观测的完整在线效应。

## 三组实际执行

- ALWAYS_A7：目标重新 SSGS 后执行既有 compression 代表策略的 ActivePack，最多3次附加 exact。
- CONDITIONAL：正等待、actual MS 不变、OS 真正改变时，用来源 residual wait 的 0.25/0.5/1.0 三尺度投影；否则执行同 RNG 的 A7。
- RANDOM：在每个 run/phase/caller-preference 桶的所有正等待且 genotype 改变目标中随机抽取与条件组相同的 TRUE 数；分配不读取候选目标，包含 assignment 改变。空等待和 unchanged 固定走 A7。

第三步随机池跨同桶的不同来源，解决真实轨迹逐来源只有单个目标时随机对照与条件组完全相同的问题；这是前瞻实验约定，区别于第二步逐来源匹配，不能把两批推断混成同一实验。0 TRUE 或全 eligible TRUE 的退化桶全部保留并单独报告，不因此重采样。

每条目标保留共同 source exact、SSGS base exact；每组最多3次附加 exact、50/100 jobs 附加预算分别1/3秒。每组支付实际 residual extraction、投影/构造/exact；匹配规划与共同工作另计。TRUE 投影失败回退到共同 base，不额外尝试 A7；晚完成提案记录及计费但不能供结果保留。unchanged 提案三组均实际执行 A7，保留其机会稀释作用。

## 归一化与主指标

各组只用自己实际按时可行候选更新 ideal，然后用 caller weight 和原人口 maximum 选择保留解，符合当前 MacroSearch 的 batch ideal 更新方式。实际执行的默认旧第二步 `routing_probe` 仍用 frozen 模式，新模式显式 opt-in，不改写旧数据。

主评估把三个组实际已评价候选的 ideal 合并成每条目标的共同评价 context，比较各组**已经保留的解**与共同 base；不会再按另一组候选重挑某组结果。如此避免不同量尺被当作改进。附带固定 ideal 对这些已保留解的敏感性和实际 Flow/Bill 值。共同评估属于离线测量，不是线上可获得的调度器信息。

每条目标相对标量改善为 $(g_{base}-g_{retained})/\max(g_{base},10^{-12})$，可能为负。桶内用 inverse-inclusion 权重 `bucket_events/bucket_sampled` 估计实际调用分布；每个阶段除以该阶段全部事件数，再等权平均三阶段、同 base 六个 H/T×seed 运行，最后16个独立基础实例等权。空阶段按0机会保留并报告；非空桶丢失样本属于错误，停止分析。

两个事前主比较：CONDITIONAL−ALWAYS_A7、CONDITIONAL−RANDOM。基础实例是独立统计单位；规模内50,000次 bootstrap、16-base精确双侧符号翻转、两项 Holm 校正。质量 screen 要求两项均平均差正、95%下界正、Holm p<0.05。即使通过也只支持机制，不直接更改默认算法。阶段/路径/规模/电价/自然频率为诊断描述，不据其挑子集或宣称新的显著发现。

## 验真、资源与失败

完整验收：ruff、format、mypy、pytest与核心 coverage；新增观测有同 RNG 决策/Archive/Q 表语义一致性测试，以及对冻结默认基线的 deterministic replay。来源最终人口与 archive exact 重评；所有离线 base、保留解、提案重新 exact 核对，base 重新 SSGS；来源 gate 与回放 gate 必须一致，每桶随机 TRUE 数严格匹配。每个来源的每个非空阶段对第一个保留事件另做完整三策略重复，最多288个配对目标；无预算绑定时非时间字段必须相同，若绑定则单列不能当作确定性通过。重复开销和规划/验真分开计费，不进入主质量样本。

campaign 保存 clean commit、canonical source hash、协议 SHA256、配置/实例/初始化/输入 hash、任务矩阵、PID、日志、heartbeat、资源记录、每个完成件 hash。使用本批次独占 supervisor lease。`stop.request` 停止新派发并收尾在途。可验证完成件可续用；失败、损坏、孤立 attempt 保留，禁止自动重试/覆盖。若迁移应重跑受影响完整配对运行，不混合宿主。四项科学边界：新实例机制实验不等于完整在线验证；同 quota 不等于同成本；预跑不等于科学完成；观察频率不等于全局算法收益。

自然调用到新目标的来源规则、权重/归一化、随机池、全部样本分母、预算和失败规则均在正式质量产生前冻结。旧 P2/D1 原件与默认 F6 保持原状态。完成后按项目规则发布摘要、代码/CI、完整结果包校验；在此之前明确标记运行中或待交付。
