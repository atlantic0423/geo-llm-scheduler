# 独立四组初始化质量与成本对照

## Material Passport

Origin Skill: academic-research-suite / experiment-agent；Origin Mode: run；日期2026-10-12；Verification Status: PLANNED；版本benchmark-v1。工程验收不代表研究效果验证。

## 问题与配方

单独比较BASE、BILL10、BILL20、DEMAND20的初始前沿与真实单臂生产成本。需求中的需量导向按前次DEMAND20处理。36个原诊断base（50/100/200任务×六场景×两个base）、native H/T、seed11001/11002，144块/576种群，P100。复用原Step-2输入，不复用旧计算成果，非全新留出。没有进化或RL/Polish。

每组单独从头生成并评价相同原100成员种群，保留同样随机流和原十类标签。BASE使用原100；BILL10保留每类前9个加10种子，BILL20/DEMAND20每类前8个加20种子。旧最快端不特殊保护。BILL10只生成10种子、40候选；其余各20种子、80候选，每位置四选一。独立库没有跨family重复集，与旧共享四库配方有这一差异，因此不假称复现旧质量数值。BILL10/BILL20同family种子流，前10新成员一致但分别实际计算和计时。

原前缀积分/900秒窗口/idle/active union不变；需量导向费用评分只用Demand，Flow权重0.1；BILL Flow权重0/0.1/0.25/0.5循环。路径近优随机范围为评分极差5%；完整候选四选一按各自费用与总Flow归一化。前缀评分是贪心，没有未来全调度保证。

## 计时与控制

计时从原初始化构造开始，包括原100成员SSGS/exact、该臂新种子构造与首次exact、筛选和最终front提取；另记CPU秒。参考初始化重放、候选重复exact、最终100成员独立SSGS/exact是审计成本，单列不算生产；完整elapsed和导出时间一起报告。每块四组同host顺序运行，四阶Williams在36base/两seed下位置与前序整体均衡，同base全部HT/seed同host。每组都重复生产，不共享bank或父种群计时。计时含独立进度观察器的微小开销；这是固定低并发负载下成本测量，不保证独占硬件延迟。

仅技术pilot选择三规模每node首个base的HT、seed11001，共12块/48种群，用于成本/内存验收，不按质量缩矩阵。正式两当前8GiB容器共享128逻辑CPU；每目标最多8低优先级worker，256MiB地址空间cap，cgroup空余至少1.5GiB才派发下一worker，实际RSS须pilot支持，保留其他项目。空间物理free至少10GiB且个人100GiB保守配额预留2GiB，守卫只停本实验派发。独立目录、OS排他锁、input/source/spec/result SHA绑定、整块完成标记。中断仅恢复完整同host块，保留partial/quarantine；真实崩溃不自动重试，停止两shard新派发。

## 预设分析与绘图

主指标HV、IGD+；IGD与wall/CPU成本必报告。每native case四组及两seed目标union，统一min-max尺度和共同经验ND参考，HV reference(1.1,1.1)。经验参考非true Pareto，比例不得跨旧批比较。36base独立，HT和两seed先均值，20k配对cluster bootstrap/100k signflip，seed2026101204；三个新臂对BASE的HV/IGD+共六项主联合Holm，IGD三项和生产wall三项各自次级Holm，CPU描述性。所有对照都报告，不事后扩样到显著；缺块则保持部分矩阵并列完整base敏感性，不按结果挑子组。

输出每个native算例seed11001的四组单P100前沿完整与快端放大PNG/PDF，共72案例；正文预定展示50计算紧张T、100混合H/T、200高需量T各首个base，必要时另给按差异中位数选例，明确规则。Flow和Bill使用物理单位，左下更好；连线仅引导，不代表可行插值；不合并seed冒充单种群。

## 验收与范围

全部144块/576种群、每成员100且unique/可行、输入/初始化/配置/source/runner/host/seed/SHA、400最终独立复验证据每块、全部worker退出及exit complete才计算验收。回收与统计/MD/前沿交付另列任务。用户确认正式运行后要求返回ETA并结束，不能把启动当完成，不创建Codex定时任务，不改main/默认初始化、旧批次或Lilanxin。
