# 实现契约

静态问题及其子对象、Genotype、Schedule、EvaluationResult、Candidate 使用 frozen dataclass 和 tuple。ProblemInstance 共享；MS 在代码写作 Genotype.ms，按 P/D 交替展开，是 assignment 唯一权威。Schedule.starts 只存开始时刻，completion 从固定 duration 推导，避免重复可变状态。

structural proposal 只携带 genotype；EvaluationGateway 必须 SSGS rebuild。timing proposal 携带原 genotype、新 Schedule、selected；MacroSearch 检查 genotype 不变及未选 timing 冻结。第一版不持久缓存资源事件，每次由 MS/timing 重建，无独立 assignment cache。

accepted A7/A8/Polish 全量刷新 diagnostics：冻结所有 timing，逐操作移除自身，查询当前 EST 后最早可行开始。不调用 SSGS，不传播移动。公式见 specifications/state_severity.md。

utils/numeric.py 集中定义相对及时间/Compute/VRAM/功率/费用绝对容差 1e-9。归一化 Tchebycheff scalar 使用独立的无量纲 scalar tolerance 1e-9，不与 CNY 费用容差耦合。A8 全窗口 gate 使用独立 power tolerance。EPS_NORM/EPS_RATIO=1e-9 为数值保护，EPS_DIRECTION=1e-6 保护零权重。身份包含 MS、OS、按时间容差量化的 starts。

每个 run 一个 master seed，RNGManager 用 SHA256 派生具名子流，不依赖 hash() 或全局 random。相同环境、固定代数、实例/配置/seed 复现核心路径。

## 强制集成不变量

1. 所有完整可行 exact candidate 独立尝试 Archive，包括未被 scalar 接受及 Polish 候选。
2. MacroSearch raw evaluation 全完成后，用统一 ideal 和冻结 population maximum 比较；相同集合的 best 不受顺序影响。
3. structural rebuild；timing-only 冻结未选操作；diagnostics 刷新不改变 timing。

exact 为可行性和目标真值；代理只排序/构造。当前未启用增量评价。

Coverage-aware 的 warm-up 与方向密度只使用按 Flow/费用容差去重后的 objective-space representatives；External Archive 本身继续保留不同 phenotype。Archive insertion 是历史成功插入事件，和最终保留贡献分开记录。
