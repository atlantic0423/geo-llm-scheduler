# 全流程语义保持性能巡检

本批只做工程优化，以 b148c4addd207fe30a6b2a43d953a04afc35d3a1 为直接参考；该参考已包含 2026-10-19 的负载、日历和 Archive 缓存优化。性能增量不得与上一批混为同一个基线。

## 已检查的运行流程

| 模块 | 重复工作与处理 | 保持条件 |
| --- | --- | --- |
| 输入、Genotype 验证 | 直接检查已指派机器，避免为每道操作重列所有 eligible 机器 | 相同错误顺序和 eligibility 条件；不跳过验证 |
| 初始化十类来源 | 同次构造复用地区平均电价；paths 的 Decode eligible 列表只建一次 | 十类配额、扰动、重复恢复、随机流、BILL4 规则保持 |
| BILL 构造器 | 已有前缀资源日历和电价索引；本批共享 paths/SSGS/exact 加速 | 不改四选一或前缀评分累加；未擅自修复唯一性失败 |
| SSGS | 保留已验收的调用内 ResourceCalendar；共享更快的 Genotype 验证 | 完整 rebuild、事件点及求和顺序保持 |
| 完整 exact | 同次评价只建一份 ResourceView；所有操作完整检查；同一索引供所有地区计费 | 不近似、不跳过候选，不跨变化输入复用旧评价 |
| 功率和分时计费 | 半开区间活动查询用排序边界；电价二分跳过零贡献项 | 全部原功率分段和机器求和顺序不变；非零费用原顺序累加 |
| 固定窗口需量 | 继续完整计算每个地区全部窗口，包括尾窗和并列峰 | 不只重算旧峰，不替换 900 秒分母 |
| 诊断刷新 | 同份冻结调度共享资源索引，仍逐道操作计算最早可行时刻 | resource/intentional wait 全量刷新，timing 不改变 |
| Severity | 共享资源边界、半开活动索引和 packing 的同一输入 view | 六项公式和全部累加顺序不变；不降低刷新频率 |
| A1–A6 结构算子 | 原 A2/A3 的 AssignmentWorkloads 已复用；本批共享 paths、验证、SSGS、exact | 原 target、候选生成顺序、预算和代理评分不变 |
| A7 packing | 原基线机器 union 和事件按调用复用；支配关系只算一次；按 operation 分组 | raw move 顺序、层、crowding、两代表、pool 和抽样逐项一致 |
| A8 | 支撑段共用机器索引；当前 coalition 的旧支撑值、singleton 资格和原 horizon 复用；同份 partial 的位置检查复用 competitors | 原峰集合、修复次序、位置/人数/尝试上限和随机流不变 |
| RightShiftPolish | 共用冻结机器事件与资源索引，受益于通用 exact | 同样枚举所有原有限候选、同样全部 Archive 尝试、同样接受与并列规则 |
| 交叉、变异 | 交换位置对用与原显式列表逐索引相同的惰性 Sequence | random.choice 及其长度、候选顺序、随机后态一致；不作拒绝抽样 |
| Q/轨迹 | 复用 Archive 已缓存的存活成员 identity，用于轨迹前后集合统计 | 奖励、状态、探索率公式、Q 更新、terminal 和动作掩码不变 |
| MOEA/D | 对静态权重和 epsilon 的参考方向建立最多16套数值缓存；检查 ideal、maximum、归一化和替换 | 参考射线计算完全相同；不跨子代缓存变动种群的 ideal/maximum，替换规则保持 |
| 动态预算 | 检查 percentile 和 coverage 缓存；保留其历史/输入键语义 | 不改变固定6、两动态策略或采样机会 |
| Archive | 新 identity_keys 复用存活键并协调公开 members 列表变化 | 不限容量、不更换 dominance/tolerance，不省略任何 attempt |
| 对照算法 | NSGA-II、SPEA2、IBEA、MOPSO 共享 gateway、exact 和相关变异加速 | 原生选择/库/位置更新规则未改 |
| observer、trace、最终审计/导出 | 检查调用链；本批保留现有记录和独立最终审计 | 不通过少记日志、少导出或省审计伪造加速 |

## 复用的边界

ResourceView 绑定一份不可变 problem/MS/timing，只在局部调用内存活。显式复用时校验三个对象身份；任一改变就拒绝旧 view。没有全局调度缓存、无限候选缓存、跨结构沿用时序，也没有把目标总额减加当成 exact。

全部原始函数/类的 Git blob、路径和 SHA256 见 REFERENCE.json 及 frozen 目录。reference_runtime 只供串行差分与 benchmark，在上下文退出时恢复所有 import binding；不能在多个线程共享该开关。

## 验收与解释边界

固定代数下比较每个候选、可行性、Flow、TOU、每地区 Demand、每个固定窗口、starts、diagnostics、Archive 顺序/计数、Q、trace 与随机后态。另覆盖错误输入、部分调度、重叠隐藏边界、小数/电价边界、尾窗、并列点、公开 Archive 列表修改、非法 view 复用以及惰性交换列表的所有索引。

计时采用旧/新、下一轮新/旧的交替次序，完整输出摘要相同才接受。最终测时与完整测试分开；早期并行定位计时不作为正式速度证据。固定墙钟下将允许更多工作，搜索轨迹自然可能变化，因此固定工作量等价不等于质量提升已经证明。

报告 RESULTS.md 只填写真实通过的最终门禁和测时结果。服务器冻结运行源、旧结果和其他工作树不改，默认不自动合并或部署。
