# 初始化第一步：原方法对照冻结计划

冻结日期：2026-10-11；状态：预跑已验收，正式质量尚未观察。

用户授权完成第一步，并最新要求确认运行后返回剩余时间、结束本轮。只运行初始化原方法对照，第二、第三步保持待实验。源代码冻结为 e57d385b2a4dcc7546e2f54e920cccbe2b6eb7a1，canonical SHA256 为 06c3a645ff4bcd24431640eacca0a42fb58309c65ac41e9480b250e5927d9436；原 Draft PR46 未合并。

## 正式矩阵

- 36 独立 base：50、100、200 个任务，mixed / burst / kv_delay / compute_tight / vram_tight / demand_high 六场景，每格两个独立 base。
- 72 个 native H/T 输入；每种电价输入分别构造，不复用 H 构造给 T。
- 10 个初始化种子：11001–11010。
- 11 个原方法：MIXED、FCFS_LOAD、FCFS_RANDOM、SPT_LOAD、SPT_RANDOM、LPT_LOAD、LPT_RANDOM、RANDOM_LOAD、RANDOM、KV_AWARE、ENERGY_AWARE。
- 共 720 个同 host 方法块、7,920 个种群，目标每种群 100 个唯一基因，合计 792,000 个候选。单方法不足 100 时保留实际数量，不用其他启发式填充。
- 原初始化扰动 0.1、尝试上限 10,000。每个候选 SSGS 后 exact，再独立 exact 复验。MIXED 完整基因及 RNG 后态与原初始化器回放核对。
- 同一 base 的 H/T、种子、全部方法固定同 host。循环平移方法顺序仅用于减轻执行位置偏差，不宣称完全平衡的 Williams 设计或相同随机流。
- 只初始化，不进行进化、Polish、时序预处理、额外候选筛选或算法晋级。

## 技术校准与资源

独立 pilot：18 另 base、36 H/T 版本、seed11901、396 种群、39,600 候选均通过原 validator、独立 exact 及 MIXED 回放，未混入正式质量统计。两节点耗时约 137/184 秒；最大单方法约 41 秒，最大 RSS 约 0.039 GiB；raw 合计约 0.687 GB。10 个正式种子在观察正式质量前依据技术成本冻结，不事后加样。

仅使用当前 20005/b518ece5cd01 与 20007/878be0fed7b0，各 24 worker，共 48；可见共享 128 逻辑 CPU，各 cgroup 8 GiB。保留原 6 GiB 可用内存及 20 GiB 物理磁盘派发守卫。新运维目录的洁净文件页辅助只调用 POSIX_FADV_DONTNEED，检查文件大小、mtime 和路径边界，不删除或改变结果字节。

SSH 实测整个个人项目目录分配量约 76.92 GB；正式 raw 按 pilot 预计约 13.75 GB，预计总量仍在历史 100 GiB 配额内。平台本轮要求重新登录，未 fresh 核对个人配额；物理盘 free 不能等同个人额度。4 小时仅为新派发安全截止并安全收尾，预计计算约 30–45 分钟、保守 60 分钟，共享负载可能延长。

## 预先规定的分析

每个 native 输入的全部正式方法和种子合并，去重非支配集仅作共同经验参考包络，不称真实 Pareto 前沿。按参考包络 ideal/nadir 归一化，退化轴 span=1；HV 参考点 (1.1,1.1)，参考点外解只在 HV 中排除，IGD+ 不裁剪。报告 HV、IGD+、补充 IGD、独特 ND 数、双向覆盖、Flow/Bill 端点、方向覆盖和构造/解码/exact/独立复验成本。

统计独立单位为 36 个 base；先对同 base 的 H/T 与 10 seeds 平均配对差，再按任务规模分层。20,000 次分层 cluster bootstrap、100,000 次配对 sign flip，统计 RNG seed2026101101。主家族为 MIXED−RANDOM 的 HV 与 IGD+ 两项联合 Holm；次家族为九个其他纯方法−MIXED 的 HV/IGD+ 共 18 项联合 Holm。正向 IGD+ 差定义为对照减处理。压力场景每格仅两个 base，以描述为主，不把所有重复种群当独立样本，不把未显著当等价。

所有正式块及每种群五项产物、input/spec/config/source/seed/hash/host 和原独立 exact 证据通过，真实 worker/supervisor 退出且 exit complete 后才验收。失败保留证据并停止新派发，不自动算法重试；中断只复用已完整验证同 host 块，部分块保留后明确恢复，不宣称搜索内部 checkpoint。完整验收、回收、统计、图、中文 MD 与 GitHub 交付待后续结果处理，不根据初始化对照宣称最终迭代效果已改善。
