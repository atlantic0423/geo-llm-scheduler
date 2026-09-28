# E15 实验协议：自适应候选预算与主干比较

E15 是待验证的实验实现；仓库默认算法仍为 MOEA/D 全算法。实验用独立的基础实例集合 A（12 个，50 jobs）、B（20 个，50 jobs）和 D（12 个，100 jobs）。同一基础实例的 H/T 版本除区域分时电价与需量费率外完全相同。每个实例与算法随机种子的所有处理组读取同一份冻结的初始基因型种群。

四组模因实验固定使用当前配置的 Preference 一致性 Trigger、工作质量容忍值 0.10、5 步轨迹、42 状态、第二版六阈值与 A1–A8/RightShiftPolish；只有本页明确列出的预算策略和 NSGA-II 局部控制移植发生变化。Trigger 质量容忍值仍属待敏感性验证的工作参数。

## 算法约定

- Fixed6：每次 MacroSearch 最多评价 6 个同源且去重的完整候选。
- Coverage-v2：当前种群与 Archive 的目标代表点求并集，目标容差内去重。令 $G_i$ 为第 $i$ 个参考方向到代表点的最小夹角。$G_i<Q_{25}$ 取 3，$G_i>Q_{75}$ 取 10，其余取 6；当四分位差不超过 $\max(10^{-3},0.05\max(Q_{75},10^{-9}))$ 弧度时，统一取 6。只在代表点集合或归一化上下文改变时重算。
- Severity-v2：沿用现行各算子的严重度来源。按算子维护独立的确定性历史；前 5 次用 6 预热。此后用旧历史的经验中秩百分位 $p$：$p<0.2$ 取 3，$0.2\le p<0.8$ 取 6，$p\ge0.8$ 取 10。第 6 次开始才按此规则选择，选择后追加当次严重度。
- Sequential：从同一冻结当前解产生最多 10 个确定性顺序候选。先评价前三个；有严格标量改善才评价第 4–6 个；第 4–6 个在统一比较上下文中进一步优于前三个最优者才评价第 7–10 个。不可构成候选或重复候选不占精确评价预算；所有完整可行精确评价仍独立尝试更新 Archive。
- NSGA-II+Memetic-RD：全局父代选择与生存选择仍是 rank+crowding。每个子代在评价前用冻结的父代种群和当时归一化上下文关联一个辅助方向；子代评价后在该上下文中独立关联方向，Trigger 比较两者，避免方向门恒真。Trigger 命中后，整条 5 步局部搜索冻结父代辅助方向，供 Preference、标量接受与奖励使用。每个独立 run 的 Q 表从零开始；最后一步的未来价值为零。子代的探索停滞次数按保留的 phenotype 身份继承，下一代重建此映射。辅助方向不进入 NSGA-II 环境选择。

## 三种公平终止

Stage A 对 Fixed6/Coverage-v2/Severity-v2/Sequential 做 200 代与同一 $T_{50}$ 秒，合计 576 条。Stage A 只按完整配对单元，先在基础实例×电价版本内平均三个种子，再以基础实例为单位比较 HV 差的中位数；三种自适应中最高者进入后续验证。若其差值未超过 Fixed6，记录 `adaptive_beats_fixed=false`。

Stage B 在全新的 20 个基础实例上比较 M0（plain MOEA/D）、N0（plain NSGA-II）、M1（MOEA/D+模因层）、N1（NSGA-II+模因层）：200 代与同一 $T_{50}$ 秒各 480 条。Stage C 在同一实例上使用统一 20,100 次精确评价上限，共 480 条；允许代中终止，精确评价网关阻止超过上限。Stage D 在全新的 100-job 实例上预先保守固定四算法、200 代与同一 $T_{100}$ 秒，共 576 条。强制矩阵合计 2,592 条。

$T_{50}$ 和 $T_{100}$ 使用不同独立预跑实例上的 plain MOEA/D、plain NSGA-II 与 Fixed6 Full 的 200 代耗时最大值乘以 1.1 后向上取整，在任何正式结果生成前冻结。24/32 worker 短预跑只有 32 的吞吐至少提高 10% 且资源门安全时才提升并行度。内存门为 cgroup 限额的 78%，按 `memory.current - memory.stat[file] + memory.stat[shmem]` 估计不可回收占用；活动和非活动文件缓存均可回收，不应阻止派发。磁盘不足 30 GiB 时进入安全失败状态。15 小时只是规划窗口，强制矩阵不会被硬截止截断。

HV 和 IGD+ 在同一实例×终止协议下，使用所有算法和种子的 pooled 非支配近似及共同归一化范围，HV 参考点为 $(1.1,1.1)$。不同协议的归一化 HV 只在各自协议内进行算法比较。统计推断的主要独立单位是基础实例；H/T 为同一基础结构的配对版本，算法种子不能当作独立实例。

## 持久化与恢复

所有条目保存 `summary.json`、Archive 目标与身份摘要、预算漏斗、anytime 检查点、输入/源码哈希和原子完成标记。按基础种子可被 3 整除且算法种子为 101 的预注册分层规则，约 1/9 条保存完整 trace。已完整且哈希一致的条目自动跳过；不完整条目进入 quarantine，单条最多重试两次。Supervisor 和独立 watchdog 写 PID、heartbeat、status 和动态 ETA；最终聚合及校验通过后才写 `finished.json.complete=true`。实验结果不会自动改变现行主干或预算策略。

服务器预跑与启动入口如下。`ROOT` 应指向 `/workspace/zhouhanyu` 内独立 campaign 目录。先确认源码来自 GitHub 已合并的同一提交，并运行仓库质量门；预跑会冻结 `resource_plan.json`。正式启动命令仅调用一次，watchdog 在 SSH 断开后独立存活。

```bash
export PYTHONPATH=src
export GEO_LLM_PYTHON=/workspace/envs/geo-llm-py313/bin/python
ROOT=/workspace/zhouhanyu/campaign_e15_20260928
$GEO_LLM_PYTHON scripts/run_e15_campaign.py pilot --root "$ROOT"
bash scripts/launch_e15_unattended.sh "$ROOT"
$GEO_LLM_PYTHON scripts/run_e15_campaign.py status --root "$ROOT"
```

需要安全暂停时使用 `stop --root "$ROOT"`：当前运行的单条作业会自然完成，后续不再派发。停止后重新启动同一个 watchdog 命令即可续跑，但必须先审查并移除有意设置的 `stop.request`；已有完整条目不会重跑。`failure.json` 或 `incident.json` 出现时先修复根因，禁止把失败状态解释为实验完成。
