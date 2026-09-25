# 60 小时无人值守实验冻结批次

## 研究范围

本批次按 E14 → E11 → E12 → E09 → E06/E07/E08 → E01 → E04 筛选临时下游配置，随后用锁定 Full 配置与 plain MOEA/D、NSGA-II 做 E13 配对比较。E14 离线部分只从 D03 Type-I/Type-II 正式运行读取原始六项 severity，不混入 D03 calibration；在线部分使用新 instance seeds 完整重训。自动筛选仅用于解锁下游，不表示统计显著或研究决策已冻结。

现行基线结构和状态、动作定义不变。E04 的 `step` 放置仅在该实验 arm 中启用；主 Full 使用自动选择的放置结果，若证据歧义则回退为整段轨迹末尾执行一次。所有选择、回退原因和证据路径写入 `selections/` 与 `locks/final_full_config.lock.json`。

## 运行矩阵与公平性

正式配置见 [60h_freeze.yaml](../../configs/campaigns/60h_freeze.yaml)。当前约定：50 Jobs、100 个体、screening 实例种子 71/72/73、最终实例种子 71–75、算法种子 101/202/303。screening 和 E13 每条采用等 wall-clock soft-stop，分别为 180 秒和 300 秒；两条结果的实际 elapsed、超时量、精确评价次数均保留。Pilot 会先测量一代实际用时、进程峰值 RSS 和输出体积，并按可用 CPU、内存和磁盘决定并行 worker 数；正式启动前可以调整 YAML，启动后哈希冻结。

每个 `(instance_seed, algorithm_seed)` 只生成一次初始 MS/OS 种群，三算法读取同一 JSON 与 SHA-256。NSGA-II 使用项目内可审计的快速非支配排序、拥挤距离、rank/crowding 二元锦标赛、父代加子代精英选择；交叉/变异、SSGS 与 exact evaluator 与 MOEA/D 一致。三算法各 worker 限制 BLAS/OMP 单线程，并在支持的平台上固定等价的单 CPU 配额。Archive 仅被动记录完整可行 exact 候选，不参与 plain MOEA/D 与 NSGA-II 选择。

E13 每个实例汇集该实例全部完整配对 run 的三算法目标值，生成共同的非支配参考近似、归一化尺度与 HV 参考点，再计算各 paired cell 的 HV/IGD+。不能从某单一算法自己的结果构造参考，也不能把中途筛选的 Full 结果混入最终 E13。

## 单命令启动与恢复

正式运行前必须确认本地 `main` 与 `origin/main` 同一合并 SHA、工作树干净、`python scripts/validate_all.py` 和 `python scripts/reproduce.py` 通过、迷你 campaign/脱离父进程测试通过、磁盘余量足够，且无其他活跃 campaign。历史 D01/D02/D03 输出不清理、不覆盖。

```powershell
python scripts/run_60h_campaign.py start --detach
python scripts/run_60h_campaign.py status
python scripts/run_60h_campaign.py resume --detach
python scripts/run_60h_campaign.py stop
```

`start` 创建 `outputs/campaign_60h/<UTC id>/`，启动脱离终端的 watchdog；watchdog 维护 supervisor，supervisor 调度相互独立的 worker。启动命令在 manifest、heartbeat、至少一个 READY/RUNNING job 出现后返回。Windows watchdog 使用进程级防睡眠请求，退出时释放。`stop` 只写停止请求：现有 worker 完成后落盘，剩余等待项不启动，随后聚合并停止。若进程意外中断，`resume --detach` 根据 SQLite 和完整 artifact 校验恢复；已验证成功的不重复，不完整目录先移入 `quarantine/`。同一 job 的源码、配置、实例或初始种群哈希不符时拒绝复用。失败有限退避重试，单个 permanent failure 触发阶段显式 fallback，不阻塞其他独立任务。

本批次 60 小时是全局单调计时截止，另有 25 分钟 grace 收尾。临近截止时根据配置预算及最近已完成 run 的 p90 估计停止启动新长任务；已运行 worker 尽量自然完成。若到 grace 结束仍未完成则终止并把该 run 标为截止取消，不伪报成功。磁盘低于安全线会暂停新 worker、保留原始数据并写 incident。

## 输出与解释

`campaign_manifest.json` 保存源码 SHA/哈希、设置哈希、实例路径及 SHA、seed 矩阵；`resource_plan.json` 保存 pilot 和资源上限；`state.sqlite3`、`heartbeat.json`、`watchdog.pid.json`、`supervisor.pid.json` 与 `logs/` 用于运行状态。每个 stage 保存 `paired_metrics.csv`、`missing_jobs.csv`、`failed_jobs.csv`、`stage_summary.json`、`selection_report.md`。最终保存 `campaign_summary.json`、`reports/FINAL_CAMPAIGN_REPORT.md`、`e13_hv_igd.csv`、`runtime_accounting.csv`、`provenance.csv` 和 `checksums.sha256`。

只有 `complete.json`、summary/config/archive/population/Q-table/trace/objectives 全部存在且可解析，输入与源码身份匹配、数值有限、结束原因有效，才标记 `SUCCEEDED`。目录存在或 exit code 为 0 本身都不足以判定完成。最终报告区分完整、部分、失败、回退、下游临时选择；自动筛选结果不写成“已验证”。完成后再由研究人员核查统计稳定性，并在下一次会话把正式结果同步到 Notion。

## 当前限制

第一版 E14 离线 shortlist 为原始六项 severity 的中位数、65% 和 80% 分位向量加基线；属于预先写定的工程筛选约定，不能用离线标签平衡代替在线质量。NSGA-II 与 Full/Plain 的 CPU 公平以单 worker 单逻辑 CPU 亲和性和单线程环境实现，若平台无法设置亲和性则必须中止正式公平实验。Windows 机器保持开机和供电仍是外部前提；软件防睡眠不能对抗断电或人工关机。
