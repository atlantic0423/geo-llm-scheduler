# D4 两项独立原型启动验收｜2026-10-07

两项已于北京时间 11:43:48/49 同时正式启动。证书门控在 20011（node0）以两个 worker 运行，supervisor 7841；全窗口位置排序在 20012（node1）以四个 worker 运行，supervisor 8064。来源均复用 D4/cfeb369 保存的完整开发调度，不重新生成来源，也不当作新的确认集。

门控四组为原共享 RNG 的 LEGACY、共同新流 REFERENCE、全 recipe 证书、仅单成员证书，每来源调用四个新 seed、每 seed 四次新鲜成本测量。位置三组为同流 REFERENCE、相同额度的 RANDOM_SCORED、PEAK_RANK，每来源八个新 seed。两项各 192 key，覆盖 32 base、H/T、50/100 jobs、三个算法 seed；保留全部阶段、偏好、重复和空调用。原始 A8 单路径、成员/位置/尝试额度与最终峰 gate 保留，两个 treatment 不合并。

## 验收与资源

运行源码固定 `46bcb3385b39969b82b46b2b46fe13136d1badad`，研究分支未合并，默认 MOEA/D/A8/Q/Archive 不变。本地 360 tests、lint/format/mypy、84.57% overall/96.70% core coverage、默认 deterministic replay 均通过；[双 Python CI](https://github.com/atlantic0423/geo-llm-scheduler/actions/runs/37567275306) success。两机各 46 项目标测试通过，源码 bundle、canonical source、全部冻结输入哈希及 clean checkout 核对通过。

八个操作预跑（每项四 key）全部原 validator 通过。门控 144 unit/864 配对来源，17569 次完整 fresh exact 与 10180 次独立证书事件积分；位置 288 unit/1728 配对来源、29247 次完整 fresh exact。`exact_audits` 原字段合计含证书事件积分，不能全称为完整目标评价次数；独立分列清单为 `pilot_audit_counts_v2.json`。未观察到 proposal、RNG、recipe、exact 或 hash 不一致。

完整 pilot 926 文件/246485812 bytes，逐成员大小/SHA256 回收通过；tar.gz SHA256 `772931fff4d184b0ab160923b7b73ad7424d9e2a1bb8fe49c205ad56440e86f1`。完整 raw、清单、输入包、源码 bundle 和复现说明已发布并回读 13 项预跑资产，[Release](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/d4-two-prototypes-preflight-20261007-46bcb33)。启动证据和分列计数作为补充资产交付，保留旧文件，不覆盖。

每 key 预跑约 39–83 秒，worker 最大 RSS 0.0418 GiB。两容器共享物理 CPU；D3 保持 6422/6707、原 2725e1d 与每节点八 source，启动时均 122/144 source，状态 running、OOM0。短预跑不外推长期 Archive RSS；本批是有界离线回放，数值耗时需按共享负载解释。

## 时间与恢复

按两种规模等量外推，门控中心估计 1.58 小时、位置 0.81 小时；只取四个校准 key，预留差异后，门控约 13:15–14:15、位置约 12:35–13:05 完成计算（Asia/Shanghai）。ETA 不包含正式全量回收、统计分析与最终 Release 交付。

独立 campaign 为 `formal_guard` / `formal_positions`，位于部署目录 `d4_prototypes_46bcb338`。每 panel/seed 单位原子 checkpoint 与 SHA sidecar；同 key 同 host、排他锁、停止派发、quarantine 和两小时 key watchdog。复用已验真的完成单位，不声称恢复函数内部状态；真实算法/哈希失败停止新派发，不自动算法重试。所有新 worker 实际命令与心跳/产物增长已检查。服务器和本地原始数据保留，未清理 E15/E16。

当前仅说明实现与启动验收通过。门控真实净成本、位置成功率及 scalar 质量统计尚未完成，不认定新颖性、在线 HV 或正式算法优越性。冻结协议见 [d4_two_prototypes.md](../experiments/d4_two_prototypes.md)，[Draft PR36](https://github.com/atlantic0423/geo-llm-scheduler/pull/36)。
