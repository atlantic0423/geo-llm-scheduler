# MacroSearch 四组正式任务的导出故障修复

2026-10-13 用户授权定位故障并续跑；确认真实计算后返回预计时间并结束，不创建定时。

原算法源为 `6a6d1bbd808ba89b8cc835975127a77fc216ea1c`，canonical 为 `17ca972cb482446d9b21c4bda5149d97a4e0dc8317513b9efa5fdbd354e3598b`。40 个 base、四组、H/T、两 seed、640 次/160 块及 1200/2400 秒引擎预算保持。运行目录继续使用原 source 和 manifest；本分支只交付独立运维包装器，不替换运行源码或晋级默认算法。

## 故障与修复范围

原 worker 在引擎结束及独立 exact 复验后，导出较大 Archive 时因 256 MiB 地址空间上限触发 `MemoryError`。两节点安全停止，OOM 计数为零。已完成的 80+158=238 次逐原 validator、完整产物/输入/初始化/配置/source/hash、seed、Q42 与同 host 复验通过；剩余 402 次按原主机补跑。

包装器保留搜索阶段 256 MiB soft limit 和 nice 10，预留 1 GiB hard limit 权限。只有原 `run` 成功返回后，才取得同一实验共享的导出锁、检查至少 1.5 GiB 可用导出空间并提升 soft limit 至 1 GiB。导出空间保守计入空余容量及可由内核回收的 clean inactive file pages，减去全部 dirty/writeback；不调用全局缓存清理。原搜索派发的 1.5 GiB 内存门与 2 GiB 磁盘门不改。

原独立 exact、JSON/CSV/gzip、SHA、最终 validator 和事务 complete marker 全部继续执行，完整 Archive 不裁剪。导出等待在原引擎预算之外，单独记录；原 hard watchdog 保持。真正的搜索失败不会进入导出门，也不会被此包装器自动重试。

已确认的旧失败文件、stop、status、exit、日志与 attempt 保留为恢复历史；只确认用户本次明确授权恢复的已知导出错误，遇到新的暂停/未知停止拒绝恢复。原恢复入口 quarantine 失败 attempt，已完成正式产物不重算。任务级恢复不等于内部搜索状态 checkpoint。

## 验证与后续边界

使用真实 Archive 内容构造 48,529,884 字节 JSON，在 256 MiB 下复现原写入 MemoryError；新导出阶段写入成功，完整文件 SHA 和原 gzip 回放一致，峰值约 0.338 GiB。测试还检查只有指定 worker 被改写、原参数完整传入、成功搜索之后才进入门、搜索失败不触发门、limit 不影响其他资源设置，以及 dirty/writeback 不计入可回收量。

两目标最终各 10 项专项通过；独立技术预跑验证原 worker 的 exact、完整产物和新 limit 阶段。正式启动另须读取真实 PID/command/limits，至少两次 fresh heartbeat 与同 PID exact 增长。技术预跑不计入640正式矩阵。最终所有正式产物通过原协议验收后才可统计，不提前宣布质量结论。

恢复后的 epoch 和导出操作版本须在最终报告中单列。原 worker 的源哈希仍标识原搜索与导出函数；包装器另有 SHA/commit，不能声称运行过程中没有独立运维干预。Windows 极短预算夹具失败保留原日志，不修改预算或将独立复验冒充单次全门通过。
