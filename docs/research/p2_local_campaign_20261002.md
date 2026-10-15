# P2 F6/CG/CT：本地启动与服务器接续协议

状态：用户于 2026-10-02 授权本地启动；算法候选尚未被质量实验验证。运行使用独立冻结研究版本，不修改默认算法或恢复 E15。

## 冻结矩阵

- 50 jobs：基实例种子 810001–810012；100 jobs：820001–820012。
- 每个基实例使用 H/T 电价，算法种子为 1101、2202、3303、4404、5505。
- F6 原算法、CG 20% 基因型复制并重新 SSGS、CT 20% 基因型与时间表共同延续。
- 同一基实例、算法种子在 H 下生成一次初始化，H/T 与三组共享不可变基因型池。
- 50 jobs 每次 600 秒，100 jobs 每次 1200 秒；种群 100，代数上限 1000000，其余取 `configs/p2_local` 的对应冻结配置。
- 共 240 个三组对照块、720 次运行、180 worker-hours 的名义引擎预算。16 个持续活跃 worker 的名义下界为 11.25 小时，实际增加进程启动、结果验真、导出、资源等待和尾部排队时间。

## 等时间与日志

三组均关闭历史 trace，使用相同有界观察器；完整最终 Archive、种群、Q 表、计数和算子累计统计保留。仅在预算 10%、25%、50%、75%、90% 和结束时保存目标快照，标注实际观察时间；每 10 秒覆盖一次心跳。观察器开销计入引擎时钟。初始化包含在引擎计时中；最终 exact 复验与导出另行计时。不得沿用带 P1 诊断观察器的旧 F6 结果作为本批等时间基线。

一个 worker 同时只跑一个 solver。每个对照块在同一机器完成三组，六种组别顺序均衡轮换；块顺序使用显式 RNG 随机化。记录执行主机、Python、并发、CPU/内存和 epoch。后续跨机器结果分析须按主机/epoch 分层，不把异构设备的等时间表现视为同一个计算条件。

## 可续跑边界

这是任务级续跑，尚未实现优化器状态的代内恢复。完整结果经所有文件 SHA256、源码、配置、初始化身份和完整标记验真后复用。意外中断的运行从头重跑；证据移入 `quarantine`，原完成结果不覆盖。记录为实际失败的运行停止派发并等待审查，不自动重试崩溃。

`stop.request` 停止派发新对照块；当前块的三组顺序完成后退出。正常迁移应先请求暂停并等到 `ops/exit.json` 为 `paused` 或 `complete`、无活跃 worker。100 jobs 一个新开始的三组块可能还需约一小时收尾。若不得不迁移中断块，异机接续会保留旧块证据并重跑整块，避免三组跨主机混配。

冻结身份只含相对路径；源码 hash 明确采用 UTF-8 换行规范化，Windows/Linux 的 CRLF/LF 差异不会导致误判。summary 原有字节级 source_hash 仍保留，另存 canonical_source_hash。不可变 manifest 和所有 input/spec 文件逐项验真。排他文件锁阻止两个 supervisor 同时派发，同机发现旧 worker 仍存活时拒绝续跑。

## 操作

在经过完整质量门的干净冻结 Git checkout 中设置 `PYTHONPATH` 为该 checkout 的 `src`，准备并启动：

```shell
python scripts/run_p2_campaign.py prepare --root CAMPAIGN
python scripts/run_p2_campaign.py run --root CAMPAIGN --workers 16
```

短预跑使用独立目录和独立种子，不进入正式统计：

```shell
python scripts/run_p2_campaign.py prepare --root PILOT --pilot --seconds 90
python scripts/run_p2_campaign.py run --root PILOT --workers 16
```

迁移时保留源目录，生成全文件清单、大小与 SHA256，传输到服务器持久盘并逐项核验；同一 commit 的完整 Git bundle 两端 verify，源码 canonical hash、输入和初始化 hash 再核对。服务器须重新检查 cgroup 内存、CPU 配额、可用盘与短预跑并发。保留暂停文件为历史证据，再由已授权操作移走暂停请求，运行同一 `run` 命令即可跳过已完成对照块。旧 E15 暂停文件不能因此移除。

启动确认以持续更新的 `ops/status.json`、活跃 PID 和 worker 心跳中增长的 exact/offspring 为准，不以进程创建成功代替真实计算。Windows supervisor 抑制系统自动休眠并在退出时释放；用户主动关机、重启或休眠仍会中断。

资源保护：最多本机逻辑 CPU / cgroup CPU 配额允许的 worker；短预跑先评估 RSS。空闲内存不足 4 GiB 或空闲盘不足 20 GiB 时暂停派发新块。正在运行的块继续收尾；实际 RSS 与吞吐仍需长运行观察，短预跑不能证明最终内存上界。任一失败停止新工作，保留日志和退出证据。

## 后续交付

源码、配置和脚本通过研究分支 PR/CI 交付；本地结果目录长期保留。实验完成或安全暂停后，完整包、manifest 与校验和再按项目规则发布 Release 并回读验证。本次启动不等于质量实验完成，也不等于服务器或 Release 已同步。
