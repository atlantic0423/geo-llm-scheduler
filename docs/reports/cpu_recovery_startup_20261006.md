# 双 CPU 历史数据回收与新 MOEA/D 实验启动验收

## Material Passport

academic-research-suite / experiment-agent / run；2026-10-06。
本报告区分历史数据交付、资源预跑验收、正式运行启动和科学质量结论。
历史数据保存/清理、双目标机资源预跑和正式启动已验收；960 次正式计算与科学质量分析尚未完成。

## 执行范围

用户授权先回收服务器项目文件、验真后清理旧测试数据，再直接运行新的 MOEA/D
改进实验。旧 E15 的 1,302/2,592 矩阵不续跑；E15 按最新用户例外只需完整迁回
本地并校验即可删除服务器副本。E16 不属于本批运行范围。

本批保留默认算法，运行独立研究版本：

| 批次 | 冻结源码 | 对照与目的 |
| --- | --- | --- |
| P2 四组确认 | `6dd6d4a84d403a2dd2737d166cb4d3ee1eca25b0` | F6、RPERM、RDIR、A7B，区分方向替换与随机邻域效果、检查 A7 电费代理 |
| D2 剩余配对 | `9e4a8510fa534f65c5116775a8c90d6cb95505eb` | 前瞻机会信息是否超越质量、距离、历史等控制；保持原主终点 |

[P2 Draft PR29](https://github.com/atlantic0423/geo-llm-scheduler/pull/29) 和
[D2 Draft PR28](https://github.com/atlantic0423/geo-llm-scheduler/pull/28)
均未合并；运行版本不称为已升级的默认算法。P2 求解器 `src` 相对原 `f6b7fba` 无差异，
本次仅新增四组冻结调度入口、测试与协议。

## 历史数据保存

原精确清单包含 27,750 个普通文件、0 个符号链接、104,188,942,164 逻辑字节。
本地全部大小/SHA256 已验真，使用 canonical 原始档案加 E15 独立目录保存，
重复的目录、tar 和分卷可按逐文件存储映射恢复到原路径。
没有声称已获得不存在的完整冗余全目录 tar。

| 类别 | 原文件数 | 保存与交付 |
| --- | ---: | --- |
| 唯一代码、配置、日志等补充包 | 2,969 | 逐成员本地验真；索引 Release 的 supplement |
| P1 原始成员 | 8,883 | 原 tar 和 35 原分卷；每个声明文件名/大小/SHA256 精确匹配 |
| P1 tar、分卷、manifest 等重复表示 | 37 | 原 canonical 字节对象按映射恢复 |
| E13 当前成员及原资产 | 3,495 | 本地重新哈希；既有完整 Release 回读核验 |
| E15 两个旧数据目录 | 12,366 | 本地独立目录逐文件验真；用户明确本地保存例外 |

[最终覆盖索引与补充包](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/server-canonical-backup-20261005)
以及 [P1 完整 raw 的 42 项资产](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/p1-fullraw-20261005-3ec2355)
均已发布、回读大小/digest 并抽样下载核验。
[E13 完整 Release](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/200gen-20260927-results)
的 13 项资产也已重新核验。
E15 完整 raw 未发布到 GitHub，本地保存例外不扩大到其他数据。

P1 原 tar 为 17,864,816,640 字节，SHA256
`07893cdc4d756e7d210d68c2ccac1dc4c504244c634d6774cc044292387fcb31`。
除 8,883 声明成员外还有原收集侧车 `artifact_manifest.json`，它的原始字节和成员数
已验证。清单明确按 UTF-8 解码，没有模糊文件名匹配或原数据修改。
源码必须先导入 `source.bundle` 再导入 `source.merge.bundle`，实际 clean checkout
`3ec2355` 已独立复现，详见最终 `ARCHIVE_REPRODUCE.md` 和 `SOURCE_REPRODUCE.md`。

E15 两目录共 38,934,150,250 字节，2026-10-05 已在完整本地验真后逐项重新哈希并删除，
释放约 36.26 GiB。原中断、失败、quarantine 和 incident 证据保存，不作为额外正式结果。
其余清理、部署与启动的实际结果由后续验收节记录。

10 月 5 日 GitHub raw 发布之后，SSH banner 与 GitHub TLS 网络失败阻断后续运维。
重新登录确认未产生剩余清理 audit/结果、未部署、无算法 worker 后才继续。
这些失败及原日志保留为运维历史，没有自动重试算法崩溃。

## 新实验矩阵与验收

P2 为 24 个新 base，50 jobs 的 970001–970012 与 100 jobs 的 980001–980012，
H/T 两电价、五 seed 1101/2202/3303/4404/5505、四 arm，共 960 次。
两服务器每台 480 次、120 个串行四组块、360 worker-hours。
同 base 的全部电价、seed 和四组都在同一主机。
50/100 jobs 分别给 1,800/3,600 秒；与旧 P2 的短预算数据独立分析，不池化。

先跑两个独立 pilot shard，每台 96 次、24 个四组块，共 192 次。
目标机验收使用冻结版本原 `validate_complete`，检查输入/初始化/配置/源码 hash、
同块同 host、最终种群和 Archive 独立 exact 复验的产物证据、无活动 worker 和
complete exit。正式最多每台 24 个 solver，至少预留 6 GiB 内存；不根据可见
128 线程直接满载。两容器各有独立 32 GiB cgroup，但共享项目盘和物理节点。

D2 原本地任务的 130 source/128 probe 已逐条验证，保留原失败现场。
两条 probe 在入口冻结源码守卫失败，失败时未保存 Git metadata，根因未知。
迁移副本经显式技术审查复用 128 完整配对，16 个未完成配对在新主机以完整原预算
重启，包括两条仅 source 完成的配对；quarantine 保留，source/probe 同 host。
计划 8 source 加最多 2 probe，不改变原 24-base、144 source/144 probe 总矩阵。

完整本地质量门：P2 263 tests、86.30% overall/96.80% core，最终 10 campaign
检查；六组 24 次回放与 10,100 独立 exact 通过，默认基线轨迹一致。
[P2 双版本 CI](https://github.com/atlantic0423/geo-llm-scheduler/actions/runs/37282794222)
及 [D2 双版本 CI](https://github.com/atlantic0423/geo-llm-scheduler/actions/runs/37213164311)
已通过。报告分支基线完整质量门 211 tests、85.96%/96.32%，lint/format/mypy 通过。

源码与输入预检：
[P2 预检 Release](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/p2-server-preflight-20261005-6dd6d4a)，
[D2 预检 Release](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/d2-opportunity-preflight-20261004-9e4a851)。
预检交付不代替目标机资源预跑或正式启动证明。

## 后续研究与结论边界

P2 三项事前主比较是 RDIR−F6、RDIR−RPERM、A7B−F6，HV 为主，IGD+、中部
覆盖与多样性为护栏。以 24 个 base 为独立单位，20,000 次 base bootstrap、
配对 signflip 和 Holm 三比较校正。RDIR 须同时超越 F6 与随机邻域；A7B 须超越 F6。
不根据中途显著性加样本或换终点。

D2 variation 的 OPPORTUNITY−BASE 与 OPPORTUNITY−SHUFFLED 须同时通过原冻结
0.001 绝对 normalized scalar gain、正 CI 下界和 Holm 校正门槛，才安排在线父代
选择器完整 HV/IGD+ 对照。离线信息有效性不等于在线优化质量已提高。
D3 算子响应和 D4 峰窗证书仍需设计与成本/正确性验证；D1 自然路径验证未通过全部
事前质量门，先保留而不直接投入完整在线矩阵。旧 E15 不在待跑队列。

本批验收只能证明数据保存与清理完整、冻结源码正确运行及正式计算已开始。
RDIR、A7B、D2 的质量收益和创新性均须完整结果与研究分析后判定。

## 实际清理、目标机验收与正式启动

最终共删除 25,705 个已保存文件、103,729,351,681 字节，约 96.61 GiB。包含前述 E15 的 36.26 GiB，不重复计数；全部逐文件删除 audit 已回收本地。代码、Git 历史和可复用环境保留。

四个部署包均完成本地/服务器 SHA256 对照、bundle verify 和 clean detached checkout。目标机的 P2 campaign tests、节点 0 的 D2 opportunity tests 全部通过。双节点 pilot 共 192 次、48 个四组块完整验真，最终种群/Archive 独立复验共 31,835 个解，OOM/OOMkill 均零。

| 节点 | 正式启动（北京时间） | supervisor PID | 真实 worker | 启动内存余量 GiB | 60 秒 exact 增长范围 |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | 2026-10-06T00:47:34+08:00 | 2311 | 24 | 19.87 | +2438 至 +9736 |
| 1 | 2026-10-06T00:47:34+08:00 | 1024 | 24 | 31.26 | +3463 至 +9757 |

两个正式 supervisor 和 48 个 worker 的真实命令、冻结 commit、两次状态及心跳已回读；每个 worker 的 exact 计数均增长，无失败、无暂停请求。D2 已显式迁移并于 2026-10-06T00:39:06+08:00 续跑，启动快照完成 source/probe=128/128，实际活跃 source/probe=8/0，上限 8/2。

P2 预计 16–19 小时，预计于 2026-10-06T16:47:34+08:00 至 2026-10-06T19:47:34+08:00 完成计算。D2 剩余配对预计 2–3 小时，即 2026-10-06T02:39:06+08:00 至 2026-10-06T03:39:06+08:00。这些估计含队列尾部、初始化与最终 exact/导出余量，不包含最终回收和科学统计。短 pilot 不证明长时间 Archive 内存无增长；完整运行仍保留资源守卫、进程状态、心跳与 epoch。两容器共享物理 CPU，D2 初期重叠负载与后续无 D2 epoch 须在等时间分析中记录并检查敏感性。

完整验收证据及运维源码见本批 startup evidence Release；可移植摘要为 `cpu_startup_20261006_summary.json`。断点续跑复用已验真完整任务/块，中断任务按原预算重跑，保留 quarantine；不声称恢复算法进程内部状态。

启动检查中的 `offspring` 是事件上下文字段，MacroSearch 心跳可写为 0，不能当作累计进度。运维验真条件已改为两次新鲜心跳与累计 exact 计数增长；原字段和检查历史保留，求解器源码及正在运行的 worker 均未重启。

[192 次目标机 pilot 完整 raw、清单与复现说明](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/p2-target-pilot-20261006-6dd6d4a)、[D2 显式迁移快照](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/d2-cpu-migration-20261006-9e4a851) 已逐资产回读 SHA256 并抽样下载核验；迁移快照仅包含已保存的技术状态，不代表 D2 完整矩阵已完成。[本批启动与清理验收证据](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/cpu-startup-evidence-20261006) 单独交付。
