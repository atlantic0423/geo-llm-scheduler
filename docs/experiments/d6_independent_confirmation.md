# D6：64 个全新基础实例的 W 独立确认

## Material Passport

- 模式：run；对象：合成调度实例与代码实验，无人类参与者。
- 用户授权：2026-10-08 直接启动下一批，启动验明后结束会话；跟进从预计完成时间开始，每半小时一次。
- 依据：D5 完整报告。W/GW 正均值只是探索信号，默认 F6 没有升级。本批独立确认，不与 D5 合并推断。
- 状态：本协议及矩阵在正式计算之前冻结。质量结论尚未产生。

## 研究问题与四组

保持原 MOEA/D/full、fixed B6、L6、共享 Q、触发器、Polish、Archive 和 exact evaluator，只增加独立实例数量。四组为 LEGACY（原 F6）、REFERENCE（独立 recipe/repair 随机流）、W（有限六位置的完整固定窗口峰评分）、GW（GUARD_ALL 与 W）。W/GW 使用 D5 已验收的事件缓存版本。G 单独组本批不跑：D5 的直接整程成本空间很小，本批优先检验 W 是否稳定改善在线解集；GW 保留检查组合。G 是 ALL，不能套用旧 SINGLE 成本。

运行源码仍冻结为 `3a1342cef98dae4b09cd370f40417a119efb4703`，规范化源码 SHA256 为 `592cf56992cfd0079f7dd9f3e581efdf01215af4cbe5f6d66ccba443cdd9e2d9`。本批仅新增协议、输入准备脚本及记录，不能把交付 commit 当运行 commit。PR38/39 未合并，main/default 不变。

## 正式矩阵与调度

- 50 jobs：base 1105001–1105032；100 jobs：1115001–1115032，共 64 独立 base，与 D5 和既有开发集不同。
- 每 base：H/T × 算法 seed 1101/2202 × 四组，共 1024 runs、256 四组块。两个算法 seed 不是独立基础实例。
- 每次 engine wall budget：50 jobs 1200 秒，100 jobs 2400 秒；population100、generations1000000。初始化在搜索 clock 内，最终独立 exact/export 单列。
- base 按位置奇偶分配两个节点；同 base 的 H/T、两个 seed、四组均同节点。块内四组串行，四种循环组序在每规模/节点各完全平衡。
- 两节点各 24 solver，共48；每节点512 runs/128 blocks/256 nominal solver-hour。长块优先、短块补尾，纯预算负载下界10小时40分；按 D5 实测与新 pilot 留调度、导出和机器负载余量，预计11–12小时。上轮八小时是 D5 的完成目标，本批是新的扩大确认批次。
- 技术 pilot：50 jobs 1120101–1120106；100 jobs1121101–1121106，H/T、seed6606/7707、四组192 runs/48 blocks、60/120秒。每节点24 blocks可同时填满24槽。pilot不计正式质量，也不按pilot质量选参数。
- 仅完整块可迁移；同机未完成 attempt 保存 quarantine，再按任务续跑。不是恢复算法内存中的搜索状态。新用户暂停优先，不移除暂停文件。

## 事前统计规则

1. 独立单位为64 base。先在每 base 内对 H/T 与两个 seed 的配对差平均，再按规模等权汇总；分层 base cluster bootstrap20000次、双侧配对 sign-flip100000次，随机 seed2026100806。报告95%CI、原始p及Holm。
2. 主确认家族固定四项：W−REFERENCE 和 W−LEGACY 各自的 normalized HV 增益与 IGD+ 减少量；四项联合Holm，alpha0.05。HV为大优，IGD+为小优。只有本批主家族通过才写对应指标获得确认；不因次指标、子组或原始p通过声称整体优势，不把未显著当等价。不自动晋级main。
3. GW−REFERENCE、GW−LEGACY、GW−W的HV/IGD+是组合次级家族，六项另作Holm，只描述对应比较，不能补救W主家族失败。其他组对、普通IGD、双向coverage、Flow/Bill极值、anytime、构造/精确评价吞吐为探索描述。
4. 每个同实例全部四组/两个seed构造共同经验非支配union，以共同ideal/nadir归一化，HV参考点(1.1,1.1)；完整沿用D5分析的退化维度/重复点/coverage容差。经验union不是已知真前沿。正式D6与旧D5不同union，不直接比较跨批normalizedHV绝对值。
5. 50/100、H/T子组全部报告，禁止事后只保留正子组。偏好/进度状态表是轨迹访问分布，不能解释为同源反事实收益，不启动自适应控制器。
6. 固定64 base为计算预算与精度折中，不能保证检出小效应。若方差不变，标准误约为D5的 $\sqrt{24/64}\approx0.612$；这只是精度估计，不是80%功效保证。正式前用D5逐base分布做保守近似，保存估计与限制，不能据正式结果追加实例直到显著。任何新一批须另行冻结。

## 准入、失败与完成验收

正式前核对两节点host、公钥入口、无活跃旧worker、冻结commit/sourcehash/inputhash、磁盘/cgroup/OOM，以及原目标相关tests。完整pilot八项产物、原P2及D5 validate_complete、每块同host、每个最终population/archive解新独立exact复验、全量pilot文件size/SHA回收通过；按pilot超时/export与RSS检查48槽。原3a运行源码不修改，现有完整424 tests及双PythonCI证据可追溯；新协议/准备器批次还须完整本地门禁及交付CI通过。

原supervisor2秒维护状态/资源守卫和硬超时；任一真实算法失败停止新派发并保留证据，另一节点安全写stop.request收尾，不自动重试。不会因减少聊天定时读取频率关闭本地守卫。新用户pause/取消立即处理。旧E15/E16及已结束D5不续跑、不清理。

完成须1024/256全部原validator/输入/初始化/配置/source/hash/seed/同host通过，真实worker0、supervisor退出、exit complete，无未处理失败；不按目录数宣布结束。原worker最终exact证据用已校验产物hash绑定，不冒充fresh全解再次复验。

结束后完整文件大小/SHA清单、全量raw独立本地回收验真，再按本协议统计、中文Markdown报告；独立交付PR/双PythonCI，完整raw/manifest/checksum/运行与交付bundle/复现说明发布新Release并回读/抽样下载。原件核验前保留。Notion04/05/06/01/07/09同步真实状态，交付完成后暂停本跟进。不自动启动新算法实验。
