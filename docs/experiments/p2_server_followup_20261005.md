# P2 方向替换与 A7 电费代理：双 CPU 独立确认

Material Passport: academic-research-suite / experiment-agent; run; 2026-10-05;
execution and quality UNVERIFIED until separately accepted.

用户授权恢复服务器后直接备份、清理并安排此前保留实验。旧 F6/CG/CT 已完成，
不重复该矩阵；本批 F6/RPERM/RDIR/A7B 为单因素四组，共同使用轻日志和当前精确评价器。
RPERM 是随机替换邻域位置的对照，RDIR 按子代方向选择替换邻域；A7B 只改每工序
第一代表的电费代理排序。所有算子预算、Q、Trigger、Archive 与初始化均保持相同。
默认算法不升级，四组不是多机制组合。

## 冻结矩阵与资源

24 个新 base：50 jobs 970001–970012；100 jobs 980001–980012。
每 base H/T 两电价、五 seed 1101/2202/3303/4404/5505、四 arm，共 960 次。
50/100 jobs 分别 1800/3600 秒，720 worker-hours。此预算比旧 P2 长，
本批独立比较，不与旧 600/1200 秒结果池化。新 base 不用于事前选择或调参。

按各规模 base 位置奇偶分两主机；每个 base 的两电价、五 seed 和四组全部在同一主机。
每台 120 个四组块、480 次、360 worker-hours。arm 顺序循环遍历 24 种排列，
块顺序用固定 RNG 打乱。正式最大 24 个 solver/主机须通过目标主机 pilot；
预留至少 6 GiB 内存，真实 cgroup/worker RSS 和 disk 监控优先于宿主显示值。
若资源不足，减少并发而不减少单次预算或删除 Archive。

pilot 每规模四个独立 base 770001–770004，三 seed 6606/7707/8808，
每台 96 次，50/100 jobs 60/120 秒；pilot 不纳入正式统计。

## 事前统计与否证

HV 为主；IGD+、中部覆盖、端点、吞吐、基因/表型多样性和替换分布为护栏。
同 base/tariff 的所有正式组/seed 构造共同归一化和 pooled 非支配近似；
HV reference 1.1/1.1，IGD+ reference 是经验近似前沿，不称真实 Pareto front。
先按 tariff/seed 配对，再按 base 等权汇总；独立样本数 24，不以 960 次作独立样本。

预注册三项主比较：RDIR−F6、RDIR−RPERM、A7B−F6；报告 base 级配对效应、
20,000 bootstrap 区间、固定随机流的 signflip 检验和 Holm 三比较校正。
RDIR 的升级须同时优于 F6 和随机邻域对照；A7B 须优于 F6；各要求 HV 改善
95% CI 下界大于零且 Holm p 小于 .05。IGD+ 或中部覆盖恶化时不直接升级。
不得观察显著性后追加 base、替换主终点或只汇报有利规模/电价。

## 续跑与交付

原始输入、初始化、配置、源码 commit/canonical hash、命令、PID、心跳、资源快照冻结。
四组块任务级续跑，已验真完整块可迁移；未完成块跨主机须保留 quarantine 并整体重跑。
排他锁防重复启动，stop.request 阻止新块派发并让已启动块收尾。
实际失败保留证据、停止派发，不自动算法重试。完成须逐条 validate_complete、
input/source hash、每块同 host、无 worker、exit complete；目录计数不足以验收。

本地与 GitHub 全量备份及资产回读核验完成后，才按精确清单删除旧服务器测试数据。
保留代码、可复用 Python 环境和本次新实验输入。正式结果走独立报告 PR 与完整
Release，附 SHA256、文件清单、源码 bundle 与复现说明；科学结论另于完整统计后记录。
