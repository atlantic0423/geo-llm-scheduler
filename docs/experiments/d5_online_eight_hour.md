# D5：W 等价加速与五组在线等时间确认

本协议按用户 2026-10-07 的八小时要求重新安排。本轮是独立研究分支；默认 F6、共享 Q、A1–A7、Polish、Archive 和 exact evaluator 不变。旧 D3/D4/P2 源码和结果保留。G/W 的构造代理结果不作为在线质量证据。

## 先验问题与五组

| 组 | A8 改动 | 比较目的 |
| --- | --- | --- |
| LEGACY | 原 PeakCoalition | 当前 F6 实际基线 |
| REFERENCE | D4 每 recipe 独立选择/repair 随机流，原首个可行位置 | 分离随机流变化 |
| G | REFERENCE 加全 recipe 必要条件证书 | 判断节省构造成本是否转化为等时间质量 |
| W | REFERENCE 加至多 L=6 个位置的窗口峰评分 | 判断更好的构造能否抵消评分成本 |
| GW | 同时采用 G 和 W | 判断两机制互补；组合尚未验证 |

G 的适用性仍要求原 horizon 有未选尾工序固定。残余活动并集证书只是必要条件；不适用时完整 repair。W 冻结原 horizon，缺失 Decode 不缩短评分区间，所有固定 900 秒窗口和尾窗口固定除数保留。非选 timing 不变；完整 repair 后保留原 feasibility 和全窗口 peak gate。

W 加速复用每次插入前的事件、整数活动计数及实例功率。试插保留全部旧事件断点并加入自身端点，仍按原实例顺序求和、原 segment/window 顺序积分。因此需逐位相等的窗口、候选列表、attempts、随机流及并列次序测试，并在已见 development panels 上测 CPU 前后变化。禁止删除小差异、改分母或追加 beam/restarts。

## 冻结正式矩阵

- 全新 base：50 jobs 995001–995012；100 jobs 1005001–1005012，共 24 base。实例结构/种子不得与 development/pilot 混用。
- 每 base：H/T，两算法 seed 1101/2202，五组，共 480 runs、96 个同主机五组块。
- 每次 engine wall budget：50 jobs 1200 秒；100 jobs 2400 秒。初始化和相同轻量 observer 在 clock 内，final exact 复验/导出单独报告。
- 所有组 full/fixed B=6、population=100、L=6、原 trigger、共享 Q 和 Polish。generations=1,000,000 只是防止提前以代数结束。
- base 按位置奇偶分到两主机，H/T 与两 seed 均跟随。同块五组串行；循环五种组序在各规模近似均衡。每节点先排 100 jobs，再排 50 jobs，24 solver 槽。
- 总 nominal 240 solver-hour，两节点每节点120小时；24长块和24短块以24槽两波执行，纯预算5小时。预留预跑、最终复验和调度余量，正式启动前用实际 pilot 最大 overshoot/export、RSS 和已有资源重算截止时间。若预跑不能支持八小时，正式冻结前减少 base 数或每次预算，另记新协议，不根据正式质量调参。
- pilot 单独 base 770101–770106、两 seed 6606/7707，H/T；两节点每节点24块120次，60/120秒，五组串行。仅用于资源、产物、RNG/正确性和超时估算，不能算正式效果。

## 统计和解释

主比较为 G/REFERENCE、W/REFERENCE、GW/REFERENCE；各组相对 LEGACY 另列。每实例 H/T、seed 为配对观测，独立推断单位是 base；先对 base 内观测平均，再分规模/H/T报告、base cluster bootstrap 95% CI、配对置换与 Holm。24 base 可用于初步确认，小效应或异质性仍需后续独立验证，不能把480次当480个独立实例。

按每个同实例全部组和 seed 的共同非支配 union 建经验参考前沿，目标用该 union 的共同 ideal/nadir 归一化，统一参考点(1.1,1.1)。主指标 normalized HV、IGD+，补充普通 IGD、双向 C(A,B)、Flow/Bill 极值、anytime曲线、exact吞吐、A8构造时间/attempts/proposals、Q调用和吞吐；union不是已知真前沿。退化维度、重复点及 coverage 容差须一致。不得混用旧D4代理增益、不同预算或不同版本在线结果。

本轮不自动上线状态选择策略。若 W 效果只在规模、H/T、偏好或阶段有价值，先以已有 causal step 统计作探索，再用独立实例验证何时投入更贵构造；不用结果反向挑最优子组宣称确认。

## 运行与验收

独立 campaign 保存 clean commit、规范化 source hash、每文件输入 SHA、初始化、配置、种子、PID、心跳和互斥锁。无历史 trace 快照膨胀。完成 run 由原 P2 validate_complete 验证八项产物哈希及来源，并在 worker 导出前逐解独立 exact 比对；块须同主机。中断未完成 attempt 隔离保存，同机按任务续跑，未宣称算法内存断点恢复。用户 stop.request 永远优先；实际失败停止派发、保存证据，不自动重试。

两节点各32GiB，共享物理CPU和持久盘；短pilot不能证明长期Archive内存稳定。并发以两节点同时的pilot验收为准。运行完立即完整回收、逐文件大小/SHA验真、原验真再统计，完整raw走Release，报告/代码走PR；旧数据和服务器原件校验前保留。
