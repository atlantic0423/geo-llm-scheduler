# 初始槽位按方向分配：四臂开发验证协议（2026-10-25）

## 背景

本地诊断（`outputs/pop_decline_diagnosis_20261025`）显示，MOEA/D 底座在前几百个子代存在种群 HV 谷底，主因是初始种群随机对应子问题槽位。按方向分配可以消除这个谷底，但 12 代小样本中最终 Archive HV 没有提升。本批在原等墙钟预算下检验最终 Archive 是否受益。

## 处理与对照

- `BILL_RANDOM`：原 `BILL_PERMUTED` 普通 MOEA/D（BILL4/P100、permuted 邻居替换、无 LS/Polish）。
- `BILL_SORTED`：同上，只把已评价的初始成员按 $(F_{flow}, C_{elec})$ 升序依次放到 Flow 权重从大到小的槽位。
- `MAIN_RANDOM`：原 `STRICT_TWO_FOCUS` 主参考（Q42、B6/L5、严格细方向门、零质量容差、两重点 Polish、A3 tariff）。
- `MAIN_SORTED`：同上，只改变初始槽位对应。

槽位 $j$ 的权重为 $\lambda^j=(j/(N-1),\,1-j/(N-1))$。排序只发生在初始化器完成之后、仍计入生产时钟；初始成员集合、初始化随机流、预算和全部搜索规则不变。RANDOM 臂通过同一入口回放原参考路径。

## 实例、随机数和预算

component33 原 32 输入（16 开发父 × 2 变体，20/50/100/200 任务），原字节与 SHA 不变；单一新 seed 16501；共 32 块 × 4 臂 = 128 run。生产预算 300/600/1200/2400 秒，含初始化。16 父是重复使用的开发面板，不是新留出；单 seed 下变体不作为独立父。

## 分析预先约定

主指标为最终 Archive 在每个输入全部四臂 Archive 共同经验非支配并集下的 HV（参考点 $(1.1,1.1)$）与 IGD+。主比较 `BILL_SORTED−BILL_RANDOM`、`MAIN_SORTED−MAIN_RANDOM` × 两指标共 4 项单独 Holm；先按父聚合两变体，16 父为独立单位，按规模分层 20k bootstrap 与 100k 配对 signflip。次级描述种群 HV 时间曲线（谷底深度）、各时点 Archive HV、端点、代数与实际墙钟。经验并集不称真实 Pareto 前沿；未显著不等于等价。

## 运行与验收

219 服务器独立根目录，最长预算优先派发、单 CPU 绑定、每 worker 2 GiB 地址空间上限，运行期保留至少 200 GB 可用内存与磁盘。每个 run 结束后独立 exact 复核最终种群与 Archive，按 SHA 写完成标记。正式前先跑 16 次短预算 pilot（30/45/60/90 秒），不计入正式样本。
