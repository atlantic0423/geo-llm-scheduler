# Jev 单候选与 A6 一次全工件排序试跑

状态：用户授权实现和本机单组运行，效果待实验。独立组名 `JEV_E09_B1_TOP10`，不覆盖原 E09/B6 或默认算法，不重跑已有 MAIN_Q 与 BILL_MOEAD。

## 协议

沿用原 component33 的 200 工件、R2/M12、KV、variant 0 输入与 seed 17401；每组生产墙钟 1200 秒，包含 BILL4/P100 初始化、特征、API 排队等待和搜索。原输入 SHA256 为 `a18b3fb3c68bcf42d4857bd0fbf16ba6bf255e5ea0a52253a61af28049ab9919`。本轮只运行这个新组，原三组作历史描述性比较，不作为同时重跑的随机化控制。

每个触发后的 LS 步重新问当前算子，并按贪心概率 $\epsilon_g=0.9$ 选择最高概率项，其余 $1-\epsilon_g=0.1$ 按完整分布抽样。每步最多构造一个完整候选（B1），不是六个；无合法候选可为零。L5、strict/delta0、真实细权重接受、所有完整可行 exact 候选的 Archive 机会及两重点 Polish 保持。

A6 是明确例外：一次 Choice 返回全部工件的概率，程序取前 $k=\min(n,\max(2,\lceil0.1n\rceil))$ 个不同工件；200 工件时是 20 个。仅概率并列使用显式 RNG 打破，不对 destroy 集再做 epsilon 抽样。随后只执行一次原随机合法 repair；如未改变基因不重问排序、不增加完整候选。其他动作的对象仍在全体合资格对象中逐层按 0.9 抽样。单请求 Choice 上限 255，因此本研究组不静默截取超过 255 工件的输入。

第 1–4 步 exact/接受/Archive 后询问停止，noul 大于 0.5 停止，等于 0.5 继续，第 5 步强制结束。600 秒累计 API 等待、512 请求、40M 计费或预留输入 token 守卫保留。费用按冻结输入单价 0.042 美元/百万 token，上界 1.68 美元，加技术余量 0.10 共 1.78 美元；未知 usage 保留上下文预留。失败或限额后使用明确标记的回退，不沿用过期分布。

这同时改变 B6→B1 与 A6 destroy 选择规则，是研究整包变化，不能当作保持原数学工作量的纯工程加速，也不能从一个种子单独归因两个模块。重点记录有效动作步覆盖、对象回退、各类 POST 数、A6 每步排序请求数、等待、exact 与最终前沿。

## 实际英文提示词与中文阅读说明

提示词版本 `jev_single_v1_b1_a6_top10`。完整请求背景来自 `jev_single.BACKGROUND`，以当前 v4 英文背景为基础，显式把 SIX 改 ONE，并声明 A6 一次排序例外。动作、其他对象、停止问题沿用 `jev_prompts.PROMPTS`；完整逐字中英文保存在 `PROMPTS_B1_EN_ZH.md`。程序实际发送英文，中文只用于阅读。

A6 英文：For A6, return one probability distribution over ALL supplied Jobs, ranking their priority for joint destroy and repair from current_solution. The program takes the top selected_count distinct Jobs together, not one Job per request and not epsilon sampling. Consider the actual weight, waits, KV, assignment, tariffs, demand and history. Proxies are not exact gains. Original random legal repair follows. Every supplied Job must have a probability; probabilities must sum to one.

A6 中文：针对 A6，一次返回全部给定工件的概率分布，对当前解中适合联合破坏修复的工件排序。程序一次取概率最高的 selected_count 个不同工件，不逐工件请求，也不做 epsilon 抽样。依据实际权重、等待、KV、分配、电价、需量与历史；代理指标不是精确收益。后续保留原随机合法修复；每个工件都必须有概率，概率和为一。

输入包括全体 Job、当前 MS/OS/时刻、真实权重、等待/KV/价格/需量、反馈、selected_count 和各 Job 的稳定 ID。输出仍是 Choice 概率分布，程序校验完整键集与概率和，排序选集，原 repair 构造结构候选，再经 SSGS 和 exact；模型不直接生成 schedule。

## 验收与比较

新增 1/20/200 工件一次请求/前 10%/原 repair 独立 exact，255 边界拒绝、并列 RNG 重放、失败随机子集、八动作 B1 exact 上限及逐步动作/停止反馈测试。旧 E05/E09 背景、B6 和抽样回归继续通过。完整工程门与准确提交 CI 必须通过后正式启动。最终对存储种群与 Archive 独立 exact，费用和 600 秒守卫/回退全部报告。

与旧三组比较时共同使用四组最终经验非支配 union 与同一归一化，另外保留旧三组冻结参考指标；经验前沿不是 true Pareto，不能把每个 Archive 成员或 LS 步当独立样本，不做单实例显著性检验。单独运行与旧并行三组有机器负载差异，不作精确纯算法因果解释。
