# Jev B1：排除 API 等待的 1200 秒搜索试跑

状态：本轮用户指定的实验口径；效果尚未验证。旧试跑与默认算法保持历史口径。

## 计时

$$
T_{search}=T_{production\ wall}-\sum_i T_{API\ blocked,i}
$$

只扣除同步 requester 调用的阻塞区间，包含请求队列、HTTPS 子进程往返及失败/超时等待。
初始化、特征和 prompt 构造、本地解析校验、候选生成、SSGS、exact、Polish、观察日志均计入搜索时间。
这是经过时间扣除 API 阻塞，并非进程 CPU time；操作系统调度与其他本地开销仍计时。
初始化与终止审计不同：初始化计预算，终止后的独立 exact 审计、导出另报。

MOEA/D 截止、阶段进度、Jev 剩余预算与动作历史收益率使用同一搜索口径。
RunResult.elapsed、summary.production_seconds、步骤 seconds 保留实际墙钟值。
请求记录新增 api_blocked_s，summary 的 Jev 部分新增 search_elapsed_s、api_blocked_s 和 clock_basis。
动作历史旧字段 wall_s 在本轮表示扣除 API 后的动作时间，英文 contract 明确说明。
最终验证逐请求阻塞和、预算达到 1200 秒、墙钟与搜索时间差一致。
搜索在安全边界结束；已经开始的原子候选或 Polish 可造成少量超时。

## 固定范围与额度

沿用原 200 工件 KV variant0 输入、seed17401、BILL4/P100 初始化、strict/delta0 触发、最多五步 LS。
只运行 JEV_E09_B1_TOP10。普通动作与对象选择使用 $\epsilon_g=0.9$ 取最高分、$1-\epsilon_g$ 按模型完整分布抽样。
A6 一次询问全部 200 工件的概率，取前 20 个不同工件，概率并列用显式 RNG，随后一次原随机合法 repair；A6 不做 epsilon 抽样。
每步最多一个完整 exact 候选；完整可行候选独立尝试 Archive；保持 TWO_FOCUS Polish。

本轮禁用累计 600 秒 API 等待门，不禁用其他额度保护：每请求 15 秒、最多 512 请求、最多 40M billed-or-reserved 输入 token。
按2026-10-24重新核对的[官方输入单价](https://docs.typesafe.ai/models)每百万 token 0.042 美元、输出免费，40M 为 1.68 美元，加 0.10 美元余量为 1.78 美元，低于 2 美元上限。
未知 usage 仍保留整段上下文 token 预留；不做自动重试。额度耗尽后仍会本地回退，必须报告模型覆盖率，不能称全部 LS 由模型决策。
总墙钟安全上限独立设为 1200 + 512×15 + 600 = 9480 秒；这只是最坏情况终止保护，非搜索预算或预期耗时。
本机单 CPU、below-normal、2 GiB Job Object 子树限制，至少保留 4 GiB 可用内存；服务器不参与。

## 比较和验收

新独立 checkout、输出目录与 prompt 版本；保留旧源码和结果。不重跑 MAIN/BILL。
旧 B1 的 1200 秒墙钟包含约 600 秒 API 等待，不能从旧最终结果还原搜索满 1200 秒的前沿。
对旧 MAIN/BILL 的历史 1200 秒结果只做描述性比较：运行负载、源码批次和搜索随机路径可能不同。
旧 Jev 与本轮的差异包含时钟和累计等待门的变化，不声称单一因素因果效果。
输入 SHA、初始种群及 Archive 指纹须与旧控制一致；最终 population + Archive 重新独立 exact。
报告两时钟、请求/有效响应、模型动作覆盖、所有回退、费用预留、exact 与前沿指标；单输入单 seed 不推广普遍收益。

Notion 决策：https://app.notion.com/p/4038878b8019812f9deeee4a38167dab

Notion 任务：https://app.notion.com/p/4038878b801981c39d93e09b2b5d8410
