# Specification Understanding Report

已完整阅读附件任务书并核对 Notion 维护协议、00、01、当前算法框架、数学模型页面、Coding Contract、40 条当前采用/暂定决策、当前任务、07 版本库和 09 同步入口。源页面获取于 2026-09-21；原始响应存于 specifications/sources.json。

当前对象为静态、离线、确定性的 P→D；P/D 同 Region，跨实例增加 KV delay；双资源并发；秒/kW/kWh/CNY；半开区间；两状态功率；batch 后功率为零；固定 900 秒尾窗。货币统一为人民币，输入代码使用 `CNY`。旧 10-action 和小时实现约定不覆盖当前 8-action/Coding Contract。

完整算法：初始化→SSGS→精确评价→MOEA/D→Trigger→Q-learning/MacroSearch→RightShiftPolish→邻居自身权重替换。Archive 与 scalar acceptance 解耦。

用户补充：assignment 仅在 MS；accepted timing move 全量刷新诊断；A7 无人工 horizon；三项 experimental severity 先写定义再实现；三项 integration invariants 为强制门禁。

数学模型本地 v8 DOCX 已阅读；与 Notion 二进制附件尚未进行字节级一致性验证。规格冲突按当前框架/Contract 与用户确认处理，不自行采用旧方案。

Phase 0 创建文档与骨架；Phase 1 domain/IO；Phase 2 evaluator；Phase 3 SSGS；Phase 4 baseline；Phase 5 structural/MacroSearch；Phase 6 timing；Phase 7 RL；Phase 8 实验；Phase 9 最终验收。未冻结参数配置化，不阻塞已明确模块。
