# 复现DeepSeek Flash三组分析

运行算法源码冻结在89f14655b27b6a5c4a49f6668705a5e090787089；本目录仅分析，不修改引擎、输入、配置或正式结果。最终分析commit/源码bundle与双Python CI见本批Release和Notion09分析记录。

原32输入、16父、两变体、两seed、64块×3组=192run；实验预算及四主比较在运行前PROTOCOL.md冻结。本目录ANALYSIS_PLAN.json落实经验union、归一化、父级聚合、20k分层bootstrap、100k符号置换、65,536全枚举审计与Holm。分规模、端点、快端窗口、anytime、动作和focus只描述。

从正式结果Release按RAW_PARTS_MANIFEST.json列出的顺序合并分卷为FORMAL_COMPLETE.tar.gz，校验完整SHA后安全解压到结果目录的formal_raw；FORMAL_COMPLETE_MANIFEST.json、FORMAL_LOCAL_RECOVERY_ACCEPTANCE.json放在结果目录根。不要把分卷分别当tar解压，不要把input变体/seed当额外独立父。

```text
python -m pip install -e ".[dev]"
python research/llm_flash_results_20261024/analyze_results.py --data <结果目录> --out <分析输出目录>
python research/llm_flash_results_20261024/audit_results.py --root <分析输出目录>
python research/llm_flash_results_20261024/write_report.py --root <分析输出目录> --report <报告路径>
python research/llm_flash_results_20261024/plot_results.py --root <分析输出目录>
python -m pytest -q tests/test_llm_flash_results_analysis.py
python scripts/validate_all.py
```

绘图需要matplotlib与NumPy，Windows使用Microsoft YaHei中文字体；其他平台须安装等价中文字体或只调整字体配置，不能改变数值。PNG/PDF只用于分享和可视检查；数值CSV/JSON才是统计重放基准，PDF字体/渲染库变化可能改变文件字节。

metric函数来自已验收cb8babc7项目分析，改为本批冻结root seed2026102401，新增非有限值拒绝和相关反例测试。inference()显式接收seed并本地创建RNG，无隐式全局随机状态。每个对比/指标seed由根seed与名称SHA256派生；bootstrap的四规模各4父内重抽，先父内汇总两变体/两seed。原始p是100k Monte Carlo＋1校正，精确枚举p只作审计，未在结果后替换主p。

分析首先重新核对3050个本地原始文件的size/SHA，再检查三组共同初始化指纹、32输入/192运行配对、全部step终止/候选计数。指标双实现、完整IGD+独立算法、4均值/精确置换和Holm独立复算见INDEPENDENT_AUDIT.json。重复分析的所有核心数值文件必须逐SHA相同，记录于NUMERIC_REPLAY_ACCEPTANCE.json。

LS提前停止只计no_gain_threshold且step<4；阈值恰在第5步满足也不算提前。LLM预算末端允许命中trigger后首步前结束，1个run实际出现一次，trigger_without_step单列。LS组件、LS整步及SSGS/exact计时存在重叠，不能全部相加。原主QArchive insertion与LLM新目标事件口径不同，未作等价比较。

本轮只验真原worker的最终exact证据和完整产物SHA，没有重新fresh评估全部187133最终成员。正式服务已正常退出，服务器原件保留；本分析不需要远端密码、API key或再次调用模型。凭据不进入Git/归档。

完整raw、报告、71对PNG/PDF、数值结果、运行/分析源码恢复入口和最终上传/下载验真范围均以Release清单为准；不得把source/startup旧Release误认作正式结果全量发布。
