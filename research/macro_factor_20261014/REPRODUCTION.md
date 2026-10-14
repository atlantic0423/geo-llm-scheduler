# MacroSearch四组分析复现

运行源冻结6a6d1bbd808ba89b8cc835975127a77fc216ea1c；canonical17ca972cb482446d9b21c4bda5149d97a4e0dc8317513b9efa5fdbd354e3598b。正式640run、160同host块、40base；pilot与前批不混入。

从Release取得formal_node0.full.tar.gz与formal_node1.full.tar.gz，各解压到独立raw/formal_node0及raw/formal_node1。两包各有FULL_MANIFEST.json，逐size/SHA核对，不丢弃quarantine/ops历史；完整大小SHA由Release校验和提供。原运行source_formal.bundle、修复export_operations.bundle及独立source_analysis.bundle保留完整Git历史，不覆盖旧checkout。

使用Python3.11或3.13与项目dev依赖，在分析bundle独立clone中设置PYTHONPATH指向该clone的src。命令中的目录是用户自己的解压目录，out必须为不存在且在raw之外的新目录。

```text
python scripts/analyze_macro_factor_results.py --raw raw --out analysis_reproduced --pre-resume pre_resume_audit.json
python -m pytest -q tests/test_macro_factor_analysis.py
```

分析ZIP内code目录的独立统计复验入口可对上述输出再执行：

```text
python code/macro_factor_statistics_replay_20261014.py --analysis analysis_reproduced
```

该入口不导入主分析统计实现，核对42项统计及六主/四次两家族；若失败则保留原输出与错误，不修改统计口径。

分析入口逐原validate_complete与全部输入SHA核查；先完整640矩阵及同base配对验真，再计算共同经验ND union归一化HV/IGD+、等总wall统计。报告与机器数据中的主6/次4家族、20k分层bootstrap/100k signflip/seed2026101310冻结；42统计与两Holm家族独立重放的回执提供。不把经验union当true Pareto，不将未显著当等价。

原238完整run的marker必须与pre_resume_audit对应SHA一致，续跑402次和旧导出故障/attempt保留。本轮没有重新执行搜索，也未fresh全量重评最终成员；原worker独立exact证据通过完整SHA绑定。

完整collection脚本及科学绘图脚本随分析ZIP的ops目录交付，未修改算法源码。所有初始化/SSGS/exact/Archive与控制耗时按冻结预算，不合并pilot或前批。GitHub完整asset digest回读及实际下载范围见final_github_acceptance.json；核验前server/local原件保留。
