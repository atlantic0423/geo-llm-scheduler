# D6正式统计与报告复现

本分析使用正式1024次数据和冻结协议，独立单位为64个base。原运行commit为`3a1342cef98dae4b09cd370f40417a119efb4703`；分析交付commit由Release的`analysis_source_identity.json`记录，二者不能混淆。

1. 从[完整原始数据Release](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/d6-formal-raw-20261008-3a1342c)下载完整tar、逐文件manifest、SHA256SUMS、运行bundle；验证SHA并解压到`raw/`，下面应有`formal_node0`和`formal_node1`。原数据和已发布资产保持不变。
2. 从本分析Release下载`d6_analysis_source.bundle`和身份文件。执行`git clone d6_analysis_source.bundle d6-analysis`，在新目录checkout身份文件中的analysis commit；执行`python -m pip install -e ".[dev]"`。
3. 在分析checkout运行以下命令，raw参数指向上一步的正式数据目录，output应使用新的空目录。

```powershell
python scripts/analyze_d6_results.py --data ../raw --output outputs/d6-reanalysis
python scripts/validate_all.py
python scripts/reproduce.py
python research/d03/verify_publication.py
```

统计脚本验明冻结输入与产物hash、四组配对、每块host及完整矩阵，生成run/paired/base/anytime/state CSV、共同参考前沿和report JSON。它不重跑搜索，也不fresh独立exact评价全部最终解；原worker的545341项最终exact证据已与产物hash绑定。

bootstrap20000次、sign-flip100000次、seed2026100806。W的四项主家族联合Holm，GW的六项次级家族独立Holm。不同平台浮点累加可有微小舍入差；逐项验证base均值、CI和p，不用图形像素相同替代统计一致性。报告与数据在`research/d6_20261008/`。

完整分析ZIP包含逐文件大小/SHA清单、PNG/PDF图、独立统计复算工具、报告构建工具、430tests本地门禁/回放/发布图/wheel/CLI日志及原始失败历史。`ops/audit_d6_statistics_20261008.py`和`ops/write_d6_analysis_report_20261008.py`是原本地运维入口；跨机器时只调整其OUT/RAW或BASE路径常量，保持统计与报告逻辑不变。独立审计不导入生产分析脚本，复算72项统计量、两组Holm及全部1024次HV。

原始wall-clock搜索有机器负载影响，以上步骤复现固定结果上的统计，并不保证重新搜索得到完全相同轨迹。pilot与D5均未混入。当前W/GW未通过主/次确认，不修改main或默认算法，不自动开始下一批实验。文献新颖性尚未完成审查。
