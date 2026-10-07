# D5 分析与复现入口

运行 source 固定3a1342cef98dae4b09cd370f40417a119efb4703；此目录为独立交付批次。完整报告见[D5_results_analysis.md](D5_results_analysis.md)，统计规则见[analysis_plan.md](analysis_plan.md)。没有自动升级默认算法或启动新实验。

完整raw与每文件manifest从[D5 complete Release](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/d5-complete-20261008-3a1342c)获取，先用SHA256SUMS及full_manifest.json验真，再向全新目录解压。只包含formal_node0/1，不把pilot混入正式统计。

在冻结运行bundle检出3a1342c，原geo_llm_scheduler.experiments.p2_worker及d5_worker的validate_complete须逐run通过。分析bundle检出独立报告commit，安装dev依赖；不以报告Git commit覆盖冻结spec的source_commit。分析命令：

```bash
python scripts/analyze_d5_results.py --data /path/to/verified/raw --output /path/to/new-analysis
python -m pytest -q tests/test_d5_analysis.py
python scripts/validate_all.py
```

分析src与原冻结src相同；新脚本只做只读后处理。独立统计复算工具及真实验收日志随完整Release中的analysis_and_acceptance.zip保存。输出JSON/CSV应与data逐数一致，bootstrap和sign-flip使用显式RNG；未重跑所有wall-clock搜索，不声称算法结果逐位重现。

当前完整发布/双Python CI及六项闭环以最终delivery验真receipt和Notion09为准；此初始summary保留发布前时间点，不覆盖历史资产。
