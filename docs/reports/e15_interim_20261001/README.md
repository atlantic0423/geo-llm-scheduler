# E15 阶段分析附件

上级目录的 `E15_已完成阶段分析_20261001.md` 是中文报告。图均有 PNG/PDF 两种版本，PNG 为 300 dpi。

- `complete_stage_metrics.csv`：A_GEN、A_TIME、B_GEN 的 1,056 条指标，不含未完整的 B_TIME。
- `arm_summary.csv`：每组等权平均摘要。
- `paired_statistics.csv`：按独立基础实例配对的 HV/IGD+ 差、置信区间与 20 项 Holm 校正。
- `statistics.json`：上述统计、原始累计预算/算子漏斗、完整阶段进度及冻结筛选结果。
- `derived_manifest.json`：附件字节 SHA256。生成文件在 Git 中保持原始字节，不随平台转换换行。

原始搜索源码为冻结提交 `3c59adaf8dbe62f822300929a30d585e476eb10f`。报告脚本不修改搜索源码。附件从已验真结果计算，不将预跑、不完整任务或未完成协议混入质量比较。

复现分析：解压已交付的数据包到 `SNAPSHOT_ROOT`，确保其中包含 `runs/` 与 `ops/completed_stage_metrics_20261001.json`。安装项目开发依赖，然后执行：

```text
python scripts/e15_interim_analysis.py SNAPSHOT_ROOT docs/reports/e15_interim_20261001
python -m pytest -q tests/test_e15_interim_analysis.py
```

图形默认使用微软雅黑；其它系统可替换为本机支持中文的字体。字体差异会改变 PNG/PDF 的字节哈希，但不会改变统计结果。Bootstrap 固定为 20,000 次，使用独立显式随机种子 20261001。
