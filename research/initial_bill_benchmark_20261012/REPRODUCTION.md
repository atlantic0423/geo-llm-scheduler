# 四组完整benchmark分析复现

下载Release两独立tar并解压到同一raw目录；全成员SHA须与full_file_manifest.json匹配，正式键以formal_execution_index.json为准，pilot单列。安装pyproject的开发/实验依赖，分析源中的CLI如下（参数按下载位置替换）：

```powershell
python scripts/analyze_initial_bill_benchmark.py --raw <raw> --index <formal_execution_index.json> --out <analysis>
python scripts/audit_initial_bill_benchmark.py --data <analysis/data>
python scripts/plot_initial_bill_benchmark.py --data <analysis/data> --out <analysis/figures>
python scripts/validate_all.py
```

本报告运行源6d3d0e6/canonical29388ddf保持。分析含20k分规模base bootstrap与100k signflip，seed2026101204，三Holm家族6/3/3；只用576正式P100，12pilot块不混。指标交叉实现、Archive和代表选择回放是后处理验收，不是再次运行原实验或fresh全部成员exact。源bundle完整可达历史可fresh clone并Git fsck；原server/local保留。完整图以seed11001单种群绘制，经验union包含两seed只是指标参考。

本机完整数据路径不提交到代码；上述CLI允许任意下载根，PYTHONPATH/包安装必须指向本分析checkout的src，不能误用别的editable worktree。复现时结果JSON/CSV字段应一致，绘图PDF元数据时间可能不同；原wall/CPU测量不能在另一主机按字节复现。
