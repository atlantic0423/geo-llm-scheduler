# 中断批次的复现与数据边界

本报告对应固定运行源 `bf719d68e882a9cc3058cd5fced3adcaa84757f8`。原主矩阵未完成，不能删除失败标记或将本批标记为 complete。分析只使用 38 个完整父实例的 1,976 条结果，其余 384 条已验证结果与失败、未运行输入全部归档。

## 环境

Python 3.11/3.13，安装仓库的开发依赖。运行源环境为 Python 3.11.16。不要依赖系统隐式随机种子；统计主 seed 为 2026101613，按对比名派生稳定盐。20,000 bootstrap 与 100,000 sign flips，四 Holm 家族完整保留。

## 恢复原始结果

独立 Release `comparison13-partial-analysis-20261019-bf719d6` 保存完整 `COMPARISON13_STOPPED_RAW.tar.gz`、`RAW_MANIFEST.json`、`recovery_receipt.json`、`execution_acceptance.json`、`LOCAL_RAW_ACCEPTANCE.json`、校验和及运行/分析源码 bundle。GitHub 最终资产验真记录另外提供，不把 Release 创建当成上传验证完成。

1. 按 SHA256SUMS 检查每个下载资产。
2. 安全解压完整 tar 到自己的 `RESULTS/raw`，应得到 `RESULTS/raw/formal`。拒绝绝对路径、父级穿越、符号链接和重复成员。不要将生成的分析文件写入 raw。
3. 对照 RAW_MANIFEST 的全部 30,733 项逐大小和 SHA256 验证，合计原始字节 735,751,798。
4. 将本批三个验收元数据 JSON 放入 RESULTS 根。可独立重算 LOCAL_RAW_ACCEPTANCE，不应盲目信任该文件的 accepted 布尔值。
5. 原 validator 将主机身份也纳入检查；离线复核需明确报告原 host 证据，不能把当前计算机名写回冻结 spec。不要启动 `run_comparison13.py run/worker` 来做分析。

## 运行分析

在包含本报告脚本的 checkout 根目录，设置 `PYTHONPATH=src`。下面 RESULTS 是用户选择的恢复目录，REPORT 是独立报告目录。

```shell
python scripts/diagnose_comparison13_failure.py RESULTS/raw/formal RESULTS/analysis/failure_diagnosis.json
python scripts/analyze_comparison13_results.py RESULTS RESULTS/analysis
python research/comparison13_20261019/build_report.py RESULTS REPORT
python -m pytest tests/test_comparison13_analysis.py -q
```

失败诊断只重放公共初始化，保持原上限和种子，不触发 IBEA 进化，不改算法，也不修改正式结果。诊断应在 slot=2、100 次生成、3/4 新候选时失败，累计 exact111；sys tracing 会增加运行时间，不能把该重放 wall 当原生产 benchmark。

分析输出五个 CSV、analysis.json、failure_diagnosis.json、diagnostic_summary.json。完整分析 ZIP 包含这些文件与图文；Git 只保存适合版本管理的汇总、脚本及四对 PNG/PDF。CSV 中每行 run 不等于独立统计单位；父实例先聚合，再四规模均权。

## 数值定义

同输入 13 臂×两个 seed 的最终非支配 union 是经验参考。HV 按参考 ideal/nadir 归一化、参考点 (1.1,1.1)。参考框外矩形贡献零，原点坐标不裁剪；IGD+ 用完整坐标计算。退化轴跨度置1，零跨度不除零。示例前沿只取预定的最小父编号、变体0、seed16101。

A3 主效应为 M10、M11 的均值减去 M00、M01 的均值；替换顺序主效应为 M01、M11 减 M00、M10；交互为 M11−M10−M01＋M00。IGD+ 先变号统一改善方向。Holm 校正保留外部10、因子6、预算4、初始化4个完整家族。本批所有 CI/p 都是探索性，未完成和压力缺失的偏差不能由校正弥补。

## 证据边界

原 worker 最终独立 exact 证据由产物 SHA 绑定，本轮没有重评所有最终解。生产时钟包含初始化；最后审计与导出另外记录；软终止可有边界超时。局部调用、SSGS、exact、Archive 计时相互嵌套，不能相加。

这是一份真实中断批次的回收分析，不是成功完成的正式主实验。没有修改默认算法、运行源、原 PR64、辅助18，也没有自动启动任何后续实验或定时任务。
