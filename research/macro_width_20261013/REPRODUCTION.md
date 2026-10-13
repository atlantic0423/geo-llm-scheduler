# 复现与数据说明

原正式运行source commit为9b2ec3dd1fed371abf382192293338cac9ee4a24，包canonical为1d9a8f4c726312e95d49e553b587f9f08359a90e85b1cd0bdeac9ced6b1daea9。运维修复6982fd2546abf1222b69fce01db417694991fc0a只以独立包装器处理du扫描心跳临时文件的ENOENT，不改原运行源码、数据或单次预算。原16次50任务补跑/400次已完成marker绑定与错误日志全部保留。

正式矩阵40独立base/640run/160同host块，H/T、seed1112/2212、四组NATIVE6/POOL3/POOL6/POOL10。192技术pilot不进入正式质量统计。每root的manifest/specs/输入及每run九项产物和complete marker全部验真；640份离线replay的源marker与完整文件SHA绑定。full raw下载10508文件/519236010字节，两个tar及对应完整文件清单见交付资产；服务器原件保留。

准备Python3.11或3.13以及项目dev依赖，先按完整manifest逐项核对解压文件SHA256。两个formal tar分别解压到同一raw目录下的formal_node0与formal_node1；tar中的FULL_MANIFEST.json是额外下载索引，不覆盖原manifest.json。不要混合pilot、旧实例或另一个结果目录。完整resume launch.json保留missing列表，作为16次补跑的正式标识。

在本报告对应checkout运行（输出必须放在raw之外的新目录）：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) "src"
$env:PYTHONUTF8 = "1"
python scripts/analyze_macro_width_results.py --raw "RAW_DIRECTORY" --out "NEW_ANALYSIS_DIRECTORY" --resume "resume_launch.json"
python scripts/audit_macro_width_results.py --raw "RAW_DIRECTORY" --analysis "NEW_ANALYSIS_DIRECTORY"
python scripts/report_macro_width_results.py --raw "RAW_DIRECTORY" --analysis "NEW_ANALYSIS_DIRECTORY" --out "NEW_REPORT_DIRECTORY" --collection "collection_acceptance.json" --completion "completion_acceptance.json"
```

替换上述目录和文件位置即可。分析使用40base等权、HT两seed平均、20k分层bootstrap/100k配对signflip/seed2026101209，六项主家族联合Holm。80个原生实例分别使用共同经验非支配union，不称真实Pareto。33项统计独立重放；640HV交叉积分、32固定IGD+独立实现。任何统计重放差异、hash变化或缺失单元都拒绝通过，不通过改seed、加样或挑子组修正。

Release中的补跑记录文件实际名为`launch.json`，对应上面占位的`resume_launch.json`；将`--resume`指向该文件即可。显式绑定当前checkout的`src`，避免误用机器上另一个实验的已安装包。

curated data目录保存主结果、配对/基数级统计及诊断汇总。完整run/anytime/operator/call-bucket表、80案例单seed前沿和全部86对PNG/PDF放完整分析图表包，避免将大型派生输出放入Git。原始640最终解复验来自原worker，并由压缩原文/产物SHA验真支持；本轮只计算重现分析，不宣称重新执行所有随机限时搜索或fresh重评所有最终解。

报告中的点wise percentile CI与主signflip/Holm判定可不同；CI没有同时校正。27项额外统计、动作/偏好/压力组、排除补跑base及原生/共同候选目标前缀差异都是诊断，不当新主验证。原生/共同池目标集合不同不证明所有genotype差异，也不能唯一归因最终收益到单一动作或RNG。

所有原文件、错误、中断与补跑记录保留。本轮没有默认main/F6/初始化晋级、下一算法派发或定时任务。GitHub/CI/Release的最终真实版本和下载验收，以最终交付receipt为准。
