# 初始化第二步分析复现

原运行源为`5bb666874a86ae59e38aa2513a1c4f7ed1fa8fcb`，canonical `34dd838db2cf62124b967b45ba5049419f6ec96ddab82e6375104d106a1a7f02`。原父种群来自第一步`e57d385b2a4dcc7546e2f54e920cccbe2b6eb7a1`。分析分支保留原源历史，未更改运行语义。

`initial_step2_full_raw.tar.gz`包含正式`formal_node0`、`formal_node1`及独立技术`pilot_node0`、`pilot_node1`。分析仅读取正式两目录。先按`full_manifest.json`核对每个成员的大小/SHA256；不要凭目录数验收。该完整raw已在本地验收，服务器保留；此文档不宣称GitHub Release全raw交付。

| 文件 | 字节 | SHA256 |
|---|---:|---|
| initial_step2_full_raw.tar.gz | 488041489 | 309c67b0f6c7ad5fdfe3dfda80436c8a530bf78a145f64e40ec04c07649fc2b7 |
| full_manifest.json | 由完整原文件记录 | 478d6054e53723851e7e9b934ef11376b6dddb56fffc49f3cd04733915c4c482 |

Python环境安装仓库test依赖及numpy、matplotlib，从分析checkout启动并将其`src`和`scripts`放在PYTHONPATH前部。可移植命令如下，`RECOVERED`为回收目录，`ANALYSIS`为同目录下analysis；分析输出不得在raw内部。

```shell
python research/initial_step2_20261011/analyze.py --raw RECOVERED/raw --out RECOVERED/analysis
python research/initial_step2_20261011/audit.py --root RECOVERED
```

分析入口严格核对运行source/canonical、输入、spec、parent、complete产物SHA、同host和两seed矩阵。每个native输入以正式五组×两seed保留代表的共同经验ND union定义ideal/nadir与参考前沿，退化轴除数为1；主种群HV参考点(1.1,1.1)，IGD+不裁剪。探索Archive指标单列。

36base独立，HT×两seed先平均、50/100/200任务分层bootstrap20k、配对signflip100k、固定seed2026101102；8主/8次级两个独立Holm家族。`audit.py`用独立重采样及校正实现重放16项统计、7项指标边界。`analyze.py`对720个HV用横向面积积分交叉核验，对720个IGD+用另一项目实现交叉核验。

完整73张图保留在回收目录，Git仅包含overview及固定50/100/200任务第一mixed-base示例，不影响全矩阵指标。禁止把pilot混入分析、选子组替代全矩阵、删除中断/失败证据，或把本次产物哈希检查表述为新exact全量复验。
