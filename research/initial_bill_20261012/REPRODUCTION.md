# 部分电费混合初始化结果复现

原正式source `a8a2afd91e2fb42978a4a4be74e29622b4346c90`，canonical `3cbb17037b480d8bcde090e48ab96e236b6a1d65909f43abe05904168fd04742`，未修改；运行分支PR51未合并。分析走独立`research/initial-bill-results-20261012`，不修改原运行仓库或默认初始化。

## 原始证据

两个tar包含原正式根目录，包括全部输入/spec/父个体、完成块产物以及停止/资源/心跳记录。node0是56完成、16未启动的部分shard，node1是72完成的完整shard；不能称两shard均complete。技术pilot不混正式。

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| node0/formal_node0.tar.gz | 98540538 | 37fecec3d2cc34735cec8af1738824cb195212e33d6f7e2b5781f981ad6538ab |
| node1/formal_node1.tar.gz | 106035171 | 8745fec801a0480fe33fbd2ff8f86d525327165f24a2d21e3cca96f90c826476 |
| node0/file_manifest.json | 由原件size记录 | 9f897a9b57d632ab629a688713fe3877d65e339a768f7fffda0fa3794211b04c |
| node1/file_manifest.json | 由原件size记录 | 6818186115e86433eded2ce3bad08cdd0fbb4868dbdaf474ef9c3264a249d582 |

全部4116文件/2965314222字节已只读流式全量回收；tar成员和解压后的文件逐size/SHA256通过。两个manifest是含`files`大小/hash映射、done/pending与原exit的JSON。原服务器文件不改、不删；本说明不声称完整raw已发布GitHub Release。

配对旧五组来自完整第二步本地raw，source `5bb666874a86ae59e38aa2513a1c4f7ed1fa8fcb`，canonical `34dd838db2cf62124b967b45ba5049419f6ec96ddab82e6375104d106a1a7f02`；旧全量tar SHA `309c67b0f6c7ad5fdfe3dfda80436c8a530bf78a145f64e40ec04c07649fc2b7`、manifest SHA `478d6054e53723851e7e9b934ef11376b6dddb56fffc49f3cd04733915c4c482`。只取新组已完成的相同key，共640旧种群；不把其余旧组当作新组配对。

## 分析入口

环境安装本仓库开发依赖。从分析checkout启动，将`src`与`scripts`加入PYTHONPATH。设`RECOVERED`为含`node0/formal_node0`与`node1/formal_node1`的正式部分回收根，`STEP2`为旧第二步含`formal_node0/1`的raw根，`ANALYSIS`为独立输出目录。输出不能在原raw内部。

```shell
python research/initial_bill_20261012/analyze.py --raw RECOVERED --baseline STEP2 --out ANALYSIS
python research/initial_bill_20261012/audit.py --out ANALYSIS
python research/initial_bill_20261012/flow_budget.py --raw RECOVERED --baseline STEP2 --out ANALYSIS
```

按原完整24产物SHA、input/parent/spec/source/runner/host校验128同host块；最终1280新种群逐100成员、唯一genes、各旧来源9/8配额、库复用、新来源及费用分解检查。原40960新候选及128000最终代表exact复验为原worker产物hash绑定证据，分析没有fresh重评所有候选。

全部1920个新旧配对种群按同native输入、可用同种子15组共同经验非支配union重建归一化ideal/nadir；退化轴除数1。HV参考点(1.1,1.1)，越参考点的近似点不贡献HV；IGD+/IGD不裁剪。严格浮点前沿与原Archive的容差前沿可能有细小差异：不匹配的原Archive另按原实现重放，并在逐种群表记录两种口径的指标差值。此统计前沿不是真实Pareto前沿。

34可用base等权，30完整base敏感性，各base先平均可用H/T与seed。两个16项Holm家族完整保留，20k规模分层bootstrap、100k配对signflip、seed2026101203。`audit.py`独立重放64统计和四个Holm家族；本批缺失/旧诊断base复用使分析只能作为开发诊断，不能替代原36base完整确认或最终迭代收益验证。

`flow_budget.py`在主分析结果读取后追加，输出全部十组在1.05/1.10/1.25/1.50/2.00倍RAW最快Flow共同上限的费用比较，以及BILL20新成员四个权重的描述性诊断。这些是探索性输出，未并入预设Holm确认家族，不把槽位当独立样本。ND占比按完整100成员计算。

共享四家族种子库成本只记录一次；10/20%和PORTFOLIO复用，不拆造未记录的单臂时间。旧预处理成本与新共享库不具相同CPU预算，不能称等时间效果比较。完整68个native算例图和统计/完整trial数值台账保存在分析输出；Git只保留聚合数据、全部缺失名单、脚本、报告及少量预先按算例编号选的图。当前不续跑、不新增算法实验、不设置定时，不触碰Lilanxin。
