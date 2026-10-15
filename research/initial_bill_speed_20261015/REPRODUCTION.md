# 初始化速度与构造强度实验复现

本批只比较初始化，P100、原十来源各9加BILL10；没有进化或默认算法晋级。正式矩阵为32全新base、每base两个seed、四臂，共256run/64块。pilot四base/16run、原200输入golden对照及8个单构造profile单列，不纳入正式质量统计。

## 运行源与环境

- 原参考祖先：6d3d0e662a72c2ff0a3ded9ad7bf4d0ccf3c97c2。
- 本批运行Git：3653d52b164382491f07b6df7bd01e6a21c5f1b7，Draft PR61未合并。
- 实际运行源/runner字节摘要：f89cf2773c741027690e273bdd0702da3732a405d6fd0c386353b1d6177e36ac。
- 运行：Python3.11.16、PyYAML6.0.3；Linux目标专项使用pytest9.1.1。
- 分析：Python3.13.5、NumPy2.2.6、Matplotlib3.10.0；不将跨Python末位浮点差宣称为bitwise一致。
- source.bundle具有完整历史；candidate_source.tar.gz保存实际运行字节（原Windows检出包含CRLF）。frozen_source_manifest.json逐109文件同时提供原字节SHA256与Git blob，规范化行尾后也逐Git对象验证。Linux直接Git检出可能改变行尾，因此重放原source摘要时应使用运行tar。

## 从完整结果包分析

将raw归档解压到独立目录，保留其formal、pilot、profiles、legacy200_reference等层次。先按完整manifest逐文件检查size/SHA256及归档校验和。运行源的resume验证限制同host；跨主机的只读结果验真使用分析脚本，它核对recorded host/spec/manifest一致性，无须更改原host。

```bash
export PYTHONPATH="$PWD/src"
python scripts/analyze_initial_bill_speed.py --raw /path/to/raw/formal --out /path/to/analysis
python scripts/plot_initial_bill_speed.py --raw /path/to/raw/formal --out /path/to/analysis
```

分析器拒绝缺失/重复矩阵键、改变的source、输入/spec/config/seed、错误host绑定或产物SHA；验证全部最终P100的唯一性、来源与原Archive回放。每个正式块还严格核对REF4/FAST4最终种群/Archive、trial评分观察与RNG后态hash相同。原worker对每trial重复exact、每最终成员独立SSGS/exact的证据由产物SHA绑定；回收后没有fresh重评全部25600最终成员，不以存储可行标记代替新的全量exact声明。

同base全部四臂、两seed最终P100共同经验ND union提供参考及ideal/nadir归一化，HV参考点(1.1,1.1)。全部256HV和IGD+由第二实现交叉检查。32base独立，两seed先平均；20k按规模分层bootstrap、100k配对signflip、seed2026101517。FAST2/FAST1相对FAST4的HV差/IGD+改善四项联合Holm。未显著不等价，经验参考不是真实Pareto。

FAST2/1会改变去重历史及后续RNG位置，属于完整构造策略比较。不同规模、压力场景使用独立base，分层图表只作描述。固定前沿图使用mixed、各规模/机器数、seed15101八案例；线段仅供观察离散解位置，不代表可行插值。

## 重新运行

在独立运行目录解压candidate_source.tar.gz，安装PyYAML，显式将该目录src置入PYTHONPATH。使用新输出目录：

```bash
python scripts/run_initial_bill_speed.py prepare --root /path/to/new/formal
python scripts/run_initial_bill_speed.py run --root /path/to/new/formal --parallel 24
python scripts/run_initial_bill_speed.py validate --root /path/to/new/formal
```

prepare生成同32输入与64spec，记录实际新host。run最多24个不同物理核，每child4GiB地址空间，合计96GiB，低于用户200GB预算；这不等于聚合200GB cgroup。主机可用内存和磁盘门、每run两小时技术保护、排他锁保持。暂停写stop.request后安全完成当前臂，不自动删除暂停请求。续跑相同run命令，先复验完整run；中断attempt进入quarantine，真实算法失败保留并停止新派发，不自动重试。原pilot全部16run上的再次run调用已验证零重算、run产物SHA零变化。

## 耗时解释边界

生产时钟分别记录原P100、BILL bank及选择；独立trial重复exact和最终100成员SSGS/exact复验单列audit。当前生产代码每trial在引导函数末尾做一次SSGS回放，进入seed_bank又decode一次，这两次都已计入生产，不是被审计时钟隐藏。可复用结果是后续工程建议，未改变本批冻结代码。

原runner没有单独插入最终JSON导出计时；summary的completed字段在导出前写入，不能将其当完整worker退出时间。报告不编造导出秒数。所有实测初始化表同时提供生产和独立审计；总工程吞吐另由真实supervisor起止观察。

profiles内fast0/fast1表示加速布尔开关，不对应正式FAST1单trial臂。8个带插桩单次引导分别为200任务、12/24实例、权重0/.5及两后端；四对candidate/starts/exact/evidence/RNG hash严格一致。cProfile累计时间有嵌套，不能相加；其带插桩秒数不替代正式未插桩benchmark，也不外推为其他状态的固定占比。
