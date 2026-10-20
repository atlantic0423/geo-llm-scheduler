# 本地复现

## 冻结参考与候选版本

参考 commit 为 b148c4addd207fe30a6b2a43d953a04afc35d3a1。REFERENCE.json 记录15个原始 Git blob 的 SHA256；frozen 中的原文件不得格式化或修改。reference_runtime 是串行测试专用的临时绑定恢复器，不是生产运行开关。

原候选实现来自 research/performance-reuse-20261020 的稳定未提交快照；续办、补充验收和交付在独立分支 research/performance-reuse-finish-20261020，不修改原工作树。候选 src 的 canonical_source_hash 为 e9f3730b9cb472f6fe4935ee0991a4803b38d102508ab60b071f5bc53b286de9，按 UTF-8 换行归一化，与 Windows/Linux 路径无关。复现前先核对该 hash 与当前 src 相符；代码提交标识见 Git 历史和 Notion09。

该研究分支继承b148c4a之前的未合并研究历史；本批工程增量用 `git diff b148c4addd207fe30a6b2a43d953a04afc35d3a1...HEAD` 审阅。PR以main触发项目既有CI，不能把其全部历史差异都算成本批优化，也不自动合并该研究历史。

## 安装与质量检查

在本工作树根目录安装开发依赖，确保所有子进程的 PYTHONPATH 指向本工作树的 src，避免加载另一工作树的 editable 安装。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONUTF8 = '1'
python -m pytest -q tests/test_reuse_equivalence.py
python -m pytest -q tests/test_reuse_variant_equivalence.py
python scripts/validate_all.py
python research/d03/verify_publication.py
python scripts/reproduce.py
python -m pip wheel . --no-deps --wheel-dir outputs/reuse-wheel
python -m geo_llm_scheduler.cli.run --config configs/smoke.yaml --seed 1 --output outputs/reuse-cli
```

Linux 使用对应的 export PYTHONPATH="$PWD/src" 和相同 Python 命令。独立 benchmark 期间不得并行运行 pytest、其他 benchmark 或修改 src。

## 固定工作量计时

```powershell
python scripts/benchmark_reuse_performance.py --output outputs/reuse-benchmark --repeats 3
```

每个工作负载执行三对旧/新测时，交替先后次序，输出摘要必须严格一致。工作负载包括20/50/100/200任务的 exact、SSGS、诊断刷新、severity、A7、A8、ENERGY构造、整轮运行，以及20/50任务的BILL4 P100整轮。20/50整轮用P100，100/200用P24，均初始化加一代；后两项不能当作P100正式服务器提速实测。

该脚本不连接服务器、不做等墙钟质量实验。candidate observer 的相同摘要记录开销包含在整轮计时内，输出摘要计算本身在模块计时之外。CPU秒保留原始值，Windows短任务的CPU计时分辨率较粗，不将零CPU秒解释为免费。

可用 --only 筛选工作负载；--profile 是额外的插桩定位，插桩时间不与未插桩时间合并。最终报告采用完整、单独运行、source hash 始终不变的计时，不采用早期并行定位数字。

## 补充大规模完整运行

```powershell
python scripts/benchmark_reuse_fullscale.py --output outputs/reuse-fullscale --size 200 --population 100 --repeats 3 --bill
```

该项固定200任务、3地区、12实例、KV压力、实例和算法seed20261020、初始化seed220261020。完整工作为BILL4混合初始化加一代M11_I4，计时包含初始化和候选观察，不包含随后计算最终摘要的开销。三对旧/新交替顺序；每半次都核对source hash，每对都比较完整结果摘要。与历史P24工作负载分别报告，不合并成多实例统计。

补充脚本只读取和执行本地代码，不登录服务器。结果中的wall为本地实测，不是服务器ETA，也不是等墙钟HV实验。

## 验真文件

REFERENCE.json及frozen保存原始15个Git blob；SOURCE_PROVENANCE.json保存参考核对和14份不变核心策略文件。benchmark_modules.json是原工作树已完成、按相同src及测时脚本hash核对后复用的38项/114对计时；benchmark_fullscale.json是续办时新测的大规模完整运行，不能将前者称为重新测量。TEST_RESULTS.json记录本地完整门禁、后收集的13专项、辅助检查及原始日志SHA；全部原始日志与中断历史留在本地outputs。

data/SOURCE_PROVENANCE.json和data/TEST_RESULTS.json是原快照携带的历史记录，分别保存全部源码内容指纹与原730项门禁（328.76秒）；根目录同名文件是本轮独立检出的补充验真（601.00秒及后收集13专项）。两次运行不能合并成独立测试样本；metadata的换行规范化不修改JSON字段值，frozen原Git blob仍严格保持原字节。

两份benchmark结果在Git中统一为UTF-8/LF，跨平台检出保留相同字节；原Windows测时JSON仍在本地outputs保留。SOURCE_PROVENANCE.json分别记录原始文件SHA和版本化LF文件SHA，并核对解析后的全部字段一致；不要把原始CRLF文件的SHA直接用于校验Git中的LF文件。完整结果摘要比较是在运行时对结果数据计算，与报告文件换行无关。

## 历史执行记录

首轮全量门禁收集的是早期版本，随后源码继续更新，使E15微型续跑测试按预期报告 Source changed；722通过/1失败/1平台跳过，原日志和失败JSON保存在本地outputs。停止源码修改后该项单独通过；最终必须以冻结版本完整重跑为准，不能以单项复验代替整套验收。

早期定位已完成30项三对摘要比较；之后进入的插桩profile被主动停止，不算一个完整正式计时交付。此过程没有服务器任务、算法崩溃或自动重试旧失败矩阵。

本批不修改正式实验的算法参数、源冻结记录或历史产物；不以软件加速证明最终HV/IGD改善。
