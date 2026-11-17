# v4 组件搜索：irace 接线与运行协议

状态：本地实验入口；正式配置实验尚未启动，搜索结果不改变默认算法。当前默认仍为 B2＋P1＋N7＋GS4。工程验收回执保存在 `outputs/irace_integration/`。

## 近期公开代码核查与采用方式

本轮优先核查 2025—2026 年论文和作者代码，而后使用官方 R irace 实现外层竞赛。本项目没有复制第三方求解器、改写 irace 的竞赛算法，也不把配置器本身作为创新。

| 来源 | 实际核查与用途 | 边界 |
|---|---|---|
| [Auto-MOEA，ASC 2026，online 2025](https://doi.org/10.1016/j.asoc.2025.114366) | 已有本地全文，第 4 节与 Algorithm 2：组件模板、参数联合配置、默认点与预算对照 | 双频迭代竞赛引用 irace 采样，不等于直接用原版 irace；本次定向检索未找到可核实的作者公开实现，不能推断不存在 |
| [SynthACticBench，GECCO 2025](https://epub.ub.uni-muenchen.de/128369/)；[作者仓库](https://github.com/annaelisalappe/SynthACticBench) | 实读 `synthacticbench/irace_optimizer.py`：参数空间、场景、目标函数返回 cost 的包装 | Python 包装层不是本项目 R 接口的直接替代；未复制其源码 |
| [jMetal Evolver](https://github.com/jMetal/Evolver)；[自动配置文档](https://jmetal.readthedocs.io/en/latest/autoconfiguration.html) | 实读 `scenario-NSGAII.txt` 与 `target-runner-server`：参数文件、实例文件、目标调用、标量 cost | 作为维护中的公开工程参考，不冒称已核实的 2025/2026 期刊论文配套代码；GPL-3.0，未复制实现 |
| [EJOR anytime portfolios](https://doi.org/10.1016/j.ejor.2025.07.024) | 已核摘要和首页；配置组合、运行时限视角 | 本轮未核实可直接使用的作者 irace 仓库；不据摘要复现其方法，不引入会截断末段精修的质量早停 |
| [MAB-CMinSA 作者仓库](https://github.com/GRAFO-URJC/MAB-CMinSA) | 核查仓库目录与配置接口 | 未找到可直接移植到本求解器的 irace 场景，不复制许可证尚未核实的代码 |
| [官方 irace](https://github.com/MLopez-Ibanez/irace)；[官方包说明](https://cran.rstudio.com/web/packages/irace/readme/README.html) | 使用已安装的标准 R irace，按官方 targetRunner API 编写本项目桥接 | 实际版本以每次运行的 `runtime.json` 为准，不以网页开发版号代替安装版号 |

实际安装验证环境为 R 4.6.2、irace 4.5.1。运行不依赖 Chrome；本轮没有下载论文 PDF。R 安装包来源与散列记录在本地回执。

## 搜索边界与公平预算

- `joint` 搜索 8 个组件位置的 30 个候选规则，以及 8 个数值字段；每个组件位置只选一种。
- `numeric` 只搜同一组 8 个数值字段；组件均保持当前默认。两种空间都包含默认点。
- 引导路径、4 Job 重建、间隙 LS 起点、积分时间分配、Q 学习、种群 100、MOEA/D 标量比较和档案规则不改。
- 末段仍预留总时限 5%，右移、压缩、削峰从同一目标调用，仅更新档案；LS 宽度与末段候选预算已经分离。
- 正式内层 50/100/200 任务分别为 600/1200/2400 秒，包含初始化、搜索与末段。独立 exact 复核属于运行后审计，另记耗时。
- 两种搜索各 3 次独立竞赛，每次最多 2400 次目标调用。关闭 adaptive capping 和额外 postselection，避免偷偷追加评价预算。串行目标调用，线程环境固定为 1；不得把不同负载下的残缺运行直接混成正式对照。
- 2400 是完整内层运行次数上限，不是不同配置数。6 次外层竞赛名义约 5600 CPU 小时，另有最多 245 小时验证和 350 小时最终评价；未计预处理与失败重跑。这里提供可运行设置，并未启动这些正式工作。

## 实例、校准与固定 HV

使用已规定的原始 `mechanism_stress120_20261031` 面板，保留原始功率模型，不转换为同名负载功率副本。输入逐文件哈希与已有 v3 划分对应：配置 60、验证 30、锁定评价 30；家族和重复编号组合不跨集合，三个规模平衡。该面板以前用于研究，不能称从未查看的独立留出集。

校准来自已冻结的原面板校准文件，按规模和排序后交替折绑定，不重新拟合，并拒绝评价实例与校准拟合集重叠。原文件路径、散列和每实例折号保存在准备包。

本轮没有可直接复用、同时匹配输入与功率模型的冻结 HV 包。因此先以相同默认配置、固定种子 20261117、一个完整代的参考运行构造每实例尺度；参考运行只用于事先固定坐标映射，其开销单列，不是性能基线或候选筛选。正式配置、验证及最终评价仍运行完整时限。

两个目标顺序为 Flow、Bill，均最小化。参考档案每目标最小/最大值构成固定尺度，归一化参考点为 $(1.1,1.1)$，返回负 HV。保持现有二维精确 HV 的去重、支配处理；不裁剪优于下界的坐标，超出参考点的点不贡献面积。尺度退化、输入非有限、校验不符直接报错，不填补任意范围继续竞赛。参考档案、尺度文件和实现代码共同进入封存清单。短参考档案对极端候选的覆盖能力有限，但所有候选使用同一预先固定映射，不根据后来的搜索结果调整。

`seal.json` 包含源码、运行脚本、配置、实例、校准和指标散列；每次目标调用与恢复前重新校验。准备目录保留机器绝对路径，仅作为本地运行包，不提交到 Git。源码基础 commit 与逐文件散列共同标识代码，不能只凭基础 commit 忽略未提交更改。

## 从竞赛到最终评价

每次竞赛保存全部精英表，向验证阶段只提交第一名。每种方法最多 3 个候选；重复候选合并，不用下一名补齐。验证阶段所有候选与默认使用 30 实例、3 个共同种子。

按种子均值、实例均值、三个规模等权的 HV 选择；平衡任务矩阵下等价于所有配对 cost 的均值。绝对 cost 容差 $10^{-12}$ 内并列时，优先选择相对默认改动字段更少的配置，再按配置散列排序。若未超过默认，则该方法明确回退默认。选择文件与校验完成标记冻结后，三个最终对象才进入 30 个锁定实例、10 个共同种子的评价。

保存逐实例、逐种子得分，后续论文统计须以实例或家族作为统计单位，不能把种子当成独立问题。这里的自动选型和原始结果汇总不等于完成置信区间、显著性检验或论文结论。

## 命令入口

在包含本文件的工作树根目录运行。下面的 `python` 必须是已安装项目依赖的 Python；显式设置 `PYTHONPATH`，避免共享虚拟环境导入别的工作树。`RSCRIPT` 和 `RLIBRARY` 替换为本机已安装路径。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python scripts/prepare_irace.py outputs/irace/formal --split SPLIT.json --calibration CALIBRATION.json
python scripts/irace_campaign.py seal outputs/irace/formal
python scripts/irace_campaign.py check outputs/irace/formal
python scripts/irace_campaign.py run outputs/irace/formal/campaign_joint_rep1.json --rscript RSCRIPT --rlibrary RLIBRARY
```

两种方法各有 `rep1`、`rep2`、`rep3` 三份场景。默认不自动派发全部竞赛。准备参考档案可在相同输入下续跑；已封存后不得重新准备或修改目录。需要变更版本时建立新包，不能覆盖旧结果并冒充续跑。

```powershell
python scripts/irace_campaign.py recover outputs/irace/formal/campaign_joint_rep1.json --rscript RSCRIPT --rlibrary RLIBRARY
python scripts/evaluate_irace.py outputs/irace/formal validation
python scripts/evaluate_irace.py outputs/irace/formal validation --run
python scripts/evaluate_irace.py outputs/irace/formal evaluation_locked
python scripts/evaluate_irace.py outputs/irace/formal evaluation_locked --run
```

评价入口不加 `--run` 时仅冻结和列出任务矩阵。正式锁定评价要求验证已完成且选择文件散列正确。

恢复时先复制 irace 原检查点再交给官方恢复接口。已完成的目标调用仅在配置、种子、实例和结果校验一致时重放；失败/中断记录默认拒绝覆盖，显式竞赛恢复会保留旧尝试后重跑。不得同时启动同一场景的两个进程。验证入口可以重放已完成任务，但不自动覆盖失败记录；先检查原因，再用新的明确批次保留诊断。

目标异常直接终止竞赛并保存状态和错误，不把接口错误伪造成差配置罚分。每次目标保留请求、完整种群与档案、LS 摘要、计数、停止原因、时间和独立 exact 审计；终态散列供结果回收核对。

## 本地技术验收

专项检查包括 30 候选编码与默认点回译、冻结 HV 手工算例、零范围/非有限/参考点外边界、真实算法调用、结果重放、损坏文件检测、失败尝试保存、显式重试、候选去重、验证回退和输入隔离。另运行正式 R irace 的 numeric/joint 短竞赛及恢复测试。

`--smoke` 的 2 秒预算只验证接线，不能评价算法质量。完整质量门为 2058 passed、11 skipped，整体覆盖率 91.61%、核心 96.67%；格式、lint、mypy 通过。最后的 UTF-8 路径修正另外通过专项回归及真实 CLI 交接验证。正式包 120 个实例完成准备、封存和校验；numeric/joint 两种实际正式场景均通过官方配置检查，但未调用正式目标。

两种 R 短竞赛均完成并导出精英；已完成检查点恢复通过，失败目标保留与重试另有测试。合成小实例验证完整跑通 3 配置×1 实例×3 种子验证、3 配置×1 实例×10 种子最终评价及完成结果重放；合成精英输入只用于交接测试，不是真实优化结果。最终回执见 `outputs/irace_integration/VALIDATION.json`。正式实验效果仍待验证。
