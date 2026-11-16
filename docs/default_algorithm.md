# 当前默认算法

2026-11-16，按用户决定将已有工作版 B2 + P1 + N7 + GS4 设为后续默认算法。内部实现标识为 `GS4_GUIDED_RAND_SWITCH_D4`。这是采用决定，不增加独立确认实验的证据，也不宣称已验证全局最优。

## 默认行为

沿用现有 BILL4 初始化、MOEA/D、MS/OS 编码、事件驱动 SSGS、exact evaluator 和 Pareto Archive。三个改进共同启用：

1. **收益反馈引导重建**：普通变异和引导重建之间，按近期档案 HV 增量/耗时调整路径概率。每次最多重建四个 Job；采用已有中性重建版本，不按子问题方向改变重建规则。基准概率 0.2，探测概率 0.02，衰减时间 300 秒，指数 2，预热 5 秒。
2. **收益驱动局部搜索时间分配**：沿用 N7 的积分额度、5%–50% 份额范围、0.2 基准、120 秒衰减和 5 秒预热。搜索期维持 Q1/NET/G9/B3L3，Q 表每次独立运行从零学习。
3. **末段档案协同精修**：最后 5% 总预算统一右移，再调用 A7/A8；结果只进入档案。搜索期不把时序精修放回种群，A7/A8 不参与搜索期 Q 动作选择。

未晋级的精修门控、区域块混合、间隙分配、分段/交错岛均未启用。默认变更复用现有实现，不另写一套算法。

## 运行入口

`python -m geo_llm_scheduler.cli.run` 不指定 `--config` 时选择当前默认。也可显式使用 `--config configs/default.yaml`。

```powershell
python -m geo_llm_scheduler.cli.run --instance path/to/instance.json --calibration path/to/frozen_calibration.json --output outputs/new_run --seed 1
```

校准文件若是按 fold 索引的字典，需要增加 `--calibration-key n100_f0`（使用对应实例实际所属 fold）。不自动估计、猜测或重新拟合校准值。输入功率模型由实例文件确定，不在默认切换时改写。

默认总墙钟包含初始化：50/100/200 任务分别 600/1200/2400 秒；其他规模必须显式提供 `--seconds`。短时间运行只用于工程检查。批量实验继续使用冻结的 120 实例 manifest 和配对种子；本次切换不启动实验。

输出沿用现有实验 runner，包括完整配置、输入和源码哈希、搜索记录、档案、独立 exact 复核和结果 manifest；输出目录已经存在时拒绝覆盖。

Python 调用可用 `geo_llm_scheduler.default_algorithm.run_default`，或通过 `experiments.runner.run_config` 读取默认 profile。低层 `Config()` 和 `engine.run()` 是历史核心引擎接口，不能单独代表当前完整默认算法。

## 历史复现

原 `configs/default.yaml` 原样保存为 `configs/legacy_core.yaml`，旧 campaign 的默认配置路径同步改为该历史文件。显式旧实验 YAML、冻结研究臂及其算法行为保留。README 中旧 `full_method.yaml`、42×8 Q 表和十类初始化描述只对应历史核心框架。

规格依据：[当前完整算法框架](https://app.notion.com/p/3e08878b80198191b9e8cb69c1b4ea91)。实现映射：`default_algorithm.py` → `experiments/trans_runner.py` → `experiments/s4_moead.py` 中的 GS4 冻结臂。
