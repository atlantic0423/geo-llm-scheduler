# 外置快速右移选解筛选：实现协议

状态：研究配置，尚无本轮效果结论，不晋级默认算法。依据用户批准的 2026-10-29 选解规划 V2。

## 矩阵与共同约束

12 配置：B0（删 P3、无外置），B1（原 Q 内 P3），以及 T_ALL、T_BILL、POOL_RANDOM、DIRECTION、REGION_TIME 各配 P/F 两种内部任务范围。P 沿用部分合格 Prefill 池；F 为全部合格 Prefill，仍保留原成功移动上限、有限关键时刻及 3 个完整候选预算。

面板为 st_n100_d06/d09/d12/d13、st_n200_d01/d07/d09/d14，seed 16901，100/200 规模分别 1200/2400 秒，均含初始化。共 96 次，每次从空 Q 表开始；沿用既有冻结校准映射，不做预训练。所有组保持 Q1/NET、$\gamma=0.9$、B3L3、其他动作与 A8 门控不变。

每条执行过 Q 步骤的轨迹终止更新后，最多服务一个解。Q 轨迹和外置模块共同准入原 20% 累计 LS 份额，不再增加第二道 20% 门；不可分割调用仍可能软超额。选解、构池、诊断、构造、exact 与回流都由外层计时一次。右移不参与 Q 奖励或 bootstrap。Decode、MS、OS 冻结，仅推迟 Prefill，Flow 不变、总电费严格下降；所有完整候选 exact 并尝试 Archive，回流仍限 Archive entrants。

## 三种新策略的同池定义

候选池为当前种群与当前轨迹终点的物理去重并集，100 个种群槽最多产生 101 个解。物理键是完整 MS 与未经取整的 starts；OS、诊断字段和目标相同不等于物理相同。相同物理输入保留最小 OS，按物理键稳定排序。被 Archive 支配的种群解仍可选。不保存跨调用失败集合，不冷却、不跳过历史失败。

|策略|组选择|组内选择|
|---|---|---|
|T_ALL|当前 Q 终点，不限区域|该终点|
|T_BILL|当前 Q 终点处于已有归一化 Flow 的电费端|该终点|
|POOL_RANDOM|共享池全部不同解|独立命名 RNG 均匀选择|
|DIRECTION|复用当前 MOEA/D context、逆权重方向关联，轮转非空方向|自身权重下标量最小；容差并列锚定全组最小值，再按稳定 identity|
|REGION_TIME|既有 FrozenFrame 三段中累计服务秒数最少的非空段；精确并列轮转|独立命名 RNG 均匀选择|

区域秒数含构造、exact 和回流，不含公共构池/分组及输入诊断。公共成本独立记录并计入 LS 总成本，不给三个区域重复记账。区域耗时排序不引入新阈值。MOEA/D 关联 context 和 LS 的 FrozenFrame 各守原定义，不混用。

POOL_RANDOM 随机的是外置右移的输入解，LS 算子依旧由 Q 选择。它不是 Uniform 算子选择对照。

## 诊断与辅助重放

本批全部十种外置配置均启用同一日志：选中物理指纹、完整输入快照、内部范围、轨迹编号、来源、候选池/方向/区域大小、选择/输入诊断/服务耗时、完整空调用与截止中断。指纹在单个实例内有效，跨实例须结合密封输入 SHA；不用于在线筛选。默认旧配置不启用新诊断，保留原路径。

`analyze_rightshift_inputs.py` 统计同输入同范围完整空调用后再次调用的次数、耗时及占比。截止跳过、区域跳过不算调用；中断不算完整失败。该耗时不是失败缓存必然可节约的时间。

显式指定日志索引可离线 exact 检查全部合格任务的第一步关键时刻，记录活动并集变化、TOU/Demand/总电费差、部分池之外的改善机会，以及 TOU 节省被 Demand 增加抵消的情形。辅助扫描不占正式搜索预算，也不反馈给算法；它不等于重放随机贪心多步过程。大量逐时刻审计保留在线下，避免污染正式调用速度。

## 执行入口与验收

在已有 campaign runner 中显式指定 `--study selection`；省略时仍走旧十配置。

```powershell
python scripts/run_lsv2_fast_rightshift_campaign.py prepare --study selection --root outputs/selection/formal --historical <历史开发面板> --prior-q <冻结校准批次> --pilot-note <技术验收说明>
python scripts/run_lsv2_fast_rightshift_campaign.py preflight --study selection --root outputs/selection/formal --job st_n100_d06_s16901
python scripts/run_lsv2_fast_rightshift.py --study selection --arm DIRECTION_P --input <实例> --calibration <单fold映射> --seconds 1200 --out <新目录>
python scripts/analyze_rightshift_inputs.py --log <rightshift_steps.json.gz> --out <新JSON路径> --run-seconds 1200
```

prepare 校验既有输入、seed、初始化指纹及非训练 fold；新矩阵严格核对这八个实例与预算。validate 同时校验源码、提交、host、全部文件哈希和配置语义，拒绝不同 study 混用。服务器完整 preflight、exact 抽查及既有资源门保持。正式派发前须记录实际资源和技术短验收，不凭旧批次假定服务器空闲。

测试覆盖物理去重、同目标不同调度、方向轮转/空组/标量并列、按秒分配、独立 RNG、terminal 信用隔离、截止、exact 不变量、96 次密封矩阵、篡改拒绝和离线诊断。基准脚本 `benchmark_rightshift_selection.py` 对同一历史输入与冻结 93137d6 的 endpoint 路径逐解比较，并报告选择与指纹成本；跨平台恢复值按 Coding Contract 数值容差核验，不能用微基准推断最终 HV。

本轮筛选结果须同时看最终 HV/IGD+、服务成本、空率、Archive 贡献、其他 Q 动作数和代数；单 seed 开发面板不能形成跨 seed 稳定性结论。
