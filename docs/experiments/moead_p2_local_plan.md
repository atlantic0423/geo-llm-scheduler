# MOEA/D P2 本地实现与验收计划

日期：2026-10-02。状态：本地工程计划；算法收益待实验；服务器部署须由用户审阅本地结果后确认。

## 依据与范围

本地基线为 GitHub main 的 `3ec2355fbc43993375daab06ec0ea39fb9d86991`。实现放在独立分支 `research/moead-p2-local-20261002`，不修改 P1 分析工作树、E15 暂停 campaign 或 E16 开发环境。

当前权威算法继续作为默认行为。P1 的 480 条诊断运行提供了三个研究信号：重新 SSGS 可能丢失有价值的 timing phenotype；出生邻域可能错过对子代有利的方向；A7 的正压缩代理可能遗漏电费改善。它们支持制作实验变体，尚未证明在线收益。P1 报告：<https://app.notion.com/p/3ed8878b80198199a5aaedc6a9f7cb5a>。

## 步骤 1：先固定单因素对照

共同底座为 F6：完整 MOEA/D、固定局部搜索预算 6、原 Trigger/Q-learning/Polish/Archive 规则。六组分别如下。

| 组 | 相对 F6 的唯一机制变化 | 用途 |
| --- | --- | --- |
| F6 | 无 | 共同基线 |
| CG | 20% 子代复制当前子问题 incumbent 的 genotype，重新 SSGS | 控制复制/减少变异的影响 |
| CT | 同样 20% 复制同一 incumbent 的 genotype 和完整 schedule，重新 exact | 对 CG 检验保留 timing 的影响 |
| RPERM | 出生邻域不变，只随机打乱替换遍历顺序 | 控制 replacement cap 下的顺序影响 |
| RDIR | 按最终子代的归一化目标方向选择替换邻域 | 检验方向关联的影响 |
| A7B | 第一代表由电费代理优先选取 | 检验 A7 单层筛选改动 |

组合组暂不运行。20% 为研究工作值，需在后续独立样本上检验，不视为最优参数。

## 步骤 2：实现边界

CG 与 CT 用同一新增独立 RNG 流作复制决策；其余 80% 沿用原交叉变异。关闭复制时不创建或消耗该随机流。CT 冻结整个时间表，不拼接两个父代的开始时间；每次仍走统一 exact gateway、可行性检查和 Archive 更新。

RDIR 使用冻结 normalization context 和现有 inverse-weight Tchebycheff reference rays。只改变替换邻域；父代池、Trigger、local acceptance、reward 和出生子问题不变。每个邻居仍用自己的权重，替换 cap 仍为 2。RPERM 使用独立替换 RNG。

A7B 只改变每工序第一代表的优先级。原合法关键时间、正 active-union gain 资格、第二 Flow 代表、二维分层/拥挤距离、2B 池和最终随机抽样保持原规则。代理只读取该实例的活动区间和 incumbent 的原峰窗，不对 raw pool 偷跑完整 evaluator。

设移动工序所在实例功率差为 $\Delta P$；冻结其余工序，令 $\Delta a(t)$ 为该实例新旧 active indicator 的差。局部 TOU 代理和冻结原峰的一阶代理为：

$$
\widehat{\Delta C}_{TOU}=\frac{\Delta P}{3600}\int\pi_r(t)\Delta a(t)\,dt.
$$

$$
\widehat{\Delta C}_{dem}=\frac{\kappa_r\Delta P}{900|\mathcal W_r^*|}\sum_{W\in\mathcal W_r^*}\int_W\Delta a(t)\,dt.
$$

两项按真实费用单位相加，分数越小优先。并列时沿用 gain、Flow、移动距离和时间的原稳定规则。它忽略全局 horizon 改变引起的 idle tail、其他 Region 尾部变化和新峰接管，不能替代 exact bill，也不保证预测改善。特别保留一个新峰反例测试。

## 步骤 3：本地工程验收

1. 配置边界与一因素差异检查；默认关闭；非 full 方法拒绝实验选项，避免静默忽略。
2. CG/CT 手工时序反例：CG 重解码，CT 保留延迟；完整 exact 计数、cap、Archive、原对象不变和可行性。
3. RDIR 方向/端点/零范围/稳定 tie；RPERM 同邻域、可复现、独立 RNG；邻居自己权重和 cap 回归。
4. A7 代理手算 TOU/峰窗/并列峰、重叠不重复计费、新峰预测失败反例；未选择工序冻结，代理阶段不调用完整 evaluator。
5. 六组在小合成实例、两个 seed 上固定代数运行和重复回放；所有在线 exact candidate 与 reference evaluator 交叉验证；不做显著性结论。
6. 与未修改基线在独立进程比较原版候选顺序、决策、population、Archive、Q 和计数的语义哈希。
7. 完整质量门：ruff、format、mypy、pytest/coverage；整体覆盖率至少 80%，核心模块至少 90%。新增代码有有意义的边界与反例测试。

共同轻日志路径为 `retain_trace=False`，不接 P1 大量候选日志观察器，不裁剪 Archive。回放检查可临时接常量历史内存的 SemanticRecorder；它不用于正式等时间实验。所有组使用同一源码与日志口径，不能与 P1 的重日志吞吐混合。

## 步骤 4：交付与用户审阅门

输出本地验收报告、六组配置、可复现验证入口和本地 commit。记录源码/config/instance/initial hashes、真实测试结果及局限，同步 Notion 09 和本地实现任务。当前服务器部署状态必须保持“未部署”，正式实验状态保持“未运行”。

用户确认后再进入服务器部署、资源预跑和正式 P2。正式第一批优先 F6/CG/CT；RPERM/RDIR 和 A7B 先保留独立筛选，不一次性展开全部组合。正式任务矩阵、50/100 jobs 的时间预算、重复数和服务器并发在该阶段单独冻结。本地配置文件中的 600 秒仅为 50-job 模板工作值，不是论文统一规定。

## Material Passport

- Type: LOCAL_IMPLEMENTATION_PLAN
- Verification: UNVERIFIED until actual local gates complete
- Dependencies: P1 diagnosis; current Notion framework/model/Coding Contract; frozen base SHA
- Scientific status: pending independent online experiments
- External actions: no server deployment/run and no GitHub push in this batch
