# Phase 9 最终审计批次（待一项语义确认）

日期：2026-09-21，Windows / Python 3.13.5。已完成文档补全、架构图、配置/输入边界审计、代码格式、计数补齐、coverage门禁、profiling与快照可复现性修复。Phase 7末步bootstrap语义待确认，故本报告不宣称整个项目最终验收关闭。

## 变更与规格映射

- config/validation：有限值、概率、正整数、合法action、Region正容量与整数索引；对应Contract与用户severity输入约定。
- engine/runner/Polish：构造时间/attempts/proposals、逐来源Archive计数、Trigger成功率、Polish枚举与exact分开、anytime Archive目标；对应任务书20。
- operators：A6比例配置化，A8 ready列表显式排序；语义不变。
- README及docs：模型、算法、算子、配置、实验、复现、开发与traceability；对应任务书22。
- package_snapshot：manifest ZIP条目固定时间戳；requirements-lock移除机器绝对editable路径。
- tests：配置非法输入、Trigger方向/质量、真实CLI、A8全Region尾部、无正费率及IO边界。

## 实际命令与结果

1. scripts/validate_all.py：ruff lint/format、mypy 51源文件、106 tests通过；整体coverage 96.56%，核心96.66%。核心范围由scripts/check_coverage.py显式列出，不排除未覆盖分支。
2. scripts/reproduce.py：两次路径一致，trace hash d2b1c8dc3546c7bafdbbcb946dfeb044ba646b99c274c38eff23cc97b3373da3。
3. baseline/full CLI seed1：各100 offspring；plain 120 exact，full 726 exact，全部feasible且Archive尝试数相同。
4. 23个experiment/ablation YAML逐个加载并运行：每个100 offspring，所有exact均可行，Archive尝试匹配，所有effective预算不超限。
5. pip wheel . --no-deps --wheel-dir outputs/wheels：成功生成0.1.0 wheel。本机执行环境3.13；CI配置3.11，本次未实际执行远端CI，不声称已验证跨版本。
6. scripts/profile_run.py：12 jobs、2 Regions、每区2实例、pop20、2代、seed7，带profiler约0.76秒；264 exact全部可行，210 SSGS，A1–A8均有exact；9次Polish，82候选时刻、23 exact。该时间含profiler开销且依赖机器，不是方法优越性证据。

## 审计结论与限制

I1独立Archive、I2顺序无关best-of-B、I3结构重建/未选时序冻结有跨模块测试。A8原支撑结构允许遗漏依赖纯idle尾部缩短的改善，新增反例明确展示该边界，不把有界失败解释成不存在解。

热点为资源fits、Compressible关键时刻枚举与exact。保留full reference，不在未证明等价前做增量优化。未提供真实profiling输入或参考前沿；不生成正式HV/IGD+研究结论。无断点续跑，时间终止为子问题边界软上限，完整trace可能较大。

用户六项修订已同步Notion当前框架和研究决策。下一步只需确认open_questions中的Q末步语义，再补对应回归、关闭Phase7/9并上传最终确认版。
