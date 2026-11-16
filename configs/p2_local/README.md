# P2 本地审阅配置

这些文件是单因素机制模板，尚未冻结正式 campaign，不会自动上传或启动服务器。

共同底座为完整 MOEA/D、固定预算 6、原版 Trigger/Q-learning/Polish/Archive。F6 为共同基线；CG/CT 对照普通复制与时间表延续；RPERM/RDIR 对照替换顺序与方向关联；A7B 只改变每工序第一代表的优先级。所有开关默认关闭，详见 [本地实施计划](../../docs/experiments/moead_p2_local_plan.md)。

模板中的 `instance: examples/smoke.json` 是占位输入。`seconds: 600` 是拟供 50-job 筛选讨论的工作值，不能直接把 smoke 运行当作正式实验。正式部署前须冻结独立实例、共用初始化、种子、50/100 jobs 的预算、任务矩阵和资源预跑结果。

本地验收入口（在工作树根目录，用该工作树的 Python 环境执行）：

```powershell
.venv/Scripts/python.exe scripts/validate_all.py
.venv/Scripts/python.exe scripts/validate_p2_local.py --output outputs/p2_local_validation_new
```

第二条命令只运行 6-job、固定代数的工程检查，覆盖六组重复回放、逐候选 exact 对照和独立进程的旧版默认语义比较；输出目录必须尚不存在。这些短运行不能证明改进后的 HV 或统计显著性。
