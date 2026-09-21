# 复现

从仓库根运行scripts/reproduce.py，同一smoke配置/实例/seed两次执行，排除计时后比较完整trace并输出hash。full/Random/Bandit回放见integration/test_full_engine.py。

保留config.json、输入JSON、summary.json、trace.jsonl、Archive、population、Q-table和objectives.csv。summary 显式记录 git_commit、git_branch、git_dirty；无法读取 Git 时写 null。source_hash是已安装Python源码hash；config_hash是完整展开配置hash；instance_hash是原始输入bytes hash。Git provenance 与这些内容指纹互补，发布ZIP的全源码manifest hash是另一标识。

Archive 的 archive_insertions 是历史成功插入次数，可能包含后来被支配移除的临时成员；它不能解释为最终 Pareto 贡献。archive_final_by_origin 记录最终成员来源，轨迹中的 archive_net_retained 记录每个 MacroSearch batch 结束时相对 batch 前新增且仍保留的成员。

重新检查保存解：
```powershell
.venv/Scripts/python -m geo_llm_scheduler.cli.inspect_schedule --instance examples/smoke.json --archive outputs/full_qlearning_fixed_seed1/archive.json --index 0
```

inspect重新exact，不信任保存目标。当前结果文件是最终快照，不支持任意代断点续跑，完全回放从seed开始。outputs不进入源码ZIP，正式数据需独立归档。

package_snapshot.py排除git/环境/缓存/outputs，逐文件SHA256，ZIP内含SOURCE_MANIFEST.json，snapshots/manifest.json供外部审计。ZIP时间戳固定，相同源码生成相同bytes。Notion记录source hash、ZIP SHA256与附件。上传请求保存在忽略的snapshots，不进入源码包。

wall-clock停止受负载影响，不要求同一路径。固定代数且相同Python/依赖用于确定性验证，不承诺不同Python随机库版本逐位一致。
