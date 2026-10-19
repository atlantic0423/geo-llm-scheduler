# 辅助轨迹三阶段结果复现

## 固定版本与数据

运行source为867c95e866b13fa3459451eb54713292b112ee03，原Draft PR63未合并。分析位于独立分支research/aux-walk-results-20261019；没有修改运行源或默认算法。统计seed2026101603，20000分层bootstrap、100000双侧sign-flip；补充38项探索家族seed2026101901。原34/6/6家族保持分开。

从本次Release取两个独立raw tar、complete_formal_control.tar.gz、两个server_node manifest、原运行source bundle及分析bundle；按raw_archives_manifest.json逐SHA检查。

- 将两raw tar解压到START/raw，分别得到formal_node0与formal_node1，每个4800run、24000文件。
- 将control tar解压到CAPTURE/control，得到formal/plan.json、selection.json、inputs、specs、indexes、ops和心跳。
- 将server_node0_manifest.json、server_node1_manifest.json置于CAPTURE根目录。
- 两个tar相互独立，不能拼成一个tar；完整raw未删除。
- 初次捕获前的旧control_snapshot存在未到达文件的空占位；复现必须用本轮complete_formal_control.tar.gz，不能使用空selection/stop占位。

## 执行

在分析分支根目录安装项目dev依赖，使用Python3.11或3.13。命令中的START、CAPTURE和ANALYSIS是调用者选择的目录；ANALYSIS必须尚不存在。

```powershell
$env:PYTHONPATH="$PWD/src;$PWD"
$env:OPENBLAS_NUM_THREADS="1"
python scripts/analyze_aux_walk_results.py --start START --capture CAPTURE --out ANALYSIS
python scripts/validate_aux_walk_results.py --start START --capture CAPTURE --analysis ANALYSIS
python scripts/plot_aux_walk_results.py --analysis ANALYSIS --out FIGURES
python -m pytest tests/test_aux_walk_results.py -q
python scripts/validate_all.py
```

第一条重新核对全部48000文件、9600冻结spec、source、输入、Q和同host，再重建指标/统计/anytime。第二条用另一个数值路径复算46项统计/三个Holm家族，生成明确标记的38项探索比较、成本、端点相对变化和分层描述；并对固定24run的48最终成员fresh exact检查。它不是全部5903619成员的fresh复评。

完整原worker最终审计证据与SHA绑定；抽查、指标复算、统计重放和重新运行优化四种验证不可混称。分析验证不重新消耗9600次科学实验预算。

## 产物解释

analysis.json包含全部原34/6/6项；run_metrics.csv保留9600运行，parent_metrics.csv先平均同父两版本/两seed；parent_effects.csv为主比较效应。补充exploratory_anchors.json的38项只为探索，不替换主家族。cost_details/summary避免相加嵌套计时。mechanisms.json保存完整事件计数；anytime_observations.csv保留实际elapsed和覆盖范围独立归一化的HV。

quality使用每输入全部方法/seed最终经验ND union，不是真实Pareto；不同stage的HV绝对值不做直接比较。anytime另有共同范围，未插值到名义阈值。图示固定case由ID规则选择，不能当总体优势。

原ru_maxrss可能包含exec前继承峰；rss_accounting_probe.json复现64MiB地址空间限制与109940KiB历史ru_maxrss、11168KiB当前VmHWM的差异，不当成上限失效或solver独占RSS。原last_best_gap为局部标量，近退化归一化会放大它，不据跨run均值推断质量。

完整分析包保留所有表与PNG/PDF；Git仅保存报告、图、紧凑父表/统计和分析实现，详细raw与大表位于Release。原始数据和运行进程均未因分析重启、迁移或清理。
