# 历史动作与反馈状态12组复现

运行源固定为 `1ec42415d0fc86f479c81ebb6e4b0601e63ece7f`，canonical `086c9a3d6545726b4bf3e8aec447610da64ad67cf6d844c6b01c080186f1ef96`。本目录只做完成结果分析，所有运行算法/source/默认初始化保持原样。

完整结果与两个独立源码版本见[GitHub Release](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/history-rl-analysis-20261023-1ec4241)。按 `RAW_RELEASE_MANIFEST.json` 逐卷及整包size/SHA核对，分别恢复正式tar和运行bundle，不能把两个group拼为一个文件。完整tar包含10155文件/464794737字节，SHA256 `96bda5afff0d8f6825ba4fe41bfaa82a00995d26cb18c2cdd958c42e30a1945a`，压缩后304881192字节。安全解包后按 `SERVER_VALIDATION_AND_MANIFEST.json` 逐成员验真，预跑与启动快照另列。

分析源码可从本独立分支或本Release的 `ANALYSIS_BUNDLE_MANIFEST.json` 绑定的分析bundle恢复。两个Git bundle各自独立，运行bundle只导出HEAD，使用 `git clone --no-checkout` 后按清单checkout指定commit，再 `git fsck --full`。分析版本标识、准确head的PR/CI与最终发布资产回执在Notion09记录；不得用分析源码canonical替代原运行封印。

准备Python3.11以上，安装本项目dev依赖，将 `PYTHONPATH` 明确指向恢复分析版本的 `src`。原始资料目录记为DATA，其下保留 `LOCAL_RECOVERY_ACCEPTANCE.json`、`SERVER_VALIDATION_AND_MANIFEST.json` 和 `raw/formal/`；输出目录记为OUT。在仓库根执行：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python research/history_rl_results_20261023/analyze_results.py --data DATA --output OUT
python research/history_rl_results_20261023/audit_results.py --data DATA --output OUT
python research/history_rl_results_20261023/plot_results.py --output OUT --figures OUT/figures
python research/history_rl_results_20261023/supplement_results.py --data DATA --output OUT
python research/history_rl_results_20261023/write_report.py --data DATA --output OUT --destination RESULTS_ANALYSIS.md
python -m pytest -q tests/test_history_rl_results_analysis.py
python scripts/validate_all.py
```

`ANALYSIS_PLAN.json`列明16父级推断、32输入经验union、10对×两指标20主Holm及11组×两指标22次Holm，根seed2026102301；符号枚举65536种是审计，原100000次Monte Carlo结果仍用于冻结检验。重复生成时比较数值CSV、statistics/references/fronts JSON；PNG/PDF元数据可能随Matplotlib或导出时刻改变，不以位图字节不变冒充数值一致。绘图单独使用seed2026102302，边际区间不增加主检验。

完整分析产生768行run、192行父级、全部42统计、动作/深度/前缀/历史转移/anytime与探索性端点表，17对PNG/PDF（12组固定四规模案例，另5概览）。所有输入和运行保留，没有按效果筛选父或run。参考front是所有组与两seed共同经验union，不是真实Pareto；重复开发面板不能称新留出。日志回放验证Q/历史机制，不重复执行算法。

验收范围：全部原始文件逐SHA，768完整原validator，768完整HV双方向积分、每run最多16参考点的IGD+交叉实现，768个Q表逐步更新回放，512历史组的因果顺序与START核对，全部42统计独立重建。原worker668168最终成员独立exact证据与产物hash绑定；本轮未fresh重评所有最终解，也未把退出cgroup当运行期OOM峰值证据。

本地完整质量门847通过/9平台跳过、436.45秒，overall86.90%、core97.42%；该收集完成后追加的探索性Flow窗口测试与其余7分析测试共同8项通过，最终完整CI覆盖全部848项候选测试的实际结果以09回执为准。新增测试用于HV参考点边界/重复点、IGD+方向、父级单位与Holm、未结束轨迹拒绝、历史反馈泄漏/terminal bootstrap和禁止窗口内虚构插值。初次新增测试的函数边界放置错误在验收前已修复，未修改算法或降低门禁。

本次不开展新的实验、没有定时跟进、没有服务器删除，默认算法不晋级。
