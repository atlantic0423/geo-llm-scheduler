# 33组逐项复盘补充：复现说明

本补充只读取已验收33组正式结果，不运行solver，不改变原冻结70项检验。源与原始结果取得方式见上一级component33_20261020/REPRODUCTION.md和原GitHub Release。

在具有Python、Matplotlib、NumPy的环境中，运行：

```powershell
python research/component33_review_20261020/build_review.py --results-dir <已解包完整结果目录> --output <新报告目录> --receipt-dir <验收目录>
```

结果目录需有summary_statistics.json、RAW_MANIFEST.json、analysis/run_metrics.csv、analysis/action_costs.csv、raw/formal/runs。脚本对1056个summary/diagnostics逐SHA与原complete marker核对；每个计时类别的origin和全局计时闭合，初始/Polish/动作/外层调用重分类后逐run非负并重构生产总时间；各组5项原均值共165项重新匹配。

这是补充计时分析与解释重放，不是再次运行1056次算法或fresh复评874110个最终成员。原四家族70项检验只引用已有冻结结果；本次没有新增显著性检验。

绘图使用Microsoft YaHei字体，当前Windows环境可直接重放；其他系统应安装合法CJK字体并相应替换字体名，字体/Matplotlib版本会影响图文件的字节，不影响CSV/JSON/Markdown中的数据。PDF元数据含生成时间，不以其字节一致冒充数学复现。

发布前验收记录包括本补充脚本复算、文档全部33组覆盖/公式/本地链接检查，以及遵守仓库约定的本地质量门和准确head CI。最终状态由本批09记录与PR回执定义，不能用本段说明当作验收已成功的证据。
