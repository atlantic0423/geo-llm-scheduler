# 初始化第一步复现说明

运行源冻结 e57d385b2a4dcc7546e2f54e920cccbe2b6eb7a1，canonical 06c3a645ff4bcd24431640eacca0a42fb58309c65ac41e9480b250e5927d9436。正式 formal_node0/1 与 pilot_node0/1 必须分开，不合并推断样本。

完整原始包为 initial_step1_full_raw.tar.gz，大小1426295175字节、SHA1c1867a677de4ab778a4089f04f1df5404126b296a4cc9af7b82f08ca6f37606。本地已经完整保存并逐成员验真；full_manifest.json 给出43998个成员的大小与 SHA，正式7920/720，pilot396单列。当前报告的 GitHub PR 提供分析及聚合表，完整 raw/compact 尚未发布到 GitHub，不能据此声称完整远程可复现包已交付。保留 operations、历史与所有原件。

克隆原运行和分析 bundle，检出记录的冻结 commit；安装项目开发依赖及 numpy/matplotlib。运行分析时在 PYTHONPATH 加本分析 checkout 的 src 与 scripts，并调用 research/initial_step1_20261011/analyze.py，传入 --input formal_analysis_inputs.json 与 --out 新分析目录。compact JSON 是从已验收原始 population/summary/marker 提取的目标、来源及基因 SHA；其 SHA0fe541a3ccd97854e45b8d9dc621b8c1634d87fd50322b86858574d292e019d0。

analyze.py 不改原始数据，不调用进化/Polish。共同经验参考按同 input 的全部 11×10 种群唯一 ND 生成，归一化/hv_reference/IGD+ 与 FROZEN_PLAN.md 一致。统计随机种子2026101101；两主、十八次家族分别 Holm。audit.py 独立复算全部20项统计并核对来源冗余。所有生成输出 CSV、JSON、72 算例 PNG 和两概览在完整分析 ZIP；Git 保留数字表与固定代表图。

exact 证据由原 worker 逐候选独立复验和完整产物 hash 支持；结果处理不宣称重新运行全部候选 exact。指标与统计复现和算法 exact 重评是不同步骤。H/T 必须分别加载自己的输入；重复 seed 是同 base 内测量，不是独立研究样本。

零 HV 来自冻结参考矩形外解，保留参与主绝对差。hv_pct 分母为0时为 null/空 CSV，有限配对的描述均值单列，不能作主提升百分比。方向后悔差阈值0.01、来源份额、各子组和成本都是诊断量，无新增显著性主张。
