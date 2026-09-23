# D03 DominantCondition Balanced Diagnostic Summary

状态：后台流水线完成后自动生成的 **D03 controlled diagnostic**。D03 Type-I、Type-II、D02 数据始终分开聚合；本报告不构成 Q-learning 优于 Random/Bandit 的性能结论。

## 冻结与完整性

- Frozen profiles：`{"Normal": "soft", "Resource": "moderate", "KV": "moderate", "Region": "soft", "TOU": "soft", "Demand": "strong", "Compressible": "soft"}`；Type-II `long`。
- Type-I：42/42 runs；Type-II：15/15 runs；每条 50 generations。
- D02 severity 回溯在任何 D03 calibration 前完成；calibration seeds 与 evaluation seeds 不相交。

## 覆盖

- D02 state coverage：19/42。
- Type-I state coverage：42/42；condition counts `{"Compressible": 13522, "Region": 119862, "Demand": 45364, "Resource": 34474, "KV": 4531, "Normal": 5857, "TOU": 33400}`；imbalance ratio 26.454。
- Type-II state coverage：18/42。
- Type-II effective distinct at n>=10：min 2，median 3，max 3。

## 解释边界

Type-I 仅用于获得诊断样本，不能反推真实 workload 中七类 condition 均匀。Severity-scale 风险、condition transition、action effectiveness、state aliasing 与 Trigger sampling bias 应结合 `aggregate/` CSV 和 `figures/d03/` 图进行人工复核。现有 severity 公式和阈值在本轮完全冻结。
