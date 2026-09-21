# 实验协议

先通过validate_all、golden与固定代数replay，再运行实验。smoke与synthetic仅为工程数据，正式实例须注明profile来源、资源需求、释放时间、Region电价/Demand费率、时区和单位预处理。

配对相同实例/seed比较plain/full/Bandit/Random AOS，再单独比较fixed/static/random/severity/coverage预算。消融包括A1–A8 leave-one-out、noPolish、always/fixed/strict Trigger和trajectory长度。控制变量来自同一基础配置，禁止无意同时改变多个因素。

正式公平性优先等wall-clock：用足够大generations和相同seconds，记录elapsed、termination_reason、time_overshoot_seconds与未完成代；初始化/SSGS/exact/coalition/Polish都计时。当前子问题边界软停止，不能声称严格等CPU时间；必须报告 overshoot 并确认其相对总预算足够小。固定代数用于回放与计数控制。

summary含seed、Git commit/branch/dirty、实例/配置/源码hash、Python、耗时/overshoot、exact/feasible/rebuild、Archive attempts/insertions/peak/final-by-origin、Trigger触发/成功率、算子统计。trace每步保存state/action/budget/effective/raw attempts/reward/scalar/目标增量/计时，并区分 archive_insertions 与 archive_net_retained；每个子代保存Archive目标集及elapsed用于anytime曲线。qtable保存值/访问/选择/更新数。

HV/IGD+在experiments/metrics.py，需要所有方法共享明确参考点/参考前沿与归一化。默认指标null，不编造参考数据。多seed原始结果与汇总都需保留，不把单seed或smoke当研究结论。

本交付是实验框架与合成实例验证，不含真实数据论文规模实验、最优参数结论或显著性结论。
