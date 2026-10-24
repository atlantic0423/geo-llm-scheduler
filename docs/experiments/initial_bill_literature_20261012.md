# 近年电费/能耗初始化文献与本批启发

本轮检索覆盖2024–2026年的TOU调度、节能初始化与多目标构造。以下逐条区分阅读范围；没有访问到完整方法的文章不被写成已精读，期刊分区未作官方年度核验。

| 文献与主来源 | 实际阅读范围 | 可借鉴与限制 |
| --- | --- | --- |
| Benedikt等，2025，[Green Scheduling with Time-of-Use Tariffs and Machine States](https://arxiv.org/pdf/2506.10405) | 公开全文；阅读引言、模型、4.4与4.5构造段落，非逐页全部精读 | 初始解由低费用松弛形成的处理块、装箱与块长度调整构造；启发是按实际执行时间费用构造，而非平均地域电价。本项目有资源并发、两阶段和常驻idle，不照搬单机开关状态或ILP装箱。 |
| Missaoui等，2025，[Refined Iterated Pareto Greedy for Energy-aware Hybrid Flowshop Scheduling with Blocking Constraints](https://arxiv.org/html/2510.03377v1) | 公开全文；阅读模型与3.1初始化、选择和构造段落 | 初始化分别用makespan NEH与按TEC准则修改的NEH得到两方向种子。可借鉴保留效率与费用方向；TEC是能量，不能当TOU加需量Bill，初始种子也不保证在最终前沿。 |
| 2025，[Learning-based collaborative optimization for multi-objective energy-aware distributed assembly blocking flow shop scheduling](https://doi.org/10.1016/j.cie.2025.111214)，Computers & Industrial Engineering 206，111214 | 出版商可见摘要与引言；完整方法未获取 | 以节能构造启发式初始化，再结合学习和协同搜索。支持专门种子这一候选方向；没有据未读方法推断具体配额或评分。 |
| 2024，[A self-learning multi-population evolutionary algorithm for flexible job shop scheduling under time-of-use pricing](https://doi.org/10.1016/j.cie.2024.110004)，Computers & Industrial Engineering 189，110004 | 出版商索引摘要；正文访问失败 | TOU与多群/自学习结合。作为实际时间电价与多样性背景，不能声称复现其未读初始化细节。 |
| 2024，[An energy-saving distributed flexible job shop scheduling with machine breakdowns](https://doi.org/10.1016/j.asoc.2024.112276)，Applied Soft Computing 167，112276 | 出版商摘要，正文未获取 | 多规则初始与双群协同作为多配方背景；故障与本项目不同，具体规则不用于本轮实现依据。 |
| 2024，[Scientific Reports 14:30816](https://www.nature.com/articles/s41598-024-81064-z)，distributed two-stage flexible job shop under TOU | 出版商检索摘要；PDF重定向失败，未读取完整初始化段落 | 问题近邻，后续待读。不能从检索片段声称其三种初始化具体实现已读或已采用。 |
| 2026，[Energy-aware flexible job shop scheduling under time-of-use pricing with renewable energy, battery storage and maintenance](https://www.nature.com/articles/s41598-026-54904-3) | 出版商可见初始化与energy-aware后处理段落 | 初始化以随机可行顺序/机器规则起步，能耗后处理使用slack和电价；启发是结构费用构造与时序后处理要分开，不引入本文电池等额外模型。 |

## 本批取舍

原第十类能源启发只按Prefill所在地域的全horizon平均电价排序，未直接考虑Decode功率、实际执行时段或需量。新TOU/DEMAND/BILL三类改为完整资源可行前缀的真实费用分解；PORTFOLIO保留不同费用与Flow偏好。此为针对本项目的可检验工程方案，不能仅凭文献筛查称创新成立。

第一类费用引导回答“是否改善TOU暴露”，第二类回答“费用端是否主要受峰值/需量影响”，第三类观察同时计Bill是否更稳，组合类观察不同方向是否互补。10%和20%观察加入份额与减少旧策略的代价。RESAMPLE控制额外随机采样/筛选，但不是相同顺序与相同CPU的纯评分对照，后续因果实验要另设计。

本轮不将第二步的时间右移加工混入新种子，否则无法区分结构构造与时序费用改进。没有做进化，初始化前沿改善是否有助于等总时间最终MOEA/D仍需独立整程验证。
