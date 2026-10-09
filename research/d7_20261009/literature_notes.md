# D7 文献与方向取舍（2026-10-09）

用户本轮明确先改善效果，不强求新颖性。本次检索重点为 2024–2025 的 RL/AOS、调度算子与 MOEA/D，并核对作者代码与可公开原文。没有声称穷尽中科院一区，也未重新核验这些期刊的当年中科院/JCR 分区。出版日期与 DOI 不等于分区证据。封闭全文只读到摘要/检索公开方法片段的，明确标出。

| 文献与原始来源 | 本次实际阅读范围 | 对本项目的启发与限制 |
| --- | --- | --- |
| Yin & Xiang, Adaptive operator selection with dueling deep Q-network for evolutionary multi-objective optimization, Neurocomputing 581 (2024), 127491，[DOI](https://doi.org/10.1016/j.neucom.2024.127491)；[作者代码](https://github.com/Shihong-Yin/AdaW-DDQN) | 实际下载并阅读作者 AdaWDDQN.m 的主循环，以及 CandidateOps/model/train_model；未获得出版全文 | 代码状态包含个体决策变量、决策/目标多样性和个体改进历史；滑动窗口算子信用来自邻域替换是否成功。说明“搜索反馈”可比纯约束标签更贴近动作选择。连续变量 DE 动作与我们的离散 SSGS 修复不同，不能直接复制网络或原始决策向量。公开代码只有一次提交，未执行其算法或复现论文实验。 |
| Dynamic operator management in meta-heuristics using reinforcement learning: an application to permutation flowshop scheduling problems，[arXiv 2408.14864v1](https://arxiv.org/html/2408.14864v1) (2024) | 公开原文方法 3.1–3.4、算法伪码及实验设计部分；预印本身份，不当已核验一区期刊 | 用粗状态、episode 收益和可恢复 tabu 管理算子；动作可组合破坏强度与构造方式。启发是状态不一定越多越好，短期失效算子应可重新进入，而非永久删除。我们的单步轨迹和其整段 IG episode 不同，不能把零收益直接解释为算子无用。 |
| A Knowledge-driven Memetic Algorithm for Energy-efficient Distributed Homogeneous Flow Shop Scheduling，[arXiv 2404.18953v1](https://arxiv.org/html/2404.18953v1) (2024) | 原文第 4 节算子/初始化/减碳策略及第 5 节实验设计、策略消融；未确认最终期刊版本 | 根据关键工厂设计局部/跨工厂动作，并单独设计能源策略。启发是算子应对准问题结构而非给通用动作换标签。其关机机制不适用我们固定 idle/active union 模型，不借其节能结果证明 A3_TOU 有效。 |
| HK-MOEA/D: A historical knowledge-guided resource allocation for decomposition multiobjective optimization, Engineering Applications of Artificial Intelligence 139 (2025), 109482，[DOI](https://doi.org/10.1016/j.engappai.2024.109482) | 出版摘要、公开方法片段及作者单位书目；全文未获得 | 历史知识/子问题演化信息引导资源分配值得作为后续方向。这里是从摘要得到的建议，不声称精读实现。先用 RPERM 检查替换因素，再考虑子问题停滞/真实改进率驱动触发，不直接叠加复杂适应机制。 |
| ADNS: An adaptive dynamic neighborhood search method guided by joint learning heuristics and corresponding hyperparameters, Applied Soft Computing 180 (2025), 113280，[DOI](https://doi.org/10.1016/j.asoc.2025.113280) | 出版摘要和公开方法片段；未获得完整出版全文 | 联合学习启发式与搜索步长提醒我们预算应与动作相容。D7 先固定 B3/B6/B10 与既有 SEQ 对照；不重复未经证实的 severity→budget 规则。后续若有稳定差异，才设计 action×budget 的小动作空间。该文 ALNS/路径问题不同，不能直接推断 B6 应被替代。 |
| An improved MOEA/D with reinforcement learning for flexible job shops incorporating manual operations and fatigue effects, Swarm and Evolutionary Computation 99 (2025), 102200，[出版页](https://www.sciencedirect.com/science/article/pii/S2210650225003578)，[DOI](https://doi.org/10.1016/j.swevo.2025.102200) | 出版摘要；未获全文 | MOEA/D 参数选择与问题专用局部搜索可同时考察，但疲劳/手工作业模型与我们不同。先单因素识别，不把“MOEA/D 加 RL”本身当新颖性或效果保证。 |
| Deep reinforcement learning-based memetic algorithm for energy-aware flexible job shop scheduling with multi-AGV, Computers & Industrial Engineering 189 (2024), 109917，[DOI](https://doi.org/10.1016/j.cie.2024.109917) | 出版摘要；未获全文 | 能源调度中学习选择算子是相关近邻。AGV 与运输能耗不能迁移成我们的 KV/区域代理；只作为后续全文候选。 |
| Reinforcement learning enhanced memetic algorithm for multiobjective flexible job shop scheduling toward Industry 5.0, International Journal of Production Research 63(1) (2025)，[DOI](https://doi.org/10.1080/00207543.2024.2357740) | 出版摘要和公开书目信息，2024 online/2025卷期；未获全文 | 候选近邻，不能因题目接近而声称已确认具体状态/奖励设计；暂不以它冻结本轮机制。 |

## 本轮确定的思路

1. **先验证状态信息有没有用。** STATE6 删除条件；STATE84 只增加条件共存一位。避免一下膨胀到六维离散状态或神经网络。若两者都无益，必须考虑条件标签本身不是动作收益的好代理。
2. **让 A3 真正看见电价。** 只改变跨区域迁移的候选排序，保留原候选额度和 exact 接受。该试验可否定廉价静态电价代理，不是否定一切状态匹配或能源算子。
3. **把 MacroSearch 的成本—质量关系重新量清楚。** B3/B10/SEQ 与 B6 同批等时间比较，记录实际 distinct exact、空调用、接受与构造时间。严重度高不必然值得更多候选。
4. **框架先查简单因素。** RPERM 使用已有随机邻域替换。不重做 D3 已未通过的方向响应策略，不把未确认的 W/GW 加入本批。

## 后续候选，尚未实现或启动

更贴近动作的状态应包含“可迁移的电价差/可压缩窗口潜力/有效邻域大小/近期动作收益与成本”，不是简单加入更多严重度。需要同源状态池，对 A1–A8 和少量预算逐一回放，按 base 完全留出检验预测关系，并把状态计算时间算进算法总成本。线上 state×action 表不能证明因果匹配。

若预算筛选显示动作差异，再考虑选择 action×budget 或轻量 contextual bandit；短期算子禁用必须有探测/恢复机会。若 RPERM 有稳定收益，再研究子问题真实替换收益与停滞驱动的局部搜索触发/资源分配，单独冻结对照。所有后续候选均不能由本轮 pilot 或有利子组直接晋级。

本批八组只是有针对性的筛选；没有已证效果，也没有已证新颖性。效果筛选后仍需独立确认与闭环原文调查。
