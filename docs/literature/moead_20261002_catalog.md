# MOEA/D 近两年文献与创新边界索引

日期：2026-10-02；版本v1.0。与[研究报告](../reports/moead_novelty_20261002.md)配套。

收录38项：2025–2026 issue/early-access期刊30篇、2024及更早回溯7项、预印本1项。对10篇近期论文和1篇经典会议论文核查了可获得全文中的关键方法/实验章节，其余为摘要、题录或既有阅读记录。**关键全文章节不等于每页、每张表及全部补充材料均读完。** 作者稿与期刊定稿分别标注。

优先窗口为2024-10-02至2026-10-02，同时保留issue 2025/2026、但early access更早的近邻。首次在线日未核者不冒充严格窗口内论文。不是对这两年所有一区相关文献的穷尽阅读，也不能保证首次创新。

分区以中科院一区为主、JCR Q1补充。本轮未取得可复核的官方年度完整表。TEVC、Swarm、ESWA按机构公开一区标注优先筛选，Swarm和Information Sciences标注冲突保留；EJOR/COR机构JCR表列2025，未替代中科院分区，SJR列也不当作JCR。下文分区证据不能直接用作学校考核认证。

## 检索和阅读矩阵

|编号|论文／primary来源|期刊／issue年|本轮阅读范围|创新边界|
|---|---|---|---|---|
|R01|[A Survey of Multiobjective Evolutionary Algorithm Based on Decomposition: Past and Future](https://ieeexplore.ieee.org/document/10750458)|TEVC／2026|关键全文章节|匹配、全局替换、AOS、资源分配和memetic均为既有研究线；先回溯经典方法再讨论新机制。|
|R02|[Learning-Based Temporal Sequence of Constrained Handling Selection for Constrained Multi-Objective Evolutionary Optimization](https://ieeexplore.ieee.org/document/11058982)|TEVC／2026|关键全文章节|DQN选择约束处理与遗传算子组合；连续时序决策已有直接近邻。|
|R03|[Dynamic Grouping With a Self-Aware Computational Resource Allocation for Large-Scale Multi-Objective Optimization](https://xplorestaging.ieee.org/document/10994449)|TEVC／2026|关键全文章节|变量重要性动态分组和自感知计算分配已有；连续变量方法不能直接当作离散资源冲突方法。|
|R04|[Bridging Evolutionary Multiobjective Optimization and GPU Acceleration via Tensorization](https://ira.lib.polyu.edu.hk/handle/10397/114103)|TEVC／2026|关键全文章节|MOEA/D批式张量化会重排搜索更新；大种群GPU速度不能外推到本项目CPU/SSGS。|
|R05|[A Decomposition-Based Evolutionary Algorithm With Clustering and Hierarchical Estimation for Multiobjective Fuzzy Flexible Jobshop Scheduling](https://doi.org/10.1109/TEVC.2024.3359120)|TEVC／2026|摘要/预览|聚类、层次估计与分解调度已有；early access在2024年1月，属于边界回溯。|
|R06|[Multiobjective cooperative multi-fitness in workflow scheduling problem](https://journals.sagepub.com/doi/abs/10.1177/10692509251363797)|ICAE／2025|关键全文章节|一染色体多表型、Lamarckian回编码与保留多输出已有，是时序继承的最强近邻。|
|R07|[Feedback-driven adaptive variable grouping for decomposition-based large-scale multiobjective optimization](https://www.sciencedirect.com/science/article/abs/pii/S2210650226001367)|Swarm／2026|摘要/预览|反馈重分组与组级算子选择已有；具体规则仍需全文确认。|
|R08|[MOEA/D-BDN: Multimodal multi-objective evolutionary algorithm based on bi-dynamic niche strategy and adaptive weight decomposition](https://www.sciencedirect.com/science/article/abs/pii/S2210650225003281)|Swarm／2025|摘要/预览|决策/目标双动态生态位、适应权重和历史存档已有，机会空间需与距离型生态位比较。|
|R09|[Decomposition-based multi-objective evolutionary algorithm for bi-optimal selection](https://www.sciencedirect.com/science/article/pii/S2210650225003852)|Swarm／2026|摘要/预览|个体探索、权重构造和双优选择已有；单换环境选择缺少新机制证据。|
|R10|[A decomposition-based many-objective evolutionary algorithm with Q-learning guide weight vectors update](https://www.sciencedirect.com/science/article/pii/S0957417424024746)|ESWA／2025|摘要/预览|Q-learning控制权重更新与算子参数已有；不以再加Q表作为独立贡献。|
|R11|[HK-MOEA/D: A historical knowledge-guided resource allocation for decomposition multiobjective optimization](https://www.sciencedirect.com/science/article/pii/S0952197624016403)|EAAI／2025|摘要/预览|历史演化能力/收敛/密度驱动资源分配已有；需证明历史收益之外的信息价值。|
|R12|[Multi-agent reinforcement learning-aided evolutionary algorithm for a many-objective distributed hybrid flow shop scheduling problem](https://doi.org/10.1016/j.swevo.2025.101991)|Swarm／2025|关键全文章节|GrEA加多目标agent选择搜索方向与算子；方向-算子联合控制本身已有。|
|R13|[An improved MOEA/D with reinforcement learning for flexible job shops incorporating manual operations and fatigue effects](https://www.sciencedirect.com/science/article/pii/S2210650225003578)|Swarm／2025|摘要/预览|子问题难度、定制LS与RL参数控制已有；具体算子效果须全文验证。|
|R14|[A Q-learning-based evolutionary algorithm for solving the low-carbon multi-objective flexible job shop scheduling problem](https://www.sciencedirect.com/science/article/pii/S0305054825002953)|COR／2026|摘要/预览|低碳FJSP结合MOEA/D、Q-learning和VNS参数控制已有。|
|R15|[A smart multi-objective differential evolution algorithm for energy-efficient scheduling in parallel batch processing machines](https://www.sciencedirect.com/science/article/pii/S1568494625017971)|ASOC／2026|摘要/预览|老虎机式算子/参数选择用于节能批处理调度已有。|
|R16|[Reinforcement learning-driven multi-objective optimization for energy-efficient flexible job shop scheduling considering steam consumption in textile production](https://www.sciencedirect.com/science/article/abs/pii/S221065022600218X)|Swarm／2026|摘要/预览|状态驱动Q-learning用于电/蒸汽能耗调度已有；总能耗与Demand收费不同。|
|R17|[Weak-prior medical image matting based on microscale-searching evolutionary optimization](https://doi.org/10.1016/j.swevo.2025.102065)|Swarm／2025|关键全文章节|按实际特征关联子问题和动态微搜索分配已有；权重邻近需与搜索邻近区分。|
|R18|[A parallel large-scale multiobjective evolutionary algorithm based on two-space decomposition](https://link.springer.com/article/10.1007/s40747-025-01835-7)|C&IS／2025|关键全文章节|决策/目标两空间分解与并行子种群已有；第二空间和增加个体数不能直接称创新。|
|R19|[Learn to optimise for job shop scheduling: a survey with comparison between genetic programming and reinforcement learning](https://link.springer.com/article/10.1007/s10462-024-11059-9)|AIR／2025|关键全文章节|状态、策略和奖励已有系统比较；局部代理奖励可能与最终目标失配。|
|R20|[Integrating adaptive divide-and-conquer and large language model for scheduling large-scale tasks in electromagnetic satellite systems](https://www.sciencedirect.com/science/article/abs/pii/S0957417426008912)|ESWA／2026|摘要/预览|时间窗口重叠分组、历史贡献分配和LLM生成策略已有。|
|R21|[A Dynamic Multi-objective Optimization approach for computing resource allocation in industrial model repository](https://www.sciencedirect.com/science/article/pii/S2210650225003761)|Swarm／2025|摘要/预览|动态环境预测/历史记忆与本项目静态离线批次不同。|
|R22|[A parallel MOEA/D-MRA algorithm for solving multi-objective optimal power flow problems](https://www.sciencedirect.com/science/article/abs/pii/S1568494626008616)|ASOC／2026|摘要/预览|迁移和资源分配型并行MOEA/D已有；两机跑不同seed属于实验加速。|
|R23|[MOEA/D with customized replacement neighborhood and dynamic resource allocation for solving 3L-SDHVRP](https://www.sciencedirect.com/science/article/pii/S2210650223002353)|Swarm／2024|摘要/预览|子代方向关联替换邻域已有；RDIR有诊断价值，单独创新风险高。|
|R24|[An energy-efficient scheduling approach for a two-stage hybrid flow shop with parallel batch machines](https://doi.org/10.1016/j.ejor.2025.07.055)|EJOR／2026|关键全文章节|参考方向分群、协同配对、区域LS和TOU时移已有；AMCEA不是标准MOEA/D。|
|R25|[Multi-objective evolutionary algorithm based on transfer learning and neural networks: Dual operator feature fusion and weight vector adaptation](https://www.sciencedirect.com/science/article/pii/S0020025524012787)|INS／2025|摘要/预览|双算子特征融合、迁移和权重适应已有；神经网络包装不足以区分机制。|
|R26|[A learning-guided multi-objective approach for energy-oriented hybrid flow shop scheduling with limited buffers](https://www.sciencedirect.com/science/article/pii/S0020025525010205)|INS／2026|摘要/预览|目标引导邻域与学习选算子已有；有限buffer和本项目约束差异需明确。|
|R27|[A twin-reinforced evolutionary algorithm for flexible job shop scheduling problem under time-of-use tariffs](https://www.sciencedirect.com/science/article/pii/S0360835225009003)|CIE／2026|摘要/预览|RL配对/算子及full-active解码、right-shift已有；本轮未取得全文。|
|R28|[A novel multi-state reinforcement learning-based multi-objective evolutionary algorithm](https://www.sciencedirect.com/science/article/pii/S0020025524013112)|INS／2025|摘要/预览|目标几何状态和权重适应已有；early access 2024年8月，属于边界回溯。|
|R29|[A decomposition-based multi-objective evolutionary algorithm using infinitesimal method](https://www.sciencedirect.com/science/article/pii/S1568494624010469)|ASOC／2024|摘要/预览|PF分区适应权重已有；early access 2024年9月在严格窗口外。|
|R30|[Adaptive operator selection with test-and-apply structure for decomposition-based multi-objective optimization](https://doi.org/10.1016/j.swevo.2021.101013)|Swarm／2022|摘要/预览|先测试再分配后续预算已有；segment资源与一次best-of-B候选数不同。|
|R31|[Adaptive Operator Selection With Bandits for a Multiobjective Evolutionary Algorithm Based on Decomposition](https://scholars.cityu.edu.hk/en/publications/adaptive-operator-selection-with-bandits-for-a-multiobjective-evo/)|TEVC／2014|摘要/预览|滑窗fitness-rate-rank老虎机AOS是经典强对照。|
|R32|[Stable Matching-Based Selection in Evolutionary Multiobjective Optimization](https://repository.essex.ac.uk/11559/)|TEVC／2014|摘要/预览|解和子问题双向偏好匹配已有，增加匹配层不构成首次创新。|
|R33|[A Weight Adaptation Trigger Mechanism in Decomposition-based Evolutionary Multi-Objective Optimisation](https://arxiv.org/abs/2502.16481)|arXiv／2025|项目既有阅读记录|权重调整触发与memetic Trigger不同；本轮核对项目记录，未重读全文。|
|R34|[Balancing convergence and diversity in resource allocation strategy for decomposition-based multi-objective evolutionary algorithm](https://www.sciencedirect.com/science/article/abs/pii/S1568494620309066)|ASOC／2021|摘要/预览|历史逃逸概率量化evolvability，兼顾密度与收益；未来机会不能仅改名历史成功率。|
|R35|[Shake Them All! Rethinking Selection and Replacement in MOEA/D](https://link.springer.com/chapter/10.1007/978-3-319-10762-2_63)|PPSN／2014|关键全文章节|被替换解失去后续繁殖机会的动机早已有，保留搜索潜力本身不是新命题。|
|R36|[A Physics-informed Memetic Framework for Multi-objective Satellite Range Scheduling](https://www.tech.dmu.ac.uk/~syang/publications.html)|TEVC／2026|题录；正文待核|作者官网确认2026-07-01 early access；primary方法全文未取得。|
|R37|[Adaptive Sampled Walk: A Simple and Efficient Autonomous Local Search](https://pubmed.ncbi.nlm.nih.gov/41666298/)|EvolComp／2026|摘要/预览|作者摘要确认滑窗距离决定邻居评价数；候选数自适应不是空白。|
|R38|[An Evolutionary Algorithm for Multi-Objective Workflow Scheduling with Adaptive Dynamic Grouping](https://www.mdpi.com/2079-9292/14/13/2586)|Electronics／2025|摘要/预览|任务依赖分组和按组贡献分配搜索资源已有；非一区直接近邻也需保留。|

## 逐项出处、读取版本与分区证据


<a id="r01"></a>

### R01｜MOEA/D综述

- 出处：IEEE Transactions on Evolutionary Computation；issue/题录年2026；[DOI](https://doi.org/10.1109/TEVC.2024.3496507)。
- 卷期页：30(3)，957-978。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：期刊early access 2024；预印本2024-04；预印本早于窗口；2026期刊issue不双重计数。
- 阅读：关键全文章节；2024作者预印本，非2026期刊定稿；PDF第8–14页：标量化、选择/匹配、AOS与混合搜索；并非41页逐页精读。
- 方法证据：[读取版本](https://arxiv.org/pdf/2404.14571)。
- 读取PDF SHA256：`a7796ebd1c5baad1c5101b2a98049ec85251e2fdfff3c90d5ae7f3fa42db6184`。
- 分区：机构标注中科院一区；官方年度表未核；证据年份/口径：2025/2026机构页面，分区版本未指定。
- 分区来源：[机构记录1](https://jwxy.xtu.edu.cn/info/1154/3452.htm)。

<a id="r02"></a>

### R02｜CMOEA-TS

- 出处：IEEE Transactions on Evolutionary Computation；issue/题录年2026；[DOI](https://doi.org/10.1109/TEVC.2025.3584207)。
- 卷期页：30(3)，1123-1136。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：2025-06-30；early access与2026 issue分开。
- 阅读：关键全文章节；作者accepted稿；PDF第4–10、13页：状态/9类动作/奖励、实验和限制。
- 方法证据：[读取版本](https://www2.scut.edu.cn/_upload/article/files/74/7c/392c47d045e3948f22ded94344af/cc732e57-84f8-44bd-94e0-5baefe749f68.pdf)。
- 读取PDF SHA256：`e586237f56245a609ff39b0113eedd764eb3f9e460b36bcb1a386b5fc7206d43`。
- 分区：机构标注中科院一区；官方年度表未核；证据年份/口径：2025/2026机构页面，分区版本未指定。
- 分区来源：[机构记录1](https://jwxy.xtu.edu.cn/info/1154/3452.htm)。

<a id="r03"></a>

### R03｜DGVI

- 出处：IEEE Transactions on Evolutionary Computation；issue/题录年2026；[DOI](https://doi.org/10.1109/TEVC.2025.3564335)。
- 卷期页：30(2)，795-809。
- 题录核验：IEEE题录与作者上传稿交叉确认；Crossref限流。
- 首次在线：2025；作者上传2025-05-19；上传日不等于正式出版日。
- 阅读：关键全文章节；作者Chun Ouyang于2025-05-19上传稿；III方法/Algorithm1–2、IV实验/消融、可微假设局限；网页行596–765、1352–1491、1850–2066、2168–2288。
- 方法证据：[读取版本](https://www.researchgate.net/publication/391622317_Dynamic_Grouping_With_a_Self-Aware_Computational_Resource_Allocation_for_Large-Scale_Multi-Objective_Optimization)。
- 分区：机构标注中科院一区；官方年度表未核；证据年份/口径：2025/2026机构页面，分区版本未指定。
- 分区来源：[机构记录1](https://jwxy.xtu.edu.cn/info/1154/3452.htm)。

<a id="r04"></a>

### R04｜GPU tensorization

- 出处：IEEE Transactions on Evolutionary Computation；issue/题录年2026；[DOI](https://doi.org/10.1109/TEVC.2025.3555605)。
- 卷期页：30(1)，420-434。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：关键全文章节；大学公开作者稿；PDF第1、4–5、7、9、13页：张量化、MOEA/D更新、硬件/规模、限制。
- 方法证据：[读取版本](https://ira.lib.polyu.edu.hk/bitstream/10397/114103/1/Liang_Bridging_Evolutionary_Multiobjective.pdf)。
- 读取PDF SHA256：`8b0ec62bdb31ca3e8cb8e98e4fe7ec1fd8b3a9c95e701ecef4e1673a6b417209`。
- 分区：机构标注中科院一区；官方年度表未核；证据年份/口径：2025/2026机构页面，分区版本未指定。
- 分区来源：[机构记录1](https://jwxy.xtu.edu.cn/info/1154/3452.htm)。

<a id="r05"></a>

### R05｜模糊FJSP分解

- 出处：IEEE Transactions on Evolutionary Computation；issue/题录年2026；[DOI](https://doi.org/10.1109/TEVC.2024.3359120)。
- 卷期页：30(1)，2-15。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：2024-01-26；严格窗口外；2026 issue不代表2026新提出。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注中科院一区；官方年度表未核；证据年份/口径：2025/2026机构页面，分区版本未指定。
- 分区来源：[机构记录1](https://jwxy.xtu.edu.cn/info/1154/3452.htm)。

<a id="r06"></a>

### R06｜MOCMF

- 出处：Integrated Computer-Aided Engineering；issue/题录年2025；[DOI](https://doi.org/10.1177/10692509251363797)。
- 卷期页：32(4)，443-464。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：2025-08-11；期刊issue2025-11；本轮读取accepted稿。
- 阅读：关键全文章节；大学公开accepted稿；PDF第1–14、17–18页：多fitness/回编码、预算/结果/限制；第9页Algorithm2/Figure3目视核查。
- 方法证据：[读取版本](https://digibuo.uniovi.es/dspace/bitstream/handle/10651/80828/ICAE_postprint.pdf?isAllowed=y&sequence=1)。
- 读取PDF SHA256：`195666f0cb067cc6f0324def2ad46f4a6e15e69c41276f20d95ebfe2b8052974`。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

<a id="r07"></a>

### R07｜MOEA/D-FAVG

- 出处：Swarm and Evolutionary Computation；issue/题录年2026；[DOI](https://doi.org/10.1016/j.swevo.2026.102416)。
- 卷期页：106(未列)，102416。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注一区；另有二区记录，待官方复核；证据年份/口径：2026公告与2025论文列表，版本未指定。
- 分区来源：[机构记录1](https://ie.hbue.edu.cn/99/b7/c317a367031/pagem.htm)；[机构记录2](https://eee.wzu.edu.cn/info/1019/32400.htm)；[机构记录3](https://www5.zzu.edu.cn/cilab/yjfxjkycg/qklw.htm)。

<a id="r08"></a>

### R08｜MOEA/D-BDN

- 出处：Swarm and Evolutionary Computation；issue/题录年2025；[DOI](https://doi.org/10.1016/j.swevo.2025.102171)。
- 卷期页：99(未列)，102171。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注一区；另有二区记录，待官方复核；证据年份/口径：2026公告与2025论文列表，版本未指定。
- 分区来源：[机构记录1](https://ie.hbue.edu.cn/99/b7/c317a367031/pagem.htm)；[机构记录2](https://eee.wzu.edu.cn/info/1019/32400.htm)；[机构记录3](https://www5.zzu.edu.cn/cilab/yjfxjkycg/qklw.htm)。

<a id="r09"></a>

### R09｜MOEA/D-BOS

- 出处：Swarm and Evolutionary Computation；issue/题录年2026；[DOI](https://doi.org/10.1016/j.swevo.2025.102228)。
- 卷期页：100(未列)，102228。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注一区；另有二区记录，待官方复核；证据年份/口径：2026公告与2025论文列表，版本未指定。
- 分区来源：[机构记录1](https://ie.hbue.edu.cn/99/b7/c317a367031/pagem.htm)；[机构记录2](https://eee.wzu.edu.cn/info/1019/32400.htm)；[机构记录3](https://www5.zzu.edu.cn/cilab/yjfxjkycg/qklw.htm)。

<a id="r10"></a>

### R10｜Q-learning weight update

- 出处：Expert Systems with Applications；issue/题录年2025；[DOI](https://doi.org/10.1016/j.eswa.2024.125607)。
- 卷期页：262(未列)，125607。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注中科院一区；官方年度表未核；证据年份/口径：2025/2026机构页面，版本未指定。
- 分区来源：[机构记录1](https://sxy.zjnu.edu.cn/_t28/2025/0623/c5995a522773/page.psp)；[机构记录2](https://dm.uestc.edu.cn/publication/)。

<a id="r11"></a>

### R11｜HK-MOEA/D

- 出处：Engineering Applications of Artificial Intelligence；issue/题录年2025；[DOI](https://doi.org/10.1016/j.engappai.2024.109482)。
- 卷期页：139(未列)，109482。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

<a id="r12"></a>

### R12｜MRLEA distributed HFS

- 出处：Swarm and Evolutionary Computation；issue/题录年2025；[DOI](https://doi.org/10.1016/j.swevo.2025.101991)。
- 卷期页：97(未列)，101991。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：关键全文章节；项目已有期刊PDF；PDF第5、7–10、12、14页：编码/方向/算子/Q、等时间实验/消融/结论；第7页Algorithm2–3目视核查。
- 方法证据：项目论文库已有PDF《Multi-agent reinforcement learning-aided evolutionary algorithm for a many-objective distributed hybrid flow shop scheduling problem.pdf》，未上传/再分发原文。
- 读取PDF SHA256：`e8c94a732e17d27f44c614e23cf62a49fb2d835e474a62351496700c9fabf386`。
- 分区：机构标注一区；另有二区记录，待官方复核；证据年份/口径：2026公告与2025论文列表，版本未指定。
- 分区来源：[机构记录1](https://ie.hbue.edu.cn/99/b7/c317a367031/pagem.htm)；[机构记录2](https://eee.wzu.edu.cn/info/1019/32400.htm)；[机构记录3](https://www5.zzu.edu.cn/cilab/yjfxjkycg/qklw.htm)。

<a id="r13"></a>

### R13｜RL MOEA/D fatigue FJSP

- 出处：Swarm and Evolutionary Computation；issue/题录年2025；[DOI](https://doi.org/10.1016/j.swevo.2025.102200)。
- 卷期页：99(未列)，102200。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注一区；另有二区记录，待官方复核；证据年份/口径：2026公告与2025论文列表，版本未指定。
- 分区来源：[机构记录1](https://ie.hbue.edu.cn/99/b7/c317a367031/pagem.htm)；[机构记录2](https://eee.wzu.edu.cn/info/1019/32400.htm)；[机构记录3](https://www5.zzu.edu.cn/cilab/yjfxjkycg/qklw.htm)。

<a id="r14"></a>

### R14｜Q-MOEA/D-AWA low carbon

- 出处：Computers &amp; Operations Research；issue/题录年2026；[DOI](https://doi.org/10.1016/j.cor.2025.107266)。
- 卷期页：185(未列)，107266。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：补充：JCR Q1机构表；中科院分区未核；证据年份/口径：机构JCR表列2025；OR&MS另有2025记录。
- 分区来源：[机构记录1](https://www.iit.comillas.edu/publicacion/info_revista/en/282/Computers_&_Operations_Research)；[机构记录2](https://scholar.pusan.ac.kr/journals/12741)。

<a id="r15"></a>

### R15｜SMODE batch machines

- 出处：Applied Soft Computing；issue/题录年2026；[DOI](https://doi.org/10.1016/j.asoc.2025.114484)。
- 卷期页：189(未列)，114484。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

<a id="r16"></a>

### R16｜SGQL textile energy

- 出处：Swarm and Evolutionary Computation；issue/题录年2026；[DOI](https://doi.org/10.1016/j.swevo.2026.102498)。
- 卷期页：107(未列)，102498。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注一区；另有二区记录，待官方复核；证据年份/口径：2026公告与2025论文列表，版本未指定。
- 分区来源：[机构记录1](https://ie.hbue.edu.cn/99/b7/c317a367031/pagem.htm)；[机构记录2](https://eee.wzu.edu.cn/info/1019/32400.htm)；[机构记录3](https://www5.zzu.edu.cn/cilab/yjfxjkycg/qklw.htm)。

<a id="r17"></a>

### R17｜Microscale search

- 出处：Swarm and Evolutionary Computation；issue/题录年2025；[DOI](https://doi.org/10.1016/j.swevo.2025.102065)。
- 卷期页：98(未列)，102065。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：关键全文章节；大学公开作者稿；PDF第1、3、5–7、10页：特征关联/微搜索/分配、实验条件和结论；未逐表复核全部胜负。
- 方法证据：[读取版本](https://www2.scut.edu.cn/_upload/article/files/74/7c/392c47d045e3948f22ded94344af/fb97dfaf-7f51-4005-892b-e7ccdd5b05d2.pdf)。
- 读取PDF SHA256：`eb09535a8574eda210e96d6f913b31b5385696b247e1c4cd4c1ce5b16e2e6468`。
- 分区：机构标注一区；另有二区记录，待官方复核；证据年份/口径：2026公告与2025论文列表，版本未指定。
- 分区来源：[机构记录1](https://ie.hbue.edu.cn/99/b7/c317a367031/pagem.htm)；[机构记录2](https://eee.wzu.edu.cn/info/1019/32400.htm)；[机构记录3](https://www5.zzu.edu.cn/cilab/yjfxjkycg/qklw.htm)。

<a id="r18"></a>

### R18｜PEATSD

- 出处：Complex &amp; Intelligent Systems；issue/题录年2025；[DOI](https://doi.org/10.1007/s40747-025-01835-7)。
- 卷期页：11(5)，article number见出处。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：2025-03-25；出版社日期。
- 阅读：关键全文章节；出版社开放HTML；两空间/MPI方法、参数/个体数与结论；网页行127–285、466–470。
- 方法证据：[读取版本](https://link.springer.com/article/10.1007/s40747-025-01835-7)。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

<a id="r19"></a>

### R19｜GP vs RL JSS survey

- 出处：Artificial Intelligence Review；issue/题录年2025；[DOI](https://doi.org/10.1007/s10462-024-11059-9)。
- 卷期页：58(6)，article number见出处。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：2025-03-15；出版社日期。
- 阅读：关键全文章节；出版社开放HTML；表示/策略/评价与奖励/讨论；网页行125–315、469–510；未逐一复核全部综述引文。
- 方法证据：[读取版本](https://link.springer.com/article/10.1007/s10462-024-11059-9)。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

<a id="r20"></a>

### R20｜ADC plus LLM satellite

- 出处：Expert Systems with Applications；issue/题录年2026；[DOI](https://doi.org/10.1016/j.eswa.2026.131978)。
- 卷期页：318(未列)，131978。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注中科院一区；官方年度表未核；证据年份/口径：2025/2026机构页面，版本未指定。
- 分区来源：[机构记录1](https://sxy.zjnu.edu.cn/_t28/2025/0623/c5995a522773/page.psp)；[机构记录2](https://dm.uestc.edu.cn/publication/)。

<a id="r21"></a>

### R21｜Dynamic industrial allocation

- 出处：Swarm and Evolutionary Computation；issue/题录年2025；[DOI](https://doi.org/10.1016/j.swevo.2025.102219)。
- 卷期页：99(未列)，102219。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注一区；另有二区记录，待官方复核；证据年份/口径：2026公告与2025论文列表，版本未指定。
- 分区来源：[机构记录1](https://ie.hbue.edu.cn/99/b7/c317a367031/pagem.htm)；[机构记录2](https://eee.wzu.edu.cn/info/1019/32400.htm)；[机构记录3](https://www5.zzu.edu.cn/cilab/yjfxjkycg/qklw.htm)。

<a id="r22"></a>

### R22｜Parallel MOEA/D-MRA

- 出处：Applied Soft Computing；issue/题录年2026；[DOI](https://doi.org/10.1016/j.asoc.2026.115413)。
- 卷期页：200(未列)，115413。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

<a id="r23"></a>

### R23｜Customized replacement

- 出处：Swarm and Evolutionary Computation；issue/题录年2024；[DOI](https://doi.org/10.1016/j.swevo.2023.101463)。
- 卷期页：85(未列)，101463。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注一区；另有二区记录，待官方复核；证据年份/口径：2026公告与2025论文列表，版本未指定。
- 分区来源：[机构记录1](https://ie.hbue.edu.cn/99/b7/c317a367031/pagem.htm)；[机构记录2](https://eee.wzu.edu.cn/info/1019/32400.htm)；[机构记录3](https://www5.zzu.edu.cn/cilab/yjfxjkycg/qklw.htm)。

<a id="r24"></a>

### R24｜AMCEA two-stage HFS

- 出处：European Journal of Operational Research；issue/题录年2026；[DOI](https://doi.org/10.1016/j.ejor.2025.07.055)。
- 卷期页：328(3)，762-784。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：2025，具体日未独立核验；issue2026-02。
- 阅读：关键全文章节；项目已有期刊PDF；PDF第6、10–14、21–22页：模型/配对/区域LS/TOU、等时间消融、限制；第12页Algorithm6目视核查。
- 方法证据：项目论文库已有PDF《An energy-efficient scheduling approach for a two-stage hybrid flow shop.pdf》，未上传/再分发原文。
- 读取PDF SHA256：`fca7091710062bcdaf020b6e8a562f4d86582956bb649dcd7147fdca60ad1d0b`。
- 分区：补充：JCR Q1机构表；中科院分区未核；证据年份/口径：机构JCR表列2025，学科未列。
- 分区来源：[机构记录1](https://www.iit.comillas.edu/publicacion/info_revista/en/50/European_Journal_of_Operational_Research)。

<a id="r25"></a>

### R25｜Dual feature transfer

- 出处：Information Sciences；issue/题录年2025；[DOI](https://doi.org/10.1016/j.ins.2024.121364)。
- 卷期页：686(未列)，121364。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：公开中科院标注冲突；本次不认证一区；证据年份/口径：2025机构论文标注，版本未指定。
- 分区来源：[机构记录1](https://it.ouc.edu.cn/xh/list.htm)；[机构记录2](https://cs.shu.edu.cn/info/1461/40401.htm)。

<a id="r26"></a>

### R26｜Learning guided buffer HFS

- 出处：Information Sciences；issue/题录年2026；[DOI](https://doi.org/10.1016/j.ins.2025.122884)。
- 卷期页：731(未列)，122884。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：公开中科院标注冲突；本次不认证一区；证据年份/口径：2025机构论文标注，版本未指定。
- 分区来源：[机构记录1](https://it.ouc.edu.cn/xh/list.htm)；[机构记录2](https://cs.shu.edu.cn/info/1461/40401.htm)。

<a id="r27"></a>

### R27｜TREA TOU FJSP

- 出处：Computers &amp; Industrial Engineering；issue/题录年2026；[DOI](https://doi.org/10.1016/j.cie.2025.111754)。
- 卷期页：212(未列)，111754。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

<a id="r28"></a>

### R28｜Multi-state RL MOEA

- 出处：Information Sciences；issue/题录年2025；[DOI](https://doi.org/10.1016/j.ins.2024.121397)。
- 卷期页：688(未列)，121397。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：2024-08-28；严格窗口外。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：公开中科院标注冲突；本次不认证一区；证据年份/口径：2025机构论文标注，版本未指定。
- 分区来源：[机构记录1](https://it.ouc.edu.cn/xh/list.htm)；[机构记录2](https://cs.shu.edu.cn/info/1461/40401.htm)。

<a id="r29"></a>

### R29｜MOEA/D-DKS

- 出处：Applied Soft Computing；issue/题录年2024；[DOI](https://doi.org/10.1016/j.asoc.2024.112272)。
- 卷期页：167(未列)，112272。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：2024-09，具体日未核；严格窗口外。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

<a id="r30"></a>

### R30｜Test-and-Apply AOS

- 出处：Swarm and Evolutionary Computation；issue/题录年2022；[DOI](https://doi.org/10.1016/j.swevo.2021.101013)。
- 卷期页：68(未列)，101013。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注一区；另有二区记录，待官方复核；证据年份/口径：2026公告与2025论文列表，版本未指定。
- 分区来源：[机构记录1](https://ie.hbue.edu.cn/99/b7/c317a367031/pagem.htm)；[机构记录2](https://eee.wzu.edu.cn/info/1019/32400.htm)；[机构记录3](https://www5.zzu.edu.cn/cilab/yjfxjkycg/qklw.htm)。

<a id="r31"></a>

### R31｜FRRMAB

- 出处：IEEE Transactions on Evolutionary Computation；issue/题录年2014；[DOI](https://doi.org/10.1109/TEVC.2013.2239648)。
- 卷期页：18(1)，114-130。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注中科院一区；官方年度表未核；证据年份/口径：2025/2026机构页面，分区版本未指定。
- 分区来源：[机构记录1](https://jwxy.xtu.edu.cn/info/1154/3452.htm)。

<a id="r32"></a>

### R32｜Stable matching

- 出处：IEEE Transactions on Evolutionary Computation；issue/题录年2014；[DOI](https://doi.org/10.1109/TEVC.2013.2293776)。
- 卷期页：18(6)，909-923。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注中科院一区；官方年度表未核；证据年份/口径：2025/2026机构页面，分区版本未指定。
- 分区来源：[机构记录1](https://jwxy.xtu.edu.cn/info/1154/3452.htm)。

<a id="r33"></a>

### R33｜Weight adaptation trigger

- 出处：arXiv:2502.16481；issue/题录年2025；[arXiv](https://arxiv.org/abs/2502.16481)。
- 题录核验：项目既有阅读记录；本轮未重读全文。
- 首次在线：2025-02，具体日未核；预印本，不计期刊一区。
- 阅读：项目既有阅读记录；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

<a id="r34"></a>

### R34｜MOEA/D-BRA

- 出处：Applied Soft Computing；issue/题录年2021；[DOI](https://doi.org/10.1016/j.asoc.2020.106968)。
- 卷期页：100(未列)，106968。
- 题录核验：Crossref title/DOI/journal/year verified。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

<a id="r35"></a>

### R35｜Shake Them All!

- 出处：PPSN XIII, LNCS 8672；issue/题录年2014；[DOI](https://doi.org/10.1007/978-3-319-10762-2_63)。
- 题录核验：Springer章节页与原论文全文核对。
- 首次在线：未独立核验；issue年份为主，严格窗口归属待核。
- 阅读：关键全文章节；公开会议全文；PDF第1–7页：动机/Algorithm1、共享组件/预算和anytime结果；未复算全部参数表。
- 方法证据：[读取版本](https://www.cmap.polytechnique.fr/~nikolaus.hansen/proceedings/2014/PPSN/papers/8672/86720641.pdf)。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

<a id="r36"></a>

### R36｜Physics-informed satellite

- 出处：IEEE Transactions on Evolutionary Computation；issue/题录年2026；[DOI](https://doi.org/10.1109/TEVC.2026.3708200)。
- 题录核验：作者大学官网题录；Crossref限流，方法正文未取得。
- 首次在线：2026-07-01；作者大学官网early access记录；issue未核。
- 阅读：题录；正文待核；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：机构标注中科院一区；官方年度表未核；证据年份/口径：2025/2026机构页面，分区版本未指定。
- 分区来源：[机构记录1](https://jwxy.xtu.edu.cn/info/1154/3452.htm)。

<a id="r37"></a>

### R37｜Adaptive Sampled Walk

- 出处：Evolutionary Computation；issue/题录年2026；[DOI](https://doi.org/10.1162/EVCO.a.382)。
- 题录核验：NLM收录作者摘要和DOI；期刊全文未取得。
- 首次在线：2026-02-11；采用NLM题录，不采用OpenAlex相差1天的日期。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

<a id="r38"></a>

### R38｜Workflow adaptive grouping

- 出处：Electronics 14(13), 2586；issue/题录年2025；[DOI](https://doi.org/10.3390/electronics14132586)。
- 题录核验：出版社检索页题录/摘要；全文429，未继续重试。
- 首次在线：2025-06-26；出版社记录。
- 阅读：摘要/预览；题录/摘要，未取得可核查全文；只使用明确摘要机制；细节和定量效果不作已核验断言。
- 分区：分区未核；机制近邻补充；证据年份/口径：未核。

## 继续阅读的优先队列

1. R06 MOCMF、R24 AMCEA：对照作者实现/补充材料，确认回编码能表达何种等待。
2. R07 FAVG、R08 BDN、R11 HK-MOEA/D：取得全文，核查组级操作、历史信息、生态位维护和计算成本。
3. R13/R14/R27：核查子问题难度、预算粒度、编码/解码和右移传承。
4. R36物理先验memetic：目前只核对作者题录。提出约束物理结构指导算子前，须读方法正文；不依据二手摘要声称已厘清边界。
5. R37 Adaptive Sampled Walk：作者摘要足以确认候选数量自适应已有，具体停步规则须读全文后设计对照。

仅为阅读优先序，不创建定时任务，不启动服务器实验。
