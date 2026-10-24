# Jev B1 + A6 Top10：实际英文提示词与中文对照

版本：jev_single_v1_b1_a6_top10。实际只发送英文；中文用于人工阅读。

## 全局背景（每次请求）

### English

Assist deterministic offline geo-distributed LLM scheduling. All jobs are known.
This request belongs to ONE treatment, identified by state.sampling_rule. Its
greedy_probability selects a maximum; the remaining branch samples your complete
distribution. The PROGRAM samples ordinary selections independently; you return probabilities. EXCEPTION: A6 ranks ALL Jobs once and takes the highest-probability 10 percent (at least two, capped by the Job count). A6 does NOT epsilon-sample destroy Jobs. Probability ties use the program's explicit RNG.
Each nonpreemptive job has Prefill P then Decode D in the SAME geographic Region;
different instances incur its KV delay. Respect releases, phase eligibility,
cumulative Compute <=1 and effective VRAM GB. Do not change constraints or durations.
Minimize total Flow=sum(D completion-release) and Bill=TOU+Demand, in seconds/CNY.
Power is kW, energy kWh: each instance uses idle/active power on [0,T_end), active
by interval UNION, zero after the last Decode. Demand checks ALL 900s windows
anchored at zero, including tied/new peaks; partial tails still divide by 900.
Flow/Balanced/Electricity label preferences, not Regions. Use the CURRENT fine weight.
The program enforces strict-direction/delta0 entry, SSGS, exact feasibility/billing,
normalized Tchebycheff acceptance, Archive and neighbor-own-weight replacement.
Each LS step freshly selects an action, constructs at most ONE complete candidate
from ONE frozen incumbent, then accepts. Every complete feasible exact candidate
may update Archive, including locally rejected ones. At most FIVE steps; stop is
asked after steps 1-4. Polish follows outside the trajectory. All preparation,
queueing, HTTP and search consume production wall time. No future/oracle is known;
proxies are not exact gains, unobserved history is null, not failure.
A1 single phase instance; A2 P/D path within Region; A3 whole-job cross-Region path
(A1-3 keep OS); A4 positive-wait phase earlier in OS; A5 positive-wait job P/D
joint earlier insertion; A6 one full-Job ranking selects top ~10%, >=2 jobs; ORIGINAL random legal repair;
A7 one legal positive-union-compression critical-time move; A8 original-peak
singleton/support-group selection and bounded precedence-aware repair, retaining
member/position/attempt limits, original horizon and strict ALL-window peak gate.
Timing moves freeze unselected operations; structural moves rebuild by SSGS.
Assess only supplied options/proposition. Program RNG samples ordinary Choice distributions; A6 instead takes the top set;
follow ACTUALLY sampled parents, not your previous maximum. Do not output a schedule,
explanation, altered constraints or extra budget. Questions are independent.
Compact tables are lossless: columns label rows. all_jobs lists EVERY Job; operation
o=2*j/2*j+1 is P/D. Its machine/start come from current_solution.MS/starts_s, Region
from instance tables, profile from operation_profile_ids into profiles; end=start+duration.
resource_wait_s/flow_wait_s index o. Actual KV is kv_if_separate_s iff its P/D machines
differ. History default_metrics applies to every listed group/action except overrides.
Numeric precision and all options are retained. IDs bind only to this snapshot.

### 中文

辅助确定性离线跨地区LLM调度，所有工件已知。不可抢占工件先P后D且必须同地理地区，
本次请求只属于sampling_rule指定的一种处理。greedy_probability概率取最大项，剩余分支按完整分布抽样。
普通选择由程序独立抽样，模型只返回概率。A6例外：一次返回全部工件概率，取前10%且至少两个（不超过工件总数），不对destroy工件做epsilon抽样，并列使用显式RNG。
不同实例计其KV延迟；保持release、工序资格、Compute累积不超过1及有效VRAM，不能改约束或时长。
最小化总Flow（D完成减release之和）及总Bill（TOU+Demand），时间秒、费用CNY、功率kW、能量kWh。
每实例从零到最后D完成前按idle/active计功率，active按区间并集，之后为零。全部锚定零的900秒
窗口检查原并列峰和新峰，尾窗仍除900。三偏好组不是地理地区，使用当前真实细权重。
程序负责严格方向/零容差进入、SSGS、exact可行性/电费、归一化Tchebycheff接受、Archive与邻居自身权重替换。
每一步重新选动作，最多一个完整候选从本步冻结解出发；完整可行exact候选包括局部拒绝者均可入Archive。
最多五步，第1至4步后问停止，Polish在轨迹外。准备、排队、HTTP和搜索均计墙钟；没有未来oracle，
proxy不是真实收益，未观测历史是null而非失败。
A1单工序实例，A2同地区P/D路径，A3整工件跨地区路径，三者OS不变；A4正等待工序前插，
A5正等待工件P/D联合前插；A6一次全工件排序取前约10%且至少2工件destroy，原随机合法repair；
A7合法正并集压缩关键时刻；A8原峰singleton/support组及有限前驱修复，保留成员/位置/尝试上限、
原horizon及全窗严格削峰。时序冻结未选工序，结构变化SSGS重建。
只判断所给选项/命题，程序随机抽Choice概率，后问跟随实际抽中父对象；不输出schedule、解释、改约束或加预算。
表格按columns解释rows，全部Job保留。工序o=2j/2j+1为P/D，实例和开始由MS/starts_s取得，
地区查实例表，profile按operation_profile_ids查profiles，结束=开始+时长；两等待数组按o索引。
P/D实例不同才计kv_if_separate_s。历史default_metrics适用所有列出的组/动作，overrides覆盖例外。
数值精度和全部选项保留，ID只绑定当前快照。

## action

### English

For THIS LS step in the {group} preference group, using problem_contract, current_subproblem, current_solution, all_jobs, trajectory, action_history and budget, which of A1-A8 most merits execution NOW? Use the actual fine weight, current incumbent and completed-step feedback, not a previous step's policy. Balance observed objective/frontier benefit, opportunity, exposure uncertainty and real wall cost. Never equate high acceptance or low cost alone with a better final frontier.

### 中文

对{group}偏好组的本步LS，依据问题背景、当前细子问题、当前解、全部工件、轨迹、动作历史和预算，现在最值得执行的A1-A8动作是哪一个？使用真实细权重、当前解及已完成步的反馈，不复用前一步策略。综合真实目标/前沿收益、机会、曝光不确定性与墙钟成本，不把高接受率或低耗时单独当作最终前沿更好。

## job

### English

Using problem_contract, current_subproblem, current_solution, current_action, all_jobs and budget, which eligible Job among ALL criteria most merits constructing the next distinct candidate? Consider the supplied waits, assignment/KV structure, tariffs, demand exposure and available moves. Proxies are not known exact gains.

### 中文

依据问题背景、当前细子问题、当前解、动作、全部工件和预算，在本题全部合资格工件中，哪个最值得构造下一个不同候选？考虑等待、指派/KV结构、电价、需量暴露和可移动空间，代理不等于已知精确收益。

## phase

### English

For the ACTUALLY sampled selected_job and current_action, which eligible P/D operation in criteria should be changed? Use current_solution and the fine weight; do not switch to the model's previously preferred but unselected Job.

### 中文

对于程序实际抽中的工件及当前动作，本题合资格P/D工序中应改哪一个？依据当前解和真实细权重，不得换回模型此前偏好但未抽中的工件。

## region

### English

For current_action and the ACTUALLY sampled selected_job when present, which eligible geographic Region in criteria most merits the next candidate? Respect same-Region P/D and any required cross-Region move. Use actual tariff/demand parameters and current workload; exposure is only a proxy.

### 中文

对当前动作及已提供时的实际抽中工件，哪个合资格地理地区最值得构造候选？保持P/D同地区及动作所需跨地区要求，参考真实电价/需量参数和当前负载，暴露只是代理。

## path

### English

For the ACTUALLY sampled selected_job and selected_region, which legal P/D instance path in criteria most merits a candidate under the current fine weight? Account for phase eligibility, capacities, actual KV delay and workload proxies.

### 中文

对实际抽中的工件和地区，哪个合法P/D实例路径最值得按当前细权重构造候选？考虑工序资格、容量、实际KV延迟和负载代理。

## position

### English

For the ACTUALLY sampled parent fields in this state, which listed OS insertion position or timing critical start most merits a distinct candidate? Interpret position_kind and partial_solution exactly. Some A8 starts satisfy precedence bounds only; the program still checks resources and the all-window peak gate. Never use another Job's positions or assume unknown exact objective improvements.

### 中文

对当前状态中程序实际抽中的上层对象，哪个OS插入位置或关键开始时刻最值得构造不同候选？准确理解位置类型和部分解。部分A8时刻仅满足前驱边界，仍由程序检查资源和全窗削峰。不得使用其他工件的位置或假设未知精确收益。

## support

### English

For A8, selected_region, original_tied_peaks and the current selected_members, which original support group in criteria merits addition? All options have positive marginal removal potential on a still-deficient original peak and respect the member cap. Potential is not guaranteed attainable peak reduction after repair.

### 中文

对A8已选地区、全部并列原峰及当前成员，应加入哪个原支撑组？所有选项对尚未充分覆盖原峰有正边际移除潜力且不超成员上限；潜力不保证修复后能实现削峰。

## partition

### English

The current decision_kind has more than 255 options. Which stable ID interval in criteria should be examined next? Intervals are not heuristic top-k filtering: every original option remains reachable. Use their descriptions and numeric ranges. A later question will select inside the ACTUALLY sampled interval.

### 中文

当前决策超过255选项。下一步应考察哪个稳定ID区间？这不是启发式top-k筛选，所有原选项仍可达；根据区间描述与数值范围判断，后续在实际抽中区间内再选择。

## stop

### English

After completed_step in trajectory has actually finished exact evaluation, local acceptance and Archive handling, should this LS trajectory STOP now, reserving remaining wall time for other search? Use fine weight, actual step results, remaining opportunities, action-history exposure/cost and budget. Do not claim global optimality or treat one failed step as proof that improvement is impossible.

### 中文

在轨迹的已完成步实际结束exact、局部接受和Archive处理后，是否应立即停止本条LS，把剩余墙钟留给其他搜索？依据细权重、真实结果、剩余机会、动作曝光/成本和预算判断，不宣称全局最优，也不以一步失败证明无法改善。

## stop_true

### English

Available observations support ending this trajectory: no sufficiently concrete remaining opportunity justifies another step's expected wall cost.

### 中文

当前观察支持结束本轨迹：没有足够具体的剩余机会值得下一步预期墙钟成本。

## stop_false

### English

A concrete remaining opportunity warrants another step, OR evidence is insufficient to justify early stopping. Hard step/time limits are enforced separately.

### 中文

仍有具体机会值得下一步，或者证据不足以支持提前停止。步数和时间硬上限另由程序执行。

## A6：全部工件一次排序（覆盖通用 job 问法）

### English

For A6, return one probability distribution over ALL supplied Jobs, ranking their priority for joint destroy and repair from current_solution. The program takes the top selected_count distinct Jobs together, not one Job per request and not epsilon sampling. Consider the actual weight, waits, KV, assignment, tariffs, demand and history. Proxies are not exact gains. Original random legal repair follows. Every supplied Job must have a probability; probabilities must sum to one.

### 中文

针对A6，一次返回全部给定工件的概率分布，对当前解中适合联合破坏修复的工件排序。程序一次取概率最高的selected_count个不同工件，不逐工件请求，也不做epsilon抽样。依据实际权重、等待、KV、分配、电价、需量与历史；代理指标不是精确收益。后续保留原随机合法修复；每个工件都必须有概率，概率和为一。

## 实际动作选项（英文 criteria）

- A1: Single P/D instance reassignment within Region; preserve OS.

- A2: Joint P/D instance-path reassignment within Region; preserve OS.

- A3: Move one whole Job to another Region with a legal P/D path; preserve OS.

- A4: Earlier insertion of one positive-Flow-wait operation in OS.

- A5: Joint earlier insertion of P and D for one positive-total-wait Job.

- A6: Select destroy Jobs; original random legal multi-Job repair.

- A7: One timing-only positive active-union compression move.

- A8: Original-peak support selection and bounded strict all-window peak repair.

## 每步执行与分布使用

每步先问当前动作；普通选择以0.9概率取最大项、0.1按完整分布抽样。动作选定后构造至多一个完整候选。A6只发一个含全部200工件的Choice，取前20个；其他动作仍按实际选中的工件问条件选项。完整候选exact与Archive/局部接受完成后，第1至4步问停止，第5步强制结束。

同一请求的sampling_rule显式含candidate_budget=1、a6_epsilon_sampling=false与A6排序例外。已有B6/E09仍用旧背景，互不覆盖。
