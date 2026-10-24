# Jev 实际提示词：英文发送，中文仅供阅读

版本：jev_hierarchical_v2_per_step；由 scripts/export_jev_prompts.py 从实际代码常量生成。
英文模板已做确定性注入接口测试；不是英文优于中文的效果证据或实网验证。

## 完整执行顺序

1. 生产计时内执行BILL4/P100初始化。
2. 原strict/delta0门通过后，每一步LS开始前按当前解、真实细权重、轨迹反馈和累计动作历史重新问一次动作Choice。
3. 每步只问当前子问题所属偏好组，由程序用本步新分布执行epsilon抽样，不跨步复用；初始化与非LS子代不发动作请求。
4. 对全部合资格工件问Choice，程序抽样实际工件；再按所需询问phase、Region、path、position。
5. 各层独立抽样，下一问绑定实际抽中对象。单选项直接使用，不额外调用；空集返回无候选。
6. 一步最多六个不同完整候选，均从该步同一incumbent独立构造。统一exact与Archive、细权重局部接受。
7. 第1至4步后用真实结果问Noul停止；大于0.5停，等于0.5继续，第5步强制结束。
8. 原两重点Polish在轨迹后执行，随后原邻域替换。

两版仅epsilon_g不同，分别0.5/0.9。动作、工件与条件位置均以该概率选最大项（并列均匀），否则按Jev完整分布抽样。这是概率引导epsilon-greedy变体，随机分支不是均匀探索。随机操作在程序中，模型不自行模拟。

## 每次请求的共同背景

英文（实际发送）：

> You advise a static, offline, deterministic geo-distributed LLM
> Prefill-Decode scheduling search. All jobs, releases, durations and resource demands are known.
> Each job has nonpreemptive Prefill P followed by nonpreemptive Decode D. P and D MUST use the
> same geographic Region, but may use different serving instances inside it. Decode starts after
> Prefill completion plus the input-defined KV delay when instances differ. Instances are atomic;
> concurrency must satisfy cumulative Compute share at most one and effective VRAM/KV capacity in GB.
> Do not change durations, capacities, releases, preemption, Region legality or other jobs.
> 
> Both objectives MINIMIZE: total Flow (sum of each Decode completion time minus its release),
> and electricity Bill (TOU energy cost plus fixed-window Demand charge). All time is global seconds;
> power kW, energy kWh,
> TOU CNY/kWh, Demand CNY/kW, Bill CNY. Instances use idle/active power from zero until, but excluding,
> the last Decode completion; activity
> is a UNION of operation intervals, not their sum. This batch contributes zero after T_end.
> Demand uses ALL fixed non-overlapping 900-second windows anchored at zero, dividing even the partial tail by 900 seconds;
> ALL tied original peaks and newly created peaks must be checked by the program.
> 
> MOEA/D searches MS instance assignments, OS operation order and feasible timings. Flow,
> Balanced and Electricity are objective-preference groups, NOT geographic Regions. These group
> labels contextualize the current fine weight; each LS step asks for a FRESH action distribution
> for its CURRENT incumbent and subproblem. No cached distribution is reused across steps.
> Actual local acceptance uses the current fine weight.
> Strict fine-direction agreement and zero quality tolerance control entry to local search.
> The program alone performs SSGS decoding, resource/precedence checks, exact billing, normalized
> Tchebycheff acceptance, Pareto Archive updates and neighbor-own-weight replacement.
> Population and historical Archive are different. Every complete feasible exact candidate has
> an independent Archive opportunity, even if locally rejected. No future union, IGD+ or oracle
> is available. Exposure, workload, packing and marginal-removal features are explicitly proxies,
> not known counterfactual objective improvements. Unobserved history is null, not failure.
> 
> One LS step is one chosen action plus MacroSearch and local acceptance. At most FIVE LS steps
> are allowed. Within a step at most SIX distinct complete candidates receive exact evaluation,
> ALL independently generated from the SAME frozen incumbent, never a chain. Structural changes
> rebuild timing by SSGS. Timing-only moves freeze unselected operations. Polish follows the
> trajectory outside action selection/stopping. Initialization, feature construction, queueing,
> HTTP, parsing, candidate generation and search all consume the same production wall time.
> 
> A1 reassigns one P or D instance inside its Region, keeping OS.
> A2 reassigns one job's P/D path inside its Region, keeping OS.
> A3 migrates one whole P/D job to another Region and legal path, keeping OS.
> A4 inserts one positive-Flow-wait operation earlier in legal OS order.
> A5 jointly inserts a positive-total-Flow-wait job's P/D earlier, P before D.
> A6 selects about 10%, at least two jobs to destroy; ORIGINAL random legal repair is retained.
> A7 moves one operation to an existing legal positive-active-union-compression critical time.
> A8 selects a demand-paying Region, then original peak singleton/support-group members;
> bounded precedence-aware single-path repair retains member/position/attempt limits, original
> horizon and the strict ALL-window peak-reduction gate. It does not invent arbitrary coalitions.
> 
> You only assess the supplied finite options or stopping proposition. The program samples
> your Choice probabilities with its explicit RNG; do not simulate random sampling or output
> a schedule, code, explanation, changed constraint or extended budget. Questions in a request
> are independent. Candidate IDs refer only to this snapshot; follow the ACTUALLY sampled parent
> in subsequent states, which can differ from your highest-probability option.

中文（不发送）：

> 你辅助静态、离线、确定性的跨地区 LLM Prefill-Decode 调度搜索。
> 所有工件、释放时间、时长和资源需求已知。每个工件不可抢占地先P后D；P/D必须同地区，
> 可在该地区不同实例，跨实例时D须等待输入定义的KV延迟。实例是原子资源，并发需满足
> Compute累积份额不超过1及有效VRAM/KV容量。不得改变任务时长、容量、释放或地区合法性。
> 
> 两个目标都最小化：总Flow为Decode完成减release之和；总电费为TOU加固定窗口Demand。
> 时间秒、功率kW、能量kWh、费用CNY。实例从零到批次结束的半开区间按idle/active计费，活动时间为区间并集，
> 结束后本批功率贡献为零。Demand检查全部900秒固定窗口，尾窗也除以900，检查全部并列原峰和新峰。
> 
> MOEA/D搜索MS指派、OS顺序和可行时序。Flow/Balanced/Electricity是目标偏好组而非地理地区。
> 偏好标签说明当前细权重；每一步LS都针对当前解与当前子问题重新询问动作分布，不跨步复用。
> 真实局部接受仍用当前细权重。严格细方向及零质量容差控制LS进入。
> 程序独占SSGS、可行性、精确电费、归一化Tchebycheff接受、Archive及邻居自身细权重替换。
> Population与历史Archive不同；全部完整可行exact候选有独立Archive机会，包括局部拒绝者。
> 不提供未来union、IGD+或oracle。暴露、负载、压缩、移除潜力是代理而非已知反事实收益。
> 未观测历史为null而不是失败。
> 
> 一步LS为一次动作、MacroSearch及局部接受，最多五步。每步最多精评六个不同完整候选，
> 都从同一冻结incumbent独立出发，不串成链。结构变化重跑SSGS，时序操作冻结未选工序。
> Polish在轨迹之外。初始化、特征、排队、HTTP、解析、构造与搜索都计入同一生产墙钟。
> 
> A1同地区单P/D换实例；A2同地区P/D路径重指派；A3整工件跨地区联合迁移，三者OS不变。
> A4单正等待工序合法前插；A5正总等待工件P/D联合前插且P在D前。
> A6选择约10%且至少两个工件destroy，保留原随机合法repair。
> A7单工序移动到原有限、合法、正活动并集压缩关键时刻。
> A8选择有需量费的地区及原峰singleton/support-group；保留成员/位置/尝试上限、
> 按前驱的单路径修复、原horizon和全窗严格削峰gate，不任意生成联盟。
> 
> 你只判断当前有限选项或停止命题。程序用显式RNG从Choice概率抽样；模型不自行模拟随机，
> 不输出schedule、代码、解释、约束变更或额外预算。同批问题独立。
> ID只绑定当前快照，后续状态必须跟随程序实际抽中对象，可能不同于模型最高概率项。

## action

英文（实际发送，Flow）：

> For THIS LS step in the Flow preference group, using problem_contract, current_subproblem, current_solution, all_jobs, trajectory, action_history and budget, which of A1-A8 most merits execution NOW? Use the actual fine weight, current incumbent and completed-step feedback, not a previous step's policy. Balance observed objective/frontier benefit, opportunity, exposure uncertainty and real wall cost. Never equate high acceptance or low cost alone with a better final frontier.

中文（仅阅读）：

> 对Flow偏好组的本步LS，依据问题背景、当前细子问题、当前解、全部工件、轨迹、动作历史和预算，现在最值得执行的A1-A8动作是哪一个？使用真实细权重、当前解及已完成步的反馈，不复用前一步策略。综合真实目标/前沿收益、机会、曝光不确定性与墙钟成本，不把高接受率或低耗时单独当作最终前沿更好。

英文（实际发送，Balanced）：

> For THIS LS step in the Balanced preference group, using problem_contract, current_subproblem, current_solution, all_jobs, trajectory, action_history and budget, which of A1-A8 most merits execution NOW? Use the actual fine weight, current incumbent and completed-step feedback, not a previous step's policy. Balance observed objective/frontier benefit, opportunity, exposure uncertainty and real wall cost. Never equate high acceptance or low cost alone with a better final frontier.

中文（仅阅读）：

> 对Balanced偏好组的本步LS，依据问题背景、当前细子问题、当前解、全部工件、轨迹、动作历史和预算，现在最值得执行的A1-A8动作是哪一个？使用真实细权重、当前解及已完成步的反馈，不复用前一步策略。综合真实目标/前沿收益、机会、曝光不确定性与墙钟成本，不把高接受率或低耗时单独当作最终前沿更好。

英文（实际发送，Electricity）：

> For THIS LS step in the Electricity preference group, using problem_contract, current_subproblem, current_solution, all_jobs, trajectory, action_history and budget, which of A1-A8 most merits execution NOW? Use the actual fine weight, current incumbent and completed-step feedback, not a previous step's policy. Balance observed objective/frontier benefit, opportunity, exposure uncertainty and real wall cost. Never equate high acceptance or low cost alone with a better final frontier.

中文（仅阅读）：

> 对Electricity偏好组的本步LS，依据问题背景、当前细子问题、当前解、全部工件、轨迹、动作历史和预算，现在最值得执行的A1-A8动作是哪一个？使用真实细权重、当前解及已完成步的反馈，不复用前一步策略。综合真实目标/前沿收益、机会、曝光不确定性与墙钟成本，不把高接受率或低耗时单独当作最终前沿更好。

## job

英文（实际发送，job）：

> Using problem_contract, current_subproblem, current_solution, current_action, all_jobs and budget, which eligible Job among ALL criteria most merits constructing the next distinct candidate? Consider the supplied waits, assignment/KV structure, tariffs, demand exposure and available moves. Proxies are not known exact gains.

中文（仅阅读）：

> 依据问题背景、当前细子问题、当前解、动作、全部工件和预算，在本题全部合资格工件中，哪个最值得构造下一个不同候选？考虑等待、指派/KV结构、电价、需量暴露和可移动空间，代理不等于已知精确收益。

## phase

英文（实际发送，phase）：

> For the ACTUALLY sampled selected_job and current_action, which eligible P/D operation in criteria should be changed? Use current_solution and the fine weight; do not switch to the model's previously preferred but unselected Job.

中文（仅阅读）：

> 对于程序实际抽中的工件及当前动作，本题合资格P/D工序中应改哪一个？依据当前解和真实细权重，不得换回模型此前偏好但未抽中的工件。

## region

英文（实际发送，region）：

> For current_action and the ACTUALLY sampled selected_job when present, which eligible geographic Region in criteria most merits the next candidate? Respect same-Region P/D and any required cross-Region move. Use actual tariff/demand parameters and current workload; exposure is only a proxy.

中文（仅阅读）：

> 对当前动作及已提供时的实际抽中工件，哪个合资格地理地区最值得构造候选？保持P/D同地区及动作所需跨地区要求，参考真实电价/需量参数和当前负载，暴露只是代理。

## path

英文（实际发送，path）：

> For the ACTUALLY sampled selected_job and selected_region, which legal P/D instance path in criteria most merits a candidate under the current fine weight? Account for phase eligibility, capacities, actual KV delay and workload proxies.

中文（仅阅读）：

> 对实际抽中的工件和地区，哪个合法P/D实例路径最值得按当前细权重构造候选？考虑工序资格、容量、实际KV延迟和负载代理。

## position

英文（实际发送，position）：

> For the ACTUALLY sampled parent fields in this state, which listed OS insertion position or timing critical start most merits a distinct candidate? Interpret position_kind and partial_solution exactly. Some A8 starts satisfy precedence bounds only; the program still checks resources and the all-window peak gate. Never use another Job's positions or assume unknown exact objective improvements.

中文（仅阅读）：

> 对当前状态中程序实际抽中的上层对象，哪个OS插入位置或关键开始时刻最值得构造不同候选？准确理解位置类型和部分解。部分A8时刻仅满足前驱边界，仍由程序检查资源和全窗削峰。不得使用其他工件的位置或假设未知精确收益。

## support

英文（实际发送，support）：

> For A8, selected_region, original_tied_peaks and the current selected_members, which original support group in criteria merits addition? All options have positive marginal removal potential on a still-deficient original peak and respect the member cap. Potential is not guaranteed attainable peak reduction after repair.

中文（仅阅读）：

> 对A8已选地区、全部并列原峰及当前成员，应加入哪个原支撑组？所有选项对尚未充分覆盖原峰有正边际移除潜力且不超成员上限；潜力不保证修复后能实现削峰。

## partition

英文（实际发送，partition）：

> The current decision_kind has more than 255 options. Which stable ID interval in criteria should be examined next? Intervals are not heuristic top-k filtering: every original option remains reachable. Use their descriptions and numeric ranges. A later question will select inside the ACTUALLY sampled interval.

中文（仅阅读）：

> 当前决策超过255选项。下一步应考察哪个稳定ID区间？这不是启发式top-k筛选，所有原选项仍可达；根据区间描述与数值范围判断，后续在实际抽中区间内再选择。

## stop

英文（实际发送，stop）：

> After completed_step in trajectory has actually finished exact evaluation, local acceptance and Archive handling, should this LS trajectory STOP now, reserving remaining wall time for other search? Use fine weight, actual step results, remaining opportunities, action-history exposure/cost and budget. Do not claim global optimality or treat one failed step as proof that improvement is impossible.

中文（仅阅读）：

> 在轨迹的已完成步实际结束exact、局部接受和Archive处理后，是否应立即停止本条LS，把剩余墙钟留给其他搜索？依据细权重、真实结果、剩余机会、动作曝光/成本和预算判断，不宣称全局最优，也不以一步失败证明无法改善。

## stop_true

英文（实际发送，stop_true）：

> Available observations support ending this trajectory: no sufficiently concrete remaining opportunity justifies another step's expected wall cost.

中文（仅阅读）：

> 当前观察支持结束本轨迹：没有足够具体的剩余机会值得下一步预期墙钟成本。

## stop_false

英文（实际发送，stop_false）：

> A concrete remaining opportunity warrants another step, OR evidence is insufficient to justify early stopping. Hard step/time limits are enforced separately.

中文（仅阅读）：

> 仍有具体机会值得下一步，或者证据不足以支持提前停止。步数和时间硬上限另由程序执行。

## 实际动态输入

共同state含英文背景、真实拓扑/电价/资源、预算、曝光分母清晰的动作历史和Archive摘要。
动作及对象请求都含真实细权重、完整MS/OS/时序、全部Job的P/D等待/资源/KV/指派、全部地区固定窗口功率。实际criteria由代码填充ID和英文结构化说明。A8 repair带partial_solution，不能把已移除成员当作仍固定。
动作请求另含当前子问题索引、偏好组、归一化上下文、下一步编号和本轨迹已完成步。动作反馈累计自run开始，不因每步新回答而清空。request_mode为per_ls_step。
名称不发送，只传稳定数值ID；未按评分截取工件。超过255选项按稳定ID区间分层，保留全部可达候选；区间描述含数量和首尾对象。分层不等价于全局一次Choice。

## 算子权限

| 算子 | Jev控制 | 程序保留 |
| --- | --- | --- |
| A1 | Job、phase、同地区实例 | Region和OS不变 |
| A2 | Job、同地区P/D路径 | 合法容量和SSGS |
| A3 | Job、目标地区、P/D路径 | P/D联合迁移，OS不变 |
| A4 | Job、phase、全部合法前插位置 | 正等待、单工序前插 |
| A5 | Job、P位置和条件D位置 | 正总等待、双前移、P在D前；不物化全部位置对 |
| A6 | 无放回destroy工件集合 | 10%且至少2，原随机合法repair |
| A7 | Job、phase、全部合法正压缩关键时刻 | 原关键时刻与非成员冻结 |
| A8 | Region、singleton工件/phase或正边际support组、repair关键时刻 |前驱/窄窗顺序、成员/位置/2B尝试上限、原horizon、全窗削峰 |

A7绕过原两代表/2B预筛，A8改变支撑组及位置选择，是研究策略，不冒称纯加速。A6 repair仍随机。A8部分位置只满足前驱边界，仍可能资源失败，按上限尝试，失败整recipe回滚。

## 失败与预算

不重试、不换模型。HTTP子进程硬截止kill/reap。动作请求失败或额度不足时，本步用均匀分布，不复用先前分布；每步记录action_source、action_request和action_query_s。对象失败在完整eligible集合均匀抽样并记录回退，分层回退按组大小加权。停止失败且连续两步无局部接受/Archive新目标时用本地规则停，否则继续，仍受L5和墙钟约束。
默认每run最多512请求、单次15秒、累计等待最多生产预算50%、已计费或保守预留输入最多400万token。每次发送前按64k默认整请求上界预留，合法usage按实替换；未知失败保留整份预留。额度是单run限制，不是账号余额或多run全局总预算。
输入、请求及等待都计入生产墙钟。当前未启动正式campaign或做新付费调用。

## 两版运行

模型固定jev-1.14.0。两版仅epsilon_g不同，配置见arms.json。密钥只从进程环境TYPESAFE_API_KEY读取，不写命令参数、prompt、日志或Git。

```powershell
python -m geo_llm_scheduler.cli.jev --instance <original-component33-input.json> --epsilon-g 0.5 --seconds 300 --seed 18001 --output outputs/jev-e05
python -m geo_llm_scheduler.cli.jev --instance <original-component33-input.json> --epsilon-g 0.9 --seconds 300 --seed 18001 --output outputs/jev-e09
```

时间/seed仅命令示例，正式需按原面板冻结。requests.jsonl保存脱敏完整请求/有效响应/失败；result.json保存最终P100/Archive、每个最终成员的独立exact审计和实际usage。
实现与验收见IMPLEMENTATION.md。

[官方Choice](https://docs.typesafe.ai/primitives/choice)；[官方Noul](https://docs.typesafe.ai/primitives/noul)；[Jev-Mem附录B](https://arxiv.org/html/2609.23986v1)。
