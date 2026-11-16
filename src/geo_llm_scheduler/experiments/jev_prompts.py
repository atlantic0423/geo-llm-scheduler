"""Versioned English wire prompts and Chinese reading-only translations."""

PROMPT_VERSION = "jev_hierarchical_v4_run_bound"

BACKGROUND_EN_V2 = """You advise a static, offline, deterministic geo-distributed LLM
Prefill-Decode scheduling search. All jobs, releases, durations and resource demands are known.
Each job has nonpreemptive Prefill P followed by nonpreemptive Decode D. P and D MUST use the
same geographic Region, but may use different serving instances inside it. Decode starts after
Prefill completion plus the input-defined KV delay when instances differ. Instances are atomic;
concurrency must satisfy cumulative Compute share at most one and effective VRAM/KV capacity in GB.
Do not change durations, capacities, releases, preemption, Region legality or other jobs.

Both objectives MINIMIZE: total Flow (sum of each Decode completion time minus its release),
and electricity Bill (TOU energy cost plus fixed-window Demand charge). All time is global seconds;
power kW, energy kWh,
TOU CNY/kWh, Demand CNY/kW, Bill CNY. Instances use idle/active power from zero until, but excluding,
the last Decode completion; activity
is a UNION of operation intervals, not their sum. This batch contributes zero after T_end.
Demand uses ALL fixed non-overlapping 900-second windows anchored at zero, dividing even the partial tail by 900 seconds;
ALL tied original peaks and newly created peaks must be checked by the program.

MOEA/D searches MS instance assignments, OS operation order and feasible timings. Flow,
Balanced and Electricity are objective-preference groups, NOT geographic Regions. These group
labels contextualize the current fine weight; each LS step asks for a FRESH action distribution
for its CURRENT incumbent and subproblem. No cached distribution is reused across steps.
Actual local acceptance uses the current fine weight.
Strict fine-direction agreement and zero quality tolerance control entry to local search.
The program alone performs SSGS decoding, resource/precedence checks, exact billing, normalized
Tchebycheff acceptance, Pareto Archive updates and neighbor-own-weight replacement.
Population and historical Archive are different. Every complete feasible exact candidate has
an independent Archive opportunity, even if locally rejected. No future union, IGD+ or oracle
is available. Exposure, workload, packing and marginal-removal features are explicitly proxies,
not known counterfactual objective improvements. Unobserved history is null, not failure.

One LS step is one chosen action plus MacroSearch and local acceptance. At most FIVE LS steps
are allowed. Within a step at most SIX distinct complete candidates receive exact evaluation,
ALL independently generated from the SAME frozen incumbent, never a chain. Structural changes
rebuild timing by SSGS. Timing-only moves freeze unselected operations. Polish follows the
trajectory outside action selection/stopping. Initialization, feature construction, queueing,
HTTP, parsing, candidate generation and search all consume the same production wall time.

A1 reassigns one P or D instance inside its Region, keeping OS.
A2 reassigns one job's P/D path inside its Region, keeping OS.
A3 migrates one whole P/D job to another Region and legal path, keeping OS.
A4 inserts one positive-Flow-wait operation earlier in legal OS order.
A5 jointly inserts a positive-total-Flow-wait job's P/D earlier, P before D.
A6 selects about 10%, at least two jobs to destroy; ORIGINAL random legal repair is retained.
A7 moves one operation to an existing legal positive-active-union-compression critical time.
A8 selects a demand-paying Region, then original peak singleton/support-group members;
bounded precedence-aware single-path repair retains member/position/attempt limits, original
horizon and the strict ALL-window peak-reduction gate. It does not invent arbitrary coalitions.

You only assess the supplied finite options or stopping proposition. The program samples
your Choice probabilities with its explicit RNG; do not simulate random sampling or output
a schedule, code, explanation, changed constraint or extended budget. Questions in a request
are independent. Candidate IDs refer only to this snapshot; follow the ACTUALLY sampled parent
in subsequent states, which can differ from your highest-probability option."""

BACKGROUND_ZH_V2 = """你辅助静态、离线、确定性的跨地区 LLM Prefill-Decode 调度搜索。
所有工件、释放时间、时长和资源需求已知。每个工件不可抢占地先P后D；P/D必须同地区，
可在该地区不同实例，跨实例时D须等待输入定义的KV延迟。实例是原子资源，并发需满足
Compute累积份额不超过1及有效VRAM/KV容量。不得改变任务时长、容量、释放或地区合法性。

两个目标都最小化：总Flow为Decode完成减release之和；总电费为TOU加固定窗口Demand。
时间秒、功率kW、能量kWh、费用CNY。实例从零到批次结束的半开区间按idle/active计费，活动时间为区间并集，
结束后本批功率贡献为零。Demand检查全部900秒固定窗口，尾窗也除以900，检查全部并列原峰和新峰。

MOEA/D搜索MS指派、OS顺序和可行时序。Flow/Balanced/Electricity是目标偏好组而非地理地区。
偏好标签说明当前细权重；每一步LS都针对当前解与当前子问题重新询问动作分布，不跨步复用。
真实局部接受仍用当前细权重。严格细方向及零质量容差控制LS进入。
程序独占SSGS、可行性、精确电费、归一化Tchebycheff接受、Archive及邻居自身细权重替换。
Population与历史Archive不同；全部完整可行exact候选有独立Archive机会，包括局部拒绝者。
不提供未来union、IGD+或oracle。暴露、负载、压缩、移除潜力是代理而非已知反事实收益。
未观测历史为null而不是失败。

一步LS为一次动作、MacroSearch及局部接受，最多五步。每步最多精评六个不同完整候选，
都从同一冻结incumbent独立出发，不串成链。结构变化重跑SSGS，时序操作冻结未选工序。
Polish在轨迹之外。初始化、特征、排队、HTTP、解析、构造与搜索都计入同一生产墙钟。

A1同地区单P/D换实例；A2同地区P/D路径重指派；A3整工件跨地区联合迁移，三者OS不变。
A4单正等待工序合法前插；A5正总等待工件P/D联合前插且P在D前。
A6选择约10%且至少两个工件destroy，保留原随机合法repair。
A7单工序移动到原有限、合法、正活动并集压缩关键时刻。
A8选择有需量费的地区及原峰singleton/support-group；保留成员/位置/尝试上限、
按前驱的单路径修复、原horizon和全窗严格削峰gate，不任意生成联盟。

你只判断当前有限选项或停止命题。程序用显式RNG从Choice概率抽样；模型不自行模拟随机，
不输出schedule、代码、解释、约束变更或额外预算。同批问题独立。
ID只绑定当前快照，后续状态必须跟随程序实际抽中对象，可能不同于模型最高概率项。"""

BACKGROUND_EN = """Assist deterministic offline geo-distributed LLM scheduling. All jobs are known.
This request belongs to ONE treatment, identified by state.sampling_rule. Its
greedy_probability selects a maximum; the remaining branch samples your complete
distribution. The PROGRAM samples each selection independently; you return probabilities.
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
Each LS step freshly selects an action, constructs at most SIX distinct candidates
from ONE frozen incumbent, then accepts. Every complete feasible exact candidate
may update Archive, including locally rejected ones. At most FIVE steps; stop is
asked after steps 1-4. Polish follows outside the trajectory. All preparation,
queueing, HTTP and search consume production wall time. No future/oracle is known;
proxies are not exact gains, unobserved history is null, not failure.
A1 single phase instance; A2 P/D path within Region; A3 whole-job cross-Region path
(A1-3 keep OS); A4 positive-wait phase earlier in OS; A5 positive-wait job P/D
joint earlier insertion; A6 destroy ~10%, >=2 jobs, ORIGINAL random legal repair;
A7 one legal positive-union-compression critical-time move; A8 original-peak
singleton/support-group selection and bounded precedence-aware repair, retaining
member/position/attempt limits, original horizon and strict ALL-window peak gate.
Timing moves freeze unselected operations; structural moves rebuild by SSGS.
Assess only supplied options/proposition. Program RNG samples your Choice distribution;
follow ACTUALLY sampled parents, not your previous maximum. Do not output a schedule,
explanation, altered constraints or extra budget. Questions are independent.
Compact tables are lossless: columns label rows. all_jobs lists EVERY Job; operation
o=2*j/2*j+1 is P/D. Its machine/start come from current_solution.MS/starts_s, Region
from instance tables, profile from operation_profile_ids into profiles; end=start+duration.
resource_wait_s/flow_wait_s index o. Actual KV is kv_if_separate_s iff its P/D machines
differ. History default_metrics applies to every listed group/action except overrides.
Numeric precision and all options are retained. IDs bind only to this snapshot."""

BACKGROUND_ZH = """辅助确定性离线跨地区LLM调度，所有工件已知。不可抢占工件先P后D且必须同地理地区，
本次请求只属于sampling_rule指定的一种处理。greedy_probability概率取最大项，剩余分支按完整分布抽样。
每次选择由程序独立抽样，模型只返回概率；不在请求中比较两种参数。
不同实例计其KV延迟；保持release、工序资格、Compute累积不超过1及有效VRAM，不能改约束或时长。
最小化总Flow（D完成减release之和）及总Bill（TOU+Demand），时间秒、费用CNY、功率kW、能量kWh。
每实例从零到最后D完成前按idle/active计功率，active按区间并集，之后为零。全部锚定零的900秒
窗口检查原并列峰和新峰，尾窗仍除900。三偏好组不是地理地区，使用当前真实细权重。
程序负责严格方向/零容差进入、SSGS、exact可行性/电费、归一化Tchebycheff接受、Archive与邻居自身权重替换。
每一步重新选动作，最多六个不同候选都从同一冻结解出发；完整可行exact候选包括局部拒绝者均可入Archive。
最多五步，第1至4步后问停止，Polish在轨迹外。准备、排队、HTTP和搜索均计墙钟；没有未来oracle，
proxy不是真实收益，未观测历史是null而非失败。
A1单工序实例，A2同地区P/D路径，A3整工件跨地区路径，三者OS不变；A4正等待工序前插，
A5正等待工件P/D联合前插；A6约10%且至少2工件destroy，原随机合法repair；
A7合法正并集压缩关键时刻；A8原峰singleton/support组及有限前驱修复，保留成员/位置/尝试上限、
原horizon及全窗严格削峰。时序冻结未选工序，结构变化SSGS重建。
只判断所给选项/命题，程序随机抽Choice概率，后问跟随实际抽中父对象；不输出schedule、解释、改约束或加预算。
表格按columns解释rows，全部Job保留。工序o=2j/2j+1为P/D，实例和开始由MS/starts_s取得，
地区查实例表，profile按operation_profile_ids查profiles，结束=开始+时长；两等待数组按o索引。
P/D实例不同才计kv_if_separate_s。历史default_metrics适用所有列出的组/动作，overrides覆盖例外。
数值精度和全部选项保留，ID只绑定当前快照。"""

# Only the English entry is ever placed on the wire. Chinese is documentation only.
PROMPTS: dict[str, tuple[str, str]] = {
    "action": (
        "For THIS LS step in the {group} preference group, using problem_contract, "
        "current_subproblem, current_solution, all_jobs, trajectory, action_history and "
        "budget, which of A1-A8 most merits execution NOW? Use the actual fine weight, "
        "current incumbent and completed-step feedback, not a previous step's policy. "
        "Balance observed objective/frontier benefit, opportunity, "
        "exposure uncertainty and real wall cost. Never equate high acceptance or low cost "
        "alone with a better final frontier.",
        "对{group}偏好组的本步LS，依据问题背景、当前细子问题、当前解、全部工件、轨迹、"
        "动作历史和预算，现在最值得执行的A1-A8动作是哪一个？使用真实细权重、当前解及"
        "已完成步的反馈，不复用前一步策略。综合真实目标/前沿收益、机会、曝光不确定性与"
        "墙钟成本，不把高接受率或低耗时单独当作最终前沿更好。",
    ),
    "job": (
        "Using problem_contract, current_subproblem, current_solution, current_action, "
        "all_jobs and budget, which eligible Job among ALL criteria most merits constructing "
        "the next distinct candidate? Consider the supplied waits, assignment/KV structure, "
        "tariffs, demand exposure and available moves. Proxies are not known exact gains.",
        "依据问题背景、当前细子问题、当前解、动作、全部工件和预算，在本题全部合资格"
        "工件中，哪个最值得构造下一个不同候选？考虑等待、指派/KV结构、电价、需量暴露"
        "和可移动空间，代理不等于已知精确收益。",
    ),
    "phase": (
        "For the ACTUALLY sampled selected_job and current_action, which eligible P/D "
        "operation in criteria should be changed? Use current_solution and the fine weight; "
        "do not switch to the model's previously preferred but unselected Job.",
        "对于程序实际抽中的工件及当前动作，本题合资格P/D工序中应改哪一个？依据当前解"
        "和真实细权重，不得换回模型此前偏好但未抽中的工件。",
    ),
    "region": (
        "For current_action and the ACTUALLY sampled selected_job when present, which "
        "eligible geographic Region in criteria most merits the next candidate? Respect "
        "same-Region P/D and any required cross-Region move. Use actual tariff/demand "
        "parameters and current workload; exposure is only a proxy.",
        "对当前动作及已提供时的实际抽中工件，哪个合资格地理地区最值得构造候选？保持"
        "P/D同地区及动作所需跨地区要求，参考真实电价/需量参数和当前负载，暴露只是代理。",
    ),
    "path": (
        "For the ACTUALLY sampled selected_job and selected_region, which legal P/D "
        "instance path in criteria most merits a candidate under the current fine weight? "
        "Account for phase eligibility, capacities, actual KV delay and workload proxies.",
        "对实际抽中的工件和地区，哪个合法P/D实例路径最值得按当前细权重构造候选？考虑"
        "工序资格、容量、实际KV延迟和负载代理。",
    ),
    "position": (
        "For the ACTUALLY sampled parent fields in this state, which listed OS insertion "
        "position or timing critical start most merits a distinct candidate? Interpret "
        "position_kind and partial_solution exactly. Some A8 starts satisfy precedence bounds "
        "only; the program still checks resources and the all-window peak gate. Never use "
        "another Job's positions or assume unknown exact objective improvements.",
        "对当前状态中程序实际抽中的上层对象，哪个OS插入位置或关键开始时刻最值得构造"
        "不同候选？准确理解位置类型和部分解。部分A8时刻仅满足前驱边界，仍由程序检查"
        "资源和全窗削峰。不得使用其他工件的位置或假设未知精确收益。",
    ),
    "support": (
        "For A8, selected_region, original_tied_peaks and the current selected_members, "
        "which original support group in criteria merits addition? All options have positive "
        "marginal removal potential on a still-deficient original peak and respect the "
        "member cap. Potential is not guaranteed attainable peak reduction after repair.",
        "对A8已选地区、全部并列原峰及当前成员，应加入哪个原支撑组？所有选项对尚未"
        "充分覆盖原峰有正边际移除潜力且不超成员上限；潜力不保证修复后能实现削峰。",
    ),
    "partition": (
        "The current decision_kind has more than 255 options. Which stable ID interval "
        "in criteria should be examined next? Intervals are not heuristic top-k filtering: "
        "every original option remains reachable. Use their descriptions and numeric ranges. "
        "A later question will select inside the ACTUALLY sampled interval.",
        "当前决策超过255选项。下一步应考察哪个稳定ID区间？这不是启发式top-k筛选，"
        "所有原选项仍可达；根据区间描述与数值范围判断，后续在实际抽中区间内再选择。",
    ),
    "stop": (
        "After completed_step in trajectory has actually finished exact evaluation, local "
        "acceptance and Archive handling, should this LS trajectory STOP now, reserving "
        "remaining wall time for other search? Use fine weight, actual step results, "
        "remaining opportunities, action-history exposure/cost and budget. Do not claim "
        "global optimality or treat one failed step as proof that improvement is impossible.",
        "在轨迹的已完成步实际结束exact、局部接受和Archive处理后，是否应立即停止本条"
        "LS，把剩余墙钟留给其他搜索？依据细权重、真实结果、剩余机会、动作曝光/成本和"
        "预算判断，不宣称全局最优，也不以一步失败证明无法改善。",
    ),
    "stop_true": (
        "Available observations support ending this trajectory: no sufficiently concrete "
        "remaining opportunity justifies another step's expected wall cost.",
        "当前观察支持结束本轨迹：没有足够具体的剩余机会值得下一步预期墙钟成本。",
    ),
    "stop_false": (
        "A concrete remaining opportunity warrants another step, OR evidence is insufficient "
        "to justify early stopping. Hard step/time limits are enforced separately.",
        "仍有具体机会值得下一步，或者证据不足以支持提前停止。步数和时间硬上限另由程序执行。",
    ),
}

ACTION_CRITERIA = {
    "A1": "Single P/D instance reassignment within Region; preserve OS.",
    "A2": "Joint P/D instance-path reassignment within Region; preserve OS.",
    "A3": "Move one whole Job to another Region with a legal P/D path; preserve OS.",
    "A4": "Earlier insertion of one positive-Flow-wait operation in OS.",
    "A5": "Joint earlier insertion of P and D for one positive-total-wait Job.",
    "A6": "Select destroy Jobs; original random legal multi-Job repair.",
    "A7": "One timing-only positive active-union compression move.",
    "A8": "Original-peak support selection and bounded strict all-window peak repair.",
}
