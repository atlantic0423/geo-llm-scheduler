# Jev 当前试跑提示词：E09 英文发送，中文仅供阅读

版本：jev_hierarchical_v4_run_bound；由 scripts/export_jev_prompts.py 从实际代码常量生成。
英文模板已做确定性注入接口测试；不是英文优于中文的效果证据或实网验证。
当前试跑固定一个原200工件实例、一个种子，JEV_E09 / MAIN_Q / BILL_MOEAD三组各1200秒。
Jev只运行epsilon_g=0.9：90%取最大项（并列均匀），10%按模型完整分布抽样。实际state.sampling_rule明确绑定本次E09；请求不让模型比较两档参数。

## 完整执行顺序

1. 生产计时内执行BILL4/P100初始化。
2. 原strict/delta0门通过后，每一步LS开始前按当前解、真实细权重、轨迹反馈和累计动作历史重新问一次动作Choice。
3. 每步只问当前子问题所属偏好组，由程序用本步新分布执行epsilon抽样，不跨步复用；初始化与非LS子代不发动作请求。
4. 对全部合资格工件问Choice，程序抽样实际工件；再按所需询问phase、Region、path、position。
5. 各层独立抽样，下一问绑定实际抽中对象。单选项直接使用，不额外调用；空集返回无候选。
6. 一步最多六个不同完整候选，均从该步同一incumbent独立构造。统一exact与Archive、细权重局部接受。
7. 第1至4步后用真实结果问Noul停止；大于0.5停，等于0.5继续，第5步强制结束。
8. 原两重点Polish在轨迹后执行，随后原邻域替换。

本次仅epsilon_g=0.9。动作、工件与条件位置均以该概率选最大项（并列均匀），否则按Jev完整分布抽样。这是概率引导epsilon-greedy变体，随机分支不是均匀探索。随机操作在程序中，模型不自行模拟。

## 每次请求的共同背景

英文（实际发送）：

> Assist deterministic offline geo-distributed LLM scheduling. All jobs are known.
> This request belongs to ONE treatment, identified by state.sampling_rule. Its
> greedy_probability selects a maximum; the remaining branch samples your complete
> distribution. The PROGRAM samples each selection independently; you return probabilities.
> Each nonpreemptive job has Prefill P then Decode D in the SAME geographic Region;
> different instances incur its KV delay. Respect releases, phase eligibility,
> cumulative Compute <=1 and effective VRAM GB. Do not change constraints or durations.
> Minimize total Flow=sum(D completion-release) and Bill=TOU+Demand, in seconds/CNY.
> Power is kW, energy kWh: each instance uses idle/active power on [0,T_end), active
> by interval UNION, zero after the last Decode. Demand checks ALL 900s windows
> anchored at zero, including tied/new peaks; partial tails still divide by 900.
> Flow/Balanced/Electricity label preferences, not Regions. Use the CURRENT fine weight.
> The program enforces strict-direction/delta0 entry, SSGS, exact feasibility/billing,
> normalized Tchebycheff acceptance, Archive and neighbor-own-weight replacement.
> Each LS step freshly selects an action, constructs at most SIX distinct candidates
> from ONE frozen incumbent, then accepts. Every complete feasible exact candidate
> may update Archive, including locally rejected ones. At most FIVE steps; stop is
> asked after steps 1-4. Polish follows outside the trajectory. All preparation,
> queueing, HTTP and search consume production wall time. No future/oracle is known;
> proxies are not exact gains, unobserved history is null, not failure.
> A1 single phase instance; A2 P/D path within Region; A3 whole-job cross-Region path
> (A1-3 keep OS); A4 positive-wait phase earlier in OS; A5 positive-wait job P/D
> joint earlier insertion; A6 destroy ~10%, >=2 jobs, ORIGINAL random legal repair;
> A7 one legal positive-union-compression critical-time move; A8 original-peak
> singleton/support-group selection and bounded precedence-aware repair, retaining
> member/position/attempt limits, original horizon and strict ALL-window peak gate.
> Timing moves freeze unselected operations; structural moves rebuild by SSGS.
> Assess only supplied options/proposition. Program RNG samples your Choice distribution;
> follow ACTUALLY sampled parents, not your previous maximum. Do not output a schedule,
> explanation, altered constraints or extra budget. Questions are independent.
> Compact tables are lossless: columns label rows. all_jobs lists EVERY Job; operation
> o=2*j/2*j+1 is P/D. Its machine/start come from current_solution.MS/starts_s, Region
> from instance tables, profile from operation_profile_ids into profiles; end=start+duration.
> resource_wait_s/flow_wait_s index o. Actual KV is kv_if_separate_s iff its P/D machines
> differ. History default_metrics applies to every listed group/action except overrides.
> Numeric precision and all options are retained. IDs bind only to this snapshot.

中文（不发送）：

> 辅助确定性离线跨地区LLM调度，所有工件已知。不可抢占工件先P后D且必须同地理地区，
> 本次请求只属于sampling_rule指定的一种处理。greedy_probability概率取最大项，剩余分支按完整分布抽样。
> 每次选择由程序独立抽样，模型只返回概率；不在请求中比较两种参数。
> 不同实例计其KV延迟；保持release、工序资格、Compute累积不超过1及有效VRAM，不能改约束或时长。
> 最小化总Flow（D完成减release之和）及总Bill（TOU+Demand），时间秒、费用CNY、功率kW、能量kWh。
> 每实例从零到最后D完成前按idle/active计功率，active按区间并集，之后为零。全部锚定零的900秒
> 窗口检查原并列峰和新峰，尾窗仍除900。三偏好组不是地理地区，使用当前真实细权重。
> 程序负责严格方向/零容差进入、SSGS、exact可行性/电费、归一化Tchebycheff接受、Archive与邻居自身权重替换。
> 每一步重新选动作，最多六个不同候选都从同一冻结解出发；完整可行exact候选包括局部拒绝者均可入Archive。
> 最多五步，第1至4步后问停止，Polish在轨迹外。准备、排队、HTTP和搜索均计墙钟；没有未来oracle，
> proxy不是真实收益，未观测历史是null而非失败。
> A1单工序实例，A2同地区P/D路径，A3整工件跨地区路径，三者OS不变；A4正等待工序前插，
> A5正等待工件P/D联合前插；A6约10%且至少2工件destroy，原随机合法repair；
> A7合法正并集压缩关键时刻；A8原峰singleton/support组及有限前驱修复，保留成员/位置/尝试上限、
> 原horizon及全窗严格削峰。时序冻结未选工序，结构变化SSGS重建。
> 只判断所给选项/命题，程序随机抽Choice概率，后问跟随实际抽中父对象；不输出schedule、解释、改约束或加预算。
> 表格按columns解释rows，全部Job保留。工序o=2j/2j+1为P/D，实例和开始由MS/starts_s取得，
> 地区查实例表，profile按operation_profile_ids查profiles，结束=开始+时长；两等待数组按o索引。
> P/D实例不同才计kv_if_separate_s。历史default_metrics适用所有列出的组/动作，overrides覆盖例外。
> 数值精度和全部选项保留，ID只绑定当前快照。

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
sampling_rule固定本次treatment=JEV_E09、greedy_probability=0.9、model_distribution_probability=0.1，随机分支仍可能抽中最大项。
动作及对象请求都含真实细权重、完整MS/OS/时序、全部Job的P/D等待/资源/KV/指派、全部地区固定窗口功率。实际criteria由代码填充ID和英文结构化说明。A8 repair带partial_solution，不能把已移除成员当作仍固定。
v3使用可逆表格：全Job rows与profile索引、MS/starts恢复重复工序字段，历史default_metrics加overrides；全部候选和数值精度保留。expand_state可还原原快照。信息一致不保证模型分布一致，不把压缩当作已验证效果无损。
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
输入、请求及等待都计入生产墙钟。已做四原规模动作Prompt技术调用，没有启动正式campaign或验证完整LS费用/效果；实测见CAMPAIGN.md。

## 当前唯一Jev版本

模型固定jev-1.14.0。本次只运行JEV_E09，当前配置见arms.json的active_pilot。密钥只从进程环境TYPESAFE_API_KEY读取，不写命令参数、prompt、日志或Git。

```powershell
python -m geo_llm_scheduler.cli.jev --instance <original-component33-input.json> --epsilon-g 0.9 --seconds 1200 --seed 17401 --max-input-tokens 40000000 --long-context --output outputs/jev-e09
```

时间/seed仅命令示例，正式需按原面板冻结。requests.jsonl保存脱敏完整请求/有效响应/失败；result.json保存最终P100/Archive、每个最终成员的独立exact审计和实际usage。
实现与验收见IMPLEMENTATION.md。
E05软件接口与原两档配置仅保留历史复现，本次不派发、不在模型请求中介绍另一档。

[官方Choice](https://docs.typesafe.ai/primitives/choice)；[官方Noul](https://docs.typesafe.ai/primitives/noul)；[Jev-Mem附录B](https://arxiv.org/html/2609.23986v1)。
