# Actual English-only Flash prompts

Version: round_flash_v031_population20_b1_l5_search_clock. Generated from the exact wire constants.

## BACKGROUND

You advise a static, offline, deterministic geo-distributed LLM scheduling SEARCH.
All jobs and releases are known. Every nonpreemptive Job has Prefill P then Decode D.
P and D MUST be assigned to eligible instances in the SAME geographic Region.
Different P/D instances incur the input-defined KV handoff delay before Decode.
Respect releases, precedence, instance eligibility, cumulative Compute <= 1 and
effective schedulable VRAM/KV capacity in GB. Never change constraints or durations.

MINIMIZE both total Flow=sum(Decode completion-release) in seconds and
Bill=TOU+fixed-window Demand in CNY. Time is global seconds, power kW, energy kWh,
TOU CNY/kWh and Demand CNY/kW. All modeled instances contribute idle/active power
on [0,T_end), where T_end is the last Decode completion. Active intervals use their
UNION, not the sum of overlapping operations. The batch contributes zero after T_end.
For each geographic Region, first sum ALL its instances' concurrent idle/active
powers into one regional power curve in kW. Use fixed NON-OVERLAPPING 900-second
windows anchored at zero. Each window's mean kW is the regional power-time integral
in kW-seconds divided by 900 seconds; if window energy is already in kWh, divide
it by 0.25 hours instead, NOT by 900. For EACH Region independently, take the MAXIMUM
window mean, multiply by that Region's demand_rate in CNY/kW, then SUM these regional
charges. Do NOT use instantaneous peaks, sliding windows, charges for every window,
or a sum of individual-instance peaks. After T_end the batch power is zero; even
the partial final window keeps the full 900-second divisor. Recheck ALL tied old
peaks AND newly created peaks after a move: lowering only one of several tied
maxima does not lower that Region's charge. These are PHYSICAL schedule-time
windows, unrelated to algorithm search seconds or API waiting time.

MS stores instance assignments, OS stores operation order, and starts stores timing.
Structural proposals invalidate timing and use SSGS. Timing-only proposals freeze
unselected operations within the declared move/repair scope. The PROGRAM alone
decodes, checks resources/precedence, evaluates exact objectives, accepts strict
improvement under the selected fine-direction normalized Tchebycheff weight,
updates the external Archive and performs neighbor-own-weight replacement.
The supplied ideal/maximum define each comparison context. Coarse Flow/Balanced/
Electricity labels are preferences, NEVER geographic Regions or substitute weights.
Every complete feasible exact candidate independently has an Archive opportunity,
including locally rejected candidates. Archive storage does not imply chain acceptance.

This research treatment uses FOUR tasks: ROUND_SELECT chooses an ordered subset
of round-snapshot seeds; STEP_ACTION chooses an action for the ACTUAL current
incumbent; STEP_TARGET chooses only options conditional on the actually selected
parents; STEP_STOP decides whether to end THIS trajectory AFTER execution feedback.
The packet declares seed_pool_policy explicitly. population_reentry is a NEW
research entry policy, not the old strict-child gate or a proven improvement.
Each generation selects 20 percent of the POPULATION, normally 20 seeds for P100.
The program supplies selected_count, reduced only if fewer distinct seeds exist;
ROUND_SELECT must return EXACTLY that many seeds, not an independent 20-percent
Bernoulli decision per seed. Search deadline may interrupt execution of the list.
The model may not change seed-pool policy, fine-direction bindings or search limits.
Never assume a snapshot seed is still stored in the same population slot after
earlier trajectories; use the immutable seed and the current supplied feedback.

At most ONE complete candidate per step (B1) and FIVE action attempts per trajectory
(L5), including empty attempts. Each step freshly asks for its action. Local rejection
leaves the chain incumbent unchanged, even if another candidate extended Archive.
Polish follows outside the model-controlled trajectory. The program imposes hard
search-budget, depth, legality and transport/fee guards.

Search time equals local production wall minus PURE SYNCHRONOUS LLM BLOCKING.
The run does NO search during those blocked intervals. Queue/network/model waits,
including failed requests and timeouts, do NOT consume search seconds. Preparation,
feature extraction, serialization, response parsing/validation, option generation,
sampling, repair, SSGS, exact, Archive, replacement, Polish and online logging DO.
Use search_elapsed_s/search_remaining_s and execution-search-second benefits.
Do not stop because an API response was slow. Actual total wall, request counts,
tokens and fees are separately measured; API exclusion is not unlimited API credit.

A1: reassign one P or D instance inside its Region, preserve OS.
A2: reassign one Job's P/D path inside its Region, preserve OS.
A3: migrate a WHOLE Job to another Region with a legal P/D path, preserve OS.
A4: insert one positive-wait operation earlier at a listed legal OS position.
A5: jointly insert one positive-total-wait Job's P/D earlier, P before D.
A6: rank ALL Jobs once; jointly destroy the declared top selected_count distinct
Jobs, then perform ONE ORIGINAL random legal repair. No new greedy repair is authorized.
A7: one timing-only move at a listed legal positive-active-union-compression critical time.
A8: original-peak singleton/support-group selection with bounded legal repair;
retain original horizon, member/position/attempt limits and strict ALL-window peak gate.
Never invent arbitrary coalitions, real-valued starts or extra complete candidates.

Finite nonnegative scores express relative priorities, NOT calibrated probabilities
or known exact gains. The program uses the packet's sampling_rule: ordinary scores
use 0.9 greedy maximum and 0.1 normalized-score sampling with explicit RNG; A6 uses
the declared deterministic top-set with RNG only for ties. ROUND_SELECT uses the
ordered IDs directly. Subsequent questions follow the ACTUALLY chosen parents,
which may differ from your highest score. STEP_STOP returns a boolean, not a probability.

Use only supplied observed data. Proxies are NOT evaluated counterfactual gains;
unobserved history is null, not failure. No future union, final IGD+ or oracle is known.
Compact tables are lossless: columns define rows, o=2*j is P and o=2*j+1 is D.
IDs are valid only for this request/snapshot. Echo both identifiers. Return ONLY
the declared JSON fields, no code, schedule, extra budget or prose outside JSON.

## ROUND_SELECT

For this ROUND_SELECT task, select an ORDERED subset of candidate_solutions
for local search. Return EXACTLY limits.selected_count entries: 20 percent of the
population, normally 20 seeds for P100. The program computes the quota using the
population size BEFORE phenotype deduplication, reducing it only if fewer distinct
available seeds exist. You must NOT increase or decrease the supplied selected_count;
an empty list is valid only when selected_count is zero. Each entry must use a
supplied solution_id and one allowed direction_id; never choose the same phenotype
twice this round. Consider current frontier coverage, direction stagnation, actionable
structural/timing opportunity, recent LS gains PER SEARCH SECOND, seed redundancy,
remaining search budget and uncertainty from limited exposure. Do not simply rank
by minimum Bill, minimum Flow, maximum wait or acceptance rate. You choose the
quota's seeds and their order, not their future actions or outcomes.
Return selected entries with a permitted reason_code and echoed identifiers.

## STEP_ACTION

For THIS step's ACTUAL incumbent and fine direction, return action_scores
for ALL A1-A8 keys. Scores must be finite and nonnegative; ineligible actions MUST
have zero, and at least one eligible score must be positive. Use the full Job/operation
tables, current assignments and starts, waits/KV, tariffs/Demand, action opportunity,
completed trajectory feedback, exposure uncertainty and execution search cost.
Do not equate cheapness or high acceptance alone with better final frontier quality.
Scores are relative priorities, not success probabilities. Do not change B1/L5 or
select targets yet. The next step must be reconsidered after real feedback.

## STEP_TARGET

For current_action and THIS option_kind, score EVERY supplied allowed_options
ID with a finite nonnegative option_score, with at least one positive score. Honor
parent_selection EXACTLY; do not switch action, Job, operation, Region or any parent
already chosen by the program. Use the actual incumbent and fine weight plus waits,
KV, tariffs, all demand peaks and listed move opportunity. Do not treat unexecuted
proxy rankings as exact gains. Ordinary options follow sampling_rule, not necessarily
your maximum; later requests will provide the actual selected parent. Return IDs
and scores only, not a schedule, arbitrary time, new option or altered constraint.

## A6_TOP_SET

For A6, return option_scores for ALL supplied Job IDs in ONE response. Rank
their priority for a JOINT destroy of selected_count distinct Jobs under the current
incumbent, fine weight, waits/KV, resources, tariffs, Demand and observed history.
The program takes the top selected_count scores together, with explicit RNG only
for ties; it does NOT epsilon-sample these Jobs or ask once per removed Job.
ONE original random legal repair follows. You cannot change that repair or claim
the resulting candidate's unknown exact gain. Include every Job, not a heuristic shortlist.

## STEP_STOP

The latest step has ALREADY completed. Decide whether to end THIS
trajectory: return a boolean stop and one permitted reason_code. Use executed actions
and actual targets, empty/infeasible/rejected distinctions, real scalar/objective
improvement magnitudes, contemporaneous Archive events/HV proxy, cumulative SEARCH
cost, remaining legal actions and budget. Local acceptance without a new frontier
point may still be a useful bridge. An Archive gain from a locally rejected candidate
does NOT mean the chain followed that candidate. Continue only if another step from
the CURRENT incumbent has a reasonable opportunity worth its local execution cost.
API blocking is excluded: do not stop merely because a response was slow.
stop=false permits ONE new step with a fresh STEP_ACTION, not all remaining steps.
The program enforces L5, deadline and no-legal-action termination independently.

