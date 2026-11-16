# D2 prospective opportunity screening and D1 cost verification

## Material Passport

Origin Skill: academic-research-suite / experiment-agent.
Mode: plan/run. Date: 2026-10-04.
Verification Status: future scientific results UNVERIFIED.
Version: D2_opportunity_v1.

The user authorized implementation, validation and direct launch of another
local experiment lasting at least ten hours, with no further confirmation.
This experiment changes observation and offline intervention only. It does
not adopt a new parent selector, Q-learning policy, trigger, replacement rule
or timing operator in the default algorithm.

## Questions and frozen scope

Primary: do pre-action scheduling opportunity proxies improve selection of
the first parent within the existing MOEA/D neighborhood, beyond objective
quality, genotype/objective distance, stage and causal past slot gains?
Use a fixed second parent and three common-seed variation draws per source.

Secondary: do the same proxies provide additional information about A1-A8
local-search donors? Analyze each action separately. These results cannot
substitute for the primary parent-selection test.

D1 engineering: can checking actual MS/OS compatibility before extracting
positive wait preserve a nonbinding conditional-route result while saving
cost? This is engineering verification, not renewed D1 quality confirmation.

## Independent instances and duration

24 new bases: 50 jobs 850001-850012; 100 jobs 860001-860012.
Each base has paired H/T tariffs and seeds 1101/2202/3303: 144 source runs.
The first eight bases at each size are development (16 bases); the remaining
four at each size are held out (8 bases). The split is fixed before labels.
All six tariff/seed repetitions of a base stay in the same split.

Frozen F6 config and operators inherit the bd358dd research checkout.
This is the existing experimental F6 baseline, not a claim that unmerged
research branches are GitHub main or that all current specification parameters
have already been adopted on main.

50/100-job source wall budgets are 1800/3600 seconds. Source workload totals
108 worker-hours, with at most 10 source workers plus 2 offline workers.
Source computation alone therefore has a 10.8-hour wall-clock lower bound.
Use a resource pilot to estimate total duration; do not idle or repeat runs
to meet the requested duration.

Pilot uses old development bases 810001/820001 and seeds 6606/7707/8808,
normally 60/120 seconds. No formal heldout tuning occurs during the pilot.

## Prospective panels, controls and features

At the first pre-offspring event reaching 10%, 50% and 90% of the source
budget, sample one caller per preference and six population slots from that
caller's existing geometric neighborhood. The caller incumbent is the fixed
second parent. Capture exact source candidates and all control inputs before
running any offline action. Nine panels per run, 1296 panels in total.
Repeated phenotypes and degenerate pools remain in the analysis.

Sampling uses independent named RNG streams. The observer records an EWMA
of each slot's past own-caller improvement, updated after offspring handling;
this is a limited causal history control, not a complete implementation of
BRA or HK-MOEA/D. The observer never consumes algorithm RNG or changes Q.
Its time is included in the F6 source wall budget.

Controls: normalized source and mate Flow/Bill, source and mate scalar quality,
explicit normalized objective distance, MS/OS Hamming distances
to the fixed mate, source/caller weight distance, stagnation, past EWMA gain,
phase, preference and job count. Objective distance is represented by the
two normalized coordinates and explicit distance; the linear baseline uses the complete control
vector rather than a distance-only heuristic.

Opportunity proxies: positive precedence/resource waiting, legal A4 order
position count, A5 coupled position count, prefill precedence slack, legal
same-region assignment count, other-region path count, current active-union
gap fraction, existing fixed-window peak multiplicity and Demand/Bill fraction.
These are upper/proxy opportunities, not certificates of feasible improvement.
Feature extraction does not run complete move pools, new SSGS or exact.
All control and feature costs are measured separately.

## Interventions and ground truth

For each source slot execute variation and A1-A8 separately, B=3 distinct
complete exact candidates per action. Use identical named action RNGs across
source alternatives, randomizing execution order independently of outcomes.
Structural candidates use SSGS; timing candidates retain their frozen operations.
For variation, the caller/mate is the retention baseline, so old source timing
is never injected as a candidate offspring. For A1-A8 the donor is the baseline.
MacroSearch's temporary ideal belongs to each source/action. Retain the relevant
baseline if no strict better candidate exists, using deterministic ties.

After actual per-source retention, evaluate retained outputs in a common
candidate-updated comparison context for the matched panel/action. Never use
other sources' candidates to reselect a source's retained output.
The label is the absolute difference between the caller/mate scalar and
retained scalar in that common normalized context, not a percentage.
Every source, mate and proposal is independently re-evaluated with exact.
Record construction, SSGS/evaluation, feature, audit and failure costs.
Empty outcomes and negative caller-relative gains remain.

D1 twins use the first sampled source and variation/A1/A4/A5 target generators.
Missing structural proposals produce an unchanged target, which remains.
Compare reference/early-guard conditional routing with the same named RNG,
three exact cap and 10/30-second safety quotas, balanced order. Audit all
proposals independently. Record binding quotas; demand identical routes,
retained starts/objectives and exact counts only for nonbinding pairs.
Do not call finite-cap differences semantic equivalence, and do not describe
this artificial target panel as natural D1 opportunity prevalence.

## Frozen prediction and inference

Fit per-action standardized linear ridge, fixed alpha=1, intercept and scaling
estimated from development bases only. No hyperparameter selection on heldout.
Compare RANDOM, source QUALITY, genotype DISTANCE, causal HISTORY, all-controls
BASE, controls plus OPPORTUNITY, and SHUFFLED opportunity features.
SHUFFLED retains extended model capacity and permutes feature vectors within
each matched source panel/action, independently in development and heldout.

Primary contrasts are OPPORTUNITY minus BASE and OPPORTUNITY minus SHUFFLED
on heldout variation parent-selection gains. Aggregate panels/preferences/
phases/tariffs/seeds equally within each base, then bases equally.
Use paired exact two-sided sign flip, Holm across these two contrasts, and
20,000 base-bootstrap draws. Screening requires both means at least 0.001
normalized scalar units, positive interval lower bounds and Holm p<0.05.
This is an internal mechanism screen, not final novelty/online confirmation.
Report MSE, all control policies, each A1-A8 exploratory response, empty rates,
cost and degenerate pools even if the primary gate fails.

Failure rejects only this frozen cheap proxy/linear predictor configuration;
it does not prove all opportunity information useless. Do not add samples
after inspecting p-values. Full MOEA/D HV/IGD+/coverage confirmation is later.

## Restart, pause and migration

Manifest, protocol, source, configuration, instance and initialization hashes
are immutable. Source and offline jobs publish atomic complete markers.
Completed paired panel checkpoints can be reused on the same host after
an accidental interruption without a recorded algorithm failure.
Resume is at job/panel level; an interrupted source solver is restarted for
its full budget, rather than claiming its process state was restored.
The supervisor lease prevents duplicate runs; heartbeats and resource guards
monitor all workers. Failures stop new dispatch and are never automatically
retried. An owned-worker hard timeout records a failure.

Create stop.request to drain active jobs; resumption requires the user's
authorization to remove that request. Preserve interrupted attempts in quarantine.
Explicit migration retains complete source/probe pairs and restarts unfinished
pairs on the new host, keeping old evidence. Source and its offline probes
must share a host. Report host/epoch provenance; do not pool paired costs
across hosts without stratification.

Run with PYTHONPATH explicitly bound to this checkout/src because the shared
venv's editable package otherwise resolves to a different checkout.
Commands: python -m geo_llm_scheduler.experiments.d2_campaign prepare/run/recover/migrate
with --root CAMPAIGN and --repository CHECKOUT for preparation.

After all 288 jobs validate, generate the analysis report and SHA256 inventory.
Complete raw Release delivery, source bundle and reproduction instructions
are separate follow-up acceptance steps. Preserve all originals until asset
lists and sample downloads have been verified.
