# D1 conditional transfer: terminal internal holdout

## Material Passport

Origin Skill: academic-research-suite / experiment-agent. Origin Mode: plan+run+validate.
Origin Date: 2026-10-04. Verification Status: UNVERIFIED until execution and audit.
Version Label: D1_terminal_routing_holdout_v1.

## Question and frozen scope

Test the rule derived from the first eight development bases: positive source
residual wait, identical actual target MS, and changed actual target OS imply TRUE
bounded wait projection; otherwise use fresh compression-based A7. This experiment
does not change the online MOEA/D, its Q-table, trigger, archive, or default operators.
The original unconditional D1 prototype did not pass its fresh-A7 quality gate.

The remaining P2 bases are 810005–810012 (50 jobs) and 820005–820012 (100 jobs),
two H/T tariffs, seeds 1101/2202/3303: 96 terminal F6 runs. They were not used to
select the D1 rule, but have already participated in P2 research. This is an
internal holdout, not entirely unseen independent instances or stage snapshots.
Source P2 code is f6b7fba; terminal budgets remain 600/1200 seconds. Do not pool
these with the first D1 campaign's 1800/3600-second early/middle/late sources.

Sample one unique phenotype in each preference stratum plus one remainder with
explicit named RNGs. Preserve empty/duplicate strata as missing. Per source, make
OS/INSTANCE/REGION perturbations at one-job and approximately ten-percent sizes.
The ceiling is 384 sampled sources and 2304 paired targets, not 2304 independent
instances or complete MOEA/D runs. Preserve unavailable or unchanged targets.

## Three actual treatments

- ALWAYS_A7: regenerate A7 candidates on every common SSGS base.
- CONDITIONAL: use TRUE only under the frozen actual-vector rule; otherwise A7.
- RANDOM: retain the positive-wait guard and assign exactly the same TRUE count
  uniformly within the source's full changed target pool. It may select assignment
  changes; it must not restrict random selection to the gate's OS cases.

Every treatment actually executes only its chosen route; there is no oracle
choice between already-evaluated TRUE/A7 outcomes. Identical A7-route named RNGs
pair candidate pools across groups. Randomize group execution order per target.
Matched routing counts are set before any candidate outcomes. Recompute and charge
source diagnostics for every guarded policy call, including A7 fallbacks.

TRUE uses the frozen first-D1 scales 0.25/0.5/1.0, 900-second individual wait cap
and batch extension cap. Reject failed projections and retain the SSGS fallback;
do not secretly run another A7. Both routes allow at most three additional exact
evaluations and use 1/3-second call envelopes for 50/100 jobs. Guard work, rejected
attempts and atomic late work count toward actual time; late proposals cannot win.
Same envelopes/upper bounds do not imply equal actual time.

All groups use the sampled source index's own subproblem weight and one common
context fixed before proposals: ideal is the terminal population/archive minimum,
extended by the target base only; maximum is the terminal population maximum.
This offline comparison convention is distinct from the online MacroSearch's
temporary ideal update. Record candidates below this frozen ideal and check a
common updated-ideal sensitivity; do not silently choose a favorable convention.

## Evidence and inference

Validate each imported P2 completion, frozen input and artifact hash. Import only
selected complete schedules and necessary instances, preserving exact provenance.
Re-evaluate sampled sources and every new proposal using the exact evaluator;
retain starts/objectives/feasibility/quota decisions for replay. Publish a worker
completion only after its hashes and paired route count pass validation. Use a
bounded four-worker queue, heartbeats and a 600-second per-worker hard timeout.
Recorded failures require review, with no automatic retry or raw-data deletion.

Primary quality is source-weighted relative scalar gain against the common base:

$$
I_a=(g_0-g_a)/\max(g_0,10^{-12}).
$$

Average targets within each source run, then six tariff/seed runs within a base,
then sixteen bases equally. The base is the independent unit. Primary comparisons
are CONDITIONAL−ALWAYS_A7 and CONDITIONAL−RANDOM. Use 50000 size-stratified base
bootstrap draws, all 65536 base sign flips, and Holm adjustment over these two
quality comparisons. The quality screen requires a positive mean and lower 95%
bootstrap bound and Holm p below .05 for both controls. This is not an algorithm
adoption gate; online HV/IGD+/coverage and new instances remain later work.

Report Flow/Bill separately, actual costs including shared validation/SSGS/exact
and target-allocation work, exact counts, success/empty-intent/transfer rates,
failure/late costs, and descriptive size/tariff/structure strata. A speed signal
alone is not a quality improvement. Do not select subgroups after confirmation.
All eleven statistical fallacy checks and independent exact/replay audit are
required before a scientific conclusion. Formal and pilot evidence remain separate.

## Portable commands

Use the frozen clean checkout and an existing Python environment with project dev
dependencies. P2_ROOT and RUN_ROOT are user-selected directories, not committed
machine paths. The four-source resource pilot uses old development bases, never
the sixteen holdout bases.

```powershell
python -m geo_llm_scheduler.experiments.d1_holdout prepare --root PILOT_ROOT --p2-root P2_ROOT --pilot
python -m geo_llm_scheduler.experiments.d1_holdout run --root PILOT_ROOT --workers 4
python -m geo_llm_scheduler.experiments.d1_holdout validate --root PILOT_ROOT --p2-root P2_ROOT
python -m geo_llm_scheduler.experiments.d1_holdout prepare --root RUN_ROOT --p2-root P2_ROOT
python -m geo_llm_scheduler.experiments.d1_holdout run --root RUN_ROOT --workers 4
python -m geo_llm_scheduler.experiments.d1_holdout validate --root RUN_ROOT --p2-root P2_ROOT
python -m geo_llm_scheduler.experiments.d1_holdout analyse --root RUN_ROOT
```

## Execution and independent audit (2026-10-04)

The frozen runtime 78659c3 completed 96/96 jobs, 384 sources and 2304 paired
targets. Every new proposal record was independently checked against exact;
48 targets were replayed and base-level inference independently recomputed.
The prespecified internal quality screen passed both controls. The source and
rule remained frozen throughout computation; the online default is unchanged.
See the [reviewed report](../reports/d1_terminal_holdout_20261004.md) for effects,
costs, the full statistical review and limitations. The protocol's original
UNVERIFIED passport describes its state at planning; the reviewed report is
ANALYZED after execution and audit.
