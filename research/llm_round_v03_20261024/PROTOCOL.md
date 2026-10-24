# Population-guided Flash: four paired blocks, three algorithms

Status: implementation and engineering acceptance; no quality claim. This study
implements the user-approved v0.3.1 design, including the subsequently confirmed
20% population quota. The source commit, canonical package hash and live model
receipts are frozen before formal dispatch.

## Matrix

| Jobs | Original input | Seeds | Search budget per arm |
|---|---|---|---:|
| 100 | component33 `b202610160112_v0.json`, R2/M12, KV pressure | 17401, 27401 | 1200 s |
| 200 | component33 `b202610160113_v0.json`, R2/M12, KV pressure | 17401, 27401 | 2400 s |

Exactly three arms: `BILL_MOEAD`, `MAIN_Q`, `LLM_ROUND20`. There are four paired
input-seed blocks and 12 formal runs, using two original physical instances.
Seeds and runs are not additional independent base instances. The original input
bytes/specification hashes, common BILL4/P100 initialization, host and budgets
are bound in the manifest. No easier replacement instances are generated.

`BILL_MOEAD` and `MAIN_Q` retain the previous reference configurations and search
path. MAIN_Q uses its strict child trigger, Q42/B6/L5, tariff A3, two-focus Polish
and permuted replacement. The new arm uses the same outer variation, BILL4,
population size and neighbor-own-weight replacement, with the changes below.

## New-arm behavior

1. Complete one ordinary MOEA/D traversal without child-triggered LS.
2. Freeze the current P100 phenotype pool. Preserve distinct timing; deduplicate
   whole phenotypes while retaining their original fine-slot bindings.
3. Ask Flash to select and order exactly 20% of the pre-deduplication population,
   normally 20 seeds. Only the program reduces the quota if fewer distinct seeds
   exist. Do not fill the quota with duplicate schedules.
4. Refine each immutable selected seed under its selected, allowed fine weight.
   Ask for a fresh A1-A8 action each step and conditionally choose legal Jobs,
   phases, Regions, paths, OS positions or timing/support options.
5. Use B1 and at most five action attempts. Full exact evaluation, independent
   Archive opportunities and strict fine-weight acceptance remain authoritative.
6. After steps 1-4, ask for a boolean stop using completed execution feedback.
   A rejected candidate does not become the chain incumbent, even if archived.
7. Apply TWO_FOCUS Polish to each started trajectory if search budget remains,
   then perform ordinary permuted-neighbor replacement using neighbor weights.
8. Continue the next outer traversal. No round-0 LS or arbitrary Archive reentry.

Population reentry is a new research entry policy. It explicitly replaces the old
strict *child-entry* gate only in this arm; it is not described as preserving that
gate through a vacuous incumbent-versus-itself comparison. Numeric acceptance,
hard constraints, exact and Archive rules remain unchanged. This is an integrated
comparison, not a causal attribution to a single controller or operator.

Ordinary choices are 0.9 greedy maximum, 0.1 normalized-score sampling, with
explicit named RNG streams. Scores are priorities, not calibrated probabilities.
A6 ranks every Job once and jointly destroys the top approximately 10%, with
explicit RNG only for ties, followed by one original random legal repair. No
greedy reconstruction, extra candidate pool or beam continuation is added.

Eligibility masks remove certainly unavailable actions. A positive mask is a
necessary opportunity test, not a guarantee that full bounded A7/A8 construction
will find a move. All option construction and failed attempts are charged.

## English-only inference and cost boundaries

Model: `deepseek-flash`, non-thinking JSON responses, maximum 8192 output tokens.
Only the English contract in `PROMPTS_EN.md` and English-schema numeric state are
sent. Requests are self-contained; dynamic states use lossless tables and retain
all eligible Jobs/operations/options without rounding or heuristic shortlisting.

Initial transport guards are 30 seconds/request, at most 8192 requests and 64M
billed-or-reserved input tokens per LLM run. Unknown usage retains a conservative
65,536-token reservation. There is no cumulative API-wait fraction cutoff.
Request/usage/transport failures do not automatically retry. Action and object
failures use recorded legal random fallbacks; invalid stop conservatively ends
the current trajectory. Every request/task coverage and quota fallback is
reported. A compute-complete run is not claimed to have full model coverage.

## Search-clock contract

Search time is production wall time minus the program's disjoint synchronous
requester intervals. The run performs no search during a requester block. Both
successful and failed/timeout transport blocks are excluded; JSON response
parsing, validation, target enumeration, prompt preparation, initialization,
outer evolution, candidate construction, SSGS/exact, Archive, Polish and online
logging are charged. API intervals, actual wall and search seconds are stored.

Deadlines, remaining budgets, per-action efficiency and anytime samples use the
same search clock. API deadlines and an independent operational safety timeout
use actual wall. Final independent exact audits/export are separate and may not
continue search. Previous end-to-end-wall results are not relabeled as these runs.

## Engineering and execution acceptance

Meaningful API-free tests cover quota/deduplication/binding, stale and malformed
responses, complete score keys, all eight legal generators, exact cross-checks,
fresh action/stop sequencing, default/reference regression, English transport,
clock exclusion and timeout kill/reap. Full local lint/format/mypy/tests/coverage,
D03/replay/wheel/CLI and exact-head Python 3.11/3.13 CI are required.

On the target host, recheck identity, immutable source and environment, independently
audit all 140 common initializer candidates for each of four blocks, and execute
bounded technical checks before starting the 12-run formal matrix. Formal workers
are isolated from existing jobs, each on one allowed physical core, with 2 GiB
address-space limits, a 32 GiB study cgroup and a conservative admission guard
that preserves at least 200 GB available memory. Existing server data and the
separate local Jev clock trial are not stopped or changed.

The final worker independently exact-checks every saved population and Archive
member. Per-file size/SHA, input/spec/source/host/initial fingerprints, both clocks,
all four valid decision types, quota and termination are verified before a run
receives its complete marker. No automatic formal retry or duplicate counting.

## Evidence scope

The user requested execution of this small development comparison. With only two
physical instances, report paired descriptive endpoints, fronts, anytime and
controller coverage. Do not treat four seed blocks as four independent parents,
claim statistical superiority or promote this integrated arm to the default.
