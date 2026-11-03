# D1 local overnight mechanism experiment

Status: experimental protocol, source `f6b7fba` plus the separately recorded D1
research commit. This campaign does not change the online MOEA/D default.

The user authorized local implementation, acceptance and immediate execution,
requesting at least ten hours of actual experiment activity during an idle window.
The full matrix contains eight development bases, four each at 50 and 100 jobs,
paired homogeneous/heterogeneous regional prices, and five seeds. These are
development instances already used in P2, not independent confirmation instances.

## Compute allocation

- 80 F6 source runs: 1800 seconds for 50 jobs, 3600 seconds for 100 jobs.
- Total source engine budgets: 60 worker-hours. At most six source workers.
- Therefore source computation alone takes at least ten wall-clock hours on a
  successful uninterrupted run; export, scheduling and probes add overhead.
- Two additional probe workers consume atomically published early/middle/late
  snapshots while sources continue. Expected window is roughly 10–12 hours,
  subject to the resource pilot. No idle sleep is used to fill the requested time.
- A positive memory reserve of 4 GiB and disk reserve of 20 GiB block dispatch.
  Source/probe processes run with one numerical-library thread each and lower
  Windows priority. The supervisor inhibits system sleep while it is alive.

## Source sampling and pairing

At elapsed fractions 0.1, 0.5 and 0.9, sample distinct phenotypes in each of the
three preference groups and one remainder, using a separate observation RNG.
Duplicated/empty strata remain missing rather than being fabricated. Store actual
elapsed time, source timings, genotype, exact objectives, population normalization
context, source weight index and sampling probabilities. Logging is bounded to
three checkpoints per run and counters; historical archives are not copied each
offspring. Observer time counts in the source engine budget. These observed F6
sources are not a new fair online performance baseline for an unobserved variant.

Each source produces six fixed perturbations: OS, same-region instance and joint
region changes, each at one job and about 10 percent of jobs. Actual MS/OS distance
is recorded. Unchanged structures and missing sources are explicit skips. There
are at most 960 sampled sources, 5760 paired quartets and 23040 treatment records.
Neither cases, time snapshots nor repeated seeds are independent base instances.

## Limited D1 representation

Recompute source residual waiting against its frozen profile. Only the positive
wait beyond the earliest feasible placement is transferred. This does not encode
negative packing displacements, tariff phase or a full peak coalition. A null
result rejects this bounded residual-wait prototype, not every possible D1 design.

Rebuild the changed genotype with the existing SSGS first. Then independently
project residual waits at scales 0.25, 0.5 and 1.0, serially using the new release,
precedence, KV and cumulative resource constraints. Each wait is capped at 900
seconds. Reject a recipe if its batch end exceeds the rebuilt batch end by over
900 seconds. Old source absolute start times are never copied. This experimental
projection cap does not modify A7 or the authoritative algorithm's horizon rules.

## Four treatments and fair cost accounting

ZERO uses the rebuilt schedule. TRUE projects true residual waits. SHUFFLED
permutes waits separately within P/D phases before the same projection. FRESH_A7
runs the existing compression-based A7 from the same rebuilt schedule.

Every treatment has the same wall-clock envelope (1 second at 50 jobs, 3 seconds
at 100 jobs) and at most three distinct additional exact evaluations. All have
the zero-intent schedule as fallback. Complete atomic construction/evaluation may
overshoot; record its full time but exclude late candidates from within-envelope
quality. Equal envelopes are upper limits, not a claim of equal actual runtime.
No arm chains candidates. Source verification, both SSGS rebuilds, baseline exact,
intent extraction, projection, failed/duplicate attempts, A7 raw construction and
additional exact evaluation are separately accounted. Actual quota overshoot and
effective candidate counts must be reviewed before a cost-efficiency conclusion.

All completed feasible candidates use a common temporary ideal and the source
population maximum for paired scalar diagnostics. Use the sampled source's own
weight. Report Flow/Bill and timing costs alongside scalar, never scalar alone.
Probes do not affect source populations, archives, ideal, controllers or search RNG.

## Acceptance and outcome boundary

Before freeze, require counterexamples for releases/KV/dual resources/horizon cap,
phase-preserving permutation, changed structures, late-candidate exclusion, fixed
generation observation replay, complete artifact hashes, source progress budgets
and interrupted-attempt preservation. Run the repository full quality gate and a
short concurrent pilot. Freeze source SHA, canonical source hash, inputs, initial
pools, configs and the whole matrix. Stop dispatch on the first worker error,
preserve its evidence, and do not retry silently. `stop.request` drains active jobs.

On completion verify every source and probe before aggregating paired effects by
independent base. Preserve stage, tariff, preference, perturbation size, actual
cost, failures and missingness in case records. The automatic report is a first
descriptive mechanism summary; independent-base statistical uncertainty and
cost sensitivity must be reviewed before considering online integration.

Success of launch/pilot is engineering evidence. Actual support requires TRUE to
beat SHUFFLED and fresh A7 across changed structures at an acceptable total cost.
Online HV, IGD+, diversity and coverage remain separate future experiments.

## Entry point

Use `python -m geo_llm_scheduler.experiments.d1_campaign prepare --root <campaign>`
to freeze a new campaign, or add `--pilot-seconds 45` for the four-source pilot.
Run with `python -m geo_llm_scheduler.experiments.d1_campaign run --root <campaign>`.
Never modify frozen source while workers run; use a separate worktree for analysis.
Worker/manifest paths are relative and checked to stay inside the campaign.
