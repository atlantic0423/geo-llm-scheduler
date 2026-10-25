# Responses restart of the four-layer round-search comparison

This restart uses the user-selected `gpt-6.1-sol` through the explicitly authorized
`https://v1.cdks.work/v1` gateway and its Responses endpoint. The first gateway
group did not expose this model and returned `404 model_not_found`. After the
user changed the entry, a fresh model catalog included the exact model and a
minimal Responses request returned `200`, `completed`, and `gpt-6.1-sol`.
No alternative model is selected automatically. These are gateway observations,
not proof of a third party's internal model implementation.

## Frozen comparison and reuse

Keep the two original component33 KV physical inputs (100 and 200 Jobs), seeds
17401 and 27401, BILL4/P100 initialization, and 1200/2400 local production search
seconds. The eight BILL_MOEAD/MAIN_Q runs from frozen source `a90004a` are retained
with their full completion/artifact hashes and original validators. Only four
LLM_ROUND20 runs restart from initialization. A formal execution index binds the
eight old reference paths and four new LLM paths: twelve formal samples total.
Aborted DeepSeek attempts are retained and excluded, never relabeled or counted.

The unchanged search semantics are population reentry of exactly 20% at each
completed generation, immutable seed/fine-direction binding, fresh action,
conditional legal targets, post-execution boolean stop, B1/L5, ordinary 0.9 greedy
and 0.1 score sampling, A6 full-Job top-10% ranking and original random repair,
strict local scalar acceptance, independent feasible-exact Archive opportunities,
TWO_FOCUS Polish and neighbor-own-weight replacement. This new entry policy is
not the original strict-child gate. Default/source/history are not promoted.

## Transport and clocks

Use only the existing English BACKGROUND and four English task prompts in
`experiments/round_prompts.py`. The full background appears in a developer input
message as well as the instructions field, so an intermediary's own instructions
do not remove the scheduling contract from the explicit message payload. Dynamic
packets retain lossless complete tables and conditional options. Existing strict
IDs, complete scores, legal bindings and quota validation remain in force.

Requested reasoning effort is medium. Requests use `store=false`, JSON-object
text output and maximum 8192 output tokens. Provider-echoed parameters and actual
usage are retained, including if an intermediary does not echo/enforce a requested
parameter. An output usage above 8192 is rejected. Per-request transport is bounded
by a 120-second isolated child deadline, killed and reaped on expiry. The original
32768 requests and 1B billed-or-reserved input-token guards are retained; these are
limits, not expected consumption or a price estimate. Cache and reasoning usage
remain observable; no third-party price is inferred from official API prices.

Only disjoint synchronous requester intervals are excluded from search time.
Initialization, local preparation, serialization, parsing, validation, candidate
generation/repair, SSGS/exact, Archive, Polish, replacement and production logging
remain charged. Actual wall and intervals are recorded separately. No search
overlaps a blocked request. Technical probes and final independent audits/export
are separate from formal production budgets.

There is no automatic retry or model fallback. Authentication/balance/model/client
errors (HTTP400/401/402/403/404), a model-identity mismatch, refusal or output-limit
violation abort the opted-in run after recording the failure. Five consecutive
transport failures also abort. Request/token guard exhaustion aborts rather than
silently finishing a nominal LLM run as mostly random search. Isolated malformed
decisions still use the existing recorded legal fallback/stop rule; their coverage
must be reported. The old Flash path is unchanged when no profile is supplied.

## Acceptance and resource boundaries

API-free tests cover actual wire fields, English developer background, credential
and redirect boundaries, exact model identity, usage normalization, HTTP classes,
fatal and repeated failures, child timeout kill/reap, excluded-wait accounting,
fixed-work search/RNG equality, reference hash tampering, and explicit execution
index integrity. Run local lint/format/type/full regression/coverage, then exact-head
dual-Python CI including D03/replay/wheel/CLI. Original tiny campaign timeout
failures and bounded rechecks must be reported truthfully, never hidden.

On the target, verify source/environment, all four initializer fingerprints against
the original references and independent exact, and real four-task inference for
both sizes. Start four new workers on distinct physical cores with 2 GiB address
limits, a 16 GiB study cgroup, a conservative 220 GiB admission reserve and a
200 GiB runtime reserve. Keep all old results and other services unchanged.
Verify real processes, commands, fresh heartbeats, search/exact progress and actual
model responses before reporting startup. No recurring Codex task is requested.

Only two physical development bases are present. Four seed blocks are paired
observations, not four independent bases or evidence of general superiority.
