# Concurrent right shift × LS share: approved eight-arm development screen

Status: experimental, not a default algorithm change. User approved implementation and launch on 2026-10-30. Preceding research: `outputs/group_rightshift_research_20261030/DESIGN_V2.md` in the local project.

## Frozen matrix

| Method | 20% shared LS | 30% shared LS |
| --- | --- | --- |
| No external right shift | NONE_20 | NONE_30 |
| Original single-target fast right shift | SINGLE_20 | SINGLE_30 |
| Single-target plus common-translation cover groups | COMMON_20 | COMMON_30 |
| Previous candidates plus coordinated insertion | COORD_20 | COORD_30 |

Eight original development inputs: n100 d06/d09/d12/d13 and n200 d01/d07/d09/d14, seed16901. Total 64 runs; 1200/2400 seconds including initialization. All use Q1/NET/gamma0.9/B3L3, fresh online Q table, original calibration and initializer. P3 is absent from every Q action pool. No pretraining. All external methods use POOL_RANDOM selection from the same current-population-plus-terminal physical pool. This randomizes the polish input, not the learning controller. No new failure memory or region-selection factor.

Time share means cumulative LS time divided by production search time after initialization. One complete admitted Q-plus-polish episode can overshoot the share; its cost delays the next admission. Total wall deadline and all selection/construction/exact/bookkeeping costs remain charged. 30% does not extend total time or mean 30% of the initial population.

## Neighborhoods and finite work

The single-target path is unchanged: partial targets, original greedy multi-move cap and up to three proposals. Both new methods first retain these proposals, then inspect every distinct natural active Prefill set of size at least two with positive right slack and no frozen Decode covering that elementary segment. The additional group layer has no arbitrary pair enumeration, size threshold or target-pool truncation. Singleton opportunities remain in the original partial pool; this experiment does not claim exhaustive single-target enumeration.

Group rank is whole-group removal Bill, not summed individual exclusive contributions. Common translation enumerates positive shifts aligning member starts/ends with frozen resource events, tariff boundaries, fixed 900-second window boundaries, and the common latest-start bound. Groups are always moved relative to one frozen input, not chained. Capacity checks precede incremental billing. An unchanged full active union is skipped safely. This event search is heuristic: it does not include every intersection of regional demand-window cost curves.

Coordinated insertion additionally removes the group and replaces its members one at a time: least slack first, reversed order, and one seeded shuffled order (the existing proposal budget is three). Orders are deduplicated. Each member considers its original start through its latest legal start using resource/tariff/window events. Pick least incremental full Bill; ties within cost tolerance prefer earlier starts. Intermediate insertions need not improve the parent. A failed reconstruction is rolled back. Completed groups compete with original single and common candidates, so a failed reconstruction cannot discard a known candidate. No beam search or new tuning thresholds.

All completed constructions are scored incrementally; at most three distinct best-Bill full proposals are emitted to the unified exact gateway. This cap is an explicit research construction/selection rule, not a claim of exhaustive search or pure engineering speedup. Every complete feasible exact evaluation retains its Archive opportunity. Diagnostics are refreshed for admitted timing candidates. Every candidate keeps MS/OS, all Decode, and unselected timing fixed; every changed Prefill start moves right and satisfies the frozen-Decode bound. Full Flow and horizon remain fixed. Polish occurs after terminal credit and generates no reward or Q update.

## Diagnostics and acceptance

Log all attempted inputs and old selection diagnostics, group count, slack/Decode-blocked event sets, resource-rejected positions, unchanged unions, Bill checks, nonimproving complete candidates, reconstruction failures, proposal sources, incremental deltas, exact/accepted counts and wall seconds. Counts of intermediate insertion positions are not independent schedules. Inputs with no proposals remain in the empty-call denominator. Evaluate online HV/IGD+ alongside accepted Bill gain per second, not empty rate alone.

Local tests cover zero-single/joint-positive cases, unequal displacements, resource pressure, frozen timing, incremental/exact agreement, deterministic RNG, single-path retention, all eight configs, terminal credit, sealed campaign tamper rejection and reference-path equality. Fixed historical snapshot benchmark measures extra search work honestly; a short Linux fixed-work check of both sizes precedes formal launch. Full source/input/config/initialization/exact preflight remains required.

## Literature boundary

Rocholl, Mönch and Fowler (2020), *Bi-criteria parallel batch machine scheduling to minimize total weighted tardiness and electricity cost*, DOI 10.1007/s11573-020-00970-6, provides explicit batch postponement. Khandekar et al. (2015), *Real-time scheduling to minimize machine busy times*, DOI 10.1007/s10951-014-0411-z, models concurrent starts and busy-time unions. Neither validates this dual-resource, frozen-Decode, TOU-plus-Demand implementation. The preceding 14 hotspot diagnostic inputs establish missed joint moves only; they are not an online effect estimate.
