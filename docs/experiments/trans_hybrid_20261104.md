# Three continuous-key comparison methods: shared initialization and Polish

Status: development implementation; quality gate and real-instance smoke results
must be recorded separately before claiming delivery. No campaign authorized.

The existing five enhanced baselines construct 100 mixed individuals: 20 original
heuristic candidates, one cost-guided candidate with a single trial, and 79
random legal genotypes. They apply the existing single-Prefill Polish to 20% of
the population sampled without replacement after a generation. These are not the
full S4 initializer or its FAST COMMON group right-shift.

The opt-in methods `CMOEA_CD_HYBRID_PD`, `CMOEA_AOP_HYBRID_PD`, and
`CMOFLCS_HYBRID_PD` invoke that same initializer and Polish. Their original
continuous-key entries stay available. Initialization and Polish are charged to
the same wall budget. Each dual-population method constructs two separately
seeded mixed populations and polishes 20% of each. The first population uses the
same explicit seed and must reproduce the existing baseline's initial population
and initial archive. The second uses a deterministic distinct seed. The
three-archive method polishes generated offspring before the next archive update;
the two dual-population methods polish survivors after selection.

An existing Polish candidate is always offered to the exact-evaluation archive.
Its improved schedule is fed back to a continuous-key population only if the
inverse keys replay the identical MS, OS and starts to project tolerances through
the normal decoder. Some feasible timing shifts are not representable by these
keys; these are counted and remain eligible for the measurement archive, but are
not silently substituted into a mismatched genotype. Re-encoding and replay
evaluation are charged; their costs and rejected round trips matter when
interpreting any future comparison. Disabling wait genes is not supported for
these enhanced variants. Small population overrides construct the complete
100-member mixture before returning the first requested members; they are for
engineering tests only.

The user's explicit choice leaves `PSLGP_H_PD` for separate planning: its
population contains rule trees rather than schedule individuals. A schedule
cannot be placed into the rule population without a separately specified mapping
and experimental semantics. No change is made to that method.

The prior 120-instance results remain historical; these changed variants have no
performance result until they are run under a frozen, paired protocol. Original
paper source attribution and the limits of our Python adaptations remain in
`trans_python.md`.
