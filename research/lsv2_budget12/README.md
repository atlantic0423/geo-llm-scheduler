# Twelve-configuration development screen: recovery analysis

This analysis uses the frozen 192-run campaign (16 development instances, seed
16901). The solver source is commit `670af6e`; these scripts do not modify or rerun
the solver. All 3,338 recovered files passed size/SHA checks. Server originals are
retained. Historical 64 runs supply descriptive baseline comparisons only.

Run from this checkout, supplying the project root that contains `outputs/`:

```powershell
python research/lsv2_budget12/analyze.py E:/LLM调度
python research/lsv2_budget12/diagnose.py E:/LLM调度
python research/lsv2_budget12/validate_analysis.py E:/LLM调度
python research/lsv2_budget12/plot.py E:/LLM调度
python research/lsv2_budget12/report.py E:/LLM调度
pytest -q tests/test_lsv2_budget12_analysis.py
```

Outputs are written to `outputs/lsv2_budget12_20261028/`: the Chinese `REPORT.md`,
per-run CSV tables, diagnostic JSON, 16 per-instance Pareto plots and five summary
plots. The raw input is `results/formal/`; the two historical campaign directories
are `outputs/lsv2_panel_20261026/results/formal/` and
`outputs/ext_hybrid_panel_20261026/results/formal/`.

The analysis checks 768 fresh exact candidates, replays 412,002 Q updates and
independently cross-checks 448 indicator pairs and 16 paired contrasts. Eight
primary contrasts use Holm correction as one family; eight secondary contrasts
form a separate family, with an additional all-16 sensitivity check. Bootstrap
intervals stratify by instance size. Each instance has only one seed; inference
does not estimate within-instance seed variance.

Uniform B6L5 has the highest mean HV (0.97245). Q gamma=.9 B3L3 improves over
Q gamma=.9 B6L5 in 12/16 instances (mean HV difference +0.01937), but no primary
contrast passes Holm 0.05. B1L1 creates more LS trajectories with lower positive-HV
rates. Removing A8 helps the Q arm on average but hurts the uniform arm. Increasing
P3 selection and frequent empty proposals suggest a reward/cost interaction;
the observational logs do not establish causality. No default algorithm change,
new campaign, PR, or Release is implied by this development analysis.

## B3L3 operator follow-up

Run operator_profile.py and then operator_report.py with the same workspace
argument. Per the user's narrowed scope, these read only Q gamma 0.9, Q gamma 0
and uniform B3L3 (48 runs). They write operator_profile_b3l3/ in the campaign,
verify input hashes, and cross-check action counts with previously audited tables.
Ratios average runs equally. Large-reward events are the upper 5% of each run's
positive net rewards, a descriptive rule rather than a new solver threshold.

P3 has high empty-proposal rates and rising selection shares. A4 is mostly
nonempty but low-yield; its B3L3 selection share does not rise. A7 gives frequent
immediate improvements; P2_FLOW/P2_MIXED give larger occasional net rewards.
A3 is costly and low-hit, A8 has gated rare opportunities, and A1/A2/A5 supply
basic neighborhood changes. No action removal is established by these logs.

The two targeted test modules are tests/test_lsv2_operator_profile.py and
tests/test_lsv2_budget12_analysis.py. Solver behavior and raw results are unchanged.
