# B3L3 deletion-screen analysis

Run the scripts in order with the workspace directory as their sole argument: `analyze.py`, `supplement.py`, `history_check.py`, `validate_analysis.py`, `report.py`.

Inputs: SHA-verified `outputs/lsv2_ablation3_20261028/results`, frozen pre-run `a3_diagnostic`, and the previous budget12 full control for a descriptive repeat check. No solver runs are launched. The primary reference contains only the eight current arms. Seven full-control contrasts share Holm correction; the factorial and A3 conditions are exploratory. Full numerical tables, figures, and report are written under the campaign output directory. `validate_analysis.py` independently recomputes HV, IGD+, sign-flip tests and Holm adjustment. Formal solver commit remains recorded in the manifest; this analysis commit is separate.
