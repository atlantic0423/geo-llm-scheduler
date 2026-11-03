"""Additional descriptive action/phase diagnostics, never used to retune this batch."""

import json
from collections import defaultdict

import analyze as a
import numpy as np


def main() -> None:
    """Aggregate per-run conditional behavior; retain denominators and source scope."""
    manifest = a.read(a.ROOT / "manifest.json")
    op_rows = []
    phase_rows = []
    final_rows = []
    for j in manifest["jobs"]:
        if not j["arm"].startswith("Q1"):
            continue
        key = j["runs"][0]["key"]
        p = a.ROOT / "results" / j["id"] / ("00_" + key)
        steps = a.read(p / "lsv2_steps.json.gz")
        s = a.read(p / "summary.json")
        cfg = a.read(p / "config.json")
        total = s["lsv2"]["ls_seconds"]
        n = cfg["spec"]["n"]
        arm = j["arm"]
        for op in sorted({r["operator"] for r in steps}):
            rr = [r for r in steps if r["operator"] == op]
            empty = [r for r in rr if not r["proposals"]]
            op_rows.append(
                dict(
                    key=key,
                    n=n,
                    arm=arm,
                    operator=op,
                    calls=len(rr),
                    call_share=len(rr) / len(steps),
                    empty_share=len(empty) / len(rr),
                    mean_seconds=float(np.mean([r["charged_step_seconds"] for r in rr])),
                    empty_seconds=float(np.mean([r["charged_step_seconds"] for r in empty]))
                    if empty
                    else 0.0,
                    reward_mean=float(np.mean([r["q_reward"] for r in rr])),
                    reward_median=float(np.median([r["q_reward"] for r in rr])),
                    empty_reward_median=float(np.median([r["q_reward"] for r in empty]))
                    if empty
                    else 0.0,
                    greedy_share=float(
                        np.mean(
                            [
                                r["bandit"]["probability"] > 0.2 / len(r["available"]) + 1e-12
                                for r in rr
                            ]
                        )
                    ),
                    positive_share=float(np.mean([r["hv_gain"] > 1e-12 for r in rr])),
                )
            )
        for phase in range(4):
            rr = [r for r in steps if min(3, int(r["elapsed_ls"] / total * 4)) == phase]
            assert rr
            phase_rows.append(
                dict(
                    key=key,
                    n=n,
                    arm=arm,
                    phase=phase,
                    steps=len(rr),
                    p3_share=sum(r["operator"] == "P3" for r in rr) / len(rr),
                    empty_share=sum(not r["proposals"] for r in rr) / len(rr),
                    positive_share=sum(r["hv_gain"] > 1e-12 for r in rr) / len(rr),
                    negative_reward_share=sum(r["q_reward"] < 0 for r in rr) / len(rr),
                )
            )
        q = a.read(p / "q_after.json")
        best = defaultdict(int)
        for state, values in q["values"].items():
            mask = [
                op for op in cfg["lsv2"]["operators"] if op != "A8" or state.split(":")[2] == "1"
            ]
            top = max(values.get(op, 0.0) for op in mask)
            for op in mask:
                if values.get(op, 0.0) == top:
                    best[op] += 1
        final_rows.append(
            dict(
                key=key,
                n=n,
                arm=arm,
                states=len(q["values"]),
                p3_best_states=best["P3"],
                a7_best_states=best["A7"],
            )
        )
    a.csvout("action_reward_diagnostics", op_rows)
    a.csvout("phase_diagnostics", phase_rows)
    a.csvout("final_greedy_states", final_rows)
    (a.OUT / "diagnostic_scope.json").write_text(
        json.dumps(
            dict(
                phase="cumulative LS time quarters, not wall clock quarters",
                causal=False,
                local_gain_warning="frozen frames change; summed local HV is not final global HV",
                empty="zero complete proposals, not necessarily zero internal construction work",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
