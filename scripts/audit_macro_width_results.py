"""Independently replay cluster statistics and compare saved native/common objective prefixes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from geo_llm_scheduler.experiments.macro_width import read_json

CONTRASTS = ("POOL3-POOL6", "POOL10-POOL6", "POOL6-NATIVE6")
METRICS = (
    "hv_diff",
    "igd_plus_gain",
    "hv_pct",
    "igd_gain",
    "coverage_net",
    "flow_gain_pct",
    "bill_gain_pct",
    "exact_pct",
    "offspring_pct",
    "macro_call_pct",
    "unique_nd_diff",
)


def audit(raw: Path, analysis: Path) -> dict:
    """Recompute the complete random sequence without calling the production estimator."""
    if analysis.resolve().is_relative_to(raw.resolve()):
        raise ValueError("Audit output must stay outside immutable raw")
    summary = read_json(analysis / "analysis_summary.json")
    with (analysis / "base_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rng = np.random.default_rng(summary["seed"])
    raw_ps = {}
    checks = 0
    for contrast in CONTRASTS:
        for metric in METRICS:
            selected = sorted(
                [r for r in rows if r["contrast"] == contrast and r["metric"] == metric],
                key=lambda r: (int(r["jobs"]), int(r["base_seed"])),
            )
            values = np.array([float(r["value"]) for r in selected])
            if len(values) != 40:
                raise ValueError("Independent-base count changed")
            replicate_sums = np.zeros(summary["bootstrap"])
            for size in (50, 100):
                group = np.array([float(r["value"]) for r in selected if int(r["jobs"]) == size])
                sampled = rng.choice(group, (summary["bootstrap"], len(group)), replace=True)
                replicate_sums += sampled.sum(axis=1)
            ci = np.percentile(replicate_sums / 40, (2.5, 97.5))
            observed = abs(math.fsum(values) / 40)
            exceed = 0
            for begin in range(0, summary["signflips"], 4096):
                count = min(4096, summary["signflips"] - begin)
                signs = rng.integers(0, 2, (count, 40)) * 2 - 1
                null = np.einsum("ij,j->i", signs, values) / 40
                exceed += int(np.count_nonzero(np.abs(null) >= observed - 1e-14))
            p = (exceed + 1) / (summary["signflips"] + 1)
            expected = summary["statistics"][contrast][metric]
            if not np.allclose(ci, expected["ci95"], rtol=0, atol=1e-12):
                raise ValueError("Independent bootstrap mismatch")
            if not math.isclose(p, expected["p_signflip"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("Independent sign-flip mismatch")
            if not math.isclose(math.fsum(values) / 40, expected["mean"], abs_tol=1e-12):
                raise ValueError("Independent paired mean mismatch")
            raw_ps[(contrast, metric)] = p
            checks += 1
    primary = [(c, m) for c in CONTRASTS for m in ("hv_diff", "igd_plus_gain")]
    running = 0.0
    for rank, key in enumerate(sorted(primary, key=raw_ps.__getitem__)):
        running = max(running, (6 - rank) * raw_ps[key])
        if not math.isclose(
            min(1, running),
            summary["statistics"][key[0]][key[1]]["p_holm_primary_six"],
            abs_tol=1e-12,
        ):
            raise ValueError("Independent six-test Holm mismatch")
    with (analysis / "run_metrics.csv").open(encoding="utf-8", newline="") as stream:
        runs = list(csv.DictReader(stream))
    rows = []
    for run in runs:
        if run["arm"] != "NATIVE6":
            continue
        root = raw / ("formal_node" + run["node"])
        sample = read_json(root / "runs" / run["key"] / "diagnostics.json.gz")
        replay = read_json(root / "replay" / (run["key"] + ".json.gz"))
        for bucket, original in sample["calls"].items():
            full = replay["calls"][bucket]
            native = original["online_evaluated"]
            eligible = (
                all(c["feasible"] for c in native)
                and len(full["full_objectives"]) == full["prefix"]["10"]["evaluated"]
            )
            equal = (
                (
                    [tuple(c["objectives"]) for c in native]
                    == [tuple(c) for c in full["full_objectives"][:6]]
                )
                if eligible
                else None
            )
            equal_set = (
                (
                    {tuple(c["objectives"]) for c in native}
                    == {tuple(c) for c in full["full_objectives"][:6]}
                )
                if eligible
                else None
            )
            rows.append(
                {
                    "key": run["key"],
                    "jobs": run["jobs"],
                    "base_seed": run["base_seed"],
                    "tariff": run["tariff"],
                    "algorithm_seed": run["algorithm_seed"],
                    "action": original["action"],
                    "eligible": int(eligible),
                    "objective_prefix_equal": int(equal) if eligible else None,
                    "objective_prefix_set_equal": int(equal_set) if eligible else None,
                }
            )
    with (analysis / "native_common_prefix_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_action = {}
    for action in range(1, 9):
        rr = [r for r in rows if r["action"] == action and r["eligible"]]
        by_action[str(action)] = {
            "eligible_buckets": len(rr),
            "different_objective_prefixes": sum(r["objective_prefix_equal"] == 0 for r in rr),
            "difference_rate": float(np.mean([1 - r["objective_prefix_equal"] for r in rr])),
            "different_objective_sets": sum(r["objective_prefix_set_equal"] == 0 for r in rr),
            "set_difference_rate": float(
                np.mean([1 - r["objective_prefix_set_equal"] for r in rr])
            ),
        }
    result = {
        "state": "verified",
        "cluster_statistics_crosschecks": checks,
        "holm_joint_family_size": 6,
        "native_common_prefix_buckets": len(rows),
        "by_action": by_action,
        "interpretation": "Objective-prefix differences do not isolate genotype identity, RNG advancement or the action causing final online gain.",
    }
    (analysis / "independent_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """Audit an already completed immutable-data analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.raw, args.analysis)))


if __name__ == "__main__":
    main()
