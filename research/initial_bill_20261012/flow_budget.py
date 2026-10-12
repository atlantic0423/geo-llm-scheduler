"""Exploratory electricity comparison under identical absolute Flow caps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze import NEW, points, read, table
from analyze_d5_results import nondominated


def analyze_caps(raw: Path, baseline: Path, out: Path) -> None:
    """Use predeclared 5/10/25/50/100% caps relative to each RAW fastest solution."""
    rows = []
    for node in (0, 1):
        root = raw / f"node{node}" / f"formal_node{node}"
        oldroot = baseline / f"formal_node{node}"
        for key in read(root / "manifest.json")["blocks"]:
            if not (root / "runs" / key / "complete.json").exists():
                continue
            spec = read(root / "specs" / (key + ".json"))
            parent = points(read(oldroot / "runs" / key / "RAW" / "population.json"))
            fast = min(p[0] for p in parent)
            fronts = {
                arm: points(read(root / "runs" / key / arm / "population.json")) for arm in NEW
            }
            for factor in (1.05, 1.1, 1.25, 1.5, 2.0):
                cap = fast * factor
                raw_bill = min(p[1] for p in parent if p[0] <= cap + 1e-9)
                for arm, front in fronts.items():
                    eligible = [p for p in front if p[0] <= cap + 1e-9]
                    bill = min(p[1] for p in eligible) if eligible else None
                    rows.append(
                        {
                            k: spec[k]
                            for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")
                        }
                        | dict(
                            arm=arm,
                            flow_cap_factor=factor,
                            raw_min_flow=fast,
                            flow_cap=cap,
                            raw_bill_under_cap=raw_bill,
                            new_bill_under_cap=bill,
                            bill_saving_pct=100 * (1 - bill / raw_bill)
                            if bill is not None
                            else None,
                        )
                    )
    table(out / "data/flow_cap_metrics.csv", rows)
    summaries = []
    for factor in (1.05, 1.1, 1.25, 1.5, 2.0):
        for arm in NEW:
            these = [r for r in rows if r["arm"] == arm and r["flow_cap_factor"] == factor]
            assert all(r["bill_saving_pct"] is not None for r in these)
            groups = defaultdict(list)
            for row in these:
                groups[row["jobs"], row["base_seed"]].append(row)
            summaries.append(
                dict(
                    arm=arm,
                    flow_cap_factor=factor,
                    n_bases=len(groups),
                    mean_bill_saving_pct=float(
                        np.mean(
                            [np.mean([r["bill_saving_pct"] for r in g]) for g in groups.values()]
                        )
                    ),
                    minimum_paired_population_saving_pct=min(r["bill_saving_pct"] for r in these),
                    positive_populations=sum(r["bill_saving_pct"] > 1e-9 for r in these),
                    negative_populations=sum(r["bill_saving_pct"] < -1e-9 for r in these),
                )
            )
    table(out / "data/flow_cap_summary.csv", summaries)
    print(json.dumps(summaries), flush=True)


def analyze_weights(raw: Path, out: Path) -> None:
    """Describe BILL's four effective weights without treating slots as replicates."""
    with (out / "data/run_metrics.csv").open(encoding="utf-8") as handle:
        original = {r["key"]: r for r in csv.DictReader(handle) if r["arm"] == "RAW"}
    rows = []
    for node in (0, 1):
        root = raw / f"node{node}" / f"formal_node{node}"
        for key in read(root / "manifest.json")["blocks"]:
            path = root / "runs" / key / "BILL20/population.json"
            if not path.exists():
                continue
            spec = read(root / "specs" / (key + ".json"))
            population = points(read(path))
            front = set(nondominated(population))
            for slot, p in enumerate(population[80:]):
                rows.append(
                    {k: spec[k] for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")}
                    | dict(
                        slot=slot,
                        weight=(0, 0.1, 0.25, 0.5)[slot % 4],
                        flow_over_raw_best=p[0] / float(original[key]["min_flow"]),
                        bill_over_raw_lowest=p[1] / float(original[key]["min_bill"]),
                        is_nd=int(p in front),
                    )
                )
    table(out / "data/bill_weight_diagnostics.csv", rows)
    summaries = []
    for weight in (0, 0.1, 0.25, 0.5):
        groups = defaultdict(list)
        for row in rows:
            if row["weight"] == weight:
                groups[row["jobs"], row["base_seed"]].append(row)
        summaries.append(
            dict(
                weight=weight,
                **{
                    k: float(np.mean([np.mean([r[k] for r in g]) for g in groups.values()]))
                    for k in ("flow_over_raw_best", "bill_over_raw_lowest", "is_nd")
                },
            )
        )
    (out / "data/bill_weight_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze_caps(args.raw, args.baseline, args.out)
    analyze_weights(args.raw, args.out)
