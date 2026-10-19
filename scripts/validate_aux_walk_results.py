"""Independent numerical checks and explicitly exploratory mechanism contrasts."""

from __future__ import annotations

import csv
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import analyze_aux_walk_results as primary
import numpy as np

from geo_llm_scheduler.domain.models import Genotype, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.utils.numeric import TOL, close


def _run(start: Path, capture: Path, out: Path) -> None:
    START, CAP, OUT = start, capture, out
    result = primary.read(OUT / "analysis.json")
    with (OUT / "parent_metrics.csv").open(encoding="utf-8") as f:
        parents = list(csv.DictReader(f))
    with (OUT / "run_metrics.csv").open(encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    lookup = {(r["stage"], r["parent"], r["arm"]): r for r in parents}
    rng = np.random.default_rng(2026101603)
    checked = []
    for item in result["statistics"]:
        stage, a, b, metric = [item[k] for k in ("stage", "treatment", "control", "metric")]
        ps = sorted({r["parent"] for r in parents if r["stage"] == stage})
        values = np.array(
            [
                (float(lookup[stage, p, a][metric]) - float(lookup[stage, p, b][metric]))
                * (1 if metric == "hv" else -1)
                for p in ps
            ]
        )
        sizes = np.array([int(lookup[stage, p, a]["jobs"]) for p in ps])
        samples = []
        for size in sorted(set(sizes)):
            group = values[sizes == size]
            idx = rng.integers(len(group), size=(20000, len(group)))
            samples.append(np.take(group, idx).sum(axis=-1))
        ci = np.quantile(np.sum(samples, axis=0) / len(values), [0.025, 0.975])
        extreme = 0
        for start in range(0, 100000, 1000):
            signs = rng.integers(0, 2, size=(min(1000, 100000 - start), len(values))) * 2 - 1
            random_means = np.einsum("ij,j->i", signs, values) / len(values)
            extreme += np.count_nonzero(np.abs(random_means) >= abs(np.mean(values)) - 1e-14)
        p = (1 + int(extreme)) / 100001
        assert np.allclose(ci, item["ci95"], rtol=0, atol=1e-14) and p == item["p"]
        checked.append([stage, a, b, metric])
    for stage in primary.STAGES:
        rows = [r for r in result["statistics"] if r["stage"] == stage]
        ordered = sorted(rows, key=lambda r: r["p"])
        adjusted = np.minimum(
            1, np.maximum.accumulate([r["p"] * (len(rows) - i) for i, r in enumerate(ordered)])
        )
        assert all(p == r["p_holm"] for p, r in zip(adjusted, ordered))
    primary.write(
        OUT / "independent_statistics_acceptance.json",
        {
            "statistics": len(checked),
            "families": [34, 6, 6],
            "status": "passed",
            "seed": 2026101603,
            "bootstrap": 20000,
            "signflips": 100000,
        },
    )
    anchors = [
        ("W0", "S0"),
        ("W1", "S1"),
        ("W1", "W0"),
        ("S1", "S0"),
        ("W1", "S0"),
        ("F1", "W1"),
        ("F2", "W1"),
        ("D1", "D2"),
        ("D3", "D2"),
        ("D4", "D2"),
        ("D5", "D2"),
        ("D6", "D2"),
        ("C0", "D2"),
        ("L0", "W1"),
        ("L1", "D2"),
        ("A1", "D2"),
        ("A1", "A0"),
    ]
    requests = [("screen", a, b) for a, b in anchors] + [
        (stage, "W1", "S0") for stage in ("confirm", "long")
    ]
    rng = np.random.default_rng(2026101901)
    secondary = []
    for stage, a, b in requests:
        ps = sorted({r["parent"] for r in parents if r["stage"] == stage})
        for metric, sign in [("hv", 1), ("igd", -1)]:
            values = [
                (float(lookup[stage, p, a][metric]) - float(lookup[stage, p, b][metric])) * sign
                for p in ps
            ]
            strata = [int(lookup[stage, p, a]["jobs"]) for p in ps]
            secondary.append(
                {
                    "stage": stage,
                    "treatment": a,
                    "control": b,
                    "metric": metric,
                    **primary.inference(values, strata, rng),
                }
            )
    for r, p in zip(secondary, primary.holm([r["p"] for r in secondary])):
        r["p_holm_exploratory_38"] = p
    primary.write(
        OUT / "exploratory_anchors.json",
        {
            "status": "postprocessing exploratory family; not an additional preregistered confirmation",
            "seed": 2026101901,
            "statistics": secondary,
        },
    )

    def one(r):
        s = primary.read(START / "raw" / ("formal_node" + r["node"]) / r["id"] / "summary.json")
        t = s["timings"]
        p = s["production_seconds"]
        counts = s["counts"]
        return {
            "id": r["id"],
            "stage": r["stage"],
            "arm": r["arm"],
            "jobs": int(r["jobs"]),
            "seconds": float(r["seconds"]),
            "production_seconds": p,
            "overshoot": s["overshoot"],
            "overshoot_pct": 100 * s["overshoot"] / float(r["seconds"]),
            "initialization_seconds": s["initialization_seconds"],
            "polish_seconds": t.get("polish", 0),
            "polish_pct": 100 * t.get("polish", 0) / p,
            "budget_seconds": t.get("budget", 0),
            "history_seconds": t.get("auxiliary_history", 0),
            "observer_seconds": s["observer_seconds"],
            "auxiliary_exact": counts.get("auxiliary_exact", 0),
            "polish_exact": counts.get("polish_exact", 0),
            "exact": counts["exact"],
            "rss_recorded": s["peak_rss_gib"],
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        costs = list(pool.map(one, runs))
    primary.table(OUT / "cost_details.csv", costs)
    costs_summary = []
    for stage in primary.STAGES:
        for arm in sorted({r["arm"] for r in runs if r["stage"] == stage}):
            rs = [r for r in costs if r["stage"] == stage and r["arm"] == arm]
            costs_summary.append(
                {
                    "stage": stage,
                    "arm": arm,
                    "runs": len(rs),
                    **{
                        k: float(np.mean([r[k] for r in rs]))
                        for k in (
                            "production_seconds",
                            "initialization_seconds",
                            "overshoot",
                            "overshoot_pct",
                            "polish_seconds",
                            "polish_pct",
                            "budget_seconds",
                            "history_seconds",
                            "observer_seconds",
                            "auxiliary_exact",
                            "polish_exact",
                            "exact",
                        )
                    },
                    "max_overshoot": max(r["overshoot"] for r in rs),
                    "p95_overshoot": float(np.quantile([r["overshoot"] for r in rs], 0.95)),
                }
            )
    primary.table(OUT / "cost_summary.csv", costs_summary)
    heterogeneity = []
    for item in result["statistics"]:
        stage, a, b, metric = [item[k] for k in ("stage", "treatment", "control", "metric")]
        if stage == "screen":
            continue
        ps = sorted({r["parent"] for r in parents if r["stage"] == stage})
        for factor in ("jobs", "topology", "pressure", "node"):
            for level in sorted({lookup[stage, p, a][factor] for p in ps}):
                selected = [p for p in ps if lookup[stage, p, a][factor] == level]
                vals = [
                    (float(lookup[stage, p, a][metric]) - float(lookup[stage, p, b][metric]))
                    * (1 if metric == "hv" else -1)
                    for p in selected
                ]
                heterogeneity.append(
                    {
                        "stage": stage,
                        "treatment": a,
                        "control": b,
                        "metric": metric,
                        "factor": factor,
                        "level": level,
                        "parents": len(vals),
                        "mean": float(np.mean(vals)),
                        "positive": sum(v > 0 for v in vals),
                    }
                )
    primary.table(OUT / "heterogeneity.csv", heterogeneity)
    relative = []
    for stage in primary.STAGES:
        for a, b in [("W1", "S0"), ("D2", "W1"), ("D2", "R0"), ("D2", "S0")]:
            ps = sorted({r["parent"] for r in parents if r["stage"] == stage})
            relative.append(
                {
                    "stage": stage,
                    "treatment": a,
                    "control": b,
                    "paired_relative_hv_pct": float(
                        np.mean(
                            [
                                100
                                * (
                                    float(lookup[stage, p, a]["hv"])
                                    / float(lookup[stage, p, b]["hv"])
                                    - 1
                                )
                                for p in ps
                            ]
                        )
                    ),
                    "paired_bill_end_improvement_pct": float(
                        np.mean(
                            [
                                100
                                * (
                                    1
                                    - float(lookup[stage, p, a]["bill"])
                                    / float(lookup[stage, p, b]["bill"])
                                )
                                for p in ps
                            ]
                        )
                    ),
                    "paired_flow_end_improvement_pct": float(
                        np.mean(
                            [
                                100
                                * (
                                    1
                                    - float(lookup[stage, p, a]["flow"])
                                    / float(lookup[stage, p, b]["flow"])
                                )
                                for p in ps
                            ]
                        )
                    ),
                }
            )
    primary.write(OUT / "descriptive_relative.json", relative)
    # Predetermined lexical first S0 run in every stage/size/host, first population
    # and first archive solution; no selection by objective value or discrepancies.
    checks = []
    maxdiff = 0
    for stage in primary.STAGES:
        for size in (20, 50, 100, 200):
            for node in (0, 1):
                r = sorted(
                    [
                        r
                        for r in runs
                        if r["stage"] == stage
                        and r["arm"] == "S0"
                        and int(r["jobs"]) == size
                        and int(r["node"]) == node
                    ],
                    key=lambda r: r["id"],
                )[0]
                root = START / "raw" / ("formal_node" + r["node"]) / r["id"]
                s = primary.read(root / "summary.json")
                problem = load_instance(CAP / "control/formal" / s["spec"]["input"])
                with gzip.open(root / "solutions.json.gz", "rt", encoding="utf-8") as f:
                    solutions = json.load(f)
                for label in ("population", "archive"):
                    sol = next(v for v in solutions if v["set"] == label)
                    g = Genotype(**{k: tuple(v) for k, v in sol["genotype"].items()})
                    sch = Schedule(**{k: tuple(v) for k, v in sol["schedule"].items()})
                    fresh = evaluate(problem, g, sch)
                    original = sol["evaluation"]
                    assert fresh.feasible and original["feasible"]
                    expected = (original["flow"], original["tou"] + sum(original["demand"]))
                    assert all(
                        close(a, b, TOL.time if k == 0 else TOL.cost)
                        for k, (a, b) in enumerate(zip(fresh.objectives, expected))
                    )
                    maxdiff = max(
                        maxdiff, *[abs(a - b) for a, b in zip(fresh.objectives, expected)]
                    )
                    checks.append({"id": r["id"], "set": label, "objectives": fresh.objectives})
    primary.write(
        OUT / "fresh_exact_spotcheck.json",
        {
            "status": "passed",
            "members": len(checks),
            "runs": 24,
            "max_absolute_objective_difference": maxdiff,
            "selection": "lexical first S0 per stage/size/host; first population and archive members; not all-solution re-evaluation",
            "checks": checks,
        },
    )
    print(
        json.dumps(
            {
                "independent_statistics": len(checked),
                "exploratory_anchors": len(secondary),
                "cost_runs": len(costs),
                "fresh_exact": len(checks),
            }
        ),
        flush=True,
    )


def main() -> None:
    """Recheck statistics, export exploratory anchors and cost, and spot-check fresh exact."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    _run(args.start, args.capture, args.analysis)


if __name__ == "__main__":
    main()
