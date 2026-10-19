"""Analyze complete paired parents from the stopped thirteen-arm comparison."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from analyze_d6_results import coverage, distance, holm, nondominated, normalization

from geo_llm_scheduler.experiments.comparison13 import ARMS, SIZES
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus

SOURCE = "bf719d68e882a9cc3058cd5fced3adcaa84757f8"
CANONICAL = "85c187874a18c9225c303fe3c45df25ac4668d41a759ab2e8a2add36e39e7540"
BASE = "M11_I4"
FAMILIES = {
    "external": tuple(("M11_I4-" + a, m) for a in ARMS[8:] for m in ("hv", "igd_plus")),
    "factorial": tuple((a, m) for a in ("A3", "RPERM", "interaction") for m in ("hv", "igd_plus")),
    "budget": tuple((a + "-M11_I4", m) for a in ("DS_I4", "DC_I4") for m in ("hv", "igd_plus")),
    "initialization": tuple(
        (a + "-M11_I4", m) for a in ("M11_I2", "M11_I1") for m in ("hv", "igd_plus")
    ),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    """Export deterministic tables with a single schema per file."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def equal_scale_mean(values: list[float], sizes: list[int]) -> float:
    """Give each of the four frozen task scales one quarter of total weight."""
    if len(values) != len(sizes) or not values or set(sizes) != set(SIZES):
        raise ValueError("Require paired values at all four scales")
    if not all(map(math.isfinite, values)):
        raise ValueError("Nonfinite effect")
    return float(np.mean([np.mean([v for v, s in zip(values, sizes) if s == n]) for n in SIZES]))


def cluster_statistics(
    values: list[float],
    sizes: list[int],
    rng: np.random.Generator,
    bootstrap: int = 20000,
    signflips: int = 100000,
) -> dict:
    """Use independent parent clusters, scale-equal bootstrap and paired sign flips."""
    observed = equal_scale_mean(values, sizes)
    data = np.asarray(values)
    groups = [data[np.asarray(sizes) == n] for n in SIZES]
    draws = sum(g[rng.integers(len(g), size=(bootstrap, len(g)))].mean(axis=1) / 4 for g in groups)
    weights = np.asarray([1 / (4 * sizes.count(n)) for n in sizes])
    extreme = 0
    for start in range(0, signflips, 4096):
        signs = 2 * rng.integers(0, 2, size=(min(4096, signflips - start), len(data))) - 1
        extreme += int(
            (np.abs((signs * data * weights).sum(axis=1)) >= abs(observed) - 1e-14).sum()
        )
    return dict(
        mean=observed,
        ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
        p_signflip=(extreme + 1) / (signflips + 1),
        positive=int((data > 1e-12).sum()),
        negative=int((data < -1e-12).sum()),
        n_parents=len(data),
        per_scale={str(n): float(g.mean()) for n, g in zip(SIZES, groups)},
        exploratory=True,
    )


def contrast_effects(scores: dict[str, float]) -> dict[str, float]:
    """Return predeclared effects; input scores must already orient higher as better."""
    effects = {"M11_I4-" + a: scores[BASE] - scores[a] for a in ARMS[8:]}
    effects.update({a + "-M11_I4": scores[a] - scores[BASE] for a in ARMS[1:3] + ARMS[6:8]})
    effects.update(
        A3=(scores["M10_I4"] + scores[BASE] - scores["M00_I4"] - scores["M01_I4"]) / 2,
        RPERM=(scores["M01_I4"] + scores[BASE] - scores["M00_I4"] - scores["M10_I4"]) / 2,
        interaction=scores[BASE] - scores["M10_I4"] - scores["M01_I4"] + scores["M00_I4"],
    )
    return effects


def points(path: Path) -> list[tuple[float, float]]:
    """Read only exact-feasible objective pairs from full candidate artifacts."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        candidates = json.load(stream)
    if not candidates or not all(c["evaluation"]["feasible"] for c in candidates):
        raise ValueError("Empty or infeasible archive")
    return nondominated(
        [
            (c["evaluation"]["flow"], c["evaluation"]["tou"] + sum(c["evaluation"]["demand"]))
            for c in candidates
        ]
    )


def independent_hv(front: list) -> float:
    """Integrate vertical slabs independently of the project rectangle sweep."""
    xs = sorted({p[0] for p in front if p[0] < 1.1} | {1.1})
    return math.fsum(
        (b - a) * max(0, 1.1 - min((y for x, y in front if x <= a), default=1.1))
        for a, b in zip(xs, xs[1:])
    )


def hv_score(front: list) -> float:
    """Measure intersection with the HV box; outside points contribute zero area.

    Coordinates remain unchanged for all distance and coverage metrics. Filtering
    zero-area rectangles is necessary because the package HV API rejects them.
    """
    return hypervolume([tuple(p) for p in front if p[0] <= 1.1 and p[1] <= 1.1], (1.1, 1.1))


def analyze(results: Path, output: Path) -> dict:
    """Validate the selected complete-parent set and export exploratory comparisons."""
    output.mkdir(parents=True, exist_ok=True)
    acceptance = json.loads((results / "execution_acceptance.json").read_text())
    assert acceptance["commit"] == SOURCE and acceptance["source"] == CANONICAL
    assert acceptance["validated_runs"] == 2360 and len(acceptance["complete_parents"]) == 38
    assert json.loads((results / "LOCAL_RAW_ACCEPTANCE.json").read_text())["accepted"]
    raw = results / "raw" / "formal"
    parents = set(acceptance["complete_parents"])
    selected = [r for r in acceptance["rows"] if r["base_seed"] in parents]
    assert len(selected) == 38 * 4 * 13
    groups: dict[tuple, list] = defaultdict(list)
    for row in selected:
        groups[(row["base_seed"], row["variant"])].append(row)
    rows, snapshots, references, fronts = [], [], {}, {}
    metric_checks = dict(hv=0, igd_plus=0)
    runtime_fields = (
        "offspring",
        "production_seconds",
        "requested_seconds",
        "budget_overshoot_seconds",
        "audit_seconds",
        "archive",
        "max_rss_bytes",
    )
    for case, records in sorted(groups.items()):
        assert {(r["arm"], r["seed"]) for r in records} == {
            (a, s) for a in ARMS for s in (16101, 16201)
        }
        loaded = {}
        for record in records:
            folder = raw / "runs" / record["key"] / record["arm"]
            loaded[(record["arm"], record["seed"])] = points(folder / "archive.json.gz")
        ref = nondominated([p for ps in loaded.values() for p in ps])
        lo, span = normalization(ref)
        normalized_ref = ((np.asarray(ref) - lo) / span).tolist()
        references[str(case)] = dict(ideal=lo.tolist(), span=span.tolist(), points=ref)
        for record in records:
            arm, seed = record["arm"], record["seed"]
            folder = raw / "runs" / record["key"] / arm
            original = loaded[(arm, seed)]
            norm = [tuple(p) for p in ((np.asarray(original) - lo) / span).tolist()]
            fronts[(case[0], case[1], seed, arm)] = norm
            hv = hv_score(norm)
            assert math.isclose(hv, independent_hv(norm), rel_tol=1e-11, abs_tol=1e-11)
            metric_checks["hv"] += 1
            igdp = distance(norm, normalized_ref, True)
            if metric_checks["igd_plus"] < 52:
                assert math.isclose(
                    igdp, igd_plus(norm, normalized_ref), rel_tol=1e-11, abs_tol=1e-11
                )
                metric_checks["igd_plus"] += 1
            summary = record["summary"]
            anytime = json.loads((folder / "anytime.json").read_text())
            init = [r for r in anytime if r["target"] == "initialization"]
            assert len(init) == 1
            init_norm = [
                tuple(p) for p in ((np.asarray(init[0]["objectives"]) - lo) / span).tolist()
            ]
            init_hv = hv_score(init_norm)
            assert hv >= init_hv - 1e-8
            row = {
                k: record[k]
                for k in (
                    "key",
                    "arm",
                    "n",
                    "base_seed",
                    "variant",
                    "seed",
                    "regions",
                    "machines",
                    "pressure",
                )
            }
            row.update(
                hv=hv,
                igd_plus=igdp,
                igd=distance(norm, normalized_ref, False),
                unique_nd=len(original),
                flow=min(p[0] for p in original),
                bill=min(p[1] for p in original),
                initial_hv=init_hv,
                hv_gain_after_init=hv - init_hv,
                init_seconds=summary["initialization"]["initialization_seconds"],
                bill_seed_seconds=summary["initialization"]["bank_stats"]["production_seconds"],
                exact=summary["counts"]["exact"],
                export_seconds=summary["completed"]
                - summary["production_end"]
                - summary["audit_seconds"],
                **{f: summary[f] for f in runtime_fields},
            )
            rows.append(row)
            for snap in anytime:
                norm_snap = [
                    tuple(p) for p in ((np.asarray(snap["objectives"]) - lo) / span).tolist()
                ]
                snapshots.append(
                    dict(
                        key=record["key"],
                        arm=arm,
                        n=record["n"],
                        target=snap["target"],
                        elapsed=snap["elapsed"],
                        hv=hv_score(norm_snap),
                    )
                )
    clusters: dict[tuple, list] = defaultdict(list)
    for row in rows:
        clusters[(row["base_seed"], row["arm"])].append(row)
    parent_rows = []
    numeric = [
        k
        for k, v in rows[0].items()
        if isinstance(v, (float, int))
        and k not in {"n", "base_seed", "variant", "seed", "regions", "machines"}
    ]
    for (parent, arm), items in sorted(clusters.items()):
        assert len(items) == 4 and {(r["variant"], r["seed"]) for r in items} == {
            (v, s) for v in (0, 1) for s in (16101, 16201)
        }
        parent_rows.append(
            dict(
                base_seed=parent,
                arm=arm,
                n=items[0]["n"],
                pressure=items[0]["pressure"],
                regions=items[0]["regions"],
                machines=items[0]["machines"],
                **{k: float(np.mean([r[k] for r in items])) for k in numeric},
            )
        )
    parent_map = {(r["base_seed"], r["arm"]): r for r in parent_rows}
    differences = []
    for parent in sorted(parents):
        for metric in ("hv", "igd_plus"):
            scores = {
                a: parent_map[(parent, a)][metric] * (1 if metric == "hv" else -1) for a in ARMS
            }
            for name, effect in contrast_effects(scores).items():
                differences.append(
                    dict(
                        base_seed=parent,
                        n=parent_map[(parent, BASE)]["n"],
                        contrast=name,
                        metric=metric,
                        effect=effect,
                    )
                )
    statistics = {}
    for family, tests in FAMILIES.items():
        entries = []
        for name, metric in tests:
            subset = [r for r in differences if r["contrast"] == name and r["metric"] == metric]
            salt = int.from_bytes(hashlib.sha256((name + metric).encode()).digest()[:4], "big")
            stat = cluster_statistics(
                [r["effect"] for r in subset],
                [r["n"] for r in subset],
                np.random.default_rng(np.random.SeedSequence([2026101613, salt])),
            )
            entries.append(dict(contrast=name, metric=metric, **stat))
        for entry, adjusted in zip(entries, holm([s["p_signflip"] for s in entries])):
            entry["p_holm_exploratory"] = adjusted
        statistics[family] = entries
    summaries = []
    for arm in ARMS:
        items = [r for r in parent_rows if r["arm"] == arm]
        summaries.append(
            dict(
                arm=arm,
                **{
                    k: equal_scale_mean([r[k] for r in items], [r["n"] for r in items])
                    for k in numeric
                },
                per_scale={
                    str(n): {
                        k: float(np.mean([r[k] for r in items if r["n"] == n])) for k in numeric
                    }
                    for n in SIZES
                },
            )
        )
    paired = []
    for parent in sorted(parents):
        for arm in ARMS:
            if arm == BASE:
                continue
            case_rows = [r for r in rows if r["base_seed"] == parent and r["arm"] == arm]
            nets = []
            for row in case_rows:
                key = (parent, row["variant"], row["seed"])
                a, b = fronts[(*key, arm)], fronts[(*key, BASE)]
                nets.append(coverage(a, b) - coverage(b, a))
            a, b = parent_map[(parent, arm)], parent_map[(parent, BASE)]
            paired.append(
                dict(
                    base_seed=parent,
                    n=a["n"],
                    arm=arm,
                    hv_gain=a["hv"] - b["hv"],
                    relative_hv_pct=100 * (a["hv"] / b["hv"] - 1),
                    igd_plus_gain=b["igd_plus"] - a["igd_plus"],
                    coverage_net=float(np.mean(nets)),
                    flow_gain_pct=100 * (1 - a["flow"] / b["flow"]),
                    bill_gain_pct=100 * (1 - a["bill"] / b["bill"]),
                )
            )
    metadata = [parent_map[(p, BASE)] for p in sorted(parents)]
    execution = dict(
        exit=acceptance["exit"],
        validated_runs=2360,
        primary_planned_runs=8320,
        complete_blocks=acceptance["complete_blocks"],
        complete_parents=len(parents),
        selected_runs=len(rows),
        per_scale=dict(Counter(r["n"] for r in metadata)),
        per_pressure=dict(Counter(r["pressure"] for r in metadata)),
        per_topology=dict(Counter(f"R{r['regions']}M{r['machines']}" for r in metadata)),
        failure=acceptance["failures"],
        no_confirmatory_claim=True,
    )
    results_summary = dict(
        source=SOURCE,
        canonical=CANONICAL,
        execution=execution,
        arms=summaries,
        statistics=statistics,
        independent_metric_checks=metric_checks,
        references=references,
        all_run_max_overshoot=max(
            r["summary"]["budget_overshoot_seconds"] for r in acceptance["rows"]
        ),
        all_run_max_rss=max(r["summary"]["max_rss_bytes"] for r in acceptance["rows"]),
        all_run_total_production=sum(
            r["summary"]["production_seconds"] for r in acceptance["rows"]
        ),
    )
    for name, data in (
        ("run_metrics.csv", rows),
        ("parent_metrics.csv", parent_rows),
        ("parent_contrasts.csv", differences),
        ("paired_diagnostics.csv", paired),
        ("anytime_metrics.csv", snapshots),
    ):
        write_csv(output / name, data)
    (output / "analysis.json").write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.results, args.output)
    print(
        json.dumps(
            {"execution": result["execution"], "metric_checks": result["independent_metric_checks"]}
        )
    )
