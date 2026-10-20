"""Reproduce the prespecified complete 33-arm development-panel analysis."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_d6_results import distance, holm, nondominated, normalization

from geo_llm_scheduler.experiments.component_random import ARMS, EXTERNAL_ARMS
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus

SOURCE = "b148c4addd207fe30a6b2a43d953a04afc35d3a1"
CANONICAL = "37f3153178bb7d3238750ec1ac85ebe7871c18a317eb5f2dc8025a25497dfaba"
BASE = "FULL_CURRENT_I4"
SIZES = (20, 50, 100, 200)
SEED = 2026101908
FAMILIES = {
    "main": tuple((BASE, a) for a in EXTERNAL_ARMS),
    "component": (
        ("MIXED_MOEAD", "EXT_MOEAD"),
        ("BILL_MOEAD", "MIXED_MOEAD"),
        *((f"ONLY_A{i}", "MIXED_MOEAD") for i in range(1, 9)),
        ("POLISH_ONLY", "MIXED_MOEAD"),
        ("RPERM_ONLY", "MIXED_MOEAD"),
    ),
    "hierarchy": (
        ("POOL_RANDOM_B1_L1", "MIXED_MOEAD"),
        ("POOL_RANDOM_B6_L1", "POOL_RANDOM_B1_L1"),
        ("POOL_Q_B6_L1", "POOL_RANDOM_B6_L1"),
        ("POOL_Q_B6_L5", "POOL_Q_B6_L1"),
        *(
            (a, "POOL_Q_B6_L5")
            for a in ("POOL_Q_STRICT_B6_L5", "POOL_Q_SEVERITY_L5", "POOL_Q_COVERAGE_L5")
        ),
        *(
            ("FULL_STRICT_NO_POLISH_I4", a)
            for a in (BASE, "EXT_MOEAD", "MIXED_MOEAD", "BILL_MOEAD")
        ),
    ),
    "full": tuple((a, BASE) for a in ARMS[26:]),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write deterministic scalar tables with a shared row schema."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def equal_scale_mean(values: list[float], sizes: list[int]) -> float:
    """Give all four task sizes equal weight, after parent-level reduction."""
    if len(values) != len(sizes) or set(sizes) != set(SIZES):
        raise ValueError("Require paired effects at all four scales")
    if not all(map(math.isfinite, values)):
        raise ValueError("Nonfinite effects")
    return float(np.mean([np.mean([v for v, s in zip(values, sizes) if s == n]) for n in SIZES]))


def cluster_statistics(
    values: list[float],
    sizes: list[int],
    rng: np.random.Generator,
    bootstrap: int = 20000,
    signflips: int = 100000,
) -> dict:
    """Resample parents within size and test the paired symmetric sign null."""
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
    )


def points(path: Path) -> list[tuple[float, float]]:
    """Load objective pairs and reject empty or infeasible saved candidates."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        candidates = json.load(stream)
    if not candidates or not all(c["evaluation"]["feasible"] for c in candidates):
        raise ValueError("Empty or infeasible candidates")
    return nondominated(
        [
            (c["evaluation"]["flow"], c["evaluation"]["tou"] + sum(c["evaluation"]["demand"]))
            for c in candidates
        ]
    )


def hv_score(front: list) -> float:
    """Integrate only nonnegative rectangles inside the unchanged HV reference box."""
    return hypervolume([tuple(p) for p in front if max(p) <= 1.1], (1.1, 1.1))


def independent_hv(front: list) -> float:
    """Independently integrate vertical strips, including zero-contribution points."""
    xs = sorted({p[0] for p in front if p[0] < 1.1} | {1.1})
    return math.fsum(
        (b - a) * max(0, 1.1 - min((y for x, y in front if x <= a), default=1.1))
        for a, b in zip(xs, xs[1:])
    )


def weak_coverage(a: list, b: list) -> float:
    """Measure weak domination of B by A with normalized 1e-12 tolerance."""
    if not a or not b:
        raise ValueError("Empty coverage set")
    aa, bb = np.asarray(a), np.asarray(b)
    count = 0
    for start in range(0, len(bb), 128):
        count += ((aa[None, :, :] <= bb[start : start + 128, None, :] + 1e-12).all(2).any(1)).sum()
    return float(count / len(bb))


def analyze(results: Path, output: Path) -> dict:
    """Analyze all frozen runs and emit the complete 70-test family set."""
    output.mkdir(parents=True, exist_ok=True)
    acceptance = json.loads((results / "execution_acceptance.json").read_text())
    assert acceptance["commit"] == SOURCE and acceptance["source"] == CANONICAL
    assert acceptance["validated_runs"] == 1056 and len(acceptance["complete_parents"]) == 16
    assert json.loads((results / "LOCAL_RAW_ACCEPTANCE.json").read_text())["accepted"]
    raw = results / "raw" / "formal"
    groups: dict[str, list] = defaultdict(list)
    for row in acceptance["rows"]:
        groups[row["key"]].append(row)
    assert len(groups) == 32
    rows, snapshots, actions, paired = [], [], [], []
    references, saved_fronts = {}, {}
    checks = dict(hv=0, igd_plus=0, feasible_artifacts=0)
    for number, (key, records) in enumerate(sorted(groups.items())):
        assert {r["arm"] for r in records} == set(ARMS) and len(records) == 33
        loaded = {
            r["arm"]: points(raw / "runs" / key / r["arm"] / "archive.json.gz") for r in records
        }
        ref = nondominated([p for ps in loaded.values() for p in ps])
        lo, span = normalization(ref)
        norm_ref = ((np.asarray(ref) - lo) / span).tolist()
        references[key] = dict(ideal=lo.tolist(), span=span.tolist(), points=ref)
        norm_fronts = {a: ((np.asarray(ps) - lo) / span).tolist() for a, ps in loaded.items()}
        saved_fronts[key] = norm_fronts
        for record in records:
            arm = record["arm"]
            folder = raw / "runs" / key / arm
            original, norm = loaded[arm], norm_fronts[arm]
            hv = hv_score(norm)
            assert math.isclose(hv, independent_hv(norm), rel_tol=1e-11, abs_tol=1e-11)
            checks["hv"] += 1
            igdp = distance(norm, norm_ref, True)
            assert math.isclose(igdp, igd_plus(norm, norm_ref), rel_tol=1e-11, abs_tol=1e-11)
            checks["igd_plus"] += 1
            summary = record["summary"]
            diag = json.loads((folder / "diagnostics.json").read_text())
            counts, seconds = diag["counts"], diag["seconds"]
            assert counts == summary["counts"]
            anytime = json.loads((folder / "anytime.json").read_text())
            init = [r for r in anytime if r["target"] == "initialization"]
            assert len(init) == 1
            norm_init = ((np.asarray(init[0]["objectives"]) - lo) / span).tolist()
            init_hv = hv_score(norm_init)
            assert hv >= init_hv - 1e-8
            selected_init = points(folder / "initial.json.gz")
            selected_norm = ((np.asarray(selected_init) - lo) / span).tolist()
            final_population = points(folder / "population.json.gz")
            checks["feasible_artifacts"] += 3
            production = summary["production_seconds"]
            search_structure = summary["structural_diagnostics"]["counts"]
            structural = search_structure.get("search:structural_evaluations", 0)
            novel = search_structure.get("search:novel_structure_in_run", 0)
            q = json.loads((folder / "qtable.json").read_text())
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
                igd=distance(norm, norm_ref, False),
                unique_nd=len(original),
                flow=min(p[0] for p in original),
                bill=min(p[1] for p in original),
                initial_hv=init_hv,
                initial_p100_hv=hv_score(selected_norm),
                population_hv=hv_score(((np.asarray(final_population) - lo) / span).tolist()),
                hv_gain_after_init=hv - init_hv,
                init_seconds=summary["initialization"]["initialization_seconds"],
                init_share=summary["initialization"]["initialization_seconds"] / production,
                offspring=summary["offspring"],
                generations_proxy=summary["offspring"] / 100,
                exact=counts["exact"],
                exact_per_second=counts["exact"] / production,
                production_seconds=production,
                requested_seconds=summary["requested_seconds"],
                overshoot_seconds=max(0, production - summary["requested_seconds"]),
                audit_seconds=summary["audit_seconds"],
                nonproduction_seconds=summary["completed"] - summary["started"] - production,
                max_rss_bytes=summary["max_rss_bytes"],
                archive=summary["archive"],
                trigger_hits=counts.get("trigger_hits", 0),
                trigger_rate=counts.get("trigger_hits", 0)
                / max(1, counts.get("trigger_attempts", 0)),
                trigger_success_rate=counts.get("trigger_successes", 0)
                / max(1, counts.get("trigger_hits", 0)),
                polish_seconds=seconds.get("polish", 0),
                polish_share=seconds.get("polish", 0) / production,
                polish_calls=counts.get("polish_calls", 0),
                polish_accepted=counts.get("polish_accepted", 0),
                polish_exact=counts.get("polish_exact", 0),
                polish_acceptance=counts.get("polish_accepted", 0)
                / max(1, counts.get("polish_candidates", 0)),
                construction_seconds=sum(
                    v for k, v in seconds.items() if k.startswith("construction:A")
                ),
                exact_seconds=seconds.get("exact", 0),
                ssgs_seconds=seconds.get("ssgs", 0),
                archive_seconds=seconds.get("archive", 0),
                budget_seconds=seconds.get("budget", 0),
                search_structural_evaluations=structural,
                novel_structures=novel,
                repeat_structure_fraction=search_structure.get(
                    "search:repeated_structure_in_run", 0
                )
                / max(1, structural),
                timing_evaluations=search_structure.get("search:timing_evaluations", 0),
                population_unique_genotypes=summary["structural_diagnostics"][
                    "population_unique_genotypes"
                ],
                archive_unique_genotypes=summary["structural_diagnostics"][
                    "archive_unique_genotypes"
                ],
                q_selections=int(np.asarray(q["selections"]).sum()),
                q_updates=int(np.asarray(q["updates"]).sum()),
                q_visited_states=int(np.count_nonzero(q["visits"])),
            )
            rows.append(row)
            for action in range(1, 9):
                name = f"A{action}"
                actions.append(
                    dict(
                        key=key,
                        arm=arm,
                        n=record["n"],
                        base_seed=record["base_seed"],
                        action=name,
                        selections=int(np.asarray(q["selections"])[:, action - 1].sum()),
                        construction_seconds=seconds.get("construction:" + name, 0),
                        exact=counts.get("exact:" + name, 0),
                        ssgs_seconds=seconds.get("ssgs:" + name, 0),
                        exact_seconds=seconds.get("exact:" + name, 0),
                        archive_seconds=seconds.get("archive:" + name, 0),
                        insertions=counts.get("archive_insertions:" + name, 0),
                        attempts=counts.get("construction_attempts:" + name, 0),
                        proposals=counts.get("proposals:" + name, 0),
                    )
                )
            for snap in anytime:
                ns = ((np.asarray(snap["objectives"]) - lo) / span).tolist()
                snapshots.append(
                    dict(
                        key=key,
                        arm=arm,
                        n=record["n"],
                        base_seed=record["base_seed"],
                        variant=record["variant"],
                        target=snap["target"],
                        elapsed=snap["elapsed"],
                        fraction=snap["elapsed"] / summary["requested_seconds"],
                        hv=hv_score(ns),
                    )
                )
            other, base = norm_fronts[arm], norm_fronts[BASE]
            paired.append(
                dict(
                    key=key,
                    arm=arm,
                    base_seed=record["base_seed"],
                    n=record["n"],
                    coverage_to_full=weak_coverage(other, base),
                    coverage_from_full=weak_coverage(base, other),
                    coverage_net=weak_coverage(other, base) - weak_coverage(base, other),
                    relative_hv_pct=100 * (hv / hv_score(base) - 1) if hv_score(base) else 0,
                )
            )
        print(json.dumps(dict(input_complete=number + 1, inputs=32, key=key)), flush=True)
    clusters: dict[tuple, list] = defaultdict(list)
    for row in rows:
        clusters[(row["base_seed"], row["arm"])].append(row)
    numeric = [
        k
        for k, v in rows[0].items()
        if isinstance(v, (float, int))
        and k not in {"n", "base_seed", "variant", "seed", "regions", "machines"}
    ]
    parent_rows = []
    for (parent, arm), items in sorted(clusters.items()):
        assert len(items) == 2 and {r["variant"] for r in items} == {0, 1}
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
    parents = sorted(acceptance["complete_parents"])
    differences, statistics = [], {}
    for family, contrasts in FAMILIES.items():
        entries = []
        for left, right in contrasts:
            name = left + "-" + right
            for metric in ("hv", "igd_plus"):
                subset = []
                for parent in parents:
                    effect = (
                        parent_map[(parent, left)][metric] - parent_map[(parent, right)][metric]
                    ) * (1 if metric == "hv" else -1)
                    item = dict(
                        family=family,
                        base_seed=parent,
                        n=parent_map[(parent, left)]["n"],
                        contrast=name,
                        metric=metric,
                        effect=effect,
                    )
                    differences.append(item)
                    subset.append(item)
                salt = int.from_bytes(hashlib.sha256((name + metric).encode()).digest()[:4], "big")
                stat = cluster_statistics(
                    [r["effect"] for r in subset],
                    [r["n"] for r in subset],
                    np.random.default_rng(np.random.SeedSequence([SEED, salt])),
                )
                entries.append(dict(contrast=name, left=left, right=right, metric=metric, **stat))
        for entry, adjusted in zip(entries, holm([s["p_signflip"] for s in entries])):
            entry["p_holm"] = adjusted
        statistics[family] = entries
    assert [len(statistics[f]) for f in FAMILIES] == [10, 24, 22, 14]
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
    result = dict(
        source=SOURCE,
        canonical=CANONICAL,
        seed=SEED,
        bootstrap=20000,
        signflips=100000,
        execution=dict(
            validated_runs=1056,
            complete_blocks=32,
            independent_parents=16,
            algorithm_seeds=[16101],
            per_scale_parents={str(n): 4 for n in SIZES},
            failures=[],
            exit=acceptance["exit"],
        ),
        arms=summaries,
        statistics=statistics,
        references=references,
        independent_metric_checks=checks,
        max_overshoot_seconds=max(r["overshoot_seconds"] for r in rows),
        max_rss_bytes=max(r["max_rss_bytes"] for r in rows),
        total_production_seconds=sum(r["production_seconds"] for r in rows),
        final_exact_evidence=sum(r["summary"]["final_exact_verified"] for r in acceptance["rows"]),
    )
    for name, data in (
        ("run_metrics.csv", rows),
        ("parent_metrics.csv", parent_rows),
        ("parent_contrasts.csv", differences),
        ("paired_diagnostics.csv", paired),
        ("anytime_metrics.csv", snapshots),
        ("action_costs.csv", actions),
    ):
        write_csv(output / name, data)
    (output / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "normalized_fronts.json").write_text(json.dumps(saved_fronts), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.results, args.output)
    print(
        json.dumps(
            {"execution": result["execution"], "checks": result["independent_metric_checks"]}
        )
    )
