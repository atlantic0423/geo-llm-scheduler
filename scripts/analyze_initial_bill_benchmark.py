"""Analyze the frozen complete four-arm benchmark without modifying raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from analyze_d5_results import (
    cluster_summary,
    coverage,
    distance,
    holm,
    nondominated,
    normalization,
)

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.experiments.p1_observation import restore_candidate

ARMS = ("BASE", "BILL10", "BILL20", "DEMAND20")
SOURCE = "6d3d0e662a72c2ff0a3ded9ad7bf4d0ccf3c97c2"
CANONICAL = "29388ddf2e5b03593b4d51b0bcf02d81906218d3d8469e157100e89a3f41def7"
STAT_SEED = 2026101204


def read(path: Path) -> object:
    """Load a UTF-8 artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    """Hash all bytes of an artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path: Path, value: object) -> None:
    """Save derived data outside immutable raw evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def table(path: Path, rows: list[dict]) -> None:
    """Save complete flat tables for independent statistical replay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def points(pop: list[dict]) -> list[tuple[float, float]]:
    """Extract physical Flow seconds and total Bill CNY from exact records."""
    return [
        (p["evaluation"]["flow"], p["evaluation"]["tou"] + math.fsum(p["evaluation"]["demand"]))
        for p in pop
    ]


def hv_oracle(front: list[tuple[float, float]]) -> float:
    """Integrate horizontal strips independently of the project HV implementation."""
    valid = [p for p in front if p[0] <= 1.1 and p[1] <= 1.1]
    ys = sorted({p[1] for p in valid} | {1.1})
    return math.fsum(
        (b - a) * max(0, 1.1 - min((x for x, y in valid if y <= a), default=1.1))
        for a, b in zip(ys, ys[1:])
    )


def fast_bill(front: list[tuple[float, float]], limit: float) -> float | None:
    """Return cheapest observed feasible point at a fixed physical Flow bound."""
    eligible = [bill for flow, bill in front if flow <= limit + 1e-9]
    return min(eligible) if eligible else None


def base_reduce(rows: list[dict], field: str) -> tuple[list[float], list[int], list[tuple]]:
    """Average four paired native-H/T and seed observations within each independent base."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["jobs"], row["base_seed"]].append(row)
    keys = sorted(groups)
    for key in keys:
        assert len(groups[key]) == 4
        assert {(r["tariff"], r["seed"]) for r in groups[key]} == {
            (t, s) for t in ("H", "T") for s in (11001, 11002)
        }
    return (
        [float(np.mean([r[field] for r in groups[k]])) for k in keys],
        [k[0] for k in keys],
        keys,
    )


def analyze(raw: Path, index_path: Path, out: Path) -> dict:
    """Validate all 144 indexed blocks; compute frozen indicators, costs and Holm families."""
    assert not out.resolve().is_relative_to(raw.resolve())
    index = read(index_path)
    assert (
        index["source_commit"] == SOURCE
        and index["source_hash"] == CANONICAL
        and len(index["blocks"]) == 144
    )
    groups = defaultdict(list)
    hosts = defaultdict(set)
    checks = Counter()
    ledger = []
    order_counts = Counter()
    parents = {}
    for key, entry in sorted(index["blocks"].items()):
        root = raw / Path(entry["root"]).name
        manifest = read(root / "manifest.json")
        assert sha(root / "manifest.json") == read(root / "manifest.sha256.json")["sha256"]
        spec = read(root / "specs" / (key + ".json"))
        folder = root / "runs" / key
        assert (
            key in manifest["blocks"]
            and sha(root / spec["instance"]) == spec["instance_sha256"] == entry["instance_sha256"]
        )
        marker = read(folder / "complete.json")
        assert marker["spec_sha256"] == sha(root / "specs" / (key + ".json"))
        assert marker["host"] == spec["expected_host"] == entry["host"]
        assert len(marker["files"]) == 18
        for name, digest in marker["files"].items():
            assert sha(folder / name) == digest
        hosts[spec["base_seed"]].add(entry["host"])
        order_counts[tuple(spec["methods"])] += 1
        for arm in ARMS:
            sub = folder / arm
            pop = read(sub / "population.json")
            summary = read(sub / "summary.json")
            assert summary["source_commit"] == spec["source_commit"] == SOURCE
            assert summary["source_hash"] == spec["source_hash"] == CANONICAL
            assert summary["runner_sha256"] == spec["runner_sha256"]
            assert (
                len(pop) == 100
                and len({json.dumps(p["genotype"], sort_keys=True) for p in pop}) == 100
            )
            assert all(
                p["evaluation"]["feasible"]
                and p["evaluation"]["flow"] > 0
                and p["evaluation"]["tou"] >= 0
                and all(x >= 0 for x in p["evaluation"]["demand"])
                for p in pop
            )
            assert summary["final_independent_exact_verified"] == 100
            expected = 0 if arm == "BASE" else int(arm[-2:]) * 4
            assert (
                summary["new_candidate_exact"]
                == summary["candidate_independent_exact_verified"]
                == expected
            )
            assert summary["production_wall_seconds"] > 0 and summary["production_cpu_seconds"] > 0
            assert np.isclose(
                summary["production_wall_seconds"],
                summary["parent_wall_seconds"]
                + summary["bank_wall_seconds"]
                + summary["selection_wall_seconds"],
                rtol=1e-12,
            )
            parents.setdefault(key, summary["parent_genotypes_sha256"])
            assert parents[key] == summary["parent_genotypes_sha256"]
            nd = nondominated(points(pop))
            saved = read(sub / "front.json")
            # Archive uses centralized objective tolerances; reproduce that separate behavior.
            archive = Archive()
            for p in pop:
                archive.consider(restore_candidate(p))
            assert [c.genotype for c in archive.members] == [
                restore_candidate(p).genotype for p in saved
            ]
            checks["stored_archive_replay"] += 1
            if nondominated(points(saved)) != nd:
                checks["strict_vs_tolerance_front"] += 1
            retained = Counter(p["origin"] for p in pop if p["origin"].startswith("initial_mode_"))
            quota = 10 if arm == "BASE" else (100 - int(arm[-2:])) // 10
            assert set(retained.values()) == {quota} and len(retained) == 10
            observations = read(sub / "observations.json")
            assert len(observations) == expected
            for obs in observations:
                assert np.isclose(obs["bill"], obs["tou"] + obs["demand"], rtol=1e-12, atol=1e-8)
                assert np.isclose(obs["prefix_tou"], obs["tou"], rtol=1e-10, atol=1e-8)
                assert np.isclose(obs["prefix_demand"], obs["demand"], rtol=1e-10, atol=1e-8)
                ledger.append({"key": key, "arm": arm} | obs)
            if arm != "BASE":
                family = arm[:-2]
                seeds = [p for p in pop if p["origin"].startswith("bill_seed_")]
                assert len(seeds) == int(arm[-2:])
                for slot, p in enumerate(seeds):
                    pool = [r for r in observations if r["slot"] == slot]
                    assert len(pool) == 4
                    costs = [r["bill"] if family == "BILL" else r["demand"] for r in pool]
                    flows = [r["flow"] for r in pool]
                    w = (0, 0.1, 0.25, 0.5)[slot % 4] if family == "BILL" else 0.1
                    scores = [
                        (1 - w) * (c - min(costs)) / (max(costs) - min(costs) or 1)
                        + w * (f - min(flows)) / (max(flows) - min(flows) or 1)
                        for c, f in zip(costs, flows)
                    ]
                    pf, pb = points([p])[0]
                    match = [
                        i
                        for i, r in enumerate(pool)
                        if np.isclose(r["flow"], pf, rtol=1e-12)
                        and np.isclose(r["bill"], pb, rtol=1e-12)
                    ]
                    assert match and min(scores[i] for i in match) <= min(scores) + 1e-9
                    checks["tournament_replay"] += 1
            groups[spec["jobs"], spec["scenario"], spec["base_seed"], spec["tariff"]].append(
                {
                    "spec": spec,
                    "pop": pop,
                    "arm": arm,
                    "nd": nd,
                    "summary": summary,
                    "root": root.name,
                    "host": entry["host"],
                    "cohort": entry["status_at_redistribution"],
                }
            )
            checks["valid_members"] += 100
            checks["populations"] += 1
            checks["independent_new_exact_bound"] += expected
        a = read(folder / "BILL10/population.json")
        b = read(folder / "BILL20/population.json")
        assert [p for p in a if p["origin"].startswith("bill_seed_")] == [
            p for p in b if p["origin"].startswith("bill_seed_")
        ][:10]
        checks["bill_rng_prefix"] += 1
    assert len(hosts) == 36 and all(len(h) == 1 for h in hosts.values())
    assert len(order_counts) == 4 and set(order_counts.values()) == {36}
    records = []
    fronts = {}
    refs = {}
    normalized = {}
    for case, members in sorted(groups.items()):
        assert len(members) == 8 and {(r["arm"], r["spec"]["seed"]) for r in members} == {
            (a, s) for a in ARMS for s in (11001, 11002)
        }
        union = nondominated([p for r in members for p in r["nd"]])
        lo, span = normalization(union)
        ref = [tuple(p) for p in ((np.asarray(union) - lo) / span).tolist()]
        refhv = hypervolume(ref, (1.1, 1.1))
        name = "_".join(map(str, case))
        refs[name] = {
            "ideal": lo.tolist(),
            "divisor": span.tolist(),
            "raw_front": union,
            "front": ref,
            "hv": refhv,
        }
        for r in members:
            s = r["spec"]
            arm = r["arm"]
            nd = r["nd"]
            norm = [tuple(p) for p in ((np.asarray(nd) - lo) / span).tolist()]
            hv = hypervolume([p for p in norm if max(p) <= 1.1], (1.1, 1.1))
            ip = distance(norm, ref, True)
            assert np.isclose(hv, hv_oracle(norm), rtol=1e-12, atol=1e-12)
            assert np.isclose(ip, igd_plus(norm, ref), rtol=1e-12, atol=1e-12)
            checks["hv_crosscheck"] += 1
            checks["igd_plus_crosscheck"] += 1
            fronts[s["key"], arm] = nd
            normalized[s["key"], arm] = norm
            row = {k: s[k] for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")} | {
                "arm": arm,
                "host": r["host"],
                "root": r["root"],
                "execution_cohort": r["cohort"],
                "hv": hv,
                "hv_reference_fraction": hv / refhv,
                "igd_plus": ip,
                "igd": distance(norm, ref, False),
                "nd_count": len(nd),
                "min_flow": min(p[0] for p in nd),
                "min_bill": min(p[1] for p in nd),
                "flow_gap_pct": 100 * (min(p[0] for p in nd) / lo[0] - 1),
                "bill_gap_pct": 100 * (min(p[1] for p in nd) / lo[1] - 1),
            }
            row.update(
                {
                    k: r["summary"][k]
                    for k in (
                        "production_wall_seconds",
                        "production_cpu_seconds",
                        "parent_wall_seconds",
                        "bank_wall_seconds",
                        "audit_wall_seconds",
                        "audit_cpu_seconds",
                        "export_wall_seconds",
                        "total_elapsed_seconds",
                        "new_candidate_exact",
                    )
                }
            )
            records.append(row)
    assert len(records) == 576
    lookup = {(r["key"], r["arm"]): r for r in records}
    pairs = []
    for key in sorted(index["blocks"]):
        b = lookup[key, "BASE"]
        basefront = fronts[key, "BASE"]
        for arm in ARMS[1:]:
            a = lookup[key, arm]
            af = fronts[key, arm]
            row = {k: b[k] for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")} | {
                "arm": arm,
                "hv_diff": a["hv"] - b["hv"],
                "hv_pct": 100 * (a["hv"] - b["hv"]) / b["hv"],
                "igd_plus_gain": b["igd_plus"] - a["igd_plus"],
                "igd_gain": b["igd"] - a["igd"],
                "wall_diff": a["production_wall_seconds"] - b["production_wall_seconds"],
                "wall_ratio": a["production_wall_seconds"] / b["production_wall_seconds"],
                "cpu_ratio": a["production_cpu_seconds"] / b["production_cpu_seconds"],
                "bill_gain_pct": 100 * (b["min_bill"] - a["min_bill"]) / b["min_bill"],
                "fast_flow_loss_pct": 100 * (a["min_flow"] - b["min_flow"]) / b["min_flow"],
                "coverage_new_base": coverage(normalized[key, arm], normalized[key, "BASE"]),
                "coverage_base_new": coverage(normalized[key, "BASE"], normalized[key, arm]),
            }
            for multiplier in (1.1, 1.25):
                limit = b["min_flow"] * multiplier
                bc = fast_bill(basefront, limit)
                ac = fast_bill(af, limit)
                label = str(multiplier).replace(".", "_")
                row["fast_bill_gain_pct_" + label] = None if ac is None else 100 * (bc - ac) / bc
                row["fast_bill_available_" + label] = ac is not None
            pairs.append(row)
    rng = np.random.default_rng(STAT_SEED)
    stats = {}
    families = {"primary": [], "igd_secondary": [], "wall_secondary": []}
    for arm in ARMS[1:]:
        selected = [r for r in pairs if r["arm"] == arm]
        stats[arm] = {}
        for metric, family in (
            ("hv_diff", "primary"),
            ("igd_plus_gain", "primary"),
            ("igd_gain", "igd_secondary"),
            ("wall_diff", "wall_secondary"),
        ):
            values, strata, keys = base_reduce(selected, metric)
            assert len(values) == 36
            stat = cluster_summary(values, strata, rng)
            stat["base_effects"] = values
            stat["base_keys"] = keys
            stats[arm][metric] = stat
            families[family].append((arm, metric))
    for family in families.values():
        ps = holm([stats[a][m]["p_signflip"] for a, m in family])
        for (a, m), p in zip(family, ps):
            stats[a][m]["p_holm"] = p
    fields = [
        "hv",
        "hv_reference_fraction",
        "igd_plus",
        "igd",
        "nd_count",
        "flow_gap_pct",
        "bill_gap_pct",
        "production_wall_seconds",
        "production_cpu_seconds",
        "bank_wall_seconds",
        "audit_wall_seconds",
        "export_wall_seconds",
        "total_elapsed_seconds",
    ]
    aggregates = {}
    stratified = []
    for arm in ARMS:
        selected = [r for r in records if r["arm"] == arm]
        aggregates[arm] = {f: float(np.mean(base_reduce(selected, f)[0])) for f in fields}
        for jobs in (50, 100, 200):
            sub = [r for r in selected if r["jobs"] == jobs]
            stratified.append(
                {
                    "arm": arm,
                    "jobs": jobs,
                    **{f: float(np.mean(base_reduce(sub, f)[0])) for f in fields},
                }
            )
    for arm in ARMS[1:]:
        selected = [r for r in pairs if r["arm"] == arm]
        aggregates[arm].update(
            {
                f: float(np.mean(base_reduce(selected, f)[0]))
                for f in (
                    "hv_pct",
                    "wall_ratio",
                    "cpu_ratio",
                    "bill_gain_pct",
                    "fast_flow_loss_pct",
                    "coverage_new_base",
                    "coverage_base_new",
                )
            }
        )
        for mult in ("1_1", "1_25"):
            selected_values = [
                r["fast_bill_gain_pct_" + mult]
                for r in selected
                if r["fast_bill_available_" + mult]
            ]
            aggregates[arm]["fast_bill_gain_pct_" + mult] = float(np.mean(selected_values))
            aggregates[arm]["fast_bill_unavailable_" + mult] = 144 - len(selected_values)
        aggregates[arm]["fast_flow_loss_bases"] = sum(
            x > 1e-8 for x in base_reduce(selected, "fast_flow_loss_pct")[0]
        )
    for tariff in ("H", "T"):
        for arm in ARMS:
            sub = [r for r in records if r["arm"] == arm and r["tariff"] == tariff]
            stratified.append(
                {
                    "arm": arm,
                    "jobs": tariff,
                    **{f: float(np.mean([r[f] for r in sub])) for f in fields},
                }
            )
    report = {
        "source": SOURCE,
        "canonical": CANONICAL,
        "formal_blocks": 144,
        "populations": 576,
        "independent_bases": 36,
        "stat_seed": STAT_SEED,
        "bootstrap": 20000,
        "signflips": 100000,
        "aggregates": aggregates,
        "statistics": stats,
        "families": families,
        "checks": dict(checks),
        "method_orders": {"-".join(k): v for k, v in order_counts.items()},
        "definitions": {
            "normalization": "common empirical nondominated union ideal/nadir, degenerate range=1",
            "hv_reference": [1.1, 1.1],
            "positive_gains": "HV new-base; IGD/IGD+ base-new; wall new-base means slower",
            "fast_bound_analysis": "posthoc exploratory physical Flow bound relative to BASE fastest; unavailable counted, not imputed",
        },
    }
    data = out / "data"
    dump(data / "report.json", report)
    dump(data / "references.json", refs)
    dump(
        data / "fronts_seed11001.json",
        {
            key: {a: fronts[key, a] for a in ARMS}
            for key in index["blocks"]
            if key.endswith("_11001")
        },
    )
    table(data / "metrics.csv", records)
    table(data / "pairs.csv", pairs)
    table(data / "stratified.csv", stratified)
    table(data / "candidate_observations.csv", ledger)
    dump(data / "integrity_checks.json", dict(checks))
    print(
        json.dumps(
            {
                "validated_populations": 576,
                "aggregates": aggregates,
                "primary_holm": {
                    a: {m: stats[a][m]["p_holm"] for m in ("hv_diff", "igd_plus_gain")}
                    for a in ARMS[1:]
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.raw, args.index, args.out)
