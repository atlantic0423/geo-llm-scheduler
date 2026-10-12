"""Analyze only verified completed bill-mixture blocks and matched old populations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

OLD = ("RAW", "TIME16", "TIME64", "STRUCT16", "STRUCT64")
FAMILIES = ("RESAMPLE", "TOU", "DEMAND", "BILL", "PORTFOLIO")
NEW = tuple(f"{f}{q}" for f in FAMILIES for q in (10, 20))
ARMS = OLD + NEW
SOURCE = "a8a2afd91e2fb42978a4a4be74e29622b4346c90"
CANONICAL = "3cbb17037b480d8bcde090e48ab96e236b6a1d65909f43abe05904168fd04742"
STAT_SEED = 2026101203


def read(path: Path):
    """Read an immutable UTF-8 JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    """Return the digest of the whole artifact, not a sample."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    """Write a deterministic derived artifact outside raw evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def table(path: Path, rows: list[dict]) -> None:
    """Write a flat analysis table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def points(population: list[dict]) -> list[tuple[float, float]]:
    """Use exact serialized TOU plus every regional demand charge."""
    return [
        (p["evaluation"]["flow"], p["evaluation"]["tou"] + sum(p["evaluation"]["demand"]))
        for p in population
    ]


def genes(candidate: dict) -> str:
    """Canonicalize genes for equality; schedules are not assignment authority."""
    return json.dumps(candidate["genotype"], sort_keys=True, separators=(",", ":"))


def hv_oracle(front: list) -> float:
    """Independently integrate horizontal strips for a (1.1,1.1) reference."""
    valid = [p for p in front if max(p) <= 1.1]
    ys = sorted({p[1] for p in valid} | {1.1})
    return sum(
        (b - a) * max(0.0, 1.1 - min((x for x, y in valid if y <= a), default=1.1))
        for a, b in zip(ys, ys[1:])
    )


def directions(front: list, reference: list) -> float:
    """Fraction of 11 weighted directions within 0.001 of empirical best scalar value."""
    hit = 0
    for weight in np.linspace(0, 1, 11):

        def score(p):
            return weight * p[0] + (1 - weight) * p[1]

        hit += min(map(score, front)) <= min(map(score, reference)) + 0.001
    return hit / 11


def summarize_by_base(rows: list[dict], fields: list[str]) -> dict:
    """Average available H/T and seeds within each base, then weight bases equally."""
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["jobs"], row["base_seed"]].append(row)
    return {
        field: float(np.mean([np.mean([r[field] for r in group]) for group in clusters.values()]))
        for field in fields
    }


def statistical_family(pairs: list[dict], complete_only: bool = False) -> dict:
    """Keep all 16+16 originally specified tests, even with a partial matrix."""
    stats = {}
    rng = np.random.default_rng(STAT_SEED)
    families = {"primary": [], "secondary": []}
    for family, comparator in (("primary", "RAW"), ("secondary", "RESAMPLE")):
        for arm in NEW[2:]:
            other = comparator if comparator == "RAW" else comparator + arm[-2:]
            label = arm + "-" + other
            selected = [
                r
                for r in pairs
                if r["contrast"] == label and (not complete_only or r["complete_base"])
            ]
            clusters = defaultdict(list)
            for row in selected:
                clusters[row["jobs"], row["base_seed"]].append(row)
            keys = sorted(clusters)
            stats[label] = {}
            for metric in ("hv_diff", "igd_plus_gain"):
                values = [float(np.mean([r[metric] for r in clusters[key]])) for key in keys]
                stats[label][metric] = cluster_summary(values, [key[0] for key in keys], rng)
                stats[label][metric]["base_effects"] = values
                families[family].append((label, metric))
    for family in families.values():
        adjusted = holm([stats[label][metric]["p_signflip"] for label, metric in family])
        for (label, metric), value in zip(family, adjusted, strict=True):
            stats[label][metric]["p_holm"] = value
    return {"statistics": stats, "families": families}


def analyze(raw: Path, baseline: Path, out: Path) -> None:
    """Validate paired provenance, recompute front metrics, and export partial diagnostics."""
    assert not out.resolve().is_relative_to(raw.resolve())
    data = out / "data"
    pending, blocks, groups, costs, ledger = [], [], defaultdict(list), [], []
    checks = Counter()
    for node, host in enumerate(("b518ece5cd01", "878be0fed7b0")):
        root = raw / f"node{node}" / f"formal_node{node}"
        oldroot = baseline / f"formal_node{node}"
        manifest = read(root / "manifest.json")
        assert manifest["source_commit"] == SOURCE and manifest["source_hash"] == CANONICAL
        assert sha(root / "manifest.json") == read(root / "manifest.sha256.json")["sha256"]
        for path, digest in manifest["input_files"].items():
            assert sha(root / path) == digest
        for key in manifest["blocks"]:
            spec = read(root / "specs" / (key + ".json"))
            folder = root / "runs" / key
            if not (folder / "complete.json").exists():
                assert not folder.exists()
                pending.append(
                    {k: spec[k] for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")}
                )
                continue
            marker, summary = read(folder / "complete.json"), read(folder / "summary.json")
            assert marker["host"] == summary["host"] == spec["expected_host"] == host
            assert marker["spec_sha256"] == sha(root / "specs" / (key + ".json"))
            assert len(marker["files"]) == 24
            for path, digest in marker["files"].items():
                assert sha(folder / path) == digest
            for field in (
                "source_commit",
                "source_hash",
                "runner_sha256",
                "parent_population_sha256",
            ):
                assert summary[field] == spec[field]
            assert summary["exact"] == summary["independent_exact_verified"] == 320
            assert summary["selected_exact_verified"] == 1000 and summary["populations"] == 10
            parent = read(root / spec["parent_population"])
            assert sha(root / spec["parent_population"]) == spec["parent_population_sha256"]
            assert parent == read(oldroot / "runs" / key / "RAW" / "population.json")
            assert (
                sha(oldroot / "runs" / key / "RAW" / "complete.json")
                == spec["baseline_raw_marker_sha256"]
            )
            assert sha(root / spec["instance"]) == sha(
                oldroot / read(oldroot / "specs" / (key + ".json"))["instance"]
            )
            case = tuple(spec[k] for k in ("jobs", "scenario", "base_seed", "tariff"))
            blocks.append(spec)
            bank, observations = read(folder / "bank.json"), read(folder / "observations.json")
            assert len(observations) == 320 and all(len(bank[f]) == 20 for f in FAMILIES[:-1])
            for family in FAMILIES[:-1]:
                for slot in range(20):
                    pool = [r for r in observations if r["family"] == family and r["slot"] == slot]
                    assert len(pool) == 4
                    cost = [
                        r["tou"]
                        if family == "TOU"
                        else r["demand"]
                        if family == "DEMAND"
                        else r["bill"]
                        for r in pool
                    ]
                    flow = [r["flow"] for r in pool]
                    weight = (0, 0.1, 0.25, 0.5)[slot % 4] if family == "BILL" else 0.1
                    scores = [
                        (1 - weight) * (c - min(cost)) / (max(cost) - min(cost) or 1)
                        + weight * (f - min(flow)) / (max(flow) - min(flow) or 1)
                        for c, f in zip(cost, flow, strict=True)
                    ]
                    winner = bank[family][slot]["evaluation"]
                    bill = winner["tou"] + sum(winner["demand"])
                    eligible = [
                        i
                        for i, r in enumerate(pool)
                        if np.isclose(r["flow"], winner["flow"], rtol=1e-12, atol=1e-8)
                        and np.isclose(r["bill"], bill, rtol=1e-12, atol=1e-8)
                    ]
                    assert eligible and min(scores[i] for i in eligible) <= min(scores) + 1e-9
                    checks["tournament_winners"] += 1
            for row in observations:
                assert np.isclose(row["bill"], row["tou"] + row["demand"], rtol=1e-12, atol=1e-8)
                if row["family"] != "RESAMPLE":
                    assert np.isclose(row["prefix_tou"], row["tou"], rtol=1e-10, atol=1e-8)
                    assert np.isclose(row["prefix_demand"], row["demand"], rtol=1e-10, atol=1e-8)
                ledger.append(
                    {k: spec[k] for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")}
                    | row
                )
            costs.append(
                {k: spec[k] for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")}
                | {
                    k: summary[k]
                    for k in (
                        "production_seconds",
                        "verification_seconds",
                        "total_elapsed",
                        "baseline_replay_seconds",
                        "rss_peak_gib",
                        "exact",
                    )
                }
            )
            for arm in ARMS:
                afolder = folder / arm if arm in NEW else oldroot / "runs" / key / arm
                pop = read(afolder / "population.json")
                if arm in OLD:
                    mark, osu = read(afolder / "complete.json"), read(afolder / "summary.json")
                    oldspec = read(oldroot / "specs" / (key + ".json"))
                    assert mark["host"] == osu["host"] == host and mark["spec_sha256"] == sha(
                        oldroot / "specs" / (key + ".json")
                    )
                    for path, digest in mark["files"].items():
                        assert sha(afolder / path) == digest
                    assert osu["source_commit"] == "5bb666874a86ae59e38aa2513a1c4f7ed1fa8fcb"
                    assert osu["parent_population_sha256"] == spec["parent_population_sha256"]
                    assert oldspec["seed"] == spec["seed"]
                    production = osu["production_seconds"]
                else:
                    share = int(arm[-2:])
                    family = arm[:-2]
                    quota = (100 - share) // 10
                    counts = [0] * 10
                    retained = []
                    for member in parent:
                        mode = int(member["origin"].removeprefix("initial_mode_"))
                        if counts[mode] < quota:
                            retained.append(member)
                            counts[mode] += 1
                    assert counts == [quota] * 10
                    seeds = (
                        [bank[("TOU", "DEMAND", "BILL")[i % 3]][i // 3] for i in range(share)]
                        if family == "PORTFOLIO"
                        else bank[family][:share]
                    )
                    assert pop == retained + seeds
                    production = None
                    checks["new_members"] += share
                assert len(pop) == 100, (key, arm, "population size")
                if arm in NEW:
                    assert len({genes(p) for p in pop}) == 100, (key, arm, "gene uniqueness")
                for member in pop:
                    e = member["evaluation"]
                    assert (
                        e["feasible"]
                        and e["flow"] > 0
                        and e["tou"] >= 0
                        and all(x >= 0 for x in e["demand"])
                    )
                nd = nondominated(points(pop))
                stored_front = nd
                if arm in NEW:
                    saved = read(afolder / "front.json")
                    bygenes = {genes(p): p for p in pop}
                    assert all(bygenes[genes(p)] == p for p in saved)
                    stored_front = nondominated(points(saved))
                    if stored_front != nd:
                        archive = Archive()
                        for member in pop:
                            archive.consider(restore_candidate(member))
                        assert [genes(p) for p in saved] == [
                            json.dumps(
                                {"ms": c.genotype.ms, "os": c.genotype.os},
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            for c in archive.members
                        ], (key, arm, "tolerance-aware archive replay")
                        checks["strict_vs_tolerance_fronts"] += 1
                groups[case].append(
                    {
                        "spec": spec,
                        "arm": arm,
                        "pop": pop,
                        "nd": nd,
                        "stored_front": stored_front,
                        "production_seconds": production,
                    }
                )
                checks["population_integrity"] += 1
                checks["representative_numeric_checks"] += 100
    assert len(blocks) == 128 and len(pending) == 16
    available = defaultdict(list)
    for s in blocks:
        available[s["jobs"], s["base_seed"]].append(s)
    assert len(available) == 34 and sum(len(v) == 4 for v in available.values()) == 30
    complete_bases = {key for key, value in available.items() if len(value) == 4}
    records, fronts, cases, references = [], {}, {}, {}
    for case, members in sorted(groups.items()):
        seeds = sorted({r["spec"]["seed"] for r in members})
        assert len(members) == len(ARMS) * len(seeds)
        union = nondominated([p for r in members for p in r["nd"]])
        lo, span = normalization(union)
        ref = ((np.array(union) - lo) / span).tolist()
        refhv = hypervolume([tuple(p) for p in ref], (1.1, 1.1))
        name = "_".join(map(str, case))
        references[name] = {
            "ideal": lo.tolist(),
            "divisor": span.tolist(),
            "raw_front": union,
            "normalized_front": ref,
            "hv": refhv,
            "available_seeds": seeds,
        }
        cases[name] = {
            arm: nondominated([p for r in members if r["arm"] == arm for p in r["nd"]])
            for arm in ARMS
        }
        for r in members:
            s, arm, pop, nd = r["spec"], r["arm"], r["pop"], r["nd"]
            norm = ((np.array(nd) - lo) / span).tolist()
            hv = hypervolume([tuple(p) for p in norm if max(p) <= 1.1], (1.1, 1.1))
            ip = distance(norm, ref, True)
            assert np.isclose(hv, hv_oracle(norm), rtol=1e-12, atol=1e-12)
            assert np.isclose(ip, igd_plus(norm, ref), rtol=1e-12, atol=1e-12)
            checks["hv_independent"] += 1
            checks["igd_plus_independent"] += 1
            saved_norm = ((np.array(r["stored_front"]) - lo) / span).tolist()
            saved_hv = hypervolume([tuple(p) for p in saved_norm if max(p) <= 1.1], (1.1, 1.1))
            saved_ip = distance(saved_norm, ref, True)
            fronts[s["key"], arm] = norm
            ep = min(
                pop,
                key=lambda p: (
                    p["evaluation"]["tou"] + sum(p["evaluation"]["demand"]),
                    p["evaluation"]["flow"],
                ),
            )
            new_nd = {
                p
                for member, p in zip(pop, points(pop), strict=True)
                if member["origin"].startswith("bill_seed_") and p in set(nd)
            }
            old_nd = {
                p
                for member, p in zip(pop, points(pop), strict=True)
                if not member["origin"].startswith("bill_seed_") and p in set(nd)
            }
            records.append(
                {k: s[k] for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")}
                | dict(
                    arm=arm,
                    complete_base=(s["jobs"], s["base_seed"]) in complete_bases,
                    hv=hv,
                    stored_archive_hv_abs_difference=abs(hv - saved_hv),
                    stored_archive_igd_plus_abs_difference=abs(ip - saved_ip),
                    hv_reference_fraction=hv / refhv,
                    igd_plus=ip,
                    igd=distance(norm, ref, False),
                    nd_count=len(nd),
                    direction_hits=directions(norm, ref),
                    min_flow=min(p[0] for p in nd),
                    min_bill=min(p[1] for p in nd),
                    flow_gap_pct=100 * (min(p[0] for p in nd) / lo[0] - 1),
                    bill_gap_pct=100 * (min(p[1] for p in nd) / lo[1] - 1),
                    bill_endpoint_flow=ep["evaluation"]["flow"],
                    bill_endpoint_tou=ep["evaluation"]["tou"],
                    bill_endpoint_demand=sum(ep["evaluation"]["demand"]),
                    new_nd_points=len(new_nd),
                    new_exclusive_nd_points=len(new_nd - old_nd),
                    old_nd_points=len(old_nd),
                    production_seconds=r["production_seconds"],
                )
            )
    byrun = {(r["key"], r["arm"]): r for r in records}
    contrasts = []
    for arm in NEW:
        contrasts.append((arm, "RAW"))
        if not arm.startswith("RESAMPLE"):
            contrasts.append((arm, "RESAMPLE" + arm[-2:]))
        for other in ("STRUCT16", "STRUCT64"):
            contrasts.append((arm, other))
    contrasts.extend((family + "20", family + "10") for family in FAMILIES)
    pairs = []
    for arm, other in contrasts:
        for key in sorted({r["key"] for r in records}):
            a, b = byrun[key, arm], byrun[key, other]
            pairs.append(
                {
                    k: a[k]
                    for k in (
                        "key",
                        "jobs",
                        "scenario",
                        "base_seed",
                        "tariff",
                        "seed",
                        "complete_base",
                    )
                }
                | dict(
                    contrast=arm + "-" + other,
                    hv_diff=a["hv"] - b["hv"],
                    hv_gain_pct=100 * (a["hv"] / b["hv"] - 1) if b["hv"] else 0.0,
                    igd_plus_gain=b["igd_plus"] - a["igd_plus"],
                    min_bill_gain_pct=100 * (1 - a["min_bill"] / b["min_bill"]),
                    min_flow_gain_pct=100 * (1 - a["min_flow"] / b["min_flow"]),
                    nd_diff=a["nd_count"] - b["nd_count"],
                    direction_diff=a["direction_hits"] - b["direction_hits"],
                    coverage_a_b=coverage(fronts[key, arm], fronts[key, other]),
                    coverage_b_a=coverage(fronts[key, other], fronts[key, arm]),
                    bill_endpoint_flow_gain_pct=100
                    * (1 - a["bill_endpoint_flow"] / b["bill_endpoint_flow"]),
                )
            )
    numeric = [
        "hv",
        "hv_reference_fraction",
        "igd_plus",
        "igd",
        "nd_count",
        "direction_hits",
        "min_flow",
        "min_bill",
        "flow_gap_pct",
        "bill_gap_pct",
        "bill_endpoint_tou",
        "bill_endpoint_demand",
        "new_nd_points",
        "new_exclusive_nd_points",
        "old_nd_points",
    ]
    effect_fields = [
        "hv_diff",
        "hv_gain_pct",
        "igd_plus_gain",
        "min_bill_gain_pct",
        "min_flow_gain_pct",
        "nd_diff",
        "direction_diff",
        "coverage_a_b",
        "coverage_b_a",
        "bill_endpoint_flow_gain_pct",
    ]
    overall = {a: summarize_by_base([r for r in records if r["arm"] == a], numeric) for a in ARMS}
    paired_overall = {
        a + "-" + b: summarize_by_base(
            [r for r in pairs if r["contrast"] == a + "-" + b], effect_fields
        )
        for a, b in contrasts
    }
    subgroups = []
    for field in ("jobs", "tariff", "scenario"):
        for value in sorted({r[field] for r in records}, key=str):
            for arm in ARMS:
                rows = [r for r in records if r["arm"] == arm and r[field] == value]
                subgroups.append(
                    dict(
                        group=field,
                        value=value,
                        arm=arm,
                        n_populations=len(rows),
                        n_bases=len({(r["jobs"], r["base_seed"]) for r in rows}),
                        **summarize_by_base(rows, numeric),
                    )
                )
    table(data / "run_metrics.csv", records)
    table(data / "paired_metrics.csv", pairs)
    table(data / "subgroups.csv", subgroups)
    table(data / "shared_bank_cost.csv", costs)
    table(data / "trial_observations.csv", ledger)
    table(data / "missing_blocks.csv", pending)
    report = dict(
        verification_status="ANALYZED: derived metrics cross-checked; original worker exact evidence is hash bound, not freshly rerun",
        source_commit=SOURCE,
        canonical=CANONICAL,
        stat_seed=STAT_SEED,
        new_populations=1280,
        matched_old_populations=640,
        completed_blocks=128,
        pending_blocks=16,
        independent_available_bases=34,
        complete_four_block_bases=30,
        available_base_counts={str(k): len(v) for k, v in sorted(available.items())},
        main=statistical_family(pairs),
        sensitivity_complete_bases=statistical_family(pairs, True),
        overall_equal_base=overall,
        paired_overall_equal_base=paired_overall,
        audit=dict(checks),
        missing=pending,
        costs_by_size={
            str(j): {
                k: float(np.mean([r[k] for r in costs if r["jobs"] == j]))
                for k in (
                    "production_seconds",
                    "verification_seconds",
                    "total_elapsed",
                    "baseline_replay_seconds",
                    "rss_peak_gib",
                )
            }
            for j in (50, 100, 200)
        },
    )
    dump(data / "report.json", report)
    dump(data / "references.json", references)
    dump(data / "case_fronts.json", cases)
    make_figures(out, cases, overall)
    print(
        json.dumps({"audit": dict(checks), "overall": overall, "paired_overall": paired_overall}),
        flush=True,
    )


def make_figures(out: Path, cases: dict, overall: dict) -> None:
    """Show all arms; do not select examples by observed gains."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, field, title in zip(
        axes.flat,
        ("hv_reference_fraction", "igd_plus", "nd_count", "direction_hits"),
        (
            "HV / common empirical envelope",
            "IGD+ (lower is better)",
            "Unique nondominated objective points",
            "11-direction empirical hit fraction",
        ),
        strict=True,
    ):
        ax.bar(
            range(len(ARMS)),
            [overall[a][field] for a in ARMS],
            color=["#444444"] * 5
            + [
                "#498bb8",
                "#75b2d7",
                "#269b6d",
                "#57ba98",
                "#cc8438",
                "#e9ad66",
                "#ad5770",
                "#cf8b9c",
                "#7b6bb3",
                "#a99acd",
            ],
        )
        ax.set_xticks(range(len(ARMS)), ARMS, rotation=65, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Partial bill-mixture initialization: 34 bases, equal base weight")
    fig.tight_layout()
    fig.savefig(folder / "overview.png", dpi=150)
    plt.close(fig)
    colors = plt.get_cmap("tab10").colors
    for name, fronts in cases.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, arms, title in (
            (axes[0], OLD, "Matched Step-2 methods"),
            (axes[1], ("RAW",) + NEW, "New 10% / 20% mixtures"),
        ):
            for i, arm in enumerate(arms):
                p = np.asarray(fronts[arm])
                ax.plot(
                    p[:, 0],
                    p[:, 1],
                    ".-",
                    label=arm,
                    linewidth=0.9,
                    markersize=3,
                    color=colors[i % 10],
                    alpha=0.85,
                )
            ax.set(xlabel="Total Flow Time (seconds)", ylabel="Electricity Bill (CNY)", title=title)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, ncol=2)
        fig.suptitle(name + " / matched available-seed empirical fronts")
        fig.tight_layout()
        fig.savefig(folder / (name + ".png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    analyze(arguments.raw, arguments.baseline, arguments.out)
