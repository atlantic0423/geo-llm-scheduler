"""Postprocess the frozen initialization-only matrix without modifying raw data.

Run with the repository src and scripts on PYTHONPATH. Inference is at base level;
the 7,920 populations are repeated measurements, not independent instances.
"""

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

from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus

METHODS = (
    "MIXED",
    "FCFS_LOAD",
    "FCFS_RANDOM",
    "SPT_LOAD",
    "SPT_RANDOM",
    "LPT_LOAD",
    "LPT_RANDOM",
    "RANDOM_LOAD",
    "RANDOM",
    "KV_AWARE",
    "ENERGY_AWARE",
)
SCENARIOS = ("mixed", "burst", "kv_delay", "compute_tight", "vram_tight", "demand_high")
SOURCE = "e57d385b2a4dcc7546e2f54e920cccbe2b6eb7a1"
CANONICAL = "06c3a645ff4bcd24431640eacca0a42fb58309c65ac41e9480b250e5927d9436"
STAT_SEED = 2026101101


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a complete metric table using its explicit column order."""
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def raw_points(row: dict) -> list[tuple[float, float]]:
    """Reconstruct the exact worker Bill using its original TOU plus Demand sum."""
    return [
        (p["evaluation"]["flow"], p["evaluation"]["tou"] + sum(p["evaluation"]["demand"]))
        for p in row["points"]
    ]


def independent_hv(points: list) -> float:
    """Compute the HV by horizontal strips for independent metric verification."""
    # Horizontal strips, independent of the project's incremental rectangle sweep.
    p = [q for q in points if max(q) <= 1.1]
    ys = sorted({q[1] for q in p} | {1.1})
    return sum(
        (b - a) * max(0.0, 1.1 - min((x for x, y in p if y <= a), default=1.1))
        for a, b in zip(ys, ys[1:])
    )


def base_effect(rows: list[dict], field: str, missing: bool = False) -> tuple[list, list]:
    """Reduce paired repeats to 36 independent base effects before inference.

    Missing values are permitted only for descriptive ratios with zero denominators.
    All confirmatory differences retain every observation and base.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r["jobs"], r["base_seed"])].append(r)
    keys = sorted(groups)
    assert len(keys) == 36
    for k in keys:
        assert len(groups[k]) == 20 and {(r["tariff"], r["seed"]) for r in groups[k]} == {
            (t, s) for t in ("H", "T") for s in range(11001, 11011)
        }
    if missing:
        keys = [k for k in keys if any(r[field] is not None for r in groups[k])]
    return [float(np.mean([r[field] for r in groups[k] if r[field] is not None])) for k in keys], [
        k[0] for k in keys
    ]


def analyze(input_path: Path, out: Path) -> None:
    """Export the complete prespecified matrix, inferences and diagnostic figures."""
    source = json.loads(input_path.read_text(encoding="utf-8"))
    assert source["source_commit"] == SOURCE and source["canonical"] == CANONICAL
    records = source["rows"]
    assert len(records) == 7920
    out.mkdir(parents=True, exist_ok=True)
    data = out / "data"
    data.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for r in records:
        s = r["spec"]
        groups[(s["jobs"], s["scenario"], s["base_seed"], s["tariff"])].append(r)
        assert (
            s["source_commit"] == SOURCE
            and s["source_hash"] == CANONICAL
            and len(r["points"]) == 100
        )
        assert len({p["genotype_sha256"] for p in r["points"]}) == 100
    assert len(groups) == 72
    rows = []
    normalized = {}
    references = {}
    sources = []
    audit_hv = 0
    audit_igd = 0
    fig_inputs = {}
    for key, members in sorted(groups.items()):
        assert len(members) == 110 and {(r["method"], r["spec"]["seed"]) for r in members} == {
            (m, s) for m in METHODS for s in range(11001, 11011)
        }
        union = nondominated([p for r in members for p in raw_points(r)])
        lo, span = normalization(union)
        front = ((np.asarray(union) - lo) / span).tolist()
        ref_hv = hypervolume([tuple(p) for p in front], (1.1, 1.1))
        refname = "_".join(map(str, key))
        references[refname] = {
            "ideal": lo.tolist(),
            "divisor": span.tolist(),
            "raw_front": union,
            "front": front,
            "hv": ref_hv,
        }
        for r in members:
            s = r["spec"]
            summary = r["summary"]
            points = raw_points(r)
            nd = nondominated(points)
            p = ((np.asarray(nd) - lo) / span).tolist()
            idx = (s["key"], r["method"])
            normalized[idx] = p
            hv = hypervolume([tuple(q) for q in p if max(q) <= 1.1], (1.1, 1.1))
            assert np.isclose(hv, independent_hv(p), rtol=1e-12, atol=1e-12)
            audit_hv += 1
            ip = distance(p, front, True)
            if r["method"] in ("MIXED", "RANDOM"):
                assert np.isclose(ip, igd_plus(p, front), rtol=1e-12, atol=1e-12)
                audit_igd += 1
            weights = np.column_stack((np.linspace(0, 1, 100), np.linspace(1, 0, 100)))
            best = np.max(weights[:, None, :] * np.asarray(p)[None, :, :], axis=2).min(axis=1)
            refbest = np.max(weights[:, None, :] * np.asarray(front)[None, :, :], axis=2).min(
                axis=1
            )
            row = {k: s[k] for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")}
            row.update(
                method=r["method"],
                host=summary["host"],
                hv=hv,
                hv_reference_fraction=hv / ref_hv,
                igd_plus=ip,
                igd=distance(p, front, False),
                nd_count=len(nd),
                min_flow=min(x for x, y in nd),
                min_bill=min(y for x, y in nd),
                flow_gap_pct=100 * (min(x for x, y in nd) / lo[0] - 1),
                bill_gap_pct=100 * (min(y for x, y in nd) / lo[1] - 1),
                direction_regret_mean=float(np.mean(best - refbest)),
                direction_within_001=float(np.mean(best - refbest <= 0.01)),
                production_seconds=summary["generation_seconds"]
                + sum(summary["seconds"].get(k, 0.0) for k in ("ssgs", "exact", "archive")),
                generation_seconds=summary["generation_seconds"],
                ssgs_seconds=summary["seconds"]["ssgs"],
                exact_seconds=summary["seconds"]["exact"],
                replay_seconds=summary["original_replay_seconds"],
                verification_seconds=summary["independent_exact_seconds"],
                measured_elapsed=summary["elapsed"],
                duplicates=summary["duplicates"],
            )
            rows.append(row)
            if s["seed"] == 11001:
                fig_inputs[(refname, r["method"])] = p
            if r["method"] == "MIXED":
                bypoint = defaultdict(list)
                for point, item in zip(points, r["points"], strict=True):
                    bypoint[tuple(point)].append(item["origin"])
                credit = Counter()
                flow_credit = Counter()
                bill_credit = Counter()
                for point in nd:
                    origins = bypoint[point]
                    for origin in origins:
                        credit[origin] += 1 / len(origins)
                        if point[0] == row["min_flow"]:
                            flow_credit[origin] += 1 / len(origins)
                        if point[1] == row["min_bill"]:
                            bill_credit[origin] += 1 / len(origins)
                initial = Counter(p["origin"] for p in r["points"])
                for mode in range(10):
                    origin = f"initial_mode_{mode}"
                    sources.append(
                        {
                            **{
                                k: s[k]
                                for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")
                            },
                            "method": METHODS[mode + 1],
                            "initial_count": initial[origin],
                            "nd_credit": credit[origin],
                            "nd_credit_fraction": credit[origin] / len(nd),
                            "flow_endpoint_credit": flow_credit[origin],
                            "bill_endpoint_credit": bill_credit[origin],
                        }
                    )
    lookup = {(r["key"], r["method"]): r for r in rows}
    contrasts = [("MIXED", "RANDOM")] + [
        (m, "MIXED") for m in METHODS if m not in ("MIXED", "RANDOM")
    ]
    pairs = []
    stats = {}
    rng = np.random.default_rng(STAT_SEED)
    for a, b in contrasts:
        selected = []
        for key in sorted({r["key"] for r in rows}):
            ra, rb = lookup[(key, a)], lookup[(key, b)]
            pa, pb = normalized[(key, a)], normalized[(key, b)]
            d = {k: ra[k] for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")}
            d.update(
                contrast=a + "-" + b,
                hv_diff=ra["hv"] - rb["hv"],
                igd_plus_gain=rb["igd_plus"] - ra["igd_plus"],
                hv_pct=100 * (ra["hv"] / rb["hv"] - 1) if rb["hv"] > 0 else None,
                igd_gain=rb["igd"] - ra["igd"],
                flow_gain_pct=100 * (rb["min_flow"] - ra["min_flow"]) / rb["min_flow"],
                bill_gain_pct=100 * (rb["min_bill"] - ra["min_bill"]) / rb["min_bill"],
                coverage_a_b=coverage(pa, pb),
                coverage_b_a=coverage(pb, pa),
                production_seconds_diff=ra["production_seconds"] - rb["production_seconds"],
            )
            selected.append(d)
            pairs.append(d)
        stats[a + "-" + b] = {}
        for metric in ("hv_diff", "igd_plus_gain"):
            values, strata = base_effect(selected, metric)
            stats[a + "-" + b][metric] = cluster_summary(values, strata, rng)
        for metric in (
            "hv_pct",
            "igd_gain",
            "flow_gain_pct",
            "bill_gain_pct",
            "coverage_a_b",
            "coverage_b_a",
            "production_seconds_diff",
        ):
            values, _ = base_effect(selected, metric, missing=metric == "hv_pct")
            stats[a + "-" + b][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "n_bases": len(values),
                "undefined_population_pairs": sum(r[metric] is None for r in selected),
                "inference": "descriptive_only",
            }
    primary = [("MIXED-RANDOM", m) for m in ("hv_diff", "igd_plus_gain")]
    secondary = [(a + "-" + b, m) for a, b in contrasts[1:] for m in ("hv_diff", "igd_plus_gain")]
    for family, name in ((primary, "primary2"), (secondary, "secondary18")):
        adjusted = holm([stats[c][m]["p_signflip"] for c, m in family])
        for (c, m), pvalue in zip(family, adjusted, strict=True):
            stats[c][m]["p_holm"] = pvalue
            stats[c][m]["family"] = name
    method_means = {
        m: {
            k: float(np.mean([r[k] for r in rows if r["method"] == m]))
            for k in (
                "hv",
                "hv_reference_fraction",
                "igd_plus",
                "igd",
                "nd_count",
                "flow_gap_pct",
                "bill_gap_pct",
                "direction_within_001",
                "direction_regret_mean",
                "production_seconds",
                "generation_seconds",
                "ssgs_seconds",
                "exact_seconds",
                "verification_seconds",
                "replay_seconds",
                "duplicates",
            )
        }
        for m in METHODS
    }
    subgroup = []
    for field, values in (
        ("jobs", (50, 100, 200)),
        ("tariff", ("H", "T")),
        ("scenario", SCENARIOS),
    ):
        for value in values:
            for a, b in contrasts:
                ss = [r for r in pairs if r["contrast"] == a + "-" + b and r[field] == value]
                subgroup.append(
                    {
                        "group": field,
                        "value": value,
                        "contrast": a + "-" + b,
                        "n_bases": len({r["base_seed"] for r in ss}),
                        **{
                            k: float(np.mean([r[k] for r in ss if r[k] is not None]))
                            if any(r[k] is not None for r in ss)
                            else None
                            for k in (
                                "hv_diff",
                                "igd_plus_gain",
                                "hv_pct",
                                "flow_gain_pct",
                                "bill_gain_pct",
                            )
                        },
                    }
                )
    base_rows = []
    for base in sorted({r["base_seed"] for r in rows}):
        for m in METHODS:
            rr = [r for r in rows if r["base_seed"] == base and r["method"] == m]
            assert len(rr) == 20
            base_rows.append(
                {
                    "base_seed": base,
                    "jobs": rr[0]["jobs"],
                    "scenario": rr[0]["scenario"],
                    "method": m,
                    **{
                        k: float(np.mean([r[k] for r in rr]))
                        for k in ("hv", "igd_plus", "nd_count", "bill_gap_pct", "flow_gap_pct")
                    },
                }
            )
    report = {
        "run_source": SOURCE,
        "canonical": CANONICAL,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "populations": 7920,
        "individuals": 792000,
        "bases": 36,
        "native_inputs": 72,
        "stat_seed": STAT_SEED,
        "bootstrap": 20000,
        "signflips": 100000,
        "primary": primary,
        "secondary": secondary,
        "statistics": stats,
        "method_means": method_means,
        "reference_nd_mean": float(np.mean([len(v["front"]) for v in references.values()])),
        "mixed_sources": {
            m: {
                k: float(np.mean([r[k] for r in sources if r["method"] == m]))
                for k in (
                    "initial_count",
                    "nd_credit",
                    "nd_credit_fraction",
                    "flow_endpoint_credit",
                    "bill_endpoint_credit",
                )
            }
            for m in METHODS[1:]
        },
    }
    report["zero_hv_populations"] = {
        m: sum(r["hv"] == 0 for r in rows if r["method"] == m) for m in METHODS
    }
    for name, table in (
        ("population_metrics", rows),
        ("paired_metrics", pairs),
        ("base_metrics", base_rows),
        ("subgroups_descriptive", subgroup),
        ("mixed_source_credits", sources),
    ):
        write_csv(data / (name + ".csv"), table)
    (data / "references.json").write_text(json.dumps(references, indent=2), encoding="utf-8")
    (data / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (data / "metric_audit.json").write_text(
        json.dumps(
            {
                "independent_hv_populations": audit_hv,
                "independent_igd_plus_populations": audit_igd,
                "all_unique_genotypes_verified": 792000,
                "status": "passed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    figures(out, report, references, fig_inputs, rows, pairs)
    print(
        json.dumps(
            {
                "primary": stats["MIXED-RANDOM"],
                "method_means": method_means,
                "audit_hv": audit_hv,
                "audit_igd": audit_igd,
            }
        ),
        flush=True,
    )


def figures(
    out: Path, report: dict, references: dict, fig_inputs: dict, rows: list, pairs: list
) -> None:
    """Plot aggregate metrics and fixed-seed initialization fronts for every input."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    methods = sorted(METHODS, key=lambda m: report["method_means"][m]["hv"], reverse=True)
    for ax, metric, title in zip(
        axes,
        ("hv_reference_fraction", "igd_plus", "nd_count"),
        ("HV / empirical reference HV", "IGD+ (lower is better)", "Unique nondominated points"),
        strict=True,
    ):
        ax.barh(methods[::-1], [report["method_means"][m][metric] for m in methods[::-1]])
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "method_overview.png", dpi=180)
    plt.close(fig)
    # Fixed seed illustrative figures for all 72 inputs; pooling is reference only.
    for refname, ref in references.items():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
        parts = refname.split("_")
        jobs = int(parts[0])
        base = int(parts[-2])
        tariff = parts[-1]
        pure = max(
            METHODS[1:],
            key=lambda m: float(
                np.mean(
                    [
                        r["hv"]
                        for r in rows
                        if r["jobs"] == jobs
                        and r["base_seed"] == base
                        and r["tariff"] == tariff
                        and r["method"] == m
                    ]
                )
            ),
        )
        for ax, m in zip(axes, ("MIXED", "RANDOM", pure), strict=True):
            front = np.asarray(ref["front"])
            p = np.asarray(fig_inputs[(refname, m)])
            ax.plot(
                front[:, 0],
                front[:, 1],
                color="gray",
                lw=1,
                label="110 populations union reference",
            )
            ax.scatter(p[:, 0], p[:, 1], s=20, label=m + " seed11001")
            outside = sum(max(q) > 1.15 for q in p)
            ax.set_title(m + f"; ND={len(p)}, outside zoom={outside}")
            ax.set_xlabel("Normalized Flow (lower)")
            ax.set_xlim(-0.03, 1.15)
            ax.set_ylim(-0.03, 1.15)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Normalized Bill (lower)")
        axes[0].legend(fontsize=7)
        fig.suptitle(refname + "; seed11001, P100; zoom 0-1.15; pure arm selected descriptively")
        fig.tight_layout()
        fig.savefig(folder / (refname + ".png"), dpi=130)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    matrix = np.array(
        [
            [
                np.mean(
                    [
                        r["hv_diff"]
                        for r in pairs
                        if r["jobs"] == jobs
                        and r["scenario"] == scenario
                        and r["contrast"] == "MIXED-RANDOM"
                    ]
                )
                for scenario in SCENARIOS
            ]
            for jobs in (50, 100, 200)
        ]
    )
    image = ax.imshow(matrix, cmap="RdYlGn")
    fig.colorbar(image, ax=ax, label="MIXED minus RANDOM normalized HV (descriptive)")
    ax.set_xticks(range(6), SCENARIOS, rotation=25, ha="right")
    ax.set_yticks(range(3), (50, 100, 200))
    for i in range(3):
        for j in range(6):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(folder / "mixed_random_cells.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.out)
