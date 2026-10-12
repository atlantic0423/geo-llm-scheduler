"""Analyze the frozen Step-2 matrix; never mutate its source or raw artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
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

ARMS = ("RAW", "TIME16", "TIME64", "STRUCT16", "STRUCT64")
SOURCE = "5bb666874a86ae59e38aa2513a1c4f7ed1fa8fcb"
CANONICAL = "34dd838db2cf62124b967b45ba5049419f6ec96ddab82e6375104d106a1a7f02"
SEED = 2026101102
CONTRASTS = tuple((a, "RAW") for a in ARMS[1:]) + (
    ("TIME16", "STRUCT16"),
    ("TIME64", "STRUCT64"),
    ("TIME64", "TIME16"),
    ("STRUCT64", "STRUCT16"),
)


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _points(population: list[dict]) -> list[tuple[float, float]]:
    return [
        (p["evaluation"]["flow"], p["evaluation"]["tou"] + sum(p["evaluation"]["demand"]))
        for p in population
    ]


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _hv_oracle(points: list) -> float:
    p = [q for q in points if max(q) <= 1.1]
    ys = sorted({q[1] for q in p} | {1.1})
    return sum(
        (b - a) * max(0.0, 1.1 - min((x for x, y in p if y <= a), default=1.1))
        for a, b in zip(ys, ys[1:])
    )


def _analyze(raw: Path, out: Path) -> None:
    assert not out.resolve().is_relative_to(raw.resolve())
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    records, fronts, refs, case_figs = [], {}, {}, {}
    groups = defaultdict(list)
    base_hosts = defaultdict(set)
    checked = 0
    for node, host in enumerate(("b518ece5cd01", "878be0fed7b0")):
        root = raw / f"formal_node{node}"
        m = _read(root / "manifest.json")
        assert m["source_commit"] == SOURCE and m["source_hash"] == CANONICAL
        assert len(m["blocks"]) == 72
        for path, sha in m["input_files"].items():
            assert _sha(root / path) == sha
        for key in m["blocks"]:
            s = _read(root / "specs" / (key + ".json"))
            assert set(s["methods"]) == set(ARMS) and s["seed"] in (11001, 11002)
            case = tuple(s[k] for k in ("jobs", "scenario", "base_seed", "tariff"))
            base_hosts[s["jobs"], s["base_seed"]].add(host)
            parent = _read(root / s["parent_population"])
            assert len(parent) == 100
            for arm in ARMS:
                folder = root / "runs" / key / arm
                marker, summary = _read(folder / "complete.json"), _read(folder / "summary.json")
                assert marker["host"] == summary["host"] == host
                assert marker["spec_sha256"] == _sha(root / "specs" / (key + ".json"))
                for path, sha in marker["files"].items():
                    assert _sha(folder / path) == sha
                for field in (
                    "source_commit",
                    "source_hash",
                    "runner_sha256",
                    "parent_population_sha256",
                ):
                    assert summary[field] == s[field]
                pop, obs = _read(folder / "population.json"), _read(folder / "observations.json")
                assert len(pop) == len(obs) == summary["selected_exact_verified"] == 100
                assert summary["exact"] == summary["independent_exact_verified"]
                gains, flow_gains, tou_gains, demand_gains = [], [], [], []
                for a, b, o in zip(parent, pop, obs, strict=True):
                    ea, eb = a["evaluation"], b["evaluation"]
                    bill_a, bill_b = ea["tou"] + sum(ea["demand"]), eb["tou"] + sum(eb["demand"])
                    assert (
                        eb["feasible"]
                        and eb["flow"] <= ea["flow"] + 1e-8
                        and bill_b <= bill_a + 1e-8
                    )
                    assert o["parent_flow"] == ea["flow"] and np.isclose(o["parent_bill"], bill_a)
                    assert o["exact"] == o["independent_exact_verified"] == len(o["ledger"])
                    assert o["exact"] <= o["exact_budget"]
                    if arm == "RAW":
                        assert a == b
                    if arm.startswith("TIME"):
                        assert a["genotype"] == b["genotype"] and eb["flow"] == ea["flow"]
                        sa, sb = a["schedule"]["starts"], b["schedule"]["starts"]
                        changed = [i for i, (x, y) in enumerate(zip(sa, sb, strict=True)) if x != y]
                        assert len(changed) <= 1 and all(
                            i % 2 == 0 and sb[i] >= sa[i] for i in changed
                        )
                        assert o["ssgs"] == 0
                    gains.append(100 * (bill_a - bill_b) / bill_a)
                    flow_gains.append(100 * (ea["flow"] - eb["flow"]) / ea["flow"])
                    tou_gains.append(ea["tou"] - eb["tou"])
                    demand_gains.append(sum(ea["demand"]) - sum(eb["demand"]))
                    checked += 1
                groups[case].append(
                    dict(
                        spec=s,
                        arm=arm,
                        summary=summary,
                        points=_points(pop),
                        explored=_points(_read(folder / "archive.json")),
                        unique_genes=len({json.dumps(p["genotype"], sort_keys=True) for p in pop}),
                        accepted_fraction=summary["accepted"] / 100,
                        mean_parent_bill_gain_pct=float(np.mean(gains)),
                        mean_parent_flow_gain_pct=float(np.mean(flow_gains)),
                        mean_parent_tou_gain=float(np.mean(tou_gains)),
                        mean_parent_demand_gain=float(np.mean(demand_gains)),
                        cap_fraction=float(np.mean([o["cap_truncated"] for o in obs])),
                        proposal_seconds=sum(o["proposal_seconds"] for o in obs),
                        exact_seconds=sum(o["seconds"].get("exact", 0.0) for o in obs),
                        ssgs_seconds=sum(o["seconds"].get("ssgs", 0.0) for o in obs),
                    )
                )
    assert (
        len(groups) == 72
        and len(base_hosts) == 36
        and all(len(v) == 1 for v in base_hosts.values())
    )
    for case, members in sorted(groups.items()):
        assert {(r["arm"], r["spec"]["seed"]) for r in members} == {
            (a, s) for a in ARMS for s in (11001, 11002)
        } and len(members) == 10
        union = nondominated([p for r in members for p in r["points"]])
        lo, span = normalization(union)
        ref = ((np.array(union) - lo) / span).tolist()
        name = "_".join(map(str, case))
        refhv = hypervolume([tuple(p) for p in ref], (1.1, 1.1))
        refs[name] = dict(
            ideal=lo.tolist(), divisor=span.tolist(), front=ref, raw_front=union, hv=refhv
        )
        case_figs[name] = {
            a: nondominated([p for r in members if r["arm"] == a for p in r["points"]])
            for a in ARMS
        }
        for r in members:
            s, su = r["spec"], r["summary"]
            nd = nondominated(r["points"])
            norm = ((np.array(nd) - lo) / span).tolist()
            hv = hypervolume([tuple(p) for p in norm if max(p) <= 1.1], (1.1, 1.1))
            assert np.isclose(hv, _hv_oracle(norm), atol=1e-12, rtol=1e-12)
            ip = distance(norm, ref, True)
            assert np.isclose(ip, igd_plus(norm, ref), atol=1e-12, rtol=1e-12)
            fronts[s["key"], r["arm"]] = norm
            ex = ((np.array(nondominated(r["explored"])) - lo) / span).tolist()
            row = {k: s[k] for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")}
            row.update(
                arm=r["arm"],
                host=su["host"],
                hv=hv,
                hv_reference_fraction=hv / refhv,
                igd_plus=ip,
                igd=distance(norm, ref, False),
                nd_count=len(nd),
                unique_genes=r["unique_genes"],
                min_flow=min(p[0] for p in nd),
                min_bill=min(p[1] for p in nd),
                flow_gap_pct=100 * (min(p[0] for p in nd) / lo[0] - 1),
                bill_gap_pct=100 * (min(p[1] for p in nd) / lo[1] - 1),
                production_seconds=su["production_seconds"],
                verification_seconds=su["verification_seconds"],
                measured_elapsed=su["elapsed"],
                exact=su["exact"],
                ssgs=su["ssgs"],
                explored_nd=len(ex),
                explored_hv=hypervolume([tuple(p) for p in ex if max(p) <= 1.1], (1.1, 1.1)),
                **{
                    k: r[k]
                    for k in (
                        "accepted_fraction",
                        "mean_parent_bill_gain_pct",
                        "mean_parent_flow_gain_pct",
                        "mean_parent_tou_gain",
                        "mean_parent_demand_gain",
                        "cap_fraction",
                        "proposal_seconds",
                        "exact_seconds",
                        "ssgs_seconds",
                    )
                },
            )
            records.append(row)
    assert len(records) == 720 and checked == 72000
    _csv(data / "run_metrics.csv", records)
    byrun = {(r["key"], r["arm"]): r for r in records}
    pairs, stats = [], {}
    primary, secondary = [], []
    rng = np.random.default_rng(SEED)
    for index, (a, b) in enumerate(CONTRASTS):
        label = a + "-" + b
        these = []
        for key in sorted({r["key"] for r in records}):
            x, y = byrun[key, a], byrun[key, b]
            row = {k: x[k] for k in ("key", "jobs", "scenario", "base_seed", "tariff", "seed")}
            row.update(
                contrast=label,
                hv_diff=x["hv"] - y["hv"],
                igd_plus_gain=y["igd_plus"] - x["igd_plus"],
                coverage_a_b=coverage(fronts[key, a], fronts[key, b]),
                coverage_b_a=coverage(fronts[key, b], fronts[key, a]),
                min_bill_gain_pct=100 * (y["min_bill"] - x["min_bill"]) / y["min_bill"],
                min_flow_gain_pct=100 * (y["min_flow"] - x["min_flow"]) / y["min_flow"],
            )
            these.append(row)
        clusters = defaultdict(list)
        for r in these:
            clusters[r["jobs"], r["base_seed"]].append(r)
        keys = sorted(clusters)
        assert len(keys) == 36 and all(len(v) == 4 for v in clusters.values())
        stats[label] = {}
        for metric in ("hv_diff", "igd_plus_gain"):
            effects = [float(np.mean([r[metric] for r in clusters[k]])) for k in keys]
            stats[label][metric] = cluster_summary(effects, [k[0] for k in keys], rng)
            (primary if index < 4 else secondary).append((label, metric))
        pairs.extend(these)
    for family in (primary, secondary):
        adjusted = holm([stats[c][m]["p_signflip"] for c, m in family])
        for (c, m), p in zip(family, adjusted, strict=True):
            stats[c][m]["p_holm"] = p
    _csv(data / "paired_metrics.csv", pairs)
    numeric = [
        k
        for k, v in records[0].items()
        if isinstance(v, (float, int)) and k not in ("jobs", "seed", "base_seed")
    ]
    overall = {
        a: {k: float(np.mean([r[k] for r in records if r["arm"] == a])) for k in numeric}
        for a in ARMS
    }
    subgroup = []
    for field in ("tariff", "jobs", "scenario"):
        for value in sorted({r[field] for r in records}, key=str):
            for a in ARMS:
                rows = [r for r in records if r[field] == value and r["arm"] == a]
                subgroup.append(
                    dict(
                        group=field,
                        value=value,
                        arm=a,
                        n_populations=len(rows),
                        **{k: float(np.mean([r[k] for r in rows])) for k in numeric},
                    )
                )
    _csv(data / "subgroups.csv", subgroup)
    report = dict(
        source_commit=SOURCE,
        canonical=CANONICAL,
        stat_seed=SEED,
        independent_bases=36,
        populations=720,
        selection="one Flow-safe strict Bill improvement per frozen parent; population front primary, explored archive supplementary",
        overall=overall,
        statistics=stats,
        primary=primary,
        secondary=secondary,
        audit=dict(
            hv_independent=720,
            igd_plus_independent=720,
            selected_safety_checks=checked,
            base_same_host=36,
        ),
        verification_status="ANALYZED; original worker exact evidence bound by verified artifact hashes, no fresh full solver reevaluation this analysis",
    )
    (data / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (data / "references.json").write_text(json.dumps(refs, indent=2), encoding="utf-8")
    _figures(out, case_figs, records)
    print(
        json.dumps(
            {"populations": 720, "audits": report["audit"], "overall": overall, "statistics": stats}
        ),
        flush=True,
    )


def _figures(out: Path, cases: dict, records: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    colors = {
        "RAW": "black",
        "TIME16": "#69b3a2",
        "TIME64": "#008c95",
        "STRUCT16": "#efab45",
        "STRUCT64": "#c4461c",
    }
    for name, fronts in cases.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        for arm in ARMS:
            p = np.array(fronts[arm])
            ax.plot(p[:, 0], p[:, 1], ".-", label=arm, color=colors[arm], alpha=0.85)
        ax.set(
            xlabel="Total Flow Time (seconds)",
            ylabel="Electricity Bill (CNY)",
            title=name + " / two-seed empirical fronts",
        )
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / (name + ".png"), dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field, title in zip(
        axes,
        ("hv_reference_fraction", "mean_parent_bill_gain_pct", "production_seconds"),
        (
            "HV / common initial envelope",
            "Bill saving per original individual (%)",
            "Production seconds per population",
        ),
        strict=True,
    ):
        ax.boxplot(
            [[r[field] for r in records if r["arm"] == a] for a in ARMS],
            tick_labels=ARMS,
            showfliers=False,
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "overview.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _analyze(args.raw, args.out)
