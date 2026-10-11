"""Plot and quantify frozen initial populations against the matched D8 BASE archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_d8_results import CANONICAL, SOURCE, coverage, distance, nondominated, write_csv
from initial_trace_support import sample_population

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.metrics import hypervolume
from geo_llm_scheduler.io.loaders import load_instance


def objective(row: dict) -> tuple[float, float]:
    """Read exact Flow/Bill decomposition without estimating or rescheduling."""
    e = row["evaluation"]
    if not e["feasible"]:
        raise ValueError("Initial observation contains an infeasible candidate")
    return float(e["flow"]), float(e["tou"] + sum(e["demand"]))


def construction_problem_path(root: Path, spec: dict) -> Path:
    """D8 deliberately generated one H-based population for both tariff versions."""
    if spec["tariff"] not in ("H", "T"):
        raise ValueError("Unknown frozen D8 tariff")
    return root / "inputs" / f"{spec['jobs']}_{spec['base_seed']}_H.json"


def front_metrics(initial: list, final: list, lo: np.ndarray, span: np.ndarray) -> dict:
    """Use one externally frozen normalization for the paired initial/final sets."""
    if not initial or not final:
        raise ValueError("Matched front comparison needs two nonempty sets")
    a = [tuple(map(float, p)) for p in (np.asarray(initial) - lo) / span]
    b = [tuple(map(float, p)) for p in (np.asarray(final) - lo) / span]
    ha = hypervolume([p for p in a if max(p) <= 1.1], (1.1, 1.1))
    hb = hypervolume([p for p in b if max(p) <= 1.1], (1.1, 1.1))
    gaps = [
        min(max(w * p[0], (1 - w) * p[1]) for p in a)
        - min(max(w * p[0], (1 - w) * p[1]) for p in b)
        for w in np.linspace(0, 1, 11)
    ]
    return {
        "initial_hv": ha,
        "final_hv": hb,
        "hv_retention_pct": 100 * ha / hb if hb else None,
        "igd_plus_to_BASE_final": distance(a, b, True),
        "igd_to_BASE_final": distance(a, b, False),
        "initial_covers_final": coverage(a, b),
        "final_covers_initial": coverage(b, a),
        "flow_gap_pct": (
            100 * (min(p[0] for p in initial) / min(p[0] for p in final) - 1)
            if min(p[0] for p in final)
            else None
        ),
        "bill_gap_pct": (
            100 * (min(p[1] for p in initial) / min(p[1] for p in final) - 1)
            if min(p[1] for p in final)
            else None
        ),
        "mean_direction_gap": float(np.mean(gaps)),
        "max_direction_gap": float(max(gaps)),
    }


def analyze(raw: Path, d8_analysis: Path, output: Path) -> dict:
    """Verify every matched initial artifact, draw all cases and retain repeated-seed limits."""
    if output.resolve().is_relative_to(raw.resolve()):
        raise ValueError("Keep analysis outside immutable raw")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    references = json.loads((d8_analysis / "references.json").read_text())
    rows = []
    mode_rows = []
    cases: dict[tuple, list] = defaultdict(list)
    for node in (0, 1):
        for folder in sorted((raw / "initial_observation" / f"node{node}").iterdir()):
            if not folder.is_dir():
                continue
            marker = json.loads((folder / "complete.json").read_text())
            if set(marker["files"]) != {"population.json", "archive.json", "summary.json"}:
                raise ValueError("Unexpected matched initial artifact set")
            for name, digest in marker["files"].items():
                if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("Matched initial observation hash mismatch")
            s = json.loads((folder / "summary.json").read_text())
            if (
                s["source_commit"] != SOURCE
                or s["source_hash"] != CANONICAL
                or s["population"] != 100
                or s["independent_exact_verified"] != 100
            ):
                raise ValueError("Unexpected initial observation provenance")
            matches = [raw / f"formal_node{n}" / "specs" / (s["key"] + ".json") for n in range(4)]
            matches = [p for p in matches if p.exists()]
            if len(matches) != 1:
                raise ValueError("Initial observation must map to exactly one BASE run")
            spec = json.loads(matches[0].read_text())
            if (
                spec["factor_mask"] != 0
                or spec["instance_hash"] != s["instance_hash"]
                or spec["initial_hash"] != s["initial_hash"]
            ):
                raise ValueError("Wrong matched input/seed/arm")
            root = matches[0].parent.parent
            population = json.loads((folder / "population.json").read_text())
            genes = json.loads((root / spec["initial"]).read_text())
            if [p["genotype"] for p in population] != genes:
                raise ValueError(
                    "Observed genotypes differ from the original frozen initialization"
                )
            final_path = root / "runs" / s["key"] / "archive.json"
            if (
                hashlib.sha256(final_path.read_bytes()).hexdigest()
                != s["original_BASE_archive_sha256"]
            ):
                raise ValueError("Changed matched BASE archive")
            initial = [objective(p) for p in population]
            final = nondominated([objective(p) for p in json.loads(final_path.read_text())])
            front = nondominated(initial)
            trace = sample_population(
                load_instance(construction_problem_path(root, spec)),
                Config(seed=s["algorithm_seed"]),
                "MIXED",
            )
            if [{"ms": list(g.ms), "os": list(g.os)} for g in trace.genotypes] != genes:
                raise ValueError("Construction-source replay differs from frozen genotypes")
            sources: dict[tuple, set] = defaultdict(set)
            for point, mode in zip(initial, trace.modes, strict=True):
                if point in front:
                    sources[point].add(mode)
            for mode in range(10):
                indexes = [i for i, m in enumerate(trace.modes) if m == mode]
                mode_rows.append(
                    {
                        **{
                            k: s[k]
                            for k in ("key", "jobs", "base_seed", "tariff", "algorithm_seed")
                        },
                        "mode": mode,
                        "individuals": len(indexes),
                        "ND_candidates": sum(initial[i] in sources for i in indexes),
                        "unique_ND_split_credit": sum(
                            1 / len(labels) for labels in sources.values() if mode in labels
                        ),
                    }
                )
            key = (s["jobs"], s["base_seed"], s["tariff"])
            ref = references[str(key)]
            stats = front_metrics(
                front, final, np.asarray(ref["ideal"]), np.asarray(ref["divisor"])
            )
            if stats["hv_retention_pct"] is not None and stats["hv_retention_pct"] > 100 + 1e-8:
                raise ValueError(
                    "Final archive lost dominated initial evidence or normalization mismatch"
                )
            row = {
                k: s[k] for k in ("key", "jobs", "base_seed", "tariff", "algorithm_seed", "host")
            }
            row.update(
                population=100,
                construction_tariff="H",
                unique_genotypes=len(
                    {json.dumps(p["genotype"], sort_keys=True) for p in population}
                ),
                unique_objectives=len(set(initial)),
                initial_ND_objectives=len(front),
                initial_archive_candidates=s["archive_size"],
                final_ND_objectives=len(final),
                exact_verified=100,
                observation_elapsed=s["elapsed"],
                **stats,
            )
            rows.append(row)
            cases[key].append(
                {"seed": s["algorithm_seed"], "initial": initial, "front": front, "final": final}
            )
    if len(rows) != 256 or len(cases) != 128:
        raise ValueError("Incomplete matched initialization matrix")
    for key, members in sorted(cases.items()):
        if {r["seed"] for r in members} != {1101, 2202}:
            raise ValueError("Incomplete repeated initialization seeds")
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in members:
            points = np.asarray(r["initial"])
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=10,
                alpha=0.22,
                label=f"Initial 100 (seed {r['seed']})",
            )
        initial_union = np.asarray(nondominated([p for r in members for p in r["front"]]))
        final_union = np.asarray(nondominated([p for r in members for p in r["final"]]))
        ax.plot(
            initial_union[:, 0],
            initial_union[:, 1],
            "o-",
            markersize=4,
            color="#e68613",
            label="Initial empirical ND union",
        )
        ax.plot(
            final_union[:, 0],
            final_union[:, 1],
            "-",
            linewidth=1.5,
            color="#161616",
            label="BASE final empirical ND union",
        )
        ax.set(
            xlabel="Total Flow Time (seconds)",
            ylabel="Electricity Bill (CNY)",
            title=f"{key[0]} jobs / base {key[1]} / tariff {key[2]}",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"front_{key[0]}_{key[1]}_{key[2]}.png", dpi=160)
        plt.close(fig)
    fields = (
        "unique_objectives",
        "initial_ND_objectives",
        "final_ND_objectives",
        "hv_retention_pct",
        "igd_plus_to_BASE_final",
        "flow_gap_pct",
        "bill_gap_pct",
        "mean_direction_gap",
        "max_direction_gap",
        "initial_covers_final",
        "final_covers_initial",
    )
    means = {
        field: float(np.mean([r[field] for r in rows if r[field] is not None])) for field in fields
    }
    report = {
        "populations": 256,
        "individuals": 25600,
        "independent_bases": 64,
        "instance_versions": 128,
        "seeds_per_case": 2,
        "source_commit": SOURCE,
        "scope": "Matched unchanged initialization versus evolved BASE; no random/pure-method controls or new stress cases yet",
        "shared_initialization": "D8 generated each initial population under H and reused identical genes under H/T; these observations do not test native T-price-aware construction",
        "means": means,
        "subgroups": {
            f"{size}_{t}": {
                f: float(
                    np.mean(
                        [
                            r[f]
                            for r in rows
                            if r["jobs"] == size and r["tariff"] == t and r[f] is not None
                        ]
                    )
                )
                for f in fields
            }
            for size in (50, 100)
            for t in ("H", "T")
        },
        "ranges": {
            f: [
                float(min(r[f] for r in rows if r[f] is not None)),
                float(max(r[f] for r in rows if r[f] is not None)),
            ]
            for f in fields
        },
        "all_initial_genotypes_and_exact_evidence_verified": True,
        "all_construction_source_replays_match": True,
        "mixed_pool_source_contributions": {
            str(mode): {
                field: float(np.mean([r[field] for r in mode_rows if r["mode"] == mode]))
                for field in ("individuals", "ND_candidates", "unique_ND_split_credit")
            }
            for mode in range(10)
        },
        "evolutionary_compute_budget_is_additional": True,
        "figures": 128,
    }
    write_csv(output / "initial_run_metrics.csv", rows)
    write_csv(output / "initial_source_metrics.csv", mode_rows)
    (output / "initial_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field, label in zip(
        axes,
        ("hv_retention_pct", "flow_gap_pct", "bill_gap_pct"),
        (
            "Initial / final HV (%)",
            "Initial Flow endpoint gap (%)",
            "Initial Bill endpoint gap (%)",
        ),
        strict=True,
    ):
        ax.boxplot(
            [[r[field] for r in rows if r["jobs"] == j] for j in (50, 100)],
            tick_labels=["50 jobs", "100 jobs"],
        )
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "initial_quality_overview.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [
        "FCFS load",
        "FCFS random",
        "SPT load",
        "SPT random",
        "LPT load",
        "LPT random",
        "Random load",
        "Random",
        "KV aware",
        "Energy aware",
    ]
    ax.bar(
        labels,
        [
            report["mixed_pool_source_contributions"][str(m)]["unique_ND_split_credit"]
            for m in range(10)
        ],
        color="#4279a9",
    )
    ax.set(
        ylabel="Mean unique initial ND points (shared credit)",
        title="Contribution within the existing mixed population",
    )
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "initial_source_contributions.png", dpi=160)
    plt.close(fig)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--d8-analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(analyze(a.raw, a.d8_analysis, a.output), ensure_ascii=False), flush=True)
