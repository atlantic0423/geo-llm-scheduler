"""Read-only frozen initialization cost/quality analysis; no outcome selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from analyze_d5_results import (
    cluster_summary,
    distance,
    holm,
    nondominated,
    normalization,
    write_csv,
)

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.experiments.initial_bill_speed import ARMS, SEEDS, speed_specs
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.io.loaders import load_instance

SEED = 2026101517
RUN_SOURCE = "f89cf2773c741027690e273bdd0702da3732a405d6fd0c386353b1d6177e36ac"


def sha(path: Path) -> str:
    """Hash immutable artifact bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> object:
    """Read an exact UTF-8 artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def hv_oracle(points: list[tuple[float, float]]) -> float:
    """Compute two-dimensional HV independently by horizontal strips."""
    valid = [p for p in points if max(p) <= 1.1]
    edges = sorted({y for _, y in valid} | {1.1})
    return math.fsum(
        (b - a) * max(0.0, 1.1 - min((x for x, y in valid if y <= a), default=1.1))
        for a, b in zip(edges, edges[1:])
    )


def points(population: list[dict]) -> list[tuple[float, float]]:
    """Retain the exact evaluator's original Python regional sum order."""
    return [
        (c["evaluation"]["flow"], c["evaluation"]["tou"] + sum(c["evaluation"]["demand"]))
        for c in population
    ]


def validate(raw: Path) -> tuple[dict, list[dict]]:
    """Verify all 256 runs, identities, hashes, original Archive and strict pairs."""
    manifest = read(raw / "manifest.json")
    assert manifest["source"] == RUN_SOURCE
    assert (
        manifest["kind"] == "formal" and manifest["runs"] == 256 and len(manifest["blocks"]) == 64
    )
    assert sha(raw / "manifest.json") == read(raw / "manifest.sha256.json")["sha256"]
    assert read(raw / "exit.json")["state"] == "complete"
    for name, digest in manifest["files"].items():
        assert sha(raw / name) == digest, name
    rows = []
    expected_specs = {s["key"]: s for s in speed_specs()}
    assert set(manifest["blocks"]) == set(expected_specs)
    for key in manifest["blocks"]:
        spec = read(raw / "specs" / (key + ".json"))
        assert all(spec[k] == v for k, v in expected_specs[key].items())
        instance = load_instance(raw / spec["input"])
        assert len(instance.jobs) == spec["n"] and len(instance.instances) == spec["machines"]
        assert len(instance.regions) == 3
        summaries = []
        for arm in ARMS:
            folder = raw / "runs" / key / arm
            marker = read(folder / "complete.json")
            assert marker["source"] == manifest["source"]
            assert marker["host"] == spec["host"] == manifest["host"]
            assert marker["key"] == key and marker["arm"] == arm
            assert marker["input_sha256"] == spec["input_sha256"] == sha(raw / spec["input"])
            assert marker["spec_sha256"] == sha(raw / "specs" / (key + ".json"))
            assert set(marker["files"]) == {
                "population.json",
                "archive.json",
                "summary.json",
                "config.json",
            }
            for name, evidence in marker["files"].items():
                assert (folder / name).stat().st_size == evidence["size"] and sha(
                    folder / name
                ) == evidence["sha256"], name
            cfg = read(folder / "config.json")
            assert cfg["seed"] == spec["seed"] and cfg["bank_seed"] == spec["bank_seed"]
            assert cfg["population"] == 100 and cfg["share"] == 10 and cfg["trials"] == int(arm[-1])
            assert cfg["accelerated"] == (arm != "REF4")
            pop = read(folder / "population.json")
            candidates = [restore_candidate(c) for c in pop]
            assert len(candidates) == len({c.genotype for c in candidates}) == 100
            assert all(c.evaluation.feasible and not c.evaluation.violations for c in candidates)
            assert Counter(c.origin for c in candidates) == {
                **{f"initial_mode_{i}": 9 for i in range(10)},
                "bill_seed_BILL": 10,
            }
            archive = Archive()
            for c in candidates:
                archive.consider(c)
            assert json.loads(json.dumps([solution(c) for c in archive.members])) == read(
                folder / "archive.json"
            )
            stats = read(folder / "summary.json")
            assert (
                stats["new_candidate_exact"]
                == stats["candidate_independent_exact_verified"]
                == 10 * int(arm[-1])
            )
            assert (
                stats["final_independent_exact_verified"] == 100
                and stats["host"] == manifest["host"]
            )
            assert len(stats["bank_stats"]["observations"]) == 10 * int(arm[-1])
            summaries.append(stats)
            rows.append(
                dict(
                    key=key,
                    arm=arm,
                    **{k: spec[k] for k in ("n", "machines", "scenario", "base_seed", "seed")},
                    points=points(pop),
                    stats=stats,
                )
            )
        for name in ("population.json", "archive.json"):
            assert sha(raw / "runs" / key / "REF4" / name) == sha(
                raw / "runs" / key / "FAST4" / name
            )
        assert len({s["parent_sha256"] for s in summaries}) == 1
        for field in ("observations", "rng_state_sha256"):
            assert summaries[0]["bank_stats"][field] == summaries[1]["bank_stats"][field]
    return manifest, rows


def analyze(raw: Path, out: Path) -> dict:
    """Compute prespecified base-level paired quality and production-time contrasts."""
    assert not out.resolve().is_relative_to(raw.resolve())
    out.mkdir(parents=True, exist_ok=True)
    manifest, records = validate(raw)
    groups = defaultdict(list)
    for record in records:
        groups[record["base_seed"]].append(record)
    assert len(groups) == 32
    metrics = []
    references = {}
    for base, group in sorted(groups.items()):
        assert {(r["arm"], r["seed"]) for r in group} == {(a, s) for a in ARMS for s in SEEDS}
        union = nondominated([p for r in group for p in r["points"]])
        lo, span = normalization(union)

        def norm(ps: list[tuple[float, float]]) -> list[tuple[float, float]]:
            return [tuple(map(float, p)) for p in (np.array(ps) - lo) / span]

        reference = norm(union)
        references[str(base)] = dict(raw_front=union, ideal=lo.tolist(), divisor=span.tolist())
        direction_best = [
            min(w * x + (1 - w) * y for x, y in reference) for w in np.linspace(0.0, 1.0, 11)
        ]
        for record in group:
            front = nondominated(record["points"])
            pp = norm(front)
            hv = hypervolume([p for p in pp if max(p) <= 1.1], (1.1, 1.1))
            igdp = distance(pp, reference, True)
            assert math.isclose(hv, hv_oracle(pp), rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(igdp, igd_plus(pp, reference), rel_tol=1e-12, abs_tol=1e-12)
            hits = sum(
                min(w * x + (1 - w) * y for x, y in pp) <= best + 1e-9
                for w, best in zip(np.linspace(0.0, 1.0, 11), direction_best)
            )
            s = record["stats"]
            metrics.append(
                {
                    k: record[k]
                    for k in ("key", "arm", "n", "machines", "scenario", "base_seed", "seed")
                }
                | dict(
                    hv=hv,
                    igd_plus=igdp,
                    igd=distance(pp, reference, False),
                    nd=len(front),
                    min_flow=min(x for x, y in front),
                    min_bill=min(y for x, y in front),
                    direction_hits=hits,
                    production_wall=s["production_wall_seconds"],
                    production_cpu=s["production_cpu_seconds"],
                    parent_wall=s["parent_wall_seconds"],
                    bank_wall=s["bank_wall_seconds"],
                    selection_wall=s["selection_wall_seconds"],
                    audit_wall=s["audit_wall_seconds"],
                    recorded_preexport_wall=s["completed"] - s["started"],
                    rss=s["max_rss_bytes"],
                )
            )
    write_csv(out / "metrics.csv", metrics)
    lookup = {(r["base_seed"], r["seed"], r["arm"]): r for r in metrics}
    paired = []
    for base, group in sorted(groups.items()):
        n = group[0]["n"]
        for seed in SEEDS:
            ref = lookup[base, seed, "REF4"]
            fast = lookup[base, seed, "FAST4"]
            for arm in ("FAST4", "FAST2", "FAST1"):
                a = lookup[base, seed, arm]
                paired.append(
                    dict(
                        base_seed=base,
                        n=n,
                        seed=seed,
                        arm=arm,
                        hv_diff=a["hv"] - fast["hv"],
                        igd_plus_gain=fast["igd_plus"] - a["igd_plus"],
                        hv_ratio=a["hv"] / fast["hv"],
                        wall_ratio_to_ref=a["production_wall"] / ref["production_wall"],
                        wall_ratio_to_fast=a["production_wall"] / fast["production_wall"],
                        cpu_ratio_to_ref=a["production_cpu"] / ref["production_cpu"],
                    )
                )
    write_csv(out / "paired.csv", paired)
    rng = np.random.default_rng(SEED)
    results = []
    base_rows = []
    for arm in ("FAST2", "FAST1"):
        for metric in ("hv_diff", "igd_plus_gain"):
            values = []
            strata = []
            for base in sorted(groups):
                pair = [p for p in paired if p["base_seed"] == base and p["arm"] == arm]
                value = float(np.mean([p[metric] for p in pair]))
                values.append(value)
                strata.append(pair[0]["n"])
                base_rows.append(
                    dict(base_seed=base, n=pair[0]["n"], arm=arm, metric=metric, value=value)
                )
            results.append(dict(arm=arm, metric=metric, **cluster_summary(values, strata, rng)))
    for row, pvalue in zip(results, holm([r["p_signflip"] for r in results])):
        row["holm_primary_four"] = pvalue
    costs = []
    for field in ("production_cpu", "production_wall"):
        values = []
        strata = []
        for base in sorted(groups):
            values.append(
                float(
                    np.mean(
                        [
                            math.log(
                                lookup[base, s, "FAST4"][field] / lookup[base, s, "REF4"][field]
                            )
                            for s in SEEDS
                        ]
                    )
                )
            )
            strata.append(groups[base][0]["n"])
        summary = cluster_summary(values, strata, rng)
        costs.append(
            dict(
                metric=field,
                geometric_ratio=math.exp(summary["mean"]),
                ratio_ci95=[math.exp(x) for x in summary["ci95"]],
                log_summary=summary,
            )
        )
    aggregates = []
    for n in (None, 20, 50, 100, 200):
        for m in (None, 12, 24):
            if n is None and m is not None:
                continue
            for arm in ARMS:
                selected = [
                    r
                    for r in metrics
                    if r["arm"] == arm
                    and (n is None or r["n"] == n)
                    and (m is None or r["machines"] == m)
                ]
                fields = (
                    "hv",
                    "igd_plus",
                    "nd",
                    "production_wall",
                    "production_cpu",
                    "parent_wall",
                    "bank_wall",
                    "audit_wall",
                    "rss",
                    "direction_hits",
                )
                aggregates.append(
                    dict(
                        n=n or "all",
                        machines=m or "all",
                        arm=arm,
                        runs=len(selected),
                        **{k: float(np.mean([r[k] for r in selected])) for k in fields},
                    )
                )
    write_csv(out / "base_effects.csv", base_rows)
    write_csv(out / "aggregate.csv", aggregates)
    result = dict(
        source=manifest["source"],
        runs=256,
        blocks=64,
        bases=32,
        strict_reference_fast_pairs=64,
        original_final_exact_hash_bound=25600,
        original_trial_exact_hash_bound=7040,
        fresh_all_final_exact=False,
        quality=results,
        cost=costs,
        aggregate=aggregates,
        bootstrap=20000,
        signflips=100000,
        statistics_seed=SEED,
        limitations=[
            "Empirical front, not true Pareto",
            "Initialization only, no final evolutionary benefit claim",
            "Two/one trials change subsequent RNG positions",
            "Exports lack a separate inserted production timer",
        ],
    )
    (out / "statistics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "references.json").write_text(json.dumps(references, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(analyze(args.raw, args.out), ensure_ascii=False))
