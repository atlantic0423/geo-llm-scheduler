"""Validate and analyze the preregistered auxiliary walk experiment without changing raw data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.experiments.runner import digest
from research.auxiliary_walk_20261016.campaign import stage_specs
from research.auxiliary_walk_20261016.common import sha, validate_complete
from research.auxiliary_walk_20261016.selection import ELIGIBLE, nondominated

SEED = 2026101603
SOURCE = "867c95e866b13fa3459451eb54713292b112ee03"
CANONICAL = "9a56d536978a9f6b5b62fa31f4b9b49c084725ac7030c36596b6a47b545317ba"
STAGES = {"screen": (5760, 80), "confirm": (2560, 160), "long": (1280, 80)}


def read(path: Path) -> dict:
    """Read UTF-8 JSON evidence, rejecting placeholder files."""
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    """Write deterministic JSON analysis outside the immutable evidence tree."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def table(path: Path, rows: list[dict]) -> None:
    """Export consistently ordered machine readable rows."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quality(points: list, front: list) -> tuple[float, float]:
    """Compute exact 2D HV and vectorized IGD+ in bounded reference chunks."""
    a = np.asarray(points, dtype=float)
    r = np.asarray(front, dtype=float)
    if not len(a) or not len(r) or not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError("Empty or nonfinite quality data")
    distances = []
    for start in range(0, len(r), 128):
        delta = np.maximum(a[None, :, :] - r[start : start + 128, None, :], 0)
        distances.extend(np.sqrt((delta * delta).sum(axis=2)).min(axis=1).tolist())
    return hypervolume([tuple(p) for p in a], (1.1, 1.1)), math.fsum(distances) / len(r)


def integral_hv(points: list) -> float:
    """Independently integrate vertical strips, including duplicate x and dominated points."""
    breaks = sorted({p[0] for p in points} | {1.1})
    return math.fsum(
        (right - left) * (1.1 - min(p[1] for p in points if p[0] <= left))
        for left, right in zip(breaks, breaks[1:])
    )


def holm(pvalues: list[float]) -> list[float]:
    """Apply Holm's step down correction while retaining contrast order."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("Invalid p-value")
    result = [0.0] * len(pvalues)
    previous = 0.0
    for rank, i in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        previous = max(previous, (len(pvalues) - rank) * pvalues[i])
        result[i] = min(previous, 1.0)
    return result


def inference(
    values: list[float],
    strata: list[int],
    rng: np.random.Generator,
    bootstrap: int = 20000,
    signflips: int = 100000,
) -> dict:
    """Resample parent clusters within size; test the symmetric paired null in batches."""
    if not values or len(values) != len(strata) or not all(map(math.isfinite, values)):
        raise ValueError("Invalid parent clusters")
    a = np.asarray(values)
    sums = np.zeros(bootstrap)
    for size in sorted(set(strata)):
        group = a[np.asarray(strata) == size]
        sums += group[rng.integers(len(group), size=(bootstrap, len(group)))].sum(axis=1)
    observed = abs(float(a.mean()))
    extreme = 0
    for start in range(0, signflips, 4096):
        signs = 2 * rng.integers(0, 2, size=(min(4096, signflips - start), len(a))) - 1
        extreme += int((np.abs((a * signs).mean(axis=1)) >= observed - 1e-14).sum())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {
        "parents": len(a),
        "mean": float(a.mean()),
        "ci95": np.quantile(sums / len(a), [0.025, 0.975]).tolist(),
        "p": (extreme + 1) / (signflips + 1),
        "positive": int((a > 1e-12).sum()),
        "negative": int((a < -1e-12).sum()),
        "tied": int((np.abs(a) <= 1e-12).sum()),
        "paired_standardized": float(a.mean()) / sd if sd else None,
    }


def validate(start: Path, capture: Path, out: Path) -> tuple[list[dict], dict, dict]:
    """Bind every run, input, configuration, host, Q count and artifact to frozen evidence."""
    control = capture / "control/formal"
    plan, selected = read(control / "plan.json"), read(control / "selection.json")
    if plan["source"] != {"head": SOURCE, "canonical": CANONICAL, "dirty": False}:
        raise ValueError("Unexpected running source")
    if sha(control / "plan.json") != read(control / "plan_hash.json")["sha256"]:
        raise ValueError("Plan hash changed")
    specs = [
        s
        for stage in STAGES
        for s in stage_specs(plan, stage, None if stage == "screen" else selected)
    ]
    manifests = [read(capture / f"server_node{node}_manifest.json") for node in (0, 1)]
    server = {(node, f["path"]): f for node, m in enumerate(manifests) for f in m["files"]}
    expected_paths = {
        (s["node"], s["id"] + "/" + name)
        for s in specs
        for name in (
            "complete.json",
            "summary.json",
            "diagnostics.json",
            "q.json",
            "solutions.json.gz",
        )
    }
    if set(server) != expected_paths or len(specs) != 9600:
        raise ValueError("Server manifest differs from the complete frozen run matrix")
    input_paths = {s["input"]: s["input_sha256"] for s in specs}
    for name, expected in input_paths.items():
        if sha(control / name) != expected:
            raise ValueError("Input changed")
    if len(input_paths) != 640:
        raise ValueError("Wrong input count")

    def one(spec: dict) -> dict:
        root = start / "raw" / f"formal_node{spec['node']}" / spec["id"]
        marker = validate_complete(root, spec, plan["source"])
        for name in (*marker["files"], "complete.json"):
            p = root / name
            entry = server[(spec["node"], spec["id"] + "/" + name)]
            if p.stat().st_size != entry["bytes"] or sha(p) != entry["sha256"]:
                raise ValueError("Server/local mismatch")
        summary, q = read(root / "summary.json"), read(root / "q.json")
        if digest(summary["spec"]) != digest(spec) or summary["host"] != spec["host"]:
            raise ValueError("Wrong executed spec/host")
        if summary["source"] != plan["source"] or summary["population"] != 100:
            raise ValueError("Wrong summary source/population")
        steps = summary["counts"]["auxiliary_steps"]
        if not all(
            len(q[k]) == 42 and all(len(row) == 8 for row in q[k])
            for k in ("q", "updates", "selections")
        ):
            raise ValueError("Wrong Q dimensions")
        if (
            sum(map(sum, q["updates"])) != steps
            or sum(map(sum, q["selections"])) != steps
            or sum(q["visits"]) != steps
        ):
            raise ValueError("Lost online Q updates")
        if not np.isfinite(q["q"]).all():
            raise ValueError("Nonfinite Q")
        diag = read(root / "diagnostics.json")
        if diag["counts"] != summary["diagnostic_counts"]:
            raise ValueError("Diagnostic counters changed")
        summary["anytime"] = diag["anytime"]
        summary["completed_at"] = marker["completed_at"]
        return summary

    with ThreadPoolExecutor(max_workers=8) as pool:
        summaries = list(pool.map(one, specs))
    parents = defaultdict(set)
    for s in summaries:
        parents[s["spec"]["parent"]].add(s["host"])
    if len(parents) != 320 or any(len(v) != 1 for v in parents.values()):
        raise ValueError("Parent host pairing broken")
    screen = [s for s in summaries if s["spec"]["stage"] == "screen"]
    if (
        digest(
            [
                digest({k: v for k, v in s.items() if k not in ("anytime", "completed_at")})
                for s in screen
            ]
        )
        != selected["screening_index_hash"]
    ):
        raise ValueError("Selection was based on a different screening index")
    last_screen = max(s["completed_at"] for s in screen)
    if selected["frozen_at"] < last_screen:
        raise ValueError("Premature selection")
    receipt = {
        "status": "accepted",
        "runs": len(summaries),
        "parents": len(parents),
        "inputs": len(input_paths),
        "files": len(server),
        "bytes": sum(v["bytes"] for v in server.values()),
        "source": plan["source"],
        "plan_sha256": sha(control / "plan.json"),
        "worker_independent_final_exact_bound_by_sha": sum(
            s["independent_exact"] for s in summaries
        ),
        "fresh_all_solutions_exact_this_pass": False,
        "completion_epoch": max(m["exit"]["time"] for m in manifests),
        "stage_completions": {
            stage: max(s["completed_at"] for s in summaries if s["spec"]["stage"] == stage)
            for stage in STAGES
        },
        "max_overshoot": max(s["overshoot"] for s in summaries),
        "max_rss_gib": max(s["peak_rss_gib"] for s in summaries),
    }
    write(out / "acceptance.json", receipt)
    print(json.dumps(receipt), flush=True)
    return summaries, plan, selected


def analyze(summaries: list[dict], selected: dict, out: Path) -> dict:
    """Separate screening and two held-out families and export complete mechanism accounting."""
    groups = defaultdict(list)
    for summary in summaries:
        groups[(summary["spec"]["stage"], summary["spec"]["input"])].append(summary)
    metrics, anytime, crosschecks = [], [], 0
    for group_index, group in enumerate(groups.values()):
        points = [p for s in group for p in s["archive_points"]]
        lower = np.min(points, axis=0)
        spans = np.maximum(np.max(points, axis=0) - lower, 1e-9)

        def norm(items: list) -> list:
            return ((np.asarray(items) - lower) / spans).tolist()

        front = nondominated([tuple(p) for p in norm(points)])
        for index, s in enumerate(group):
            spec = s["spec"]
            normalized = norm(s["archive_points"])
            hv, igd = quality(normalized, front)
            if not math.isclose(hv, integral_hv(normalized), abs_tol=1e-12):
                raise ValueError("Independent HV disagreement")
            if index == 0:
                if not math.isclose(igd, igd_plus(normalized, front), abs_tol=1e-12):
                    raise ValueError("Independent IGD+ disagreement")
                crosschecks += 1
            counts, d = s["counts"], s["diagnostic_counts"]
            episodes = d["episodes"]
            steps = counts["auxiliary_steps"]
            regressions = d.get("fixed_context_regressions", 0)
            row = {
                k: spec[k]
                for k in (
                    "id",
                    "stage",
                    "parent",
                    "case",
                    "jobs",
                    "pressure",
                    "version",
                    "seed",
                    "node",
                    "arm",
                )
            }
            row.update(
                topology="x".join(map(str, spec["topology"])),
                hv=hv,
                igd=igd,
                flow=min(p[0] for p in s["archive_points"]),
                bill=min(p[1] for p in s["archive_points"]),
                seconds=spec["config"]["seconds"],
                production_seconds=s["production_seconds"],
                initialization_seconds=s["initialization_seconds"],
                overshoot=s["overshoot"],
                audit_seconds=s["audit_seconds"],
                export_seconds=s["export_seconds"],
                observer_seconds=s["observer_seconds"],
                offspring=s["offspring"],
                exact=counts["exact"],
                episodes=episodes,
                steps=steps,
                depth=steps / episodes,
                effective_per_step=d.get("effective", 0) / steps,
                moves=d.get("accepted", 0),
                regressions=regressions,
                breakthroughs=sum(v for k, v in d.items() if k.startswith("breakthrough_after:")),
                position_recoveries=sum(
                    v for k, v in d.items() if k.startswith("position_recovery_after:")
                ),
                unrecovered=d.get("unrecovered_regressions", 0),
                unrecovered_positions=d.get("unrecovered_positions", 0),
                revisit=d.get("revisit", 0),
                zero_steps=d.get("physical_zero_steps", 0),
                last_best_gap=d.get("last_best_gap_sum", 0) / episodes,
                requested=sum(
                    int(k.split(":")[1]) * v for k, v in d.items() if k.startswith("budget:")
                ),
                archive_size=s["archive_size"],
            )
            metrics.append(row)
            for tick, snapshot in enumerate(s["anytime"]):
                # Earlier archives can lie outside the final union's range; keep a separate
                # reference covering all observed snapshots rather than clipping points.
                anytime.append(
                    {
                        "id": spec["id"],
                        "stage": spec["stage"],
                        "parent": spec["parent"],
                        "arm": spec["arm"],
                        "tick": tick,
                        "elapsed": snapshot["elapsed"],
                        "fraction": snapshot["elapsed"] / spec["config"]["seconds"],
                        "points": snapshot["points"],
                        "input": spec["input"],
                    }
                )
        if group_index % 80 == 0:
            print(f"metrics {group_index}/{len(groups)}", flush=True)
    table(out / "run_metrics.csv", metrics)
    write(
        out / "metric_crosschecks.json",
        {"hv": len(metrics), "igd_plus": crosschecks, "tolerance": 1e-12},
    )
    parent_groups = defaultdict(list)
    for row in metrics:
        parent_groups[(row["stage"], row["parent"], row["arm"])].append(row)
    pmetrics = []
    numerical = [
        k
        for k, v in metrics[0].items()
        if isinstance(v, (float, int)) and k not in ("version", "seed", "node", "jobs")
    ]
    for (stage, parent, arm), rows in sorted(parent_groups.items()):
        if len(rows) != 4 or {(r["version"], r["seed"]) for r in rows} != {
            (v, seed) for v in (0, 1) for seed in (161001, 171001)
        }:
            raise ValueError("Incomplete parent/arm cluster")
        pmetrics.append(
            {
                "stage": stage,
                "parent": parent,
                "arm": arm,
                **{k: rows[0][k] for k in ("jobs", "pressure", "topology", "node")},
                **{k: float(np.mean([r[k] for r in rows])) for k in numerical},
            }
        )
    table(out / "parent_metrics.csv", pmetrics)
    lookup = {(r["stage"], r["parent"], r["arm"]): r for r in pmetrics}
    ranks = dict.fromkeys(ELIGIBLE, 0.0)
    for parent in sorted({r["parent"] for r in pmetrics if r["stage"] == "screen"}):
        for a in ELIGIBLE:
            for metric, sign in (("hv", -1), ("igd", 1)):
                x = lookup[("screen", parent, a)][metric] * sign
                ranks[a] += sum(lookup[("screen", parent, b)][metric] * sign < x for b in ELIGIBLE)
                ranks[a] += 0.5 * sum(
                    b != a and lookup[("screen", parent, b)][metric] * sign == x for b in ELIGIBLE
                )
    if (
        ranks != selected["rank_totals"]
        or min(ranks, key=lambda a: (ranks[a], a)) != selected["selected"]
    ):
        raise ValueError("Independent screening selection disagrees")
    rng = np.random.default_rng(SEED)
    stats, effects = [], []
    for stage, (_, expected_parents) in STAGES.items():
        arms = sorted({r["arm"] for r in pmetrics if r["stage"] == stage})
        comparisons = (
            [(a, "W1") for a in arms if a != "W1"]
            if stage == "screen"
            else [(selected["selected"], b) for b in ("W1", "R0", "S0")]
        )
        parents = sorted({r["parent"] for r in pmetrics if r["stage"] == stage})
        if len(parents) != expected_parents:
            raise ValueError("Wrong independent sample size")
        family = []
        for a, b in comparisons:
            for metric, sign in (("hv", 1), ("igd", -1)):
                values = [
                    (lookup[(stage, p, a)][metric] - lookup[(stage, p, b)][metric]) * sign
                    for p in parents
                ]
                strata = [lookup[(stage, p, a)]["jobs"] for p in parents]
                item = {
                    "stage": stage,
                    "treatment": a,
                    "control": b,
                    "metric": metric,
                    **inference(values, strata, rng),
                }
                family.append(item)
                for p, v in zip(parents, values):
                    base = lookup[(stage, p, a)]
                    effects.append(
                        {
                            "stage": stage,
                            "treatment": a,
                            "control": b,
                            "metric": metric,
                            "parent": p,
                            **{k: base[k] for k in ("jobs", "pressure", "topology")},
                            "effect": v,
                        }
                    )
        for item, adjusted in zip(family, holm([r["p"] for r in family])):
            item["p_holm"] = adjusted
        stats.extend(family)
        print(json.dumps({"stage": stage, "statistics": family}), flush=True)
    table(out / "parent_effects.csv", effects)
    arm_means = []
    for stage in STAGES:
        for arm in sorted({r["arm"] for r in pmetrics if r["stage"] == stage}):
            rows = [r for r in pmetrics if r["stage"] == stage and r["arm"] == arm]
            arm_means.append(
                {
                    "stage": stage,
                    "arm": arm,
                    "parents": len(rows),
                    **{k: float(np.mean([r[k] for r in rows])) for k in numerical},
                }
            )
    table(out / "arm_means.csv", arm_means)
    # Every snapshot curve uses a separate common per-input range covering ALL final
    # and recorded intermediate points. No extrapolation or time interpolation.
    agroups = defaultdict(list)
    for row in anytime:
        agroups[(row["stage"], row["input"])].append(row)
    curves = []
    for key, group in agroups.items():
        ps = [p for r in group for p in r["points"]] + [
            p for s in groups[key] for p in s["archive_points"]
        ]
        lo = np.min(ps, axis=0)
        span = np.maximum(np.max(ps, axis=0) - lo, 1e-9)
        for r in group:
            normalized = ((np.asarray(r["points"]) - lo) / span).tolist()
            curves.append(
                {k: v for k, v in r.items() if k != "points"}
                | {"hv": hypervolume([tuple(p) for p in normalized], (1.1, 1.1))}
            )
    table(out / "anytime_observations.csv", curves)
    diagnostics = defaultdict(Counter)
    timings = defaultdict(Counter)
    for s in summaries:
        key = (s["spec"]["stage"], s["spec"]["arm"])
        diagnostics[key].update(s["diagnostic_counts"])
        timings[key].update(s["timings"])
    write(
        out / "mechanisms.json",
        [
            {"stage": k[0], "arm": k[1], "counts": dict(v), "timings": dict(timings[k])}
            for k, v in sorted(diagnostics.items())
        ],
    )
    result = {
        "seed": SEED,
        "bootstrap": 20000,
        "signflips": 100000,
        "selected": selected,
        "statistics": stats,
        "arm_means": arm_means,
    }
    write(out / "analysis.json", result)
    return result


def main() -> None:
    """Run immutable validation followed by the frozen quality and mechanism analyses."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.start.resolve()) or args.out.resolve().is_relative_to(
        args.capture.resolve() / "control"
    ):
        raise ValueError("Analysis output must be outside immutable evidence")
    args.out.mkdir(parents=True, exist_ok=False)
    rows, _, selected = validate(args.start, args.capture, args.out)
    analyze(rows, selected, args.out)


if __name__ == "__main__":
    main()
