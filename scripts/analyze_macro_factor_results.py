"""Analyze the frozen 2x2 MacroSearch campaign without modifying its raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import analyze_d6_results as shared
import numpy as np

from geo_llm_scheduler.experiments.macro_factor import ARMS
from geo_llm_scheduler.experiments.macro_factor_worker import validate_complete
from geo_llm_scheduler.experiments.macro_width import read_json
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.experiments.runner import digest as semantic_digest

SOURCE = "6a6d1bbd808ba89b8cc835975127a77fc216ea1c"
CANONICAL = "17ca972cb482446d9b21c4bda5149d97a4e0dc8317513b9efa5fdbd354e3598b"
SEEDS = (1313, 2313)
SEED = 2026101310
CONTRASTS = (
    ("POOL_SHARED", "NATIVE_SHARED"),
    ("POOL_CALL", "NATIVE_CALL"),
    ("NATIVE_CALL", "NATIVE_SHARED"),
    ("POOL_CALL", "POOL_SHARED"),
)
INTERACTION = "GENERATION_X_CALL"
MAIN = ("POOL_SHARED-NATIVE_SHARED", "POOL_CALL-NATIVE_CALL", INTERACTION)
SECONDARY = ("NATIVE_CALL-NATIVE_SHARED", "POOL_CALL-POOL_SHARED")
QUALITY = ("hv_diff", "igd_plus_gain")
DESCRIPTIVE = (
    "hv_pct",
    "igd_gain",
    "coverage_net",
    "flow_gain_pct",
    "bill_gain_pct",
    "exact_pct",
    "offspring_pct",
    "macro_call_pct",
)


def sha(path: Path) -> str:
    """Hash evidence in bounded memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write stable tabular data, rejecting empty output."""
    if not rows:
        raise ValueError("No rows")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def base_values(rows: list[dict], metric: str) -> tuple[list, list, list]:
    """Average exactly four H/T-seed cells for each independent base."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["jobs"], row["base_seed"])].append(row)
    keys = sorted(groups)
    for key in keys:
        members = groups[key]
        if len(members) != 4 or {(r["tariff"], r["algorithm_seed"]) for r in members} != {
            (t, s) for t in ("H", "T") for s in SEEDS
        }:
            raise ValueError("Incomplete or duplicate base cells")
    keys = [key for key in keys if all(r[metric] is not None for r in groups[key])]
    values = [float(np.mean([r[metric] for r in groups[key]])) for key in keys]
    if not all(map(math.isfinite, values)):
        raise ValueError("Nonfinite effect")
    return values, [k[0] for k in keys], keys


def interaction_effect(arms: dict[str, dict], metric: str) -> float:
    """Compute the CALL-minus-SHARED generation effect with improvement-positive signs."""
    if set(arms) != set(ARMS):
        raise ValueError("Incomplete four-arm cell")
    value = (arms["POOL_CALL"][metric] - arms["NATIVE_CALL"][metric]) - (
        arms["POOL_SHARED"][metric] - arms["NATIVE_SHARED"][metric]
    )
    return float(-value if metric == "igd_plus" else value)


def integral_hv(points: list, reference: tuple[float, float] = (1.1, 1.1)) -> float:
    """Independently integrate the rectangle envelope, retaining axis and tie cases."""
    eligible = [p for p in points if p[0] <= reference[0] and p[1] <= reference[1]]
    breaks = sorted({p[0] for p in eligible} | {reference[0]})
    return math.fsum(
        (right - left) * (reference[1] - min(p[1] for p in eligible if p[0] <= left))
        for left, right in zip(breaks, breaks[1:])
    )


def coverage(a: list, b: list) -> float:
    """Compute weak coverage with the inherited normalized 1e-12 tolerance."""
    if not a or not b:
        raise ValueError("Empty coverage front")
    left, right = np.asarray(a), np.asarray(b)
    dominated = 0
    for start in range(0, len(right), 128):
        chunk = right[start : start + 128]
        dominated += int((left[:, None, :] <= chunk[None, :, :] + 1e-12).all(2).any(0).sum())
    return dominated / len(right)


def inference(paired: list[dict], bootstrap: int, signflips: int) -> tuple[dict, list[dict]]:
    """Apply frozen base-cluster inference and separate six/four-test Holm families."""
    rng = np.random.default_rng(SEED)
    statistics, bases = {}, []
    names = [a + "-" + b for a, b in CONTRASTS] + [INTERACTION]
    for name in names:
        selected = [r for r in paired if r["contrast"] == name]
        statistics[name] = {}
        for metric in QUALITY + (() if name == INTERACTION else DESCRIPTIVE):
            values, strata, keys = base_values(selected, metric)
            if len(values) != 40:
                raise ValueError("Expected all forty independent bases")
            statistics[name][metric] = shared.cluster_summary(
                values, strata, rng, bootstrap=bootstrap, signflips=signflips
            )
            bases.extend(
                dict(contrast=name, metric=metric, jobs=k[0], base_seed=k[1], value=v)
                for k, v in zip(keys, values)
            )
    for names, label in ((MAIN, "p_holm_primary_six"), (SECONDARY, "p_holm_secondary_four")):
        family = [(n, m) for n in names for m in QUALITY]
        adjusted = shared.holm([statistics[n][m]["p_signflip"] for n, m in family])
        for (name, metric), p in zip(family, adjusted):
            statistics[name][metric][label] = p
    return statistics, bases


def analyze(raw: Path, out: Path, prior: Path) -> dict:
    """Validate all frozen cells before computing quality, interaction and cost diagnostics."""
    if out.resolve().is_relative_to(raw.resolve()):
        raise ValueError("Output must stay outside immutable raw")
    out.mkdir(parents=True, exist_ok=False)
    pre_resume = read_json(prior)
    preserved = {r["key"] for node in pre_resume for r in node["rows"]}
    records, input_checks = [], 0
    for node, expected in ((0, 224), (1, 416)):
        root = raw / f"formal_node{node}"
        manifest = read_json(root / "manifest.json")
        if (
            manifest["source_commit"] != SOURCE
            or manifest["source_hash"] != CANONICAL
            or manifest["run_count"] != expected
            or manifest["kind"] != "MACRO_FACTOR_FORMAL"
        ):
            raise ValueError("Wrong source or formal matrix")
        if (
            sha(root / "manifest.json")
            != read_json(root / "manifest.sha256.json")["manifest_sha256"]
        ):
            raise ValueError("Manifest hash mismatch")
        for name, digest in manifest["input_files"].items():
            if sha(root / name) != digest:
                raise ValueError("Frozen input hash mismatch")
            input_checks += 1
        for block in manifest["blocks"]:
            specs = [read_json(root / "specs" / (key + ".json")) for key in block["runs"]]
            if len(specs) != 4 or {s["arm"] for s in specs} != set(ARMS):
                raise ValueError("Incomplete four-arm block")
            if (
                len({s["initial_hash"] for s in specs}) != 1
                or len({s["instance_hash"] for s in specs}) != 1
            ):
                raise ValueError("Unpaired input or initialization")
            for order, spec in enumerate(specs):
                run = root / "runs" / spec["key"]
                valid, reason = validate_complete(run, spec)
                if not valid:
                    raise ValueError((spec["key"], reason))
                summary = read_json(run / "summary.json")
                if (
                    summary["seed"] != spec["algorithm_seed"]
                    or summary["initial_hash"] != spec["initial_hash"]
                    or summary["input_hash"] != semantic_digest(spec)
                    or summary["time_budget_seconds"] != spec["config"]["seconds"]
                    or summary["call_sample_size"] != 0
                ):
                    raise ValueError("Seed, initialization, budget or production observer mismatch")
                for p in pre_resume[node]["rows"]:
                    if p["key"] == spec["key"] and sha(run / "complete.json") != p["marker_sha256"]:
                        raise ValueError("Previously completed marker changed")
                with (run / "objectives.csv").open(encoding="utf-8", newline="") as stream:
                    points = [
                        (float(r["flow_seconds"]), float(r["bill_cny"]))
                        for r in csv.DictReader(stream)
                    ]
                front = shared.nondominated(points)
                if not front:
                    raise ValueError("Empty objective set")
                records.append(
                    {
                        **{
                            k: spec[k]
                            for k in (
                                "key",
                                "block",
                                "jobs",
                                "base_seed",
                                "scenario",
                                "tariff",
                                "algorithm_seed",
                                "arm",
                                "expected_host",
                            )
                        },
                        "node": node,
                        "order": order,
                        "resumed": spec["key"] not in preserved,
                        "points": front,
                        "summary": summary,
                        "snapshots": read_json(run / "anytime.json"),
                    }
                )
    ownership: dict[tuple, set] = defaultdict(set)
    for r in records:
        ownership[(r["jobs"], r["base_seed"])].add(r["expected_host"])
    if len(records) != 640 or len({r["key"] for r in records}) != 640 or len(ownership) != 40:
        raise ValueError("Incomplete or duplicate formal matrix")
    if any(len(h) != 1 for h in ownership.values()):
        raise ValueError("Cross-host base")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["jobs"], r["base_seed"], r["tariff"])].append(r)
    rows, anytime, costs, references, normalized = [], [], [], {}, {}
    hv_checks = igd_checks = 0
    for key, members in sorted(groups.items()):
        if len(members) != 8 or {(r["arm"], r["algorithm_seed"]) for r in members} != {
            (a, s) for a in ARMS for s in SEEDS
        }:
            raise ValueError("Incomplete native-instance group")
        union = shared.nondominated([p for r in members for p in r["points"]])
        lo, span = shared.normalization(union)

        def norm(points: list) -> list:
            return [tuple(map(float, p)) for p in (np.asarray(points) - lo) / span]

        front = norm(union)
        references[str(key)] = {"ideal": lo.tolist(), "divisor": span.tolist(), "front": union}
        for r in members:
            s, p = r["summary"], norm(r["points"])
            normalized[r["key"]] = p
            hv = hypervolume([q for q in p if max(q) <= 1.1], (1.1, 1.1))
            if not math.isclose(hv, integral_hv(p), abs_tol=1e-12):
                raise ValueError("Independent HV mismatch")
            hv_checks += 1
            igdp = shared.distance(p, front, True)
            if r["base_seed"] in (1585001, 1595001):
                if not math.isclose(igdp, igd_plus(p, front), abs_tol=1e-12):
                    raise ValueError("Independent IGD+ mismatch")
                igd_checks += 1
            timings, counts = s["timings"], s["counts"]
            calls = sum(
                v.get("calls", 0) for k, v in s["light_operator_totals"].items() if k.isdigit()
            )
            construction = sum(timings.get(f"construction:A{a}", 0) for a in range(1, 9))
            phase = construction + sum(timings.get(k, 0) for k in ("ssgs", "exact", "archive"))
            if phase > s["elapsed"] + 1e-5:
                raise ValueError("Overlapping exclusive cost accounting")
            row = {
                k: r[k]
                for k in (
                    "key",
                    "block",
                    "jobs",
                    "base_seed",
                    "scenario",
                    "tariff",
                    "algorithm_seed",
                    "arm",
                    "node",
                    "order",
                    "resumed",
                    "expected_host",
                )
            }
            row.update(
                hv=hv,
                igd_plus=igdp,
                igd=shared.distance(p, front, False),
                min_flow=min(q[0] for q in r["points"]),
                min_bill=min(q[1] for q in r["points"]),
                unique_nd=len(p),
                stored_archive=s["archive_size"],
                exact_rate=counts["exact"] / s["elapsed"],
                offspring_rate=s["offspring_count"] / s["elapsed"],
                macro_call_rate=calls / s["elapsed"],
                elapsed=s["elapsed"],
                overshoot=s["time_overshoot_seconds"],
                export_seconds=s["verification_and_export_seconds"],
                construction_share=construction / s["elapsed"],
                ssgs_share=timings.get("ssgs", 0) / s["elapsed"],
                exact_share=timings.get("exact", 0) / s["elapsed"],
                archive_share=timings.get("archive", 0) / s["elapsed"],
                residual_share=(s["elapsed"] - phase) / s["elapsed"],
                polish_share=timings.get("polish", 0) / s["elapsed"],
                polish_exact_share=counts.get("exact:polish", 0) / counts["exact"],
            )
            rows.append(row)
            for action in range(1, 9):
                v = s["light_operator_totals"].get(str(action), {})
                n = v.get("calls", 0)
                costs.append(
                    {
                        **{
                            k: row[k]
                            for k in ("key", "jobs", "base_seed", "tariff", "algorithm_seed", "arm")
                        },
                        "action": action,
                        "calls": n,
                        "proposals": counts.get(f"proposals:A{action}", 0),
                        "evaluations": counts.get(f"exact:A{action}", 0),
                        "accepted": v.get("accepted", 0),
                        "construction_ms_per_call": 1000
                        * timings.get(f"construction:A{action}", 0)
                        / n
                        if n
                        else None,
                        "construction_share": timings.get(f"construction:A{action}", 0)
                        / s["elapsed"],
                        "archive_insertions": v.get("archive_insertions", 0),
                    }
                )
            for snap in r["snapshots"]:
                pp = norm(snap["objectives"])
                anytime.append(
                    {
                        **{
                            k: row[k]
                            for k in ("key", "jobs", "base_seed", "tariff", "algorithm_seed", "arm")
                        },
                        "fraction": snap["target_seconds"] / s["time_budget_seconds"],
                        "delay_seconds": snap["elapsed"] - snap["target_seconds"],
                        "hv": hypervolume([q for q in pp if max(q) <= 1.1], (1.1, 1.1)),
                    }
                )
    blocks: dict[str, dict] = defaultdict(dict)
    for r in rows:
        if r["arm"] in blocks[r["block"]]:
            raise ValueError("Duplicate arm")
        blocks[r["block"]][r["arm"]] = r
    paired = []
    for block, arms in sorted(blocks.items()):
        if set(arms) != set(ARMS):
            raise ValueError("Incomplete block")
        for a, b in CONTRASTS:
            x, y = arms[a], arms[b]
            pair = {k: x[k] for k in ("jobs", "base_seed", "tariff", "algorithm_seed", "node")}

            def pct(field: str) -> float:
                return 100 * (x[field] / y[field] - 1)

            pair.update(
                block=block,
                contrast=a + "-" + b,
                hv_diff=x["hv"] - y["hv"],
                igd_plus_gain=y["igd_plus"] - x["igd_plus"],
                hv_pct=pct("hv"),
                igd_gain=y["igd"] - x["igd"],
                coverage_net=coverage(normalized[x["key"]], normalized[y["key"]])
                - coverage(normalized[y["key"]], normalized[x["key"]]),
                flow_gain_pct=-pct("min_flow"),
                bill_gain_pct=-pct("min_bill"),
                exact_pct=pct("exact_rate"),
                offspring_pct=pct("offspring_rate"),
                macro_call_pct=pct("macro_call_rate"),
            )
            paired.append(pair)
        row = {
            k: arms["POOL_CALL"][k]
            for k in ("jobs", "base_seed", "tariff", "algorithm_seed", "node")
        }
        row.update(
            block=block,
            contrast=INTERACTION,
            hv_diff=interaction_effect(arms, "hv"),
            igd_plus_gain=interaction_effect(arms, "igd_plus"),
        )
        paired.append(row)
    statistics, bases = inference(paired, 20000, 100000)
    fields = (
        "hv",
        "igd_plus",
        "igd",
        "unique_nd",
        "exact_rate",
        "offspring_rate",
        "macro_call_rate",
        "construction_share",
        "ssgs_share",
        "exact_share",
        "archive_share",
        "residual_share",
        "polish_share",
        "polish_exact_share",
        "export_seconds",
        "overshoot",
    )
    report = {
        "status": "ANALYZED",
        "runs": len(rows),
        "blocks": len(blocks),
        "units": len(ownership),
        "source_commit": SOURCE,
        "source_hash": CANONICAL,
        "seed": SEED,
        "bootstrap": 20000,
        "signflips": 100000,
        "statistics": statistics,
        "checks": {
            "inputs": input_checks,
            "hv_independent": hv_checks,
            "igd_plus_independent": igd_checks,
            "preserved_pre_resume_runs": len(preserved),
            "fresh_final_exact": False,
        },
        "means": {
            a: {f: float(np.mean([r[f] for r in rows if r["arm"] == a])) for f in fields}
            for a in ARMS
        },
        "descriptive_size_effects": {
            str(jobs): {
                name: {
                    metric: float(
                        np.mean(
                            [
                                r["value"]
                                for r in bases
                                if r["jobs"] == jobs
                                and r["contrast"] == name
                                and r["metric"] == metric
                            ]
                        )
                    )
                    for metric in QUALITY
                }
                for name in MAIN
            }
            for jobs in (50, 100)
        },
        "limits": [
            "empirical reference, not true Pareto",
            "generation rule bundle, not pool size alone",
            "CALL matches per-operator parent seed consumption, not identical trajectories",
            "export failure/quarantine and 238 original + 402 resumed runs retained",
            "not significant does not establish equivalence",
            "no automatic default promotion",
        ],
    }
    write_csv(out / "run_metrics.csv", rows)
    write_csv(out / "paired_effects.csv", paired)
    write_csv(out / "base_effects.csv", bases)
    write_csv(out / "anytime.csv", anytime)
    write_csv(out / "operator_costs.csv", costs)
    (out / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "references.json").write_text(json.dumps(references, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "runs": len(rows),
                "primary": {n: {m: statistics[n][m] for m in QUALITY} for n in MAIN},
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    """Run the source-pinned standalone analysis with explicit immutable input roots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pre-resume", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.raw, args.out, args.pre_resume)


if __name__ == "__main__":
    main()
