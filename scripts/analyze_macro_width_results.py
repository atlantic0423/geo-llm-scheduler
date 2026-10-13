"""Analyze the frozen MacroSearch four-arm experiment without altering raw evidence."""

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

from geo_llm_scheduler.experiments.macro_width import ARMS, read_json
from geo_llm_scheduler.experiments.macro_width_worker import validate_complete
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.rl.state import decode
from geo_llm_scheduler.utils.numeric import TOL

SOURCE = "9b2ec3dd1fed371abf382192293338cac9ee4a24"
CANONICAL = "1d9a8f4c726312e95d49e553b587f9f08359a90e85b1cd0bdeac9ced6b1daea9"
SEEDS = (1112, 2212)
SEED = 2026101209
CONTRASTS = (("POOL3", "POOL6"), ("POOL10", "POOL6"), ("POOL6", "NATIVE6"))
PRIMARY = tuple((a + "-" + b, m) for a, b in CONTRASTS for m in ("hv_diff", "igd_plus_gain"))
DIAGNOSTICS = (
    "hv_pct",
    "igd_gain",
    "coverage_net",
    "flow_gain_pct",
    "bill_gain_pct",
    "exact_pct",
    "offspring_pct",
    "macro_call_pct",
    "unique_nd_diff",
)


def digest(path: Path) -> str:
    """Hash a complete immutable artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a stable table with explicit empty cells for undefined quantities."""
    if not rows:
        raise ValueError("Cannot export an empty table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def base_values(rows: list[dict], field: str) -> tuple[list[float], list[int], list[tuple]]:
    """Average four paired H/T/seed cells, rejecting missing or duplicate cells."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["jobs"], row["base_seed"])].append(row)
    keys = sorted(groups)
    for key in keys:
        if len(groups[key]) != 4 or {(r["tariff"], r["algorithm_seed"]) for r in groups[key]} != {
            (t, s) for t in ("H", "T") for s in SEEDS
        }:
            raise ValueError("Incomplete or duplicate independent base")
    keys = [key for key in keys if all(r[field] is not None for r in groups[key])]
    return (
        [float(np.mean([r[field] for r in groups[key]])) for key in keys],
        [key[0] for key in keys],
        keys,
    )


def integral_hv(points: list, reference: tuple[float, float] = (1.1, 1.1)) -> float:
    """Independently integrate the vertical envelope on exact x-breakpoint intervals."""
    points = [p for p in points if p[0] <= reference[0] and p[1] <= reference[1]]
    breaks = sorted({p[0] for p in points} | {reference[0]})
    return math.fsum(
        (right - left) * (reference[1] - min(p[1] for p in points if p[0] <= left))
        for left, right in zip(breaks, breaks[1:])
    )


def inference(paired: list[dict], bootstrap: int, signflips: int) -> tuple[dict, list[dict]]:
    """Apply the prespecified six-test joint Holm family with base-cluster resampling."""
    rng = np.random.default_rng(SEED)
    statistics: dict[str, dict] = {}
    bases = []
    for a, b in CONTRASTS:
        name = a + "-" + b
        selected = [r for r in paired if r["contrast"] == name]
        statistics[name] = {}
        for metric in ("hv_diff", "igd_plus_gain", *DIAGNOSTICS):
            values, strata, keys = base_values(selected, metric)
            statistics[name][metric] = (
                shared.cluster_summary(values, strata, rng, bootstrap, signflips)
                if values
                else None
            )
            bases.extend(
                dict(contrast=name, metric=metric, jobs=k[0], base_seed=k[1], value=v)
                for k, v in zip(keys, values)
            )
    adjusted = shared.holm([statistics[a][m]["p_signflip"] for a, m in PRIMARY])
    for (a, m), value in zip(PRIMARY, adjusted):
        statistics[a][m]["p_holm_primary_six"] = value
    return statistics, bases


def probe_fields(row: dict, available: int) -> dict:
    """Check nested gains and quantify SEQ loss in the shared ten-candidate context."""
    gains = [row["prefix"][str(b)]["gain"] for b in (3, 6, 10)]
    counts = [row["prefix"][str(b)]["evaluated"] for b in (3, 6, 10)]
    if any(not math.isfinite(g) or g < -TOL.scalar for g in gains):
        raise ValueError("Invalid prefix gain")
    if gains[1] < gains[0] - TOL.scalar or gains[2] < gains[1] - TOL.scalar:
        raise ValueError("Nonmonotone common prefix")
    if counts != sorted(counts) or any(c > b for c, b in zip(counts, (3, 6, 10))):
        raise ValueError("Invalid nested candidate counts")
    if not row["online_rng_verified"] or available < 1:
        raise ValueError("Replay or sampling provenance missing")
    common_loss = row["prefix"]["10"]["gain"] - row["seq_common_gain"]
    if common_loss < -TOL.scalar or not math.isclose(
        common_loss, row["seq_missed_gain"], abs_tol=1e-12
    ):
        raise ValueError("SEQ loss context mismatch")
    preference, condition, stagnant = decode(row["state"])
    eligible = len(row["full_objectives"]) == counts[2]
    before = shared.nondominated(row["archive_before_objectives"])
    initial = set(before)
    additions = []
    for b in (3, 6, 10):
        additions.append(
            len(set(shared.nondominated(before + row["full_objectives"][:b])) - initial)
            if eligible
            else None
        )
    return {
        "action": row["action"],
        "state": row["state"],
        "preference": preference,
        "condition": condition,
        "stagnant": stagnant,
        "progress": row["progress"],
        "stage": min(2, int(3 * row["progress"])),
        "bucket_calls": available,
        "eval3": counts[0],
        "eval6": counts[1],
        "eval10": counts[2],
        "gain3": gains[0],
        "gain6": gains[1],
        "gain10": gains[2],
        "relative_gain3": row["prefix"]["3"]["relative_gain"],
        "relative_gain6": row["prefix"]["6"]["relative_gain"],
        "relative_gain10": row["prefix"]["10"]["relative_gain"],
        "increment_3_6": gains[1] - gains[0],
        "increment_6_10": gains[2] - gains[1],
        "zero3": int(gains[0] <= TOL.scalar),
        "zero6": int(gains[1] <= TOL.scalar),
        "late_4_6": int(row["late_4_6_after_zero3"]),
        "late_7_10": int(row["late_7_10_after_zero6"]),
        "positive_increment_3_6": int(gains[1] - gains[0] > TOL.scalar),
        "positive_increment_6_10": int(gains[2] - gains[1] > TOL.scalar),
        "seq_effective": row["seq"]["final_B_eff"],
        "seq_missed_gain": common_loss,
        "seq_missed_positive": int(common_loss > TOL.scalar),
        "short_pool": int(counts[2] < 10),
        "all_full_feasible": int(eligible),
        "objective_additions3": additions[0],
        "objective_additions6": additions[1],
        "objective_additions10": additions[2],
        "common_generation_seconds": row["common_generation_seconds"],
        "offline_total_seconds": row["total_replay_seconds"],
    }


def probe_summary(calls: list[dict], field: str) -> list[dict]:
    """Summarize sampled buckets without treating individual calls as independent bases."""
    result = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in calls:
        groups[(row["arm"], row[field])].append(row)
    for (arm, group), members in sorted(groups.items()):
        zero3 = sum(r["zero3"] for r in members)
        zero6 = sum(r["zero6"] for r in members)
        # Estimate visitation-weighted unconditional rates per run, then average HT/seeds/base.
        cells: dict[tuple, list[dict]] = defaultdict(list)
        for r in members:
            cells[(r["jobs"], r["base_seed"], r["tariff"], r["algorithm_seed"])].append(r)
        base_groups: dict[tuple, list[dict]] = defaultdict(list)
        for k, rr in cells.items():
            total = sum(r["bucket_calls"] for r in rr)
            base_groups[k[:2]].append(
                {
                    metric: sum(r["bucket_calls"] * r[metric] for r in rr) / total
                    for metric in (
                        "late_4_6",
                        "late_7_10",
                        "seq_missed_positive",
                        "short_pool",
                        "relative_gain3",
                        "relative_gain6",
                        "relative_gain10",
                    )
                }
            )
        means = {
            metric: float(
                np.mean([np.mean([r[metric] for r in rr]) for rr in base_groups.values()])
            )
            for metric in (
                "late_4_6",
                "late_7_10",
                "seq_missed_positive",
                "short_pool",
                "relative_gain3",
                "relative_gain6",
                "relative_gain10",
            )
        }
        result.append(
            {
                "arm": arm,
                "group": group,
                "sample_buckets": len(members),
                "bases": len(base_groups),
                "represented_calls": sum(r["bucket_calls"] for r in members),
                "zero3_buckets": zero3,
                "zero6_buckets": zero6,
                "late_given_zero3_bucket_rate": sum(r["late_4_6"] for r in members) / zero3
                if zero3
                else None,
                "late_given_zero6_bucket_rate": sum(r["late_7_10"] for r in members) / zero6
                if zero6
                else None,
                "positive_increment3_6_bucket_rate": float(
                    np.mean([r["positive_increment_3_6"] for r in members])
                ),
                "positive_increment6_10_bucket_rate": float(
                    np.mean([r["positive_increment_6_10"] for r in members])
                ),
                "seq_effective_bucket_mean": float(np.mean([r["seq_effective"] for r in members])),
                **{"base_visit_weighted_" + metric: value for metric, value in means.items()},
            }
        )
    return result


def analyze(
    raw: Path, out: Path, resume: Path, bootstrap: int = 20000, signflips: int = 100000
) -> dict:
    """Read and verify all frozen data before computing clustered online/offline results."""
    if out.resolve().is_relative_to(raw.resolve()):
        raise ValueError("Analysis output must stay outside raw")
    out.mkdir(parents=True, exist_ok=False)
    resumed = set(read_json(resume)["missing"])
    records = []
    input_checks = 0
    for node, expected in ((0, 224), (1, 416)):
        root = raw / f"formal_node{node}"
        manifest = read_json(root / "manifest.json")
        if (
            manifest["source_commit"] != SOURCE
            or manifest["source_hash"] != CANONICAL
            or manifest["run_count"] != expected
        ):
            raise ValueError("Wrong frozen source or matrix")
        if (
            digest(root / "manifest.json")
            != read_json(root / "manifest.sha256.json")["manifest_sha256"]
        ):
            raise ValueError("Manifest changed")
        for name, sha in manifest["input_files"].items():
            if digest(root / name) != sha:
                raise ValueError("Frozen input changed")
            input_checks += 1
        for block in manifest["blocks"]:
            specs = [read_json(root / "specs" / (key + ".json")) for key in block["runs"]]
            if {s["arm"] for s in specs} != set(ARMS) or len(specs) != 4:
                raise ValueError("Broken four-arm block")
            if (
                len({s["initial_hash"] for s in specs}) != 1
                or len({s["instance_hash"] for s in specs}) != 1
            ):
                raise ValueError("Unpaired initialization/input")
            for order, spec in enumerate(specs):
                key = spec["key"]
                run = root / "runs" / key
                valid, reason = validate_complete(run, spec)
                if not valid:
                    raise ValueError((key, reason))
                summary = read_json(run / "summary.json")
                if (
                    summary["initial_hash"] != spec["initial_hash"]
                    or summary["seed"] != spec["algorithm_seed"]
                ):
                    raise ValueError("Initialization/seed provenance mismatch")
                with (run / "objectives.csv").open(encoding="utf-8", newline="") as stream:
                    points = [
                        (float(r["flow_seconds"]), float(r["bill_cny"]))
                        for r in csv.DictReader(stream)
                    ]
                marker = read_json(root / "replay" / (key + ".json.complete.json"))
                replay_file = root / "replay" / (key + ".json.gz")
                if marker["source_marker_hash"] != digest(run / "complete.json") or marker[
                    "sha256"
                ] != digest(replay_file):
                    raise ValueError("Unbound offline replay")
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
                            )
                        },
                        "node": node,
                        "order": order,
                        "host": summary["host"],
                        "resumed": int(key in resumed),
                        "points": shared.nondominated(points),
                        "summary": summary,
                        "snapshots": read_json(run / "anytime.json"),
                        "replay_file": replay_file,
                        "replay_count": marker["calls"],
                    }
                )
    if len(records) != 640 or len({r["key"] for r in records}) != 640:
        raise ValueError("Incomplete/duplicate formal matrix")
    ownership: dict[tuple, set] = defaultdict(set)
    for r in records:
        ownership[(r["jobs"], r["base_seed"])].add(r["host"])
    if len(ownership) != 40 or any(len(hosts) != 1 for hosts in ownership.values()):
        raise ValueError("Independent base host ownership changed")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["jobs"], r["base_seed"], r["tariff"])].append(r)
    rows, anytime, costs, calls, references, normalized = [], [], [], [], {}, {}
    hv_checks, igd_checks = 0, 0
    for key, members in sorted(groups.items()):
        if len(members) != 8 or {(r["arm"], r["algorithm_seed"]) for r in members} != {
            (a, s) for a in ARMS for s in SEEDS
        }:
            raise ValueError("Incomplete instance arms/seeds")
        union = shared.nondominated([p for r in members for p in r["points"]])
        lo, span = shared.normalization(union)

        def norm(points: list) -> list:
            return [tuple(map(float, p)) for p in (np.asarray(points) - lo) / span]

        front = norm(union)
        references[str(key)] = {"ideal": lo.tolist(), "divisor": span.tolist(), "front": union}
        for r in members:
            summary = r["summary"]
            p = norm(r["points"])
            normalized[r["key"]] = p
            eligible = [q for q in p if max(q) <= 1.1]
            hv = hypervolume(eligible, (1.1, 1.1))
            if not math.isclose(hv, integral_hv(p), abs_tol=1e-12):
                raise ValueError("Independent HV mismatch")
            hv_checks += 1
            igdp = shared.distance(p, front, True)
            if r["base_seed"] in (1525001, 1535001):
                if not math.isclose(igdp, igd_plus(p, front), abs_tol=1e-12):
                    raise ValueError("Independent IGD+ mismatch")
                igd_checks += 1
            counters, timings = summary["counts"], summary["timings"]
            macro_calls = sum(
                v.get("calls", 0)
                for k, v in summary["light_operator_totals"].items()
                if k.isdigit()
            )
            construction = sum(timings.get(f"construction:A{a}", 0) for a in range(1, 9))
            measured = construction + sum(timings.get(k, 0) for k in ("ssgs", "exact", "archive"))
            if measured > summary["elapsed"] + 1e-5:
                raise ValueError("Overlapping phase accounting")
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
                    "host",
                    "order",
                    "resumed",
                )
            }
            row.update(
                hv=hv,
                igd_plus=igdp,
                igd=shared.distance(p, front, False),
                min_flow=min(q[0] for q in r["points"]),
                min_bill=min(q[1] for q in r["points"]),
                unique_nd=len(r["points"]),
                stored_archive=summary["archive_size"],
                exact_rate=counters["exact"] / summary["elapsed"],
                offspring_rate=summary["offspring_count"] / summary["elapsed"],
                macro_call_rate=macro_calls / summary["elapsed"],
                elapsed=summary["elapsed"],
                overshoot=summary["time_overshoot_seconds"],
                export_seconds=summary["verification_and_export_seconds"],
                trigger_rate=summary["trigger_rate"],
                rss_gib=summary["rss_peak_gib"],
                construction_share=construction / summary["elapsed"],
                ssgs_share=timings.get("ssgs", 0) / summary["elapsed"],
                exact_share=timings.get("exact", 0) / summary["elapsed"],
                archive_share=timings.get("archive", 0) / summary["elapsed"],
                residual_share=(summary["elapsed"] - measured) / summary["elapsed"],
                polish_share=timings.get("polish", 0) / summary["elapsed"],
                polish_exact_share=counters.get("exact:polish", 0) / counters["exact"],
            )
            rows.append(row)
            for a in range(1, 9):
                v = summary["light_operator_totals"].get(str(a), {})
                n = v.get("calls", 0)
                costs.append(
                    {
                        **{
                            k: row[k]
                            for k in ("key", "jobs", "base_seed", "tariff", "algorithm_seed", "arm")
                        },
                        "action": a,
                        "calls": n,
                        "proposals": counters.get(f"proposals:A{a}", 0),
                        "evaluations": counters.get(f"exact:A{a}", 0),
                        "attempts": counters.get(f"construction_attempts:A{a}", 0),
                        "construction_seconds": timings.get(f"construction:A{a}", 0),
                        "ssgs_seconds": timings.get(f"ssgs:A{a}", 0),
                        "exact_seconds": timings.get(f"exact:A{a}", 0),
                        "archive_seconds": timings.get(f"archive:A{a}", 0),
                        "step_seconds": v.get("seconds", 0),
                        "accepted": v.get("accepted", 0),
                        "archive_insertions": v.get("archive_insertions", 0),
                        "archive_net_retained": v.get("archive_net_retained", 0),
                    }
                )
            for snapshot in r["snapshots"]:
                pp = norm(snapshot["objectives"])
                anytime.append(
                    {
                        **{
                            k: row[k]
                            for k in ("key", "jobs", "base_seed", "tariff", "algorithm_seed", "arm")
                        },
                        "fraction": snapshot["target_seconds"] / summary["time_budget_seconds"],
                        "delay_seconds": snapshot["elapsed"] - snapshot["target_seconds"],
                        "hv": hypervolume([q for q in pp if max(q) <= 1.1], (1.1, 1.1)),
                    }
                )
            replay = read_json(r["replay_file"])
            if replay["online_feedback"] or len(replay["calls"]) != r["replay_count"]:
                raise ValueError("Offline replay semantics changed")
            for bucket, sample in replay["calls"].items():
                fields = probe_fields(sample, replay["counts"][bucket])
                calls.append(
                    {
                        **{
                            k: row[k]
                            for k in (
                                "key",
                                "jobs",
                                "base_seed",
                                "scenario",
                                "tariff",
                                "algorithm_seed",
                                "arm",
                                "resumed",
                            )
                        },
                        "bucket": bucket,
                        **fields,
                    }
                )
    blocks: dict[str, dict] = defaultdict(dict)
    for row in rows:
        if row["arm"] in blocks[row["block"]]:
            raise ValueError("Duplicate arm")
        blocks[row["block"]][row["arm"]] = row
    paired = []
    for block, arms in sorted(blocks.items()):
        if set(arms) != set(ARMS):
            raise ValueError("Incomplete paired block")
        for a, b in CONTRASTS:
            x, y = arms[a], arms[b]
            if x["host"] != y["host"]:
                raise ValueError("Cross-host paired block")

            def pct(field: str) -> float | None:
                return 100 * (x[field] / y[field] - 1) if y[field] else None

            pair = {
                k: x[k]
                for k in ("jobs", "base_seed", "scenario", "tariff", "algorithm_seed", "node")
            }
            pair.update(
                block=block,
                contrast=a + "-" + b,
                hv_diff=x["hv"] - y["hv"],
                igd_plus_gain=y["igd_plus"] - x["igd_plus"],
                hv_pct=pct("hv"),
                igd_gain=y["igd"] - x["igd"],
                coverage_net=shared.coverage(normalized[x["key"]], normalized[y["key"]])
                - shared.coverage(normalized[y["key"]], normalized[x["key"]]),
                flow_gain_pct=-pct("min_flow"),
                bill_gain_pct=-pct("min_bill"),
                exact_pct=pct("exact_rate"),
                offspring_pct=pct("offspring_rate"),
                macro_call_pct=pct("macro_call_rate"),
                unique_nd_diff=x["unique_nd"] - y["unique_nd"],
                touched_resume=int(x["resumed"] or y["resumed"]),
            )
            paired.append(pair)
    statistics, bases = inference(paired, bootstrap, signflips)
    affected = {(r["jobs"], r["base_seed"]) for r in rows if r["resumed"]}
    sensitivities = []
    for a, b in CONTRASTS:
        name = a + "-" + b
        selected = [r for r in paired if r["contrast"] == name]
        for metric in ("hv_diff", "igd_plus_gain", "hv_pct", "offspring_pct"):
            values, strata, keys = base_values(selected, metric)
            for label, keep in [
                ("50", lambda k: k[0] == 50),
                ("100", lambda k: k[0] == 100),
                ("exclude_resumed_entire_bases", lambda k: k not in affected),
            ]:
                vv = [v for v, key in zip(values, keys) if keep(key)]
                sensitivities.append(
                    dict(
                        contrast=name,
                        metric=metric,
                        group=label,
                        bases=len(vv),
                        mean=float(np.mean(vv)),
                    )
                )
            unchanged_size_means = [
                float(
                    np.mean([v for v, k in zip(values, keys) if k[0] == size and k not in affected])
                )
                for size in (50, 100)
            ]
            sensitivities.append(
                dict(
                    contrast=name,
                    metric=metric,
                    group="exclude_resumed_size_balanced",
                    bases=sum(k not in affected for k in keys),
                    mean=float(np.mean(unchanged_size_means)),
                )
            )
            for tariff in ("H", "T"):
                tariffs: dict[tuple, list[dict]] = defaultdict(list)
                for r in selected:
                    if r["tariff"] == tariff:
                        tariffs[(r["jobs"], r["base_seed"])].append(r)
                if any(
                    len(rr) != 2 or {r["algorithm_seed"] for r in rr} != set(SEEDS)
                    for rr in tariffs.values()
                ):
                    raise ValueError("Incomplete tariff sensitivity")
                sensitivities.append(
                    dict(
                        contrast=name,
                        metric=metric,
                        group=tariff,
                        bases=len(tariffs),
                        mean=float(
                            np.mean([np.mean([r[metric] for r in rr]) for rr in tariffs.values()])
                        ),
                    )
                )
    write_csv(out / "run_metrics.csv", rows)
    write_csv(out / "paired_metrics.csv", paired)
    write_csv(out / "base_metrics.csv", bases)
    write_csv(out / "anytime_metrics.csv", anytime)
    write_csv(out / "operator_costs.csv", costs)
    write_csv(out / "probe_buckets.csv", calls)
    write_csv(out / "probe_actions.csv", probe_summary(calls, "action"))
    write_csv(out / "probe_preferences.csv", probe_summary(calls, "preference"))
    write_csv(out / "probe_conditions.csv", probe_summary(calls, "condition"))
    write_csv(out / "sensitivity.csv", sensitivities)
    (out / "reference_fronts.json").write_text(
        json.dumps(references, ensure_ascii=False), encoding="utf-8"
    )
    result = {
        "source_commit": SOURCE,
        "source_hash": CANONICAL,
        "runs": len(rows),
        "blocks": len(blocks),
        "independent_bases": len(ownership),
        "input_hash_checks": input_checks,
        "offline_calls": len(calls),
        "hv_crosschecks": hv_checks,
        "igd_plus_crosschecks": igd_checks,
        "bootstrap": bootstrap,
        "signflips": signflips,
        "seed": SEED,
        "primary_family": PRIMARY,
        "statistics": statistics,
        "affected_resume_bases": sorted(affected),
        "arm_means": {
            arm: {
                field: float(np.mean([r[field] for r in rows if r["arm"] == arm]))
                for field in (
                    "hv",
                    "igd_plus",
                    "igd",
                    "unique_nd",
                    "exact_rate",
                    "offspring_rate",
                    "construction_share",
                    "ssgs_share",
                    "exact_share",
                    "archive_share",
                    "residual_share",
                    "polish_share",
                    "polish_exact_share",
                )
            }
            for arm in ARMS
        },
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(out.iterdir())
            if p.is_file()
        },
    }
    (out / "analysis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    """Run the prespecified formal analysis, never pilot or post-hoc enlarged matrices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.raw, args.out, args.resume)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "runs",
                    "blocks",
                    "independent_bases",
                    "offline_calls",
                    "hv_crosschecks",
                    "igd_plus_crosschecks",
                    "statistics",
                    "arm_means",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
