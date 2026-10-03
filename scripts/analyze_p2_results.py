"""Analyze the frozen 720-run P2 campaign without modifying its raw results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.experiments.paired_statistics import holm_adjust, paired_cluster_summary
from geo_llm_scheduler.experiments.runner import digest

ARMS = ("F6", "CG", "CT")
CONTRASTS = (("CG", "F6"), ("CT", "F6"), ("CT", "CG"))
COMMIT = "f6b7fbaf8eaf3362db45fd02ba1bb2b23e8911ce"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _nd(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result, best = [], math.inf
    for x, y in sorted(set(points)):
        if y < best:
            result.append((x, y))
            best = y
    return result


def _igd(points: list[tuple[float, float]], reference: list[tuple[float, float]]) -> float:
    measured, fixed = np.asarray(points), np.asarray(reference)
    distances = np.maximum(measured[None, :, :] - fixed[:, None, :], 0)
    return float(np.sqrt(np.square(distances).sum(axis=2)).min(axis=1).mean())


def _hv(points: list[tuple[float, float]], reference: float = 1.1) -> float:
    # A point outside the reference box contributes zero dominated area.
    return float(hypervolume([p for p in points if max(p) <= reference], (reference, reference)))


def _base_values(rows: list[dict[str, Any]], field: str) -> tuple[list[float], list[int]]:
    groups: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row[field] is not None:
            groups[(row["jobs"], row["base_seed"])][row["tariff"]].append(row[field])
    values, strata = [], []
    for (jobs, _), tariffs in sorted(groups.items()):
        values.append(float(np.mean([np.mean(samples) for samples in tariffs.values()])))
        strata.append(jobs)
    return values, strata


def _contrast_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        blocks[row["block"]][row["arm"]] = row
    result = []
    for block, arms in sorted(blocks.items()):
        for candidate, control in CONTRASTS:
            a, b = arms[candidate], arms[control]
            row = {k: a[k] for k in ("jobs", "base_seed", "tariff", "algorithm_seed", "epoch")}
            row.update(
                block=block,
                contrast=candidate + "-" + control,
                hv_diff=a["hv"] - b["hv"],
                hv_gain_pct=100 * (a["hv"] / b["hv"] - 1),
                igd_gain=b["igd_plus"] - a["igd_plus"],
                igd_gain_pct=100 * (1 - a["igd_plus"] / b["igd_plus"]) if b["igd_plus"] else None,
                flow_gain_pct=100 * (1 - a["min_flow"] / b["min_flow"]),
                bill_gain_pct=100 * (1 - a["min_bill"] / b["min_bill"]),
                offspring_gain_pct=100
                * (a["offspring_per_second"] / b["offspring_per_second"] - 1),
                exact_gain_pct=100 * (a["exact_per_second"] / b["exact_per_second"] - 1),
                trigger_rate_diff=a["trigger_rate"] - b["trigger_rate"],
                hv_101_diff=a["hv_101"] - b["hv_101"],
                hv_150_diff=a["hv_150"] - b["hv_150"],
                igd_loso_gain=b["igd_loso"] - a["igd_loso"],
            )
            result.append(row)
    return result


def _summaries(
    rows: list[dict[str, Any]], seed: int, boot: int, permutations: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for candidate, control in CONTRASTS:
        name = candidate + "-" + control
        selected = [r for r in rows if r["contrast"] == name]
        result[name] = {}
        for field in (
            "hv_diff",
            "hv_gain_pct",
            "igd_gain",
            "igd_gain_pct",
            "flow_gain_pct",
            "bill_gain_pct",
            "offspring_gain_pct",
            "exact_gain_pct",
            "trigger_rate_diff",
        ):
            values, strata = _base_values(selected, field)
            result[name][field] = paired_cluster_summary(values, strata, rng, boot, permutations)
        result[name]["run_pairs"] = len(selected)
        result[name]["hv_run_wins"] = sum(r["hv_diff"] > 1e-12 for r in selected)
        result[name]["igd_run_wins"] = sum(r["igd_gain"] > 1e-12 for r in selected)
    family = [(name, metric) for name in result for metric in ("hv_diff", "igd_gain")]
    adjusted = holm_adjust([result[name][metric]["p_signflip"] for name, metric in family])
    for (name, metric), pvalue in zip(family, adjusted, strict=True):
        result[name][metric]["p_holm_six"] = pvalue
    return result


def main() -> None:
    """Validate raw identities and export fixed-reference paired analyses to a separate folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--permutations", type=int, default=100000)
    args = parser.parse_args()
    root, output = args.campaign.resolve(), args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Analysis output must be outside the immutable raw campaign")
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": 1,
        "mode": "exploratory; analysis choices fixed before quality calculation",
        "source_commit": COMMIT,
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "independent_unit": "24 base structures",
        "aggregation": "paired seeds within tariff; equal tariffs within base; equal bases",
        "bootstrap_strata": "50/100 jobs, observed base counts retained",
        "reference": "union of all 15 final archives per base/tariff; exact objective duplicates removed",
        "normalization": "coordinate min/max of the entire union, fixed for all arms/seeds/times",
        "hv_reference": [1.1, 1.1],
        "igd_reference": "strict nondominated union",
        "primary_family": "3 contrasts x HV absolute difference and IGD+ absolute decrease; Holm six",
        "secondary": "HV relative gain, IGD+ relative decrease, extrema, throughput and mechanism counts",
        "sensitivity": "HV references1.01/1.5; leave-one-algorithm-seed-out IGD+; epoch; no mixed blocks",
        "snapshots": "first safe boundary at/after target fractions, not exact synchronized times",
    }
    _write_json(output / "analysis_plan.json", plan)
    manifest = _load(root / "manifest.json")
    if (
        manifest["source_commit"] != COMMIT
        or manifest["run_count"] != 720
        or len(manifest["blocks"]) != 240
    ):
        raise ValueError("Require the frozen complete 720-run P2 matrix")
    if _sha(root / "manifest.json") != _load(root / "manifest.sha256.json")["manifest_sha256"]:
        raise ValueError("Matrix manifest changed")
    for relative, expected in manifest["input_files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or _sha(path) != expected:
            raise ValueError("Frozen input changed: " + relative)
    status, exit_state = _load(root / "ops/status.json"), _load(root / "ops/exit.json")
    if (
        status["state"] != "complete"
        or status["active"]
        or exit_state["state"] != "complete"
        or exit_state["failed"]
    ):
        raise ValueError("Campaign execution is not complete")
    paused = set(_load(root / "ops/pause_validation_20261003.json")["completed_run_keys"])
    records, input_evidence, operators = [], {}, []
    for block in manifest["blocks"]:
        pre = sum(key in paused for key in block["runs"])
        epoch = "before_pause" if pre == 3 else "after_resume" if pre == 0 else "mixed"
        hosts, initial, instances, configs = set(), set(), set(), set()
        for order, key in enumerate(block["runs"]):
            spec = _load(root / "specs" / (key + ".json"))
            run = root / "runs" / key
            marker = _load(run / "complete.json")
            if marker["input_hash"] != digest(spec):
                raise ValueError("Run identity changed: " + key)
            for name, expected in marker["files"].items():
                path = (run / name).resolve()
                if not path.is_relative_to(run) or _sha(path) != expected:
                    raise ValueError("Published artifact changed: " + key + "/" + name)
            if set(marker["files"]) != {
                "summary.json",
                "config.json",
                "archive.json",
                "population.json",
                "qtable.json",
                "objectives.csv",
                "trace.jsonl",
                "anytime.json",
            }:
                raise ValueError("Incomplete result manifest")
            summary, archive = _load(run / "summary.json"), _load(run / "archive.json")
            if (
                summary["git_commit"] != COMMIT
                or summary["canonical_source_hash"] != manifest["source_hash"]
            ):
                raise ValueError("Source identity changed")
            hosts.add(marker["host"])
            initial.add(spec["initial_hash"])
            instances.add(spec["instance_hash"])
            configs.add(
                digest(
                    {
                        k: v
                        for k, v in spec["config"].items()
                        if k not in ("output", "offspring_policy")
                    }
                )
            )
            points = [
                (c["evaluation"]["flow"], c["evaluation"]["tou"] + sum(c["evaluation"]["demand"]))
                for c in archive
            ]
            with (run / "objectives.csv").open(encoding="utf-8", newline="") as stream:
                exported = [
                    (float(r["flow_seconds"]), float(r["bill_cny"])) for r in csv.DictReader(stream)
                ]
            if points != exported:
                raise ValueError("Archive/CSV objective decomposition mismatch")
            population = _load(run / "population.json")
            diversity = {
                "archive_unique_objectives": len(set(points)),
                "archive_unique_genotypes": len({digest(c["genotype"]) for c in archive}),
                "population_unique_genotypes": len({digest(c["genotype"]) for c in population}),
                "population_unique_schedules": len(
                    {digest(c["schedule"]["starts"]) for c in population}
                ),
            }
            if (
                not points
                or any(not c["evaluation"]["feasible"] for c in archive)
                or not all(math.isfinite(x) for p in points for x in p)
            ):
                raise ValueError("Missing, infeasible or non-finite archive")
            counts = summary["counts"]
            row = {
                k: spec[k] for k in ("key", "arm", "jobs", "base_seed", "tariff", "algorithm_seed")
            }
            row.update(
                block=block["key"],
                epoch=epoch,
                arm_order=order,
                run_epoch="before_pause" if key in paused else "after_resume",
                points=points,
                snapshots=_load(run / "anytime.json"),
                summary=summary,
                diversity=diversity,
            )
            records.append(row)
            input_evidence[key] = {
                "complete_sha256": _sha(run / "complete.json"),
                "files": marker["files"],
            }
            for action, stats in summary["light_operator_totals"].items():
                operators.append(
                    {
                        "key": key,
                        "arm": spec["arm"],
                        "jobs": spec["jobs"],
                        "base_seed": spec["base_seed"],
                        "tariff": spec["tariff"],
                        "action": action,
                        **stats,
                        "archive_insertions": counts.get("archive_insertions:A" + action, 0),
                        "exact": counts.get("exact:A" + action, 0),
                    }
                )
        if any(len(group) != 1 for group in (hosts, initial, instances, configs)):
            raise ValueError("Unpaired block: " + block["key"])
    if Counter(row["arm"] for row in records) != {arm: 240 for arm in ARMS}:
        raise ValueError("Unbalanced matrix")
    print("All 720 raw run identities and artifact hashes validated", flush=True)
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["jobs"], row["base_seed"], row["tariff"])].append(row)
    references, rows, anytime = {}, [], []
    for group, members in sorted(groups.items()):
        raw_union = [p for m in members for p in m["points"]]
        lower = np.min(raw_union, axis=0)
        span = np.max(raw_union, axis=0) - lower
        if np.any(span <= 0):
            raise ValueError("Degenerate objective normalization")

        def normalize(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
            return [
                (float((p[0] - lower[0]) / span[0]), float((p[1] - lower[1]) / span[1]))
                for p in points
            ]

        front = _nd(normalize(raw_union))
        refs = {
            seed: _nd(
                normalize([p for m in members if m["algorithm_seed"] != seed for p in m["points"]])
            )
            for seed in sorted({m["algorithm_seed"] for m in members})
        }
        name = "_".join(map(str, group))
        references[name] = {
            "lower": lower.tolist(),
            "span": span.tolist(),
            "reference_front": front,
            "reference_points": len(front),
            "all_archive_objectives": len(raw_union),
            "loso_reference_fronts": refs,
        }
        for member in members:
            points = normalize(member["points"])
            s, counts = member["summary"], member["summary"]["counts"]
            row = {
                k: member[k]
                for k in (
                    "key",
                    "block",
                    "arm",
                    "jobs",
                    "base_seed",
                    "tariff",
                    "algorithm_seed",
                    "epoch",
                    "run_epoch",
                    "arm_order",
                )
            }
            row.update(
                hv=_hv(points),
                hv_101=_hv(points, 1.01),
                hv_150=_hv(points, 1.5),
                igd_plus=_igd(points, front),
                igd_loso=_igd(points, refs[member["algorithm_seed"]]),
                min_flow=min(p[0] for p in member["points"]),
                min_bill=min(p[1] for p in member["points"]),
                archive_size=s["archive_size"],
                offspring=s["offspring_count"],
                exact=counts["exact"],
                ssgs=counts["ssgs"],
                elapsed=s["elapsed"],
                overshoot=s["time_overshoot_seconds"],
                offspring_per_second=s["offspring_count"] / s["elapsed"],
                exact_per_second=counts["exact"] / s["elapsed"],
                trigger_rate=s["trigger_rate"],
                trigger_success_rate=s["trigger_success_rate"],
                trigger_hits=counts["trigger_hits"],
                clone_count=counts.get("exact:clone_genotype", 0)
                + counts.get("exact:clone_phenotype", 0),
                clone_archive_insertions=counts.get("archive_insertions:clone_genotype", 0)
                + counts.get("archive_insertions:clone_phenotype", 0),
                ssgs_per_offspring=counts["ssgs"] / s["offspring_count"],
                polish_calls=counts.get("polish_calls", 0),
                polish_accepted=counts.get("polish_accepted", 0),
                timing_origin_share=sum(
                    s["archive_final_by_origin"].get(k, 0) for k in ("A7", "A8", "polish")
                )
                / s["archive_size"],
            )
            # Cross-check the vectorized IGD+ against the frozen project's scalar implementation.
            if member["algorithm_seed"] == 1101 and member["arm"] == "F6":
                if not math.isclose(
                    row["igd_plus"], igd_plus(points, front), rel_tol=1e-12, abs_tol=1e-14
                ):
                    raise ValueError("IGD+ reference implementation mismatch")
            rows.append(row)
            row.update(member["diversity"])
            for snapshot in member["snapshots"]:
                fraction = snapshot["target_seconds"] / s["time_budget_seconds"]
                anytime.append(
                    {
                        "key": row["key"],
                        "arm": row["arm"],
                        "jobs": row["jobs"],
                        "base_seed": row["base_seed"],
                        "tariff": row["tariff"],
                        "fraction": fraction,
                        "target_seconds": snapshot["target_seconds"],
                        "elapsed": snapshot["elapsed"],
                        "snapshot_delay_seconds": snapshot["elapsed"] - snapshot["target_seconds"],
                        "hv": _hv(normalize([tuple(p) for p in snapshot["objectives"]])),
                    }
                )
    contrasts = _contrast_rows(rows)
    summary = _summaries(contrasts, args.seed, args.bootstrap, args.permutations)
    strata_results = {}
    for jobs in (50, 100):
        for tariff in ("H", "T"):
            subset = [r for r in contrasts if r["jobs"] == jobs and r["tariff"] == tariff]
            strata_results[f"{jobs}_{tariff}"] = _summaries(
                subset, args.seed + jobs + ord(tariff), args.bootstrap, 10000
            )
    sensitivity = {}
    for label, predicate in (
        ("no_mixed", lambda r: r["epoch"] != "mixed"),
        ("before_pause", lambda r: r["epoch"] == "before_pause"),
        ("after_resume", lambda r: r["epoch"] == "after_resume"),
    ):
        subset = [r for r in contrasts if predicate(r)]
        sensitivity[label] = _summaries(subset, args.seed + len(subset), args.bootstrap, 10000)
    reference_sensitivity = {}
    for candidate, control in CONTRASTS:
        name = candidate + "-" + control
        chosen = [r for r in contrasts if r["contrast"] == name]
        reference_sensitivity[name] = {}
        for field in ("hv_101_diff", "hv_150_diff", "igd_loso_gain"):
            values, strata = _base_values(chosen, field)
            reference_sensitivity[name][field] = paired_cluster_summary(
                values, strata, np.random.default_rng(args.seed + 9), args.bootstrap, 10000
            )
    diagnostics = {
        "validated_runs": len(rows),
        "paired_blocks": len(manifest["blocks"]),
        "validated_inputs": len(manifest["input_files"]),
        "validated_result_files": sum(len(e["files"]) for e in input_evidence.values()),
        "manifest_sha256": _sha(root / "manifest.json"),
        "by_epoch": dict(Counter(r["epoch"] for r in rows)),
        "snapshot_count": len(anytime),
        "max_snapshot_delay": max(r["snapshot_delay_seconds"] for r in anytime),
        "zero_igd_plus": sum(r["igd_plus"] == 0 for r in rows),
        "reference_sizes": [r["reference_points"] for r in references.values()],
    }
    _csv(output / "run_metrics.csv", rows)
    _csv(output / "paired_contrasts.csv", contrasts)
    _csv(output / "anytime_metrics.csv", anytime)
    _csv(output / "operator_totals.csv", operators)
    _write_json(output / "references.json", references)
    _write_json(output / "input_evidence.json", input_evidence)
    _write_json(
        output / "summary.json",
        {
            "primary": summary,
            "strata": strata_results,
            "epoch_sensitivity": sensitivity,
            "reference_sensitivity": reference_sensitivity,
            "diagnostics": diagnostics,
        },
    )
    print(json.dumps({"diagnostics": diagnostics, "primary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
