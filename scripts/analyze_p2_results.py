"""Analyze the frozen dual-host 960-run P2 without changing raw artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.experiments.paired_statistics import holm_adjust, paired_cluster_summary

ARMS = ("F6", "RPERM", "RDIR", "A7B")
CONTRASTS = (("RDIR", "F6"), ("RDIR", "RPERM"), ("A7B", "F6"))


def _nd(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result, best = [], math.inf
    for x, y in sorted(set(points)):
        if y < best:
            result.append((x, y))
            best = y
    return result


def _igd(points: list[tuple[float, float]], reference: list[tuple[float, float]]) -> float:
    distances = np.maximum(np.asarray(points)[None, :, :] - np.asarray(reference)[:, None, :], 0)
    return float(np.sqrt(np.square(distances).sum(axis=2)).min(axis=1).mean())


def _hv(points: list[tuple[float, float]], reference: float = 1.1) -> float:
    return float(hypervolume([p for p in points if max(p) <= reference], (reference, reference)))


def _base_values(rows: list[dict[str, Any]], field: str) -> tuple[list[float], list[int]]:
    groups: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row[field] is not None:
            groups[(row["jobs"], row["base_seed"])][row["tariff"]].append(row[field])
    return (
        [
            float(np.mean([np.mean(v) for v in tariffs.values()]))
            for _, tariffs in sorted(groups.items())
        ],
        [key[0] for key in sorted(groups)],
    )


def main() -> None:
    """Export prespecified primary contrasts plus explicitly descriptive guardrails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.data.resolve(), args.output.resolve()
    if out.is_relative_to(root):
        raise ValueError("Keep analysis separate from immutable quality inputs")
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "source_commit": "6dd6d4a84d403a2dd2737d166cb4d3ee1eca25b0",
        "primary": "Three prespecified HV contrasts; Holm three",
        "independent_units": 24,
        "normalization": "all 20 final archives per base/tariff, pooled coordinate min/max",
        "hv_reference": [1.1, 1.1],
        "igd_reference": "pooled strict nondominated empirical approximation",
        "bootstrap": 20000,
        "signflips": 100000,
        "seed": 20261006,
        "guardrails": "IGD+ and middle third of empirical front; positive gain means better",
    }
    (out / "analysis_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    records = []
    for node in (0, 1):
        folder = root / f"formal_node{node}"
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["source_commit"] != plan["source_commit"] or manifest["run_count"] != 480:
            raise ValueError("Unexpected shard")
        for block in manifest["blocks"]:
            for order, key in enumerate(block["runs"]):
                run = folder / "runs" / key
                summary = json.loads((run / "summary.json").read_text())
                with (run / "objectives.csv").open(newline="", encoding="utf-8") as f:
                    points = [
                        (float(r["flow_seconds"]), float(r["bill_cny"])) for r in csv.DictReader(f)
                    ]
                n, base, tariff, seed, arm = key.split("_")
                records.append(
                    {
                        "key": key,
                        "block": block["key"],
                        "jobs": int(n[1:]),
                        "base_seed": int(base[1:]),
                        "tariff": tariff,
                        "algorithm_seed": int(seed[1:]),
                        "arm": arm,
                        "node": node,
                        "order": order,
                        "points": points,
                        "summary": summary,
                        "snapshots": json.loads((run / "anytime.json").read_text()),
                    }
                )
    if len(records) != 960 or {r["arm"] for r in records} != set(ARMS):
        raise ValueError("Incomplete matrix")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        groups[row["jobs"], row["base_seed"], row["tariff"]].append(row)
    rows, anytime, refs = [], [], {}
    for group, members in sorted(groups.items()):
        union = np.asarray([p for m in members for p in m["points"]])
        lo, span = union.min(axis=0), np.ptp(union, axis=0)
        if np.any(span <= 0):
            raise ValueError("Degenerate normalization")

        def norm(p: list) -> list[tuple[float, float]]:
            return [tuple(map(float, q)) for q in (np.asarray(p) - lo) / span]

        front = _nd(norm(union.tolist()))
        middle = front[len(front) // 3 : (2 * len(front)) // 3] or front
        refs[str(group)] = {
            "lower": lo.tolist(),
            "span": span.tolist(),
            "front": front,
            "middle": middle,
        }
        for member in members:
            p, s = norm(member["points"]), member["summary"]
            row = {
                k: member[k]
                for k in (
                    "key",
                    "block",
                    "jobs",
                    "base_seed",
                    "tariff",
                    "algorithm_seed",
                    "arm",
                    "node",
                    "order",
                )
            }
            row.update(
                hv=_hv(p),
                hv101=_hv(p, 1.01),
                hv150=_hv(p, 1.5),
                igd=_igd(p, front),
                middle_igd=_igd(p, middle),
                min_flow=min(q[0] for q in member["points"]),
                min_bill=min(q[1] for q in member["points"]),
                exact_rate=s["counts"]["exact"] / s["elapsed"],
                offspring_rate=s["offspring_count"] / s["elapsed"],
                elapsed=s["elapsed"],
                overshoot=s["time_overshoot_seconds"],
                rss=s["rss_peak_gib"],
                archive_size=s["archive_size"],
                trigger_rate=s["trigger_rate"],
            )
            if (
                member["arm"] == "F6"
                and member["algorithm_seed"] == 1101
                and not math.isclose(row["igd"], igd_plus(p, front), abs_tol=1e-13)
            ):
                raise ValueError("Independent IGD+ mismatch")
            rows.append(row)
            for snap in member["snapshots"]:
                anytime.append(
                    {
                        **{
                            k: row[k]
                            for k in (
                                "key",
                                "block",
                                "arm",
                                "jobs",
                                "base_seed",
                                "tariff",
                                "algorithm_seed",
                            )
                        },
                        "fraction": snap["target_seconds"] / s["time_budget_seconds"],
                        "delay": snap["elapsed"] - snap["target_seconds"],
                        "hv": _hv(norm(snap["objectives"])),
                    }
                )
    blocks: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        blocks[row["block"]][row["arm"]] = row
    paired = []
    for block, arms in sorted(blocks.items()):
        for a, b in CONTRASTS:
            x, y = arms[a], arms[b]
            paired.append(
                {
                    **{k: x[k] for k in ("jobs", "base_seed", "tariff", "algorithm_seed", "node")},
                    "block": block,
                    "contrast": a + "-" + b,
                    "hv_diff": x["hv"] - y["hv"],
                    "hv_pct": 100 * (x["hv"] / y["hv"] - 1),
                    "igd_gain": y["igd"] - x["igd"],
                    "middle_gain": y["middle_igd"] - x["middle_igd"],
                    "flow_pct": 100 * (1 - x["min_flow"] / y["min_flow"]),
                    "bill_pct": 100 * (1 - x["min_bill"] / y["min_bill"]),
                    "exact_pct": 100 * (x["exact_rate"] / y["exact_rate"] - 1),
                    "offspring_pct": 100 * (x["offspring_rate"] / y["offspring_rate"] - 1),
                    "hv101_diff": x["hv101"] - y["hv101"],
                    "hv150_diff": x["hv150"] - y["hv150"],
                }
            )
    metrics = (
        "hv_diff",
        "hv_pct",
        "igd_gain",
        "middle_gain",
        "flow_pct",
        "bill_pct",
        "exact_pct",
        "offspring_pct",
        "hv101_diff",
        "hv150_diff",
    )

    def summaries(selection: list[dict], seed: int) -> dict:
        answer: dict = {}
        rng = np.random.default_rng(seed)
        for a, b in CONTRASTS:
            name = a + "-" + b
            selected = [r for r in selection if r["contrast"] == name]
            answer[name] = {}
            for metric in metrics:
                values, strata = _base_values(selected, metric)
                answer[name][metric] = paired_cluster_summary(values, strata, rng, 20000, 100000)
        return answer

    primary = summaries(paired, 20261006)
    adjusted = holm_adjust([primary[a + "-" + b]["hv_diff"]["p_signflip"] for a, b in CONTRASTS])
    for (a, b), p in zip(CONTRASTS, adjusted, strict=True):
        primary[a + "-" + b]["hv_diff"]["p_holm_three"] = p
    report = {
        "status": "ANALYZED",
        "plan": plan,
        "runs": len(rows),
        "primary": primary,
        "strata_descriptive": {
            f"jobs{n}_{t}": {
                a + "-" + b: {
                    k: float(
                        np.mean(
                            [
                                r[k]
                                for r in paired
                                if r["jobs"] == n
                                and r["tariff"] == t
                                and r["contrast"] == a + "-" + b
                            ]
                        )
                    )
                    for k in metrics
                }
                for a, b in CONTRASTS
            }
            for n in (50, 100)
            for t in ("H", "T")
        },
        "host_sensitivity": {
            str(n): summaries([r for r in paired if r["node"] == n], 20261006 + n + 1)
            for n in (0, 1)
        },
        "arm_means": {
            arm: {
                k: float(np.mean([r[k] for r in rows if r["arm"] == arm]))
                for k in (
                    "hv",
                    "igd",
                    "middle_igd",
                    "min_flow",
                    "min_bill",
                    "exact_rate",
                    "offspring_rate",
                    "elapsed",
                    "overshoot",
                    "rss",
                    "archive_size",
                    "trigger_rate",
                )
            }
            for arm in ARMS
        },
        "limits": [
            "Shared physical CPU; D2 overlap early and D4 overlap later. Within-block arms sequential; randomized/cycled arm order mitigates, does not eliminate interference.",
            "Empirical pooled front is not the true Pareto front.",
            "No default upgrade or novelty conclusion from execution success.",
        ],
    }
    report["screen"] = {
        name: v["hv_diff"]["ci95"][0] > 0
        and v["hv_diff"]["p_holm_three"] < 0.05
        and v["igd_gain"]["mean"] >= 0
        and v["middle_gain"]["mean"] >= 0
        for name, v in primary.items()
    }
    for name, table in (
        ("run_metrics.csv", rows),
        ("paired_metrics.csv", paired),
        ("anytime_metrics.csv", anytime),
    ):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (out / "references.json").write_text(json.dumps(refs), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "primary": {k: v["hv_pct"]["mean"] for k, v in primary.items()},
                "screen": report["screen"],
            }
        )
    )


if __name__ == "__main__":
    main()
