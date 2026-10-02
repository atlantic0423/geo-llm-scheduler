"""Reproduce clustered, exploratory P1 diagnostics from hash-verified reductions.

No result below constitutes an online ablation or a confirmatory significance test.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

SEED = 20261002
BOOTSTRAPS = 20000
TOL = 1e-9


def cluster_interval(values: dict[int, float]) -> dict:
    """Average independent bases equally; bootstrap within the two size strata."""
    ids = sorted(values)
    groups = [[values[b] for b in ids if b // 10000 == size] for size in (81, 82)]
    groups = [np.asarray(g) for g in groups if g]
    rng = np.random.default_rng(SEED)
    means = sum(
        g[rng.integers(0, len(g), size=(BOOTSTRAPS, len(g)))].sum(axis=1) for g in groups
    ) / len(ids)
    return {
        "mean": float(np.mean(list(values.values()))),
        "ci": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "bases": len(ids),
        "positive_bases": sum(v > TOL for v in values.values()),
        "base_values": {str(k): v for k, v in values.items()},
    }


def read_rows(path: Path) -> list[dict]:
    """Read one compact, precomputed diagnostic stream."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def verify_reductions(root: Path) -> None:
    """Reject changed compact data before calculating any research statistic."""
    manifest = json.loads((root / "reduction_manifest.json").read_text())
    for item in manifest["files"]:
        data = (root / item["name"]).read_bytes()
        if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Reduction hash mismatch: {item['name']}")
    if (root / "online_manifest.json").exists():
        item = json.loads((root / "online_manifest.json").read_text())
        data = (root / item["file"]).read_bytes()
        if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Online reduction hash mismatch")


def metric(
    rows: list[dict],
    numerator: Callable[[dict], float],
    denominator: Callable[[dict], float] = lambda r: 1,
) -> dict:
    """Form a within-base rate or mean before averaging independent bases."""
    sums = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        sums[row["base_seed"]][0] += numerator(row)
        sums[row["base_seed"]][1] += denominator(row)
    bases = {b: n / d for b, (n, d) in sums.items() if d > 0}
    if not bases:
        return {"mean": None, "ci": [None, None], "bases": 0, "n": len(rows)}
    result = cluster_interval(bases)
    result.update(
        n=len(rows),
        numerator=sum(v[0] for v in sums.values()),
        denominator=sum(v[1] for v in sums.values()),
    )
    return result


def analyze(root: Path, output: Path) -> dict:
    """Calculate H1-H4, online action statistics and reproducible base tables."""
    verify_reductions(root)
    data = {k: read_rows(root / f"{k}.jsonl.gz") for k in ("runs", "h1", "actions", "h3")}
    manifest = json.loads((root / "reduction_manifest.json").read_text())
    assert all(len(rows) == manifest["counts"][key] for key, rows in data.items())
    results = {}

    def put(
        key: str,
        rows: list[dict],
        num: Callable[[dict], float],
        den: Callable[[dict], float] = lambda r: 1,
    ) -> None:
        results[key] = metric(rows, num, den)

    runs = data["runs"]
    assert len(runs) == 480 and len({r["key"] for r in runs}) == 480
    assert all(
        r["summary"]["source_commit"] == "3ec2355fbc43993375daab06ec0ea39fb9d86991" for r in runs
    )
    for arm in ("F6", "M0"):
        for size in (0, 50, 100):
            rows = [r for r in runs if r["arm"] == arm and (not size or r["jobs"] == size)]
            for key, fn in {
                "observer_fraction": lambda r: (
                    r["summary"]["observer_seconds"] / r["summary"]["elapsed"]
                ),
                "offspring": lambda r: r["summary"]["offspring_count"],
                "exact": lambda r: r["summary"]["counts"]["exact"],
                "archive": lambda r: r["summary"]["archive_size"],
                "archive_peak": lambda r: r["summary"]["archive_peak_size"],
                "rss_gb": lambda r: r["summary"]["peak_rss_gb"],
                "diagnostic_seconds": lambda r: r["summary"]["diagnostics"]["elapsed"],
                "overshoot_seconds": lambda r: r["summary"]["elapsed"] - r["config"]["seconds"],
                "archive_fraction": lambda r: (
                    r["summary"]["timings"]["archive"] / r["summary"]["elapsed"]
                ),
            }.items():
                put(f"runtime/{arm}/{size}/{key}", rows, fn)
    valid = [r for r in data["h1"] if not r["missing"]]
    f6 = [r for r in valid if r["arm"] == "F6"]
    for row in valid:
        row["timing_origin"] = row["origin"] in ("A7", "A8", "polish")
    groups = {
        "F6/all": f6,
        "M0/all": [r for r in valid if r["arm"] == "M0"],
        "F6/timing": [r for r in f6 if r["timing_origin"]],
        "F6/structural": [r for r in f6 if not r["timing_origin"]],
    }
    for origin in sorted({r["origin"] for r in f6}):
        groups[f"F6/origin_{origin}"] = [r for r in f6 if r["origin"] == origin]
    for group, field in [
        ("stratum", "stratum"),
        ("tariff", "tariff"),
        ("size", "jobs"),
        ("stage", "fraction"),
    ]:
        for value in sorted({str(r[field]) for r in f6}):
            groups[f"F6/{group}_{value}"] = [r for r in f6 if str(r[field]) == value]
    for label, rows in groups.items():
        for key, fn in {
            "timing_origin_rate": lambda r: r["timing_origin"],
            "bill_loss_rate": lambda r: r["bill_loss"] > TOL,
            "bill_loss": lambda r: r["bill_loss"],
            "flow_change": lambda r: r["flow_change"],
            "scalar_worse_rate": lambda r: r["scalar_change"] > TOL,
            "scalar_better_rate": lambda r: r["scalar_change"] < -TOL,
            "scalar_change": lambda r: r["scalar_change"],
            "tou_change": lambda r: r["tou_change"],
            "demand_change": lambda r: r["demand_change"],
        }.items():
            put(f"h1/{label}/{key}", rows, fn)
        if label.startswith("F6"):
            for key, fn in {
                "continuation_retained_better": lambda r: (
                    r["continuation_updated_scalar_difference"] > TOL
                ),
                "continuation_redecoded_better": lambda r: (
                    r["continuation_updated_scalar_difference"] < -TOL
                ),
                "continuation_scalar_delta": lambda r: r["continuation_updated_scalar_difference"],
                "continuation_bill_delta": lambda r: r["continuation_bill_difference"],
                "continuation_flow_delta": lambda r: r["continuation_flow_difference"],
                "continuation_seconds": lambda r: r["continuation_seconds"],
            }.items():
                put(f"h1/{label}/{key}", rows, fn)
    actions = data["actions"]
    for arm in ("F6", "M0", "both"):
        for action in range(9):
            rows = [
                r
                for r in actions
                if (arm == "both" or r["arm"] == arm) and (not action or r["action"] == action)
            ]
            for key, num, den in [
                ("improving_rate", lambda r: r["improving_candidates"], lambda r: r["exact"]),
                (
                    "outside_rate",
                    lambda r: r["outside_candidates"],
                    lambda r: r["improving_candidates"],
                ),
                (
                    "missed_rate",
                    lambda r: r["completely_missed"],
                    lambda r: r["improving_candidates"],
                ),
                ("cap_rate", lambda r: r["cap_candidates"], lambda r: r["improving_candidates"]),
                ("outside_pair_rate", lambda r: r["outside_pairs"], lambda r: r["improving_pairs"]),
            ]:
                put(f"h2/{arm}/{action}/{key}", rows, num, den)
            for label in ("random", "gain_order", "direction_neighbors"):
                put(
                    f"h2/{arm}/{action}/{label}_gain_ratio",
                    rows,
                    lambda r, k=label: r["replacement_gain"][k],
                    lambda r: r["replacement_gain"]["original"],
                )
                put(
                    f"h2/{arm}/{action}/{label}_replacement_ratio",
                    rows,
                    lambda r, k=label: r["replacement_counts"][k],
                    lambda r: r["replacement_counts"]["original"],
                )
            for key, fn in {
                "exact": lambda r: r["exact"],
                "empty_rate": lambda r: r["exact"] == 0,
                "scalar_success": lambda r: r["scalar_success"],
                "bill_success": lambda r: r["bill_success"],
                "construction_seconds": lambda r: r["construction_seconds"],
                "evaluation_seconds": lambda r: r["timings"].get("exact", 0),
                "ssgs_seconds": lambda r: r["timings"].get("ssgs", 0),
                "archive_seconds": lambda r: r["timings"].get("archive", 0),
                "late_3_6": lambda r: r["late_3_6"],
                "late_6_10": lambda r: r["late_6_10"],
                "late_3_10": lambda r: r["late_3_10"],
                "rescue_after3": lambda r: r["rescue_after3"],
                "rescue_after_flat6": lambda r: r["rescue_after_flat6"],
                "late_hv_3_6_rate": lambda r: r["late_hv_3_6"] > TOL,
                "late_hv_6_10_rate": lambda r: r["late_hv_6_10"] > TOL,
                "archive_only_late_6_10": lambda r: not r["late_6_10"] and r["late_hv_6_10"] > TOL,
                "prefix3_success": lambda r: r["prefixes"][0]["scalar_gain"] > TOL,
                "prefix6_success": lambda r: r["prefixes"][1]["scalar_gain"] > TOL,
                "prefix10_success": lambda r: r["prefixes"][2]["scalar_gain"] > TOL,
                "prefix3_hv": lambda r: r["prefixes"][0]["hv_gain"],
                "prefix6_hv": lambda r: r["prefixes"][1]["hv_gain"],
                "prefix10_hv": lambda r: r["prefixes"][2]["hv_gain"],
            }.items():
                put(f"h4/{arm}/{action}/{key}", rows, fn)
            if action == 8:
                for key in (
                    "singleton_attempts",
                    "coalition_attempts",
                    "repair_successes",
                    "strict_peak_reductions",
                    "tied_peak_count",
                ):
                    put(f"a8/{arm}/{key}", rows, lambda r, k=key: r["instrumentation"].get(k, 0))
    h3 = data["h3"]
    rng = np.random.default_rng(SEED + 1)
    for row in h3:
        raw = row["raw"]
        n = len(raw)
        row["bill_conflict_rate"] = sum(x["bill_change"] > TOL for x in raw) / n if n else 0
        row["bill_better_rate"] = sum(x["bill_change"] < -TOL for x in raw) / n if n else 0
        row["miss_better_count"] = sum(x["score"] < row["selected_best"] - TOL for x in raw)
        # Equal-sized omitted-raw draws are a sensitivity audit, not a deployable baseline.
        k = min(row["selected_count"], n)
        if k:
            scores = np.array([x["score"] for x in raw])
            row["matched_k_chance"] = float(
                np.mean(
                    [
                        np.min(scores[rng.choice(n, size=k, replace=False)])
                        < row["selected_best"] - TOL
                        for _ in range(100)
                    ]
                )
            )
        else:
            row["matched_k_chance"] = 0.0
    h3groups = {
        "all": h3,
        "F6": [r for r in h3 if r["arm"] == "F6"],
        "M0": [r for r in h3 if r["arm"] == "M0"],
    }
    for field in ("stratum", "tariff", "jobs", "fraction"):
        for value in sorted({str(r[field]) for r in h3}):
            h3groups[f"{field}_{value}"] = [r for r in h3 if str(r[field]) == value]
    for label, rows in h3groups.items():
        for key, fn in {
            "regret_rate": lambda r: r["sampled_regret"] > TOL,
            "regret": lambda r: r["sampled_regret"],
            "raw_count": lambda r: r["raw_count"],
            "selected_count": lambda r: r["selected_count"],
            "extra_count": lambda r: r["extra_count"],
            "bill_conflict": lambda r: r["bill_conflict_rate"],
            "bill_better": lambda r: r["bill_better_rate"],
            "matched_k_chance": lambda r: r["matched_k_chance"],
        }.items():
            put(f"h3/{label}/{key}", rows, fn)
        for layer in ("representative", "pool", "draw"):
            put(
                f"h3/{label}/{layer}_miss_rate",
                rows,
                lambda r, k=layer: r["layers"][k]["better"] > 0,
            )
            put(
                f"h3/{label}/{layer}_better_share",
                rows,
                lambda r, k=layer: r["layers"][k]["better"],
                lambda r: r["miss_better_count"],
            )
    total = {
        "counts": {k: len(v) for k, v in data.items()},
        "missing": [r for r in data["h1"] if r["missing"]],
        "origins_F6": dict(Counter(r["origin"] for r in f6)),
        "online_exact": sum(r["summary"]["counts"]["exact"] for r in runs),
        "offline_counts": dict(
            sum((Counter(r["summary"]["diagnostics"]) for r in runs), Counter())
        ),
        "max_rss_gb": max(r["summary"]["peak_rss_gb"] for r in runs),
        "max_overshoot": max(r["summary"]["elapsed"] - r["config"]["seconds"] for r in runs),
    }
    if (root / "online.jsonl.gz").exists():
        online = read_rows(root / "online.jsonl.gz")
        assert len(online) == 480
        total["initial_control"] = dict(
            sum((Counter(r["initial_control"]) for r in online), Counter())
        )
        total["initial_maxdiff"] = max(r["initial_maxdiff"] for r in online)
        total["online_replacements"] = sum(r["replacement"]["total"] for r in online)
        for arm in ("F6", "M0"):
            rows = [r for r in online if r["arm"] == arm]
            for key, num, den in [
                ("improving_rate", "improving", "total"),
                ("outside_rate", "outside", "improving"),
                ("missed_rate", "missed", "improving"),
                ("cap_rate", "cap", "improving"),
                ("direction_missed_rate", "direction_missed", "improving"),
            ]:
                put(
                    f"online_h2/{arm}/{key}",
                    rows,
                    lambda r, k=num: r["replacement"].get(k, 0),
                    lambda r, k=den: r["replacement"].get(k, 0),
                )
            for label in ("random", "gain_order", "direction_neighbors"):
                put(
                    f"online_h2/{arm}/{label}_gain_ratio",
                    rows,
                    lambda r, k=label: r["replacement_gain"][k],
                    lambda r: r["replacement_gain"]["original"],
                )
            for key in ("last_generation", "q_visited"):
                put(f"online/{arm}/{key}", rows, lambda r, k=key: r[k])
            for action in range(1, 9):
                for key, num, den in [
                    ("call_share", "calls", None),
                    ("accepted_rate", "accepted", "calls"),
                    ("empty_rate", "empty", "calls"),
                    ("effective", "effective", "calls"),
                    ("seconds", "seconds", "calls"),
                    ("archive_only_rate", "archive_only", "calls"),
                    ("construct_seconds", "construction_seconds", "calls"),
                ]:
                    put(
                        f"online_actions/{arm}/{action}/{key}",
                        rows,
                        lambda r, k=num, a=action: r["actions"][str(a)].get(k, 0),
                        (lambda r: sum(x.get("calls", 0) for x in r["actions"].values()))
                        if den is None
                        else (lambda r, k=den, a=action: r["actions"][str(a)].get(k, 0)),
                    )
    output.mkdir(parents=True, exist_ok=True)
    payload = {"seed": SEED, "bootstrap_draws": BOOTSTRAPS, "totals": total, "metrics": results}
    (output / "statistics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output / "base_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "base_seed", "value"])
        for key, result in results.items():
            writer.writerows((key, b, v) for b, v in result.get("base_values", {}).items())
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.input, args.output)
    print(json.dumps({"totals": payload["totals"], "metrics": len(payload["metrics"])}))
