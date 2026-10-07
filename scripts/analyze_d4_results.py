"""Independently audit immutable D4 recipe records and descriptive strata."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def summarize_calls(rows: list[dict]) -> dict:
    """Keep call/candidate/recipe denominators distinct and reject false vetoes."""
    counts: Counter = Counter()
    costs: Counter = Counter()
    for row in rows:
        counts.update(
            calls=1,
            empty=int(row["empty"]),
            exact=row["exact"],
            bill_improved=row["bill_improved"],
            scalar_improved=row["scalar_improved"],
            scalar_positive_calls=int(row["gain"] > 0),
        )
        for name in ("construction_seconds", "evaluation_seconds", "audit_seconds"):
            costs[name] += row[name]
        for recipe in row["recipes"]:
            cert = recipe["certificate"]
            if cert["denied"] and (
                not cert["applicable"] or recipe["stage"] in ("proposal", "duplicate")
            ):
                raise ValueError("Invalid applicability or false veto")
            counts.update(
                recipes=1,
                nonempty_recipes=int(bool(recipe["members"])),
                applicable=int(cert["applicable"]),
                denied=int(cert["denied"]),
                prunable_recipes=int(bool(recipe["members"]) and cert["denied"]),
            )
            counts[recipe["stage"]] += 1
            counts["denied_" + recipe["stage"]] += int(cert["denied"])
            costs["certificate_seconds"] += cert["seconds"]
            costs["repair_seconds"] += recipe.get("repair_seconds", 0)
            costs["denied_repair_seconds"] += (
                recipe.get("repair_seconds", 0) if cert["denied"] else 0
            )
    return {"counts": dict(counts), "costs": dict(costs)}


def base_bootstrap(rates: list[float]) -> dict:
    """Reproduce the frozen equal-base nonempty-recipe gate without pseudoreplication."""
    a = np.asarray(rates)
    rng = np.random.default_rng(2026100604)
    boot = a[rng.integers(0, len(a), (20000, len(a)))].mean(axis=1)
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    return {"mean": float(a.mean()), "ci": ci, "candidate_gate": ci[0] > 0.1}


def main() -> None:
    """Audit the full matrix and write compact, reproducible JSON/CSV outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root, args.output
    manifest = json.loads((root / "manifest.json").read_text())
    frozen = json.loads((root / "analysis/report.json").read_text())
    by_base: dict[int, list[dict]] = defaultdict(list)
    strata: dict[str, list[dict]] = defaultdict(list)
    for key in manifest["keys"]:
        spec = json.loads((root / "specs" / f"{key}.json").read_text())
        data = json.loads((root / "probes" / key / "data.json").read_text())
        for panel in data["panels"]:
            label = panel["d2"]["panel"]
            for row in panel["d2"]["rows"]:
                by_base[spec["base_seed"]].append(row)
                for group in (key.split("_")[0], key.split("_")[2], *label.split("_")):
                    strata[group].append(row)
    bases = {str(b): summarize_calls(rows) for b, rows in sorted(by_base.items())}
    total = summarize_calls([r for rows in by_base.values() for r in rows])
    for name, value in frozen["total"].items():
        assert total["counts"][name] == value, name
    for b, summary in bases.items():
        for name, value in frozen["bases"][b].items():
            assert summary["counts"][name] == value, (b, name)
        for name, value in frozen["base_costs"][b].items():
            assert np.isclose(summary["costs"][name], value, rtol=1e-12, atol=1e-12)
    gate = base_bootstrap(
        [v["counts"]["prunable_recipes"] / v["counts"]["nonempty_recipes"] for v in bases.values()]
    )
    assert gate["mean"] == frozen["prunable_nonempty_base_mean"]
    assert gate["ci"] == frozen["prunable_nonempty_base_ci"]
    assert gate["candidate_gate"] == frozen["candidate_gate"]
    result = {
        "status": "VERIFIED_independent_record_and_frozen_gate_recompute",
        "source_commit": manifest["source_commit"],
        "source_hash": manifest["source_hash"],
        "bases": bases,
        "total": total,
        "gate": gate,
        "descriptive_strata": {k: summarize_calls(v) for k, v in sorted(strata.items())},
        "scope": "Offline original A8; no online runtime, quality, or novelty claim.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "independent_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "base_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base", "calls", "empty", "recipes", "prunable_recipes", "deny_rate"])
        for b, v in bases.items():
            c = v["counts"]
            w.writerow(
                [
                    b,
                    c["calls"],
                    c["empty"],
                    c["recipes"],
                    c["prunable_recipes"],
                    c["prunable_recipes"] / c["nonempty_recipes"],
                ]
            )
    print(json.dumps({"total": total, "gate": gate}))


if __name__ == "__main__":
    main()
