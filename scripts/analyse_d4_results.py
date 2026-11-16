"""Read-only base-clustered analysis of validated D4 prototype artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def paired_interval(values: list[float], seed: int = 2026100703) -> dict:
    """Bootstrap independent base means, never repeated calls, with explicit RNG."""
    if not values or not np.isfinite(values).all():
        raise ValueError("Require finite, nonempty base values")
    a = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(len(a), size=(20000, len(a)))].mean(axis=1)
    return {
        "bases": len(a),
        "mean": float(a.mean()),
        "ci": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "positive_bases": int((a > 0).sum()),
    }


def scalar_value(
    objectives: list[float], ideal: list[float], maximum: list[float], weight: list[float]
) -> float:
    """Recompute the original normalized Tchebycheff from raw objective values."""
    return max(
        weight[k] * abs((objectives[k] - ideal[k]) / max(maximum[k] - ideal[k], 1e-9))
        for k in range(2)
    )


def arm_metrics(arm: dict) -> dict[str, float]:
    """Retain zero-proposal calls and average timing cycles before any aggregation."""
    times = arm["timing"]
    recipes = arm["recipes"]
    return {
        "cpu": float(np.mean([t["cpu"] for t in times])),
        "wall": float(np.mean([t["wall"] for t in times])),
        "proposals": len(arm["candidates"]),
        "nonempty": float(bool(arm["candidates"])),
        "positive": float(arm["best_scalar_gain"] > 0),
        "gain": arm["best_scalar_gain"],
        "recipes": len(recipes),
        "repaired": sum("repaired_starts" in r for r in recipes),
        "certificate": sum(r["stage"] == "certificate" for r in recipes),
        "attempts": arm["attempts"],
    }


def _objectives(evaluation: dict) -> list[float]:
    """Restore exact objective properties from their serialized decomposition."""
    return [evaluation["flow"], evaluation["tou"] + sum(evaluation["demand"])]


def analyse(root: Path, output: Path, acceptance: Path) -> dict:
    """Verify recovered completion hashes, scalar labels and paired outputs; summarize bases.

    Original frozen validation and exact evidence come from the supplied recovery receipt.
    This independently checks all completion-bound unit bytes and label calculations.
    The 32 parent bases were already observed; intervals are developmental descriptions.
    """
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Analysis must remain outside the closed campaign")
    receipt = json.loads(acceptance.read_text(encoding="utf8"))
    if receipt["status"] != "complete_archive_and_each_local_file_verified":
        raise ValueError("Full recovery required")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if (manifest["source_commit"], manifest["source_hash"]) != (
        receipt["source_commit"],
        receipt["source_hash"],
    ):
        raise ValueError("Frozen source mismatch")
    role = manifest["role"]
    output.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    stages: dict[str, Counter] = defaultdict(Counter)
    veto_reference: Counter = Counter()
    observations = units = audits = false_veto = nonempty_recipes = cert_checks = 0
    source_checks = repaired_checks = proposal_checks = output_pairs = 0
    hosts = set()
    fields = ["key", "base", "jobs", "tariff", "panel", "repeat", "source", "arm"]
    metrics = list(
        arm_metrics(
            {
                "timing": [{"cpu": 0, "wall": 0}],
                "recipes": [],
                "candidates": [],
                "best_scalar_gain": 0,
                "attempts": 0,
            }
        )
    ) + ["flow_percent_gain", "bill_percent_gain"]
    with (output / "paired_calls.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fields + metrics)
        writer.writeheader()
        for key in manifest["keys"]:
            _, base, tariff, _ = key.split("_")
            jobs = key.split("_")[0][1:]
            inp = json.loads((root / "inputs" / f"{key}_panels.json").read_text())
            folder = root / "results" / key
            marker = json.loads((folder / "complete.json").read_text())
            hosts.add(marker["host"])
            for filename, expected in sorted(marker["files"].items()):
                data = (folder / filename).read_bytes()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("Changed unit")
                unit = json.loads(data)
                units += 1
                audits += unit["exact_audits"]
                panel = inp["panels"][unit["panel"]]
                for row in unit["rows"]:
                    observations += 1
                    source_checks += 1
                    ref = row["arms"]["REFERENCE"]
                    source = panel["sources"][row["source_index"]]["candidate"]
                    objective = _objectives(source["evaluation"])
                    baseline = scalar_value(
                        objective, row["joint_ideal"], panel["context"]["maximum"], panel["weight"]
                    )
                    for arm, detail in row["arms"].items():
                        gains = [
                            baseline
                            - scalar_value(
                                _objectives(c["evaluation"]),
                                row["joint_ideal"],
                                panel["context"]["maximum"],
                                panel["weight"],
                            )
                            for c in detail["candidates"]
                        ]
                        if abs(max([0.0] + gains) - detail["best_scalar_gain"]) > 1e-12:
                            raise ValueError("Scalar label mismatch")
                        repaired_checks += sum("repaired_starts" in r for r in detail["recipes"])
                        proposal_checks += len(detail["candidates"])
                        if arm in {"GUARD_ALL", "GUARD_SINGLE", "RANDOM_SCORED"}:
                            observed = [
                                {k: c[k] for k in ("genotype", "schedule", "evaluation")}
                                for c in detail["candidates"]
                            ]
                            reference = [
                                {k: c[k] for k in ("genotype", "schedule", "evaluation")}
                                for c in ref["candidates"]
                            ]
                            if observed != reference or detail["attempts"] != ref["attempts"]:
                                raise ValueError("Paired output/attempt mismatch")
                            output_pairs += 1
                        if arm in {"GUARD_ALL", "GUARD_SINGLE"}:
                            if len(detail["recipes"]) != len(ref["recipes"]):
                                raise ValueError("Recipe sequence mismatch")
                            for r, g in zip(ref["recipes"], detail["recipes"], strict=True):
                                if r["members"] != g["members"]:
                                    raise ValueError("Recipe shifted")
                                cert_checks += bool(r["members"])
                                if g["stage"] == "certificate":
                                    veto_reference[arm + ":" + r["stage"]] += 1
                                    veto_reference[arm + ":members=" + str(len(r["members"]))] += 1
                                    false_veto += r["stage"] in {"proposal", "duplicate"}
                        values = arm_metrics(detail)
                        # Include the unchanged source if no candidate improves caller scalar.
                        winner = objective
                        if gains and max(gains) > 0:
                            winner = _objectives(
                                detail["candidates"][int(np.argmax(gains))]["evaluation"]
                            )
                        values["flow_percent_gain"] = (
                            100 * (objective[0] - winner[0]) / max(abs(objective[0]), 1e-9)
                        )
                        values["bill_percent_gain"] = (
                            100 * (objective[1] - winner[1]) / max(abs(objective[1]), 1e-9)
                        )
                        groups[base, arm, "all"].update(values)
                        groups[base, arm, "all"]["calls"] += 1
                        for stratum in (
                            "n" + jobs,
                            "tariff_" + tariff,
                            "panel_" + str(unit["panel"]),
                        ):
                            groups[base, arm, stratum].update(values)
                            groups[base, arm, stratum]["calls"] += 1
                        stages[arm].update(r["stage"] for r in detail["recipes"])
                        if arm == "REFERENCE":
                            nonempty_recipes += sum(bool(r["members"]) for r in detail["recipes"])
                        writer.writerow(
                            dict(
                                key=key,
                                base=base,
                                jobs=jobs,
                                tariff=tariff,
                                panel=unit["panel"],
                                repeat=unit["repeat"],
                                source=row["source_index"],
                                arm=arm,
                                **values,
                            )
                        )
    if false_veto:
        raise ValueError("Observed false certificate veto")
    # Empty region has no recipes; every nonempty recipe has a region and two independent bounds.
    counted_audits = source_checks + repaired_checks + proposal_checks
    if role == "guard":
        counted_audits += cert_checks
    if counted_audits != audits:
        raise ValueError(f"Audit denominator mismatch: {counted_audits} != {audits}")
    bases = sorted({k[0] for k in groups})
    arms = sorted({k[1] for k in groups})
    strata = sorted({k[2] for k in groups})
    means = {
        f"{b}:{arm}:{stratum}": {k: v / total["calls"] for k, v in total.items() if k != "calls"}
        for (b, arm, stratum), total in groups.items()
    }
    summaries = {}
    for stratum in strata:
        available = [b for b in bases if (b, "REFERENCE", stratum) in groups]
        results = {}
        for arm in arms:
            arm_means = [means[f"{b}:{arm}:{stratum}"] for b in available]
            ref_means = [means[f"{b}:REFERENCE:{stratum}"] for b in available]
            results[arm] = {
                "means": {k: float(np.mean([r[k] for r in arm_means])) for k in metrics},
                "gain_delta": paired_interval(
                    [r["gain"] - q["gain"] for r, q in zip(arm_means, ref_means)]
                ),
                "nonempty_delta": paired_interval(
                    [r["nonempty"] - q["nonempty"] for r, q in zip(arm_means, ref_means)]
                ),
                "positive_delta": paired_interval(
                    [r["positive"] - q["positive"] for r, q in zip(arm_means, ref_means)]
                ),
                "cpu_saving": paired_interval(
                    [q["cpu"] - r["cpu"] for r, q in zip(arm_means, ref_means)]
                ),
                "wall_saving": paired_interval(
                    [q["wall"] - r["wall"] for r, q in zip(arm_means, ref_means)]
                ),
                "cpu_percent_saving": paired_interval(
                    [100 * (q["cpu"] - r["cpu"]) / q["cpu"] for r, q in zip(arm_means, ref_means)]
                ),
            }
        summaries[stratum] = results
    report = dict(
        role=role,
        source_commit=manifest["source_commit"],
        source_hash=manifest["source_hash"],
        keys=len(manifest["keys"]),
        bases=len(bases),
        observations=observations,
        units=units,
        hosts=sorted(hosts),
        audit_counts=dict(
            total=audits,
            source_exact=source_checks,
            repaired_exact=repaired_checks,
            proposal_exact=proposal_checks,
            independent_certificate=cert_checks if role == "guard" else 0,
        ),
        false_veto=false_veto,
        output_pairs=output_pairs,
        reference_nonempty_recipes=nonempty_recipes,
        stages={k: dict(v) for k, v in stages.items()},
        veto_reference=dict(veto_reference),
        summaries=summaries,
        inference="32 previously observed parent bases; developmental, not held-out confirmation",
        timing="Construction only, diagnose=False, tracing/exact outside; shared D3 load; not idle/cold-start benchmark",
        interval="20000 base-cluster bootstrap, seed 2026100703; unadjusted descriptive 95% intervals; no new frozen significance gate",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    (output / "base_means.json").write_text(json.dumps(means, indent=2), encoding="utf8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    args = parser.parse_args()
    report = analyse(args.root, args.output, args.acceptance)
    print(json.dumps({k: report[k] for k in ("role", "observations", "audit_counts")}))
