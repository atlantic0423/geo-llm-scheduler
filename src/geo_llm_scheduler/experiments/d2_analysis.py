"""Frozen base-held-out ridge screening; source rows are not independent bases."""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d2_observation import CONTROL_NAMES, OPPORTUNITY_NAMES
from geo_llm_scheduler.utils.rng import RNGManager

POLICIES = ("RANDOM", "QUALITY", "DISTANCE", "HISTORY", "BASE", "OPPORTUNITY", "SHUFFLED")


def ridge_predictions(training: list[dict], testing: list[dict], extended: bool) -> list[float]:
    """Fit fixed alpha=1 standardized ridge with training-only scaling and intercept."""

    def vector(row: dict) -> list[float]:
        return list(row["controls"]) + (list(row["opportunities"]) if extended else [])

    x, test = np.asarray([vector(r) for r in training]), np.asarray([vector(r) for r in testing])
    y = np.asarray([r["gain"] for r in training])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    x, test = (x - mean) / scale, (test - mean) / scale
    center = y.mean()
    coefficient = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ (y - center))
    return [float(v) for v in center + test @ coefficient]


def shuffled_features(rows: list[dict], seed: int) -> list[dict]:
    """Permute only pre-action opportunity vectors within matched source panels."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["key"], row["panel"], row["action"]].append(i)
    result = [dict(row) for row in rows]
    for key, indices in groups.items():
        rng = RNGManager(seed).stream(f"D2:shuffle:{key}")
        permuted = list(indices)
        rng.shuffle(permuted)
        for i, j in zip(indices, permuted):
            result[i]["opportunities"] = rows[j]["opportunities"]
    return result


def paired_summary(values: list[float], seed: int) -> dict:
    """Report paired base means, bootstrap interval and exact two-sided sign flip."""
    if not values:
        return {"bases": 0, "mean": None, "ci": None, "p": None}
    array = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(20000, len(array)))].mean(axis=1)
    observed = abs(float(array.mean()))
    permutations = [
        abs(sum(a * b for a, b in zip(values, signs)) / len(values))
        for signs in itertools.product((-1, 1), repeat=len(values))
    ]
    p = sum(v >= observed - 1e-14 for v in permutations) / len(permutations)
    return {
        "bases": len(values),
        "mean": float(array.mean()),
        "ci": [float(v) for v in np.quantile(means, (0.025, 0.975))],
        "p": p,
        "positive_bases": sum(v > 0 for v in values),
    }


def analyse_opportunity(root: Path) -> dict:
    """Analyze only after complete validation; keep heldout bases wholly out of fitting."""
    from geo_llm_scheduler.experiments.d2_campaign import frozen_manifest, validate_opportunity_job

    manifest = frozen_manifest(root)
    rows, cost_rows, panel_diagnostics = [], [], []
    for key in manifest["keys"]:
        for role in ("samples", "probes"):
            valid, reason = validate_opportunity_job(root, role, key)
            if not valid:
                raise ValueError(f"{role}/{key}: {reason}")
        spec = json.loads((root / "specs" / f"{key}.json").read_text(encoding="utf-8"))
        data = json.loads((root / "probes" / key / "data.json").read_text(encoding="utf-8"))
        for panel in data["panels"]:
            panel_diagnostics.append(
                {
                    "key": key,
                    "panel": panel["d2"]["panel"],
                    "unique_sources": panel["d2"]["unique_sources"],
                    "pool_size": panel["d2"]["pool_size"],
                    # Features are extracted once per source, reused across actions.
                    "feature_seconds": sum(
                        r["feature_seconds"] for r in panel["d2"]["rows"] if r["action"] == 0
                    ),
                    "control_seconds": sum(
                        r["control_seconds"] for r in panel["d2"]["rows"] if r["action"] == 0
                    ),
                }
            )
            for row in panel["d2"]["rows"]:
                rows.append(
                    {
                        **row,
                        "key": key,
                        "base": spec["base_seed"],
                        "split": spec["split"],
                        "panel": panel["d2"]["panel"],
                        "jobs": spec["jobs"],
                    }
                )
            for row in panel["d1"]["rows"]:
                cost_rows.append({**row, "base": spec["base_seed"], "key": key})
    rows.sort(key=lambda r: (r["key"], r["panel"], r["action"], r["index"]))
    report = {
        "status": "ANALYZED",
        "source_commit": manifest["source_commit"],
        "source_hash": manifest["source_hash"],
        "rows": len(rows),
        "development_bases": manifest["development_bases"],
        "heldout_bases": manifest["heldout_bases"],
        "controls": CONTROL_NAMES,
        "opportunities": OPPORTUNITY_NAMES,
        "gain_unit": "absolute difference in common normalized Tchebycheff, not percent",
        "scope": manifest["scope"],
        "online_moead_benefit": "UNVERIFIED",
    }
    report["panel_diagnostics"] = {
        "panels": len(panel_diagnostics),
        "duplicate_phenotype_pools": sum(
            p["unique_sources"] < p["pool_size"] for p in panel_diagnostics
        ),
        "single_phenotype_pools": sum(p["unique_sources"] == 1 for p in panel_diagnostics),
        "feature_seconds": sum(p["feature_seconds"] for p in panel_diagnostics),
        "control_seconds": sum(p["control_seconds"] for p in panel_diagnostics),
    }
    report["action_diagnostics"] = {}
    for action in range(9):
        subset = [r for r in rows if r["action"] == action]
        report["action_diagnostics"][str(action)] = {
            "rows": len(subset),
            "empty_fraction": sum(r["empty"] for r in subset) / len(subset),
            "exact": sum(r["exact"] for r in subset),
            "construction_seconds": sum(r["construction_seconds"] for r in subset),
            "evaluation_seconds": sum(r["evaluation_seconds"] for r in subset),
            "audit_seconds": sum(r["audit_seconds"] for r in subset),
        }
    decisions = []
    if not manifest["pilot"]:
        train = [r for r in rows if r["split"] == "development"]
        test = [r for r in rows if r["split"] == "heldout"]
        train_shuffled, test_shuffled = (
            shuffled_features(train, 923001),
            shuffled_features(test, 923002),
        )
        errors = {}
        for action in range(9):
            tr, te = (
                [r for r in train if r["action"] == action],
                [r for r in test if r["action"] == action],
            )
            trs, tes = (
                [r for r in train_shuffled if r["action"] == action],
                [r for r in test_shuffled if r["action"] == action],
            )
            predictions = {
                "BASE": ridge_predictions(tr, te, False),
                "OPPORTUNITY": ridge_predictions(tr, te, True),
                "SHUFFLED": ridge_predictions(trs, tes, True),
            }
            errors[str(action)] = {
                policy: float(np.mean([(p - r["gain"]) ** 2 for p, r in zip(pred, te)]))
                for policy, pred in predictions.items()
            }
            groups: dict[tuple, list[int]] = defaultdict(list)
            for i, row in enumerate(te):
                groups[row["key"], row["panel"]].append(i)
            for group, indices in groups.items():
                rng = RNGManager(923003).stream(f"D2:random:{action}:{group}")
                selected = {
                    "RANDOM": rng.choice(indices),
                    "QUALITY": min(indices, key=lambda i: (te[i]["controls"][2], te[i]["index"])),
                    "DISTANCE": min(
                        indices,
                        key=lambda i: (te[i]["controls"][3] + te[i]["controls"][4], te[i]["index"]),
                    ),
                    "HISTORY": max(indices, key=lambda i: (te[i]["controls"][7], -te[i]["index"])),
                    **{
                        policy: max(indices, key=lambda i: (pred[i], -te[i]["index"]))
                        for policy, pred in predictions.items()
                    },
                }
                for policy, index in selected.items():
                    decisions.append(
                        {
                            "key": group[0],
                            "panel": group[1],
                            "action": action,
                            "base": te[index]["base"],
                            "policy": policy,
                            "selected": te[index]["index"],
                            "gain": te[index]["gain"],
                            "action_seconds": te[index]["seconds"],
                            # Opportunity selection needs features for the whole pool.
                            "pool_feature_seconds": sum(te[i]["feature_seconds"] for i in indices),
                            "pool_control_seconds": sum(te[i]["control_seconds"] for i in indices),
                        }
                    )
        report["heldout_mse"] = errors
        action_means = {}
        for action in range(9):
            base_means = {
                base: {
                    policy: float(
                        np.mean(
                            [
                                d["gain"]
                                for d in decisions
                                if d["base"] == base
                                and d["action"] == action
                                and d["policy"] == policy
                            ]
                        )
                    )
                    for policy in POLICIES
                }
                for base in manifest["heldout_bases"]
            }
            action_means[str(action)] = {
                "base_means": base_means,
                "policy_means": {
                    policy: float(np.mean([v[policy] for v in base_means.values()]))
                    for policy in POLICIES
                },
                "exploratory_contrasts": {
                    comparator: paired_summary(
                        [v["OPPORTUNITY"] - v[comparator] for v in base_means.values()],
                        924000 + action * 10 + offset,
                    )
                    for offset, comparator in enumerate(("BASE", "SHUFFLED"))
                },
            }
        report["heldout_action_selection"] = action_means
        means = {}
        for base in manifest["heldout_bases"]:
            means[base] = {
                policy: float(
                    np.mean(
                        [
                            d["gain"]
                            for d in decisions
                            if d["base"] == base and d["action"] == 0 and d["policy"] == policy
                        ]
                    )
                )
                for policy in POLICIES
            }
        contrasts = {}
        for offset, comparator in enumerate(("BASE", "SHUFFLED")):
            values = [
                means[b]["OPPORTUNITY"] - means[b][comparator] for b in manifest["heldout_bases"]
            ]
            contrasts[comparator] = paired_summary(values, 923010 + offset)
        ordered = sorted(contrasts, key=lambda k: contrasts[k]["p"])
        last = 0.0
        for i, key in enumerate(ordered):
            last = max(last, min(1.0, contrasts[key]["p"] * (2 - i)))
            contrasts[key]["holm_p"] = last
        report["parent_variation_base_means"] = means
        report["primary_contrasts"] = contrasts
        report["passes_primary_screen"] = all(
            c["mean"] >= 0.001 and c["ci"][0] > 0 and c["holm_p"] < 0.05 for c in contrasts.values()
        )
    else:
        report["passes_primary_screen"] = None
        report["pilot_note"] = "Engineering pilot; no heldout scientific inference."
    report["d1_cost"] = {
        "pairs": len(cost_rows),
        "binding": sum(r["binding"] for r in cost_rows),
        "nonbinding_mismatch": sum(not r["same"] and not r["binding"] for r in cost_rows),
        "reference_seconds": sum(
            r["twins"]["reference"]["shared_seconds"]
            + r["twins"]["reference"]["groups"]["CONDITIONAL"]["seconds"]
            for r in cost_rows
        ),
        "fast_seconds": sum(
            r["twins"]["fast"]["shared_seconds"]
            + r["twins"]["fast"]["groups"]["CONDITIONAL"]["seconds"]
            for r in cost_rows
        ),
    }
    directory = root / "analysis"
    atomic_json(directory / "report.json", report)
    if decisions:
        with (directory / "heldout_decisions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
            writer.writeheader()
            writer.writerows(decisions)
    text = (
        "# D2 opportunity / D1 cost screening\n\n"
        "## Material Passport\n\n"
        "Origin Skill: academic-research-suite / experiment-agent; Mode: run/validate; "
        "Verification Status: ANALYZED; online MOEA/D benefit: UNVERIFIED.\n\n"
        f"Source commit: {manifest['source_commit']}. "
        f"Source runs: {len(manifest['keys'])}; action rows: {len(rows)}.\n\n"
        f"Primary parent-selection screen: {report['passes_primary_screen']}. "
        "Independent units are bases; phases, tariffs, seeds and source slots are repetitions. "
        "A1-A8 donor findings are secondary exploratory diagnostics. "
        "The frozen linear proxies do not cover all possible opportunity representations.\n\n"
        f"Panel diagnostics: {report['panel_diagnostics']}.\n\n"
        "Per-action empty rates, construction/evaluation/audit costs and all heldout "
        "policy comparisons are recorded in report.json even when the primary gate fails.\n\n"
        f"D1 twin result: {report['d1_cost']}.\n\n"
        "Complete original JSON artifacts retain candidates, exact audits, costs and provenance. "
        "Execution completion does not establish novelty or online HV/IGD+ improvement.\n"
    )
    if any(delimiter in text for delimiter in (r"\(", r"\)", r"\[", r"\]")):
        raise ValueError("Legacy math delimiter")
    (directory / "report.md").write_text(text, encoding="utf-8")
    inventory = [
        {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": file_hash(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.relative_to(root).as_posix() != "ops/supervisor.lock"
        and not p.relative_to(root)
        .as_posix()
        .startswith(("ops/heartbeats/", "ops/status", "logs/"))
        and p.name != "files_manifest.json"
    ]
    atomic_json(directory / "files_manifest.json", inventory)
    return report
