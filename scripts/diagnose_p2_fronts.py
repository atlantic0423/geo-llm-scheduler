"""Supplement primary P2 metrics with geometry and diversity diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from geo_llm_scheduler.experiments.paired_statistics import paired_cluster_summary


def main() -> None:
    """Check objective-export hashes and describe front regions without changing the primary family."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    root, analysis = args.campaign.resolve(), args.analysis.resolve()
    if analysis.is_relative_to(root):
        raise ValueError("Analysis output must be outside the immutable raw campaign")
    refs = json.loads((analysis / "references.json").read_text(encoding="utf-8"))
    evidence = json.loads((analysis / "input_evidence.json").read_text(encoding="utf-8"))
    with (analysis / "run_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metrics: dict[str, dict[str, float]] = {}
    for row in rows:
        group = "_".join(row[k] for k in ("jobs", "base_seed", "tariff"))
        ref = refs[group]
        path = root / "runs" / row["key"] / "objectives.csv"
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != evidence[row["key"]]["files"]["objectives.csv"]:
            raise ValueError("Objective export changed")
        with path.open(encoding="utf-8", newline="") as stream:
            raw = [(float(r["flow_seconds"]), float(r["bill_cny"])) for r in csv.DictReader(stream)]
        points = (np.asarray(raw) - ref["lower"]) / ref["span"]
        front = np.asarray(ref["reference_front"])
        distances = np.sqrt(
            np.square(np.maximum(points[None, :, :] - front[:, None, :], 0)).sum(axis=2)
        ).min(axis=1)
        segments = np.linalg.norm(np.diff(front, axis=0), axis=1)
        weights = np.zeros(len(front))
        weights[:-1] += segments / 2
        weights[1:] += segments / 2
        parts = np.array_split(np.arange(len(front)), 3)
        metrics[row["key"]] = {
            "igd_arc_weighted": float(np.average(distances, weights=weights)),
            "igd_flow_edge": float(distances[parts[0]].mean()),
            "igd_middle": float(distances[parts[1]].mean()),
            "igd_bill_edge": float(distances[parts[2]].mean()),
        }
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[row["block"]][row["arm"]] = row
    outcomes: dict[str, dict[str, Any]] = {}
    for candidate, control in (("CG", "F6"), ("CT", "F6"), ("CT", "CG")):
        name = candidate + "-" + control
        outcomes[name] = {}
        fields = list(metrics[rows[0]["key"]]) + [
            "population_unique_genotypes",
            "population_unique_schedules",
            "structural_offspring",
            "archive_unique_objectives",
        ]
        for field in fields:
            bases: dict[tuple[int, int], list[float]] = defaultdict(list)
            for block in paired.values():
                a, b = block[candidate], block[control]
                if field.startswith("igd"):
                    difference = metrics[b["key"]][field] - metrics[a["key"]][field]
                elif field == "structural_offspring":
                    difference = 100 * (
                        (float(a["offspring"]) - float(a["clone_count"]))
                        / (float(b["offspring"]) - float(b["clone_count"]))
                        - 1
                    )
                else:
                    difference = float(a[field]) - float(b[field])
                bases[(int(a["jobs"]), int(a["base_seed"]))].append(difference)
            values = [float(np.mean(v)) for _, v in sorted(bases.items())]
            strata = [k[0] for k in sorted(bases)]
            outcomes[name][field] = paired_cluster_summary(
                values, strata, np.random.default_rng(20261004), 20000, 10000
            )
    result = {
        "status": "exploratory supplementary diagnostics; not a new confirmatory family",
        "arc_weights": "half adjacent normalized arc lengths; only actual empirical reference points",
        "front_regions": "three equal-count sections sorted by Flow; not preference bins or causal routes",
        "structural_offspring_definition": "processed offspring minus clone-route offspring; an exploration opportunity proxy, not the number of newly distinct genotypes",
        "population_unique_schedules_definition": "distinct serialized start-time vectors, not necessarily distinct complete phenotypes",
        "contrasts": outcomes,
        "arm_descriptive_means": {
            arm: {
                field: float(np.mean([float(r[field]) for r in rows if r["arm"] == arm]))
                for field in [
                    "hv",
                    "igd_plus",
                    "trigger_rate",
                    "trigger_success_rate",
                    "trigger_hits",
                    "offspring",
                    "exact",
                    "overshoot",
                    "archive_size",
                    "archive_unique_objectives",
                    "population_unique_genotypes",
                    "population_unique_schedules",
                    "clone_count",
                    "clone_archive_insertions",
                    "timing_origin_share",
                ]
            }
            for arm in ("F6", "CG", "CT")
        },
    }
    (analysis / "supplementary_diagnostics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                n: {f: {k: v[k] for k in ("mean", "ci95")} for f, v in values.items()}
                for n, values in outcomes.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
