"""Independently replay every frozen base statistic and source redundancy check."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True, help="Recovered campaign output directory")
OUT = parser.parse_args().root
DATA = OUT / "analysis/data"
report = json.loads((DATA / "report.json").read_text())
pairs = list(csv.DictReader((DATA / "paired_metrics.csv").open(encoding="utf-8")))
rng = np.random.default_rng(2026101101)
checked = []
for contrast in report["statistics"]:
    members = [r for r in pairs if r["contrast"] == contrast]
    bybase = defaultdict(list)
    for r in members:
        bybase[(int(r["jobs"]), int(r["base_seed"]))].append(r)
    keys = sorted(bybase)
    assert len(keys) == 36 and all(len(v) == 20 for v in bybase.values())
    for metric in ("hv_diff", "igd_plus_gain"):
        effects = np.array([sum(float(r[metric]) for r in bybase[k]) / 20 for k in keys])
        resamples = np.zeros(20000)
        for jobs in (50, 100, 200):
            strata = effects[[k[0] == jobs for k in keys]]
            assert len(strata) == 12
            draws = rng.integers(12, size=(20000, 12))
            resamples += np.sum(strata[draws], axis=1) / 36
        ci = np.percentile(resamples, [2.5, 97.5])
        observed = abs(effects.mean())
        extreme = 0
        for i in range(0, 100000, 4096):
            signs = rng.integers(0, 2, (min(4096, 100000 - i), 36)) * 2 - 1
            extreme += np.count_nonzero(
                abs(np.sum(signs * effects, axis=1) / 36) >= observed - 1e-14
            )
        expected = report["statistics"][contrast][metric]
        assert abs(expected["mean"] - effects.mean()) < 1e-12
        assert np.max(abs(np.array(expected["ci95"]) - ci)) < 1e-12
        assert expected["p_signflip"] == (extreme + 1) / 100001
        checked.append((contrast, metric))
for family in (report["primary"], report["secondary"]):
    original = [report["statistics"][c][m]["p_signflip"] for c, m in family]
    order = np.argsort(original, kind="stable")
    adjusted = np.maximum.accumulate(np.array(original)[order] * np.arange(len(order), 0, -1))
    for rank, i in enumerate(order):
        c, m = family[int(i)]
        assert report["statistics"][c][m]["p_holm"] == min(1.0, adjusted[rank])
source = json.loads((OUT / "formal_analysis_inputs.json").read_text())
records = {(r["spec"]["key"], r["method"]): r for r in source["rows"]}
keys = sorted({k[0] for k in records})


def genes(r: dict) -> list[str]:
    """Return an ordered candidate signature independent of its source label."""
    return [p["genotype_sha256"] for p in r["points"]]


def front(r: dict) -> list[tuple[float, float]]:
    """Build a unique exact objective front for the source redundancy check."""
    points = {
        (p["evaluation"]["flow"], p["evaluation"]["tou"] + sum(p["evaluation"]["demand"]))
        for p in r["points"]
    }
    result = []
    minimum = float("inf")
    for x, y in sorted(points):
        if y < minimum:
            result.append((x, y))
            minimum = y
    return result


same_order = sum(genes(records[k, "FCFS_LOAD"]) == genes(records[k, "KV_AWARE"]) for k in keys)
same_set = sum(
    set(genes(records[k, "FCFS_LOAD"])) == set(genes(records[k, "KV_AWARE"])) for k in keys
)
hkeys = [k for k in keys if records[k, "FCFS_LOAD"]["spec"]["tariff"] == "H"]
energy_h = sum(genes(records[k, "FCFS_LOAD"]) == genes(records[k, "ENERGY_AWARE"]) for k in hkeys)
assert len(keys) == 720 and len(hkeys) == 360
fcfs_kv_fronts = sum(front(records[k, "FCFS_LOAD"]) == front(records[k, "KV_AWARE"]) for k in keys)
receipt = {
    "status": "passed",
    "statistics_independently_reproduced": len(checked),
    "holm_family_sizes": [2, 18],
    "input_sha256": hashlib.sha256((OUT / "formal_analysis_inputs.json").read_bytes()).hexdigest(),
    "FCFS_LOAD_KV_AWARE_identical_gene_order": same_order,
    "FCFS_LOAD_KV_AWARE_identical_gene_sets": same_set,
    "FCFS_LOAD_ENERGY_AWARE_H_identical_gene_order": energy_h,
    "total_comparison_blocks": 720,
    "H_comparison_blocks": 360,
    "FCFS_LOAD_KV_AWARE_identical_objective_fronts": fcfs_kv_fronts,
    "interpretation": "Source credits are observational; no removal or quota-change benefit was tested.",
}
(DATA / "independent_statistics_audit.json").write_text(
    json.dumps(receipt, indent=2), encoding="utf-8"
)
print(json.dumps(receipt), flush=True)
