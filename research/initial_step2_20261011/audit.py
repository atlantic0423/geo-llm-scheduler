"""Independently replay the frozen Step-2 cluster statistics and metric boundaries."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_d5_results import distance, nondominated, normalization

from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root
data = root / "analysis" / "data"
report = json.loads((data / "report.json").read_text(encoding="utf-8"))
with (data / "paired_metrics.csv").open(encoding="utf-8") as stream:
    pairs = list(csv.DictReader(stream))
rng = np.random.default_rng(2026101102)
checked = []
for contrast in report["statistics"]:
    bybase = defaultdict(list)
    for row in pairs:
        if row["contrast"] == contrast:
            bybase[int(row["jobs"]), int(row["base_seed"])].append(row)
    keys = sorted(bybase)
    assert len(keys) == 36 and all(len(rows) == 4 for rows in bybase.values())
    for metric in ("hv_diff", "igd_plus_gain"):
        effects = np.array([sum(float(row[metric]) for row in bybase[key]) / 4 for key in keys])
        resamples = np.zeros(20000)
        for jobs in (50, 100, 200):
            strata = effects[[key[0] == jobs for key in keys]]
            assert len(strata) == 12
            draws = rng.integers(12, size=(20000, 12))
            resamples += np.sum(strata[draws], axis=1) / 36
        ci = np.percentile(resamples, [2.5, 97.5])
        observed = abs(effects.mean())
        extreme = 0
        for index in range(0, 100000, 4096):
            signs = rng.integers(0, 2, (min(4096, 100000 - index), 36)) * 2 - 1
            extreme += np.count_nonzero(
                abs(np.sum(signs * effects, axis=1) / 36) >= observed - 1e-14
            )
        expected = report["statistics"][contrast][metric]
        assert abs(expected["mean"] - effects.mean()) < 1e-12
        assert np.max(abs(np.array(expected["ci95"]) - ci)) < 1e-12
        assert expected["p_signflip"] == (extreme + 1) / 100001
        checked.append([contrast, metric])
for family in (report["primary"], report["secondary"]):
    original = [report["statistics"][contrast][metric]["p_signflip"] for contrast, metric in family]
    order = np.argsort(original, kind="stable")
    adjusted = np.maximum.accumulate(np.array(original)[order] * np.arange(len(order), 0, -1))
    for rank, index in enumerate(order):
        contrast, metric = family[int(index)]
        assert report["statistics"][contrast][metric]["p_holm"] == min(1.0, adjusted[rank])

# Independent known answers expose dominance direction, ties and degenerate axes.
assert nondominated([(0, 1), (1, 0), (0, 1), (1, 1), (0, 2)]) == [(0, 1), (1, 0)]
assert np.isclose(hypervolume([(0, 1), (1, 0)], (1.1, 1.1)), 0.21)
assert np.isclose(hypervolume([(0, 0)], (1.1, 1.1)), 1.21)
assert igd_plus([(0, 0)], [(0, 1), (1, 0)]) == 0.0
assert np.isclose(igd_plus([(1, 1)], [(0, 0)]), np.sqrt(2))
lo, span = normalization([(2, 3)])
assert np.array_equal(lo, [2, 3]) and np.array_equal(span, [1, 1])
assert np.isclose(distance([[1, 1]], [[0, 0]], True), np.sqrt(2))

receipt = {
    "status": "passed",
    "statistics_independently_reproduced": len(checked),
    "holm_family_sizes": [len(report["primary"]), len(report["secondary"])],
    "base_clusters": 36,
    "observations_per_base": 4,
    "bootstrap_replicates": 20000,
    "signflip_replicates": 100000,
    "stat_seed": 2026101102,
    "metric_boundary_checks": 7,
    "paired_metrics_sha256": hashlib.sha256((data / "paired_metrics.csv").read_bytes()).hexdigest(),
    "report_sha256": hashlib.sha256((data / "report.json").read_bytes()).hexdigest(),
    "checked": checked,
    "scope": "Statistical/metric audit; no fresh full exact solver reevaluation.",
}
(data / "independent_statistics_audit.json").write_text(
    json.dumps(receipt, indent=2), encoding="utf-8"
)
print(json.dumps(receipt), flush=True)
