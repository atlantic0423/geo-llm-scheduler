"""Independently replay partial-matrix statistics and cross-check effect aggregation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def audit(out: Path) -> None:
    """Reproduce all 64 main/sensitivity statistics without the analysis helper."""
    data = out / "data"
    report = json.loads((data / "report.json").read_text(encoding="utf-8"))
    with (data / "paired_metrics.csv").open(encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    checked = 0
    for field, complete_only, n in (("main", False, 34), ("sensitivity_complete_bases", True, 30)):
        result = report[field]
        rng = np.random.default_rng(2026101203)
        for label, metrics in result["statistics"].items():
            groups = defaultdict(list)
            for row in pairs:
                if row["contrast"] == label and (
                    not complete_only or row["complete_base"] == "True"
                ):
                    groups[int(row["jobs"]), int(row["base_seed"])].append(row)
            keys = sorted(groups)
            assert len(keys) == n
            for metric, expected in metrics.items():
                effects = np.array(
                    [sum(float(r[metric]) for r in groups[key]) / len(groups[key]) for key in keys]
                )
                assert np.max(np.abs(effects - np.array(expected["base_effects"]))) < 1e-12
                resamples = np.zeros(20000)
                for jobs in (50, 100, 200):
                    group = effects[[key[0] == jobs for key in keys]]
                    draws = rng.integers(len(group), size=(20000, len(group)))
                    resamples += group[draws].sum(axis=1)
                ci = np.percentile(resamples / n, [2.5, 97.5])
                extreme = 0
                for start in range(0, 100000, 4096):
                    signs = 2 * rng.integers(0, 2, size=(min(4096, 100000 - start), n)) - 1
                    extreme += int(
                        (
                            abs((signs * effects).sum(axis=1) / n) >= abs(effects.mean()) - 1e-14
                        ).sum()
                    )
                assert np.isclose(expected["mean"], effects.mean(), atol=1e-12, rtol=1e-12)
                assert np.max(abs(np.array(expected["ci95"]) - ci)) < 1e-12
                assert expected["p_signflip"] == (extreme + 1) / 100001
                checked += 1
        for family in result["families"].values():
            assert len(family) == 16
            p = np.array(
                [result["statistics"][label][metric]["p_signflip"] for label, metric in family]
            )
            order = np.argsort(p, kind="stable")
            adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
            for index, (label, metric) in enumerate(family):
                rank = int(np.flatnonzero(order == index)[0])
                assert result["statistics"][label][metric]["p_holm"] == adjusted[rank]
    # Check every reported descriptive contrast with a separately written base mean.
    for label, expected in report["paired_overall_equal_base"].items():
        groups = defaultdict(list)
        for row in pairs:
            if row["contrast"] == label:
                groups[row["jobs"], row["base_seed"]].append(row)
        for field, value in expected.items():
            calculated = sum(
                sum(float(r[field]) for r in rows) / len(rows) for rows in groups.values()
            ) / len(groups)
            assert np.isclose(value, calculated, atol=1e-12, rtol=1e-12)
    result = dict(
        status="passed",
        statistics_replayed=checked,
        holm_family_sizes=[16, 16, 16, 16],
        main_bases=34,
        sensitivity_bases=30,
        bootstrap=20000,
        signflip=100000,
        seed=2026101203,
        descriptive_contrasts_replayed=len(report["paired_overall_equal_base"]),
        report_sha256=hashlib.sha256((data / "report.json").read_bytes()).hexdigest(),
        paired_metrics_sha256=hashlib.sha256(
            (data / "paired_metrics.csv").read_bytes()
        ).hexdigest(),
        scope="Derived statistics only, not a new full exact solver run",
    )
    (data / "independent_statistics_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    audit(parser.parse_args().out)
