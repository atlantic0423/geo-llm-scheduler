"""Independently replay cluster inference and check full raw recovery receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def audit(data: Path) -> dict:
    """Recompute all twelve frozen statistics and three Holm families from pair CSV."""
    report = json.loads((data / "report.json").read_text(encoding="utf-8"))
    with (data / "pairs.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rng = np.random.default_rng(2026101204)
    checked = []
    for arm, metrics in report["statistics"].items():
        groups = defaultdict(list)
        for r in rows:
            if r["arm"] == arm:
                groups[int(r["jobs"]), int(r["base_seed"])].append(r)
        keys = sorted(groups)
        assert len(keys) == 36 and all(len(g) == 4 for g in groups.values())
        for metric, expected in metrics.items():
            effects = np.asarray([sum(float(r[metric]) for r in groups[k]) / 4 for k in keys])
            samples = np.zeros(20000)
            for jobs in (50, 100, 200):
                part = effects[[k[0] == jobs for k in keys]]
                assert len(part) == 12
                draws = rng.integers(12, size=(20000, 12))
                samples += part[draws].sum(axis=1)
            ci = np.percentile(samples / 36, [2.5, 97.5])
            extreme = 0
            for start in range(0, 100000, 4096):
                signs = 2 * rng.integers(0, 2, (min(4096, 100000 - start), 36)) - 1
                extreme += np.count_nonzero(
                    abs((signs * effects).sum(axis=1) / 36) >= abs(effects.mean()) - 1e-14
                )
            assert abs(effects.mean() - expected["mean"]) < 1e-12
            assert max(abs(ci - np.asarray(expected["ci95"]))) < 1e-12
            assert expected["p_signflip"] == (extreme + 1) / 100001
            assert np.allclose(effects, expected["base_effects"], rtol=0, atol=1e-12)
            checked.append((arm, metric))
    for family in report["families"].values():
        ps = np.asarray([report["statistics"][a][m]["p_signflip"] for a, m in family])
        order = np.argsort(ps, kind="stable")
        adjusted = np.minimum(1, np.maximum.accumulate(ps[order] * np.arange(len(ps), 0, -1)))
        for rank, i in enumerate(order):
            a, m = family[int(i)]
            assert report["statistics"][a][m]["p_holm"] == adjusted[rank]
    receipt = {
        "status": "passed",
        "independent_statistics": len(checked),
        "holm_family_sizes": [len(x) for x in report["families"].values()],
        "pair_csv_sha256": hashlib.sha256((data / "pairs.csv").read_bytes()).hexdigest(),
        "no_fresh_generation_or_all_member_exact": True,
    }
    (data / "independent_statistics_audit.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.data)))
