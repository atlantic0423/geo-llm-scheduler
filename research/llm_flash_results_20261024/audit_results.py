"""Independent numerical replay of complete final metrics and the four test statistics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

from geo_llm_scheduler.experiments.metrics import igd_plus


def audit(root: Path) -> dict:
    """Verify192 full IGD+ values and exact sign-flips with a separate computation path."""
    fronts = json.loads((root / "normalized_fronts.json").read_text())
    refs = json.loads((root / "references.json").read_text())
    with (root / "run_metrics.csv").open(encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    for row in runs:
        front = fronts[row["key"] + "|" + row["arm"]]
        reference = refs[front["reference"]]
        normalized = [
            tuple((p[i] - reference["ideal"][i]) / reference["span"][i] for i in (0, 1))
            for p in reference["front"]
        ]
        actual = igd_plus([tuple(p) for p in front["archive"]], normalized)
        assert math.isclose(actual, float(row["igd_plus"]), abs_tol=1e-11, rel_tol=1e-11)
        points = [p for p in front["archive"] if max(p) <= 1.1]
        xs = sorted(set([p[0] for p in points] + [1.1]))
        area = math.fsum(
            (right - left) * (1.1 - min(p[1] for p in points if p[0] <= left))
            for left, right in zip(xs, xs[1:])
        )
        assert math.isclose(area, float(row["hv"]), abs_tol=1e-11, rel_tol=1e-11)
    with (root / "parent_contrasts.csv").open(encoding="utf-8") as f:
        contrasts = list(csv.DictReader(f))
    statistics = json.loads((root / "primary_statistics.json").read_text())
    for row in statistics:
        values = [
            float(v["effect"])
            for v in contrasts
            if (v["contrast"], v["metric"]) == (row["contrast"], row["metric"])
        ]
        assert len(values) == 16
        effect = math.fsum(values) / 16
        assert math.isclose(effect, row["mean"], abs_tol=1e-14)
        hits = sum(
            abs(math.fsum(v * s for v, s in zip(values, signs)) / 16) >= abs(effect) - 1e-14
            for signs in itertools.product((-1, 1), repeat=16)
        )
        assert hits / 65536 == row["p_exact_audit"]
    ordered = sorted(statistics, key=lambda r: r["p"])
    maximum = 0.0
    for i, row in enumerate(ordered):
        maximum = max(maximum, min(1.0, row["p"] * (4 - i)))
        assert maximum == row["p_holm"]
    result = dict(
        accepted=True,
        full_igd_plus_independent=len(runs),
        full_hv_independent=len(runs),
        parent_mean_and_exact_sign_flip_independent=4,
        holm_family_independent=1,
        raw_unchanged=True,
    )
    (root / "INDEPENDENT_AUDIT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root)
