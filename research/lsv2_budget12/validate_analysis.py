"""Independent numerical cross-check of exported tables and all declared contrasts."""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    """Recompute indicators and sign flips from exported raw fronts using separate code."""
    root = Path(sys.argv[1]) / "outputs/lsv2_budget12_20261028/analysis"
    fronts = json.loads((root / "fronts.json").read_text())
    report = json.loads((root / "analysis.json").read_text())
    rows = list(csv.DictReader((root / "run_metrics.csv").open(encoding="utf-8-sig")))
    broad = list(csv.DictReader((root / "baseline_metrics.csv").open(encoding="utf-8-sig")))
    checks = 0
    for records, is_broad in ((rows, False), (broad, True)):
        for r in records:
            f = fronts[r["parent"]]
            lo = np.array(f["broad_lo" if is_broad else "lo"])
            span = np.array(f["broad_span" if is_broad else "span"])
            p = (np.array(f["runs"][r["arm"]]) - lo) / span
            u = (np.array(f["broad_reference" if is_broad else "reference"]) - lo) / span
            yy = 1.1
            area = 0.0
            for x, y in sorted(map(tuple, p)):
                if x <= 1.1 and y < yy:
                    area += (1.1 - x) * (yy - y)
                    yy = y
            dist = sum(
                min(math.hypot(max(x - a, 0.0), max(y - b, 0.0)) for x, y in p) for a, b in u
            ) / len(u)
            assert math.isclose(area, float(r["hv"]), abs_tol=1e-10)
            assert math.isclose(dist, float(r["igd_plus"]), abs_tol=1e-10)
            checks += 1
    by = {(r["parent"], r["arm"]): float(r["hv"]) for r in rows}
    parents = sorted(fronts)
    for t in report["tests"]:
        d = np.array([by[p, t["treat"]] - by[p, t["ctrl"]] for p in parents])
        target = abs(d.sum())
        # Separate subset-sum enumeration, rather than the analyzer's sign matrix.
        sums = np.array([0.0])
        for value in d:
            sums = np.concatenate((sums + value, sums - value))
        pv = float(np.mean(abs(sums) >= target - 16e-14))
        assert math.isclose(pv, t["p"], abs_tol=1e-12)
        assert math.isclose(float(d.mean()), t["mean"], abs_tol=1e-12)
        assert int(sum(d > 1e-12)) == t["positive"] and int(sum(d < -1e-12)) == t["negative"]
    for fam in ("primary", "secondary"):
        group = sorted([t for t in report["tests"] if t["family"] == fam], key=lambda t: t["p"])
        adjusted = 0.0
        assert len(group) == 8
        for i, t in enumerate(group):
            adjusted = max(adjusted, min(1.0, (8 - i) * t["p"]))
            assert adjusted == t["p_holm"]
    value = dict(
        accepted=True,
        indicator_pairs=checks,
        contrasts=len(report["tests"]),
        primary_holm=8,
        secondary_holm=8,
        scope="deterministic analysis cross-check only; solver not rerun",
    )
    (root / "VALIDATION.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(value))


if __name__ == "__main__":
    main()
