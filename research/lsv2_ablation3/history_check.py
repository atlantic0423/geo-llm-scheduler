"""Descriptive same-seed full-control repeat, scored on the current fixed reference."""

import csv
import hashlib
import json
from pathlib import Path

import analyze as a
import numpy as np

root = Path(__import__("sys").argv[1]).resolve() / "outputs/lsv2_ablation3_20261028"
old = root.parent / "lsv2_budget12_20261028/results"
seal = {r["path"]: r for r in a.read(old / "results_manifest.json")}
fronts = a.read(root / "analysis/fronts.json")
rows = []
for parent, f in fronts.items():
    key = parent + "_s16901"
    arm = "Q1_NET_G9_B3L3"
    folder = old / "formal/results" / (key + "_" + arm) / ("00_" + key)
    for name in ("archive.json.gz", "config.json", "summary.json"):
        path = folder / name
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == seal[path.relative_to(old).as_posix()]["sha256"]
        )
    prev = a.read(folder / "config.json")
    curr = a.read(
        root / "results/formal/results" / (key + "_" + arm) / ("00_" + key) / "config.json"
    )
    assert (
        prev["engine"] == curr["engine"]
        and prev["lsv2"] == curr["lsv2"]
        and prev["qspec"] == curr["qspec"]
        and prev["calibration"] == curr["calibration"]
    )
    oldfront = a.nd(a.points(a.read(folder / "archive.json.gz")))
    lo = np.array(f["lo"])
    span = np.array(f["span"])
    oldhv = a.hv((np.array(oldfront) - lo) / span)
    newhv = a.hv((np.array(f["runs"][arm]) - lo) / span)
    rows.append(dict(parent=parent, n=f["n"], old_hv=oldhv, new_hv=newhv, delta=newhv - oldhv))
with (root / "analysis/historical_control_repeat.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as out:
    w = csv.DictWriter(out, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(
    json.dumps(
        dict(
            mean_delta=float(np.mean([r["delta"] for r in rows])),
            mean_absolute_delta=float(np.mean([abs(r["delta"]) for r in rows])),
            max_absolute_delta=max(abs(r["delta"]) for r in rows),
            wins=int(sum(r["delta"] > 0 for r in rows)),
            per_size={
                str(n): float(np.mean([r["delta"] for r in rows if r["n"] == n]))
                for n in (20, 50, 100, 200)
            },
        )
    )
)
