"""Analyze the sealed 120 by 14 campaign using common per-instance objectives."""

import csv
import gzip
import json
import math
import os
import statistics as st
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parent
ROOT = (
    Path(os.environ["S4_ANALYSIS_ROOT"])
    if "S4_ANALYSIS_ROOT" in os.environ
    else B / "results/formal"
)


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def front(points):
    out = []
    y0 = math.inf
    for x, y in sorted(set(map(tuple, points))):
        if y < y0:
            out.append((x, y))
            y0 = y
    return out


def hv(points, ref=1.1):
    last = ref
    value = 0.0
    for x, y in points:
        if x <= ref and y < last:
            value += (ref - x) * (last - y)
            last = y
    return value


def igd(points, reference):
    p = np.array(points)
    r = np.array(reference)
    ans = []
    for chunk in np.array_split(r, max(1, math.ceil(len(r) / 200))):
        d = np.maximum(p[None, :, :] - chunk[:, None, :], 0)
        ans.extend(np.sqrt((d * d).sum(axis=2).min(axis=1)))
    return float(np.mean(ans))


def main():
    compact = None
    if os.environ.get("S4_COMPACT"):
        with gzip.open(os.environ["S4_COMPACT"], "rt", encoding="utf8") as f:
            compact = json.load(f)
        m = compact["manifest"]
        compacts = {c["key"]: c for c in compact["cases"]}
    else:
        m = read(ROOT / "manifest.json")
    arms = m["arms"]
    data = []
    cases = []
    figs = B / "figures"
    figs.mkdir(exist_ok=True)
    for key in m["blocks"]:
        spec = read(ROOT / "specs" / f"{key}.json") if not compact else {"n": compacts[key]["n"]}
        fronts = {}
        summaries = {}
        for arm in arms:
            if compact:
                fronts[arm] = compacts[key]["arms"][arm]["front"]
                summaries[arm] = compacts[key]["arms"][arm]["summary"]
            else:
                p = ROOT / "runs" / key / arm
                with gzip.open(p / "archive.json.gz", "rt", encoding="utf8") as f:
                    raw = json.load(f)
                assert raw and all(
                    x["evaluation"]["feasible"] and not x["evaluation"]["violations"] for x in raw
                )
                fronts[arm] = front(
                    [
                        (
                            x["evaluation"]["flow"],
                            x["evaluation"]["tou"] + sum(x["evaluation"]["demand"]),
                        )
                        for x in raw
                    ]
                )
                summaries[arm] = read(p / "summary.json")
        pooled = [p for f in fronts.values() for p in f]
        lo = [min(p[j] for p in pooled) for j in (0, 1)]
        span = [max(p[j] for p in pooled) - lo[j] or 1 for j in (0, 1)]

        def norm(f):
            return [((x - lo[0]) / span[0], (y - lo[1]) / span[1]) for x, y in f]

        normed = {a: norm(f) for a, f in fronts.items()}
        ref = front([p for f in normed.values() for p in f])
        vals = {a: hv(f) for a, f in normed.items()}
        base = vals["A0_FULL_S4"]
        base_igd = igd(normed["A0_FULL_S4"], ref)
        for arm in arms:
            sm = summaries[arm]
            cost = sm["counts"]
            row = dict(
                key=key,
                n=spec["n"],
                arm=arm,
                hv=vals[arm],
                hv_vs_a0_pct=100 * (vals[arm] / base - 1),
                igd_plus=base_igd if arm == "A0_FULL_S4" else igd(normed[arm], ref),
                rank=1 + sum(v > vals[arm] + 1e-12 for v in vals.values()),
                min_flow=min(x for x, y in fronts[arm]),
                min_bill=min(y for x, y in fronts[arm]),
                front_points=len(fronts[arm]),
                archive_size=sm["archive_size"],
                exact=cost.get("exact", 0),
                elapsed=sm["elapsed_seconds"],
                budget=sm["budget_seconds"],
                overrun=sm["overrun_seconds"],
            )
            data.append(row)
        cases.append(dict(key=key, n=spec["n"], values=vals))
        print(key, flush=True)
    with (B / "per_run_analysis.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    result = {}
    for n in ("all", 50, 100, 200):
        rs = [r for r in data if n == "all" or r["n"] == n]
        group = {}
        for a in arms:
            ar = [r for r in rs if r["arm"] == a]
            delta = [r["hv_vs_a0_pct"] for r in ar]
            group[a] = dict(
                mean=st.mean(delta),
                median=st.median(delta),
                wins=sum(x > 1e-8 for x in delta),
                losses=sum(x < -1e-8 for x in delta),
                ties=sum(abs(x) <= 1e-8 for x in delta),
                mean_rank=st.mean(r["rank"] for r in ar),
                first=sum(r["rank"] == 1 for r in ar),
                igd=st.mean(r["igd_plus"] for r in ar),
                exact=st.mean(r["exact"] for r in ar),
                archive_size=st.mean(r["archive_size"] for r in ar),
            )
        result[str(n)] = group
    pvalues = []
    for a in ["A1_NO_SHIFT", "A2_RANDOM_OPERATOR", "A3_RANDOM_INIT", "A4_DIRECT_SHIFT"]:
        diffs = [c["values"]["A0_FULL_S4"] - c["values"][a] for c in cases]
        p = (
            float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
            if any(x != 0 for x in diffs)
            else 1.0
        )
        pvalues.append((a, p))
    ordered = sorted(pvalues, key=lambda x: x[1])
    prev = 0
    corrected = {}
    for i, (a, p) in enumerate(ordered):
        prev = max(prev, min(1, p * (len(ordered) - i)))
        corrected[a] = dict(raw=p, holm=prev)
    summary = dict(
        groups=result,
        ablation_wilcoxon=corrected,
        instances=120,
        runs=1680,
        seed=1,
        method="All 14 final fronts jointly normalized per instance, reference 1.1, IGD+ to pooled nondominated front; Wilcoxon paired HV across 120 cases Holm four tests",
        caveat="One run per instance and arm, non-blind constructed panel, 4 Trans are Python adaptations, BILL4+MOEAD not in matrix",
    )
    (B / "ANALYSIS_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8"
    )
    labels = arms
    means = [result["all"][a]["mean"] for a in arms]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(
        labels[::-1],
        means[::-1],
        color=["#d45a62" if a == "A0_FULL_S4" else "#487ca0" for a in labels[::-1]],
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Mean per-instance final HV change vs S4 (%)",
        title="120 instances | seed 1 | common reference",
    )
    fig.tight_layout()
    fig.savefig(figs / "hv_comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
