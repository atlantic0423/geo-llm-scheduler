"""Independent CSV base aggregation of paired cost and quality evidence."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(sys.argv[1])
for role in sys.argv[2:]:
    out = BASE / ("d4_" + role)
    a = out / "analysis"
    r = json.loads((a / "report.json").read_text())
    b = json.loads((a / "base_means.json").read_text())
    totals = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    timing = defaultdict(list)
    with (a / "paired_calls.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            key = row["base"], row["arm"]
            counts[key] += 1
            for metric in (
                "cpu",
                "wall",
                "gain",
                "nonempty",
                "positive",
                "recipes",
                "certificate",
                "proposals",
                "flow_percent_gain",
                "bill_percent_gain",
            ):
                totals[key][metric] += float(row[metric])
            timing[row["arm"]].append(float(row["cpu"]))
    assert len({key[0] for key in counts}) == 32
    assert all(count == r["observations"] // 32 for count in counts.values())
    for (base, arm), values in totals.items():
        for metric, value in values.items():
            assert abs(value / counts[base, arm] - b[f"{base}:{arm}:all"][metric]) < 1e-12, (
                base,
                arm,
                metric,
            )
    bases = sorted({key[0] for key in counts})
    tests = {}
    for arm in sorted(timing):
        cpu = [
            totals[base, "REFERENCE"]["cpu"] / counts[base, "REFERENCE"]
            - totals[base, arm]["cpu"] / counts[base, arm]
            for base in bases
        ]
        gain = [
            totals[base, arm]["gain"] / counts[base, arm]
            - totals[base, "REFERENCE"]["gain"] / counts[base, "REFERENCE"]
            for base in bases
        ]
        for name, values in [("cpu_saving", cpu), ("gain_delta", gain)]:
            values = np.array(values)
            rng = np.random.default_rng(2026100703)
            ci = np.quantile(values[rng.integers(32, size=(20000, 32))].mean(1), [0.025, 0.975])
            expected = r["summaries"]["all"][arm][name]
            assert np.allclose(ci, expected["ci"], atol=1e-12, rtol=1e-12)
            assert abs(float(values.mean()) - expected["mean"]) < 1e-12
            tests[arm + ":" + name] = dict(mean=float(values.mean()), ci=ci.tolist())
    additional = {}
    if role == "guard":
        for arm in ("GUARD_ALL", "GUARD_SINGLE"):
            rates = [
                totals[base, arm]["certificate"] / totals[base, "REFERENCE"]["recipes"]
                for base in bases
            ]
            rng = np.random.default_rng(2026100703)
            arr = np.array(rates)
            additional[arm + "_base_equal_deny"] = dict(
                mean=float(arr.mean()),
                ci=np.quantile(
                    arr[rng.integers(32, size=(20000, 32))].mean(1), [0.025, 0.975]
                ).tolist(),
            )
        delta = np.array(
            [
                totals[base, "GUARD_ALL"]["cpu"] / counts[base, "GUARD_ALL"]
                - totals[base, "GUARD_SINGLE"]["cpu"] / counts[base, "GUARD_SINGLE"]
                for base in bases
            ]
        )
        rng = np.random.default_rng(2026100703)
        additional["SINGLE_vs_ALL_cpu_saving"] = dict(
            mean=float(delta.mean()),
            ci=np.quantile(
                delta[rng.integers(32, size=(20000, 32))].mean(1), [0.025, 0.975]
            ).tolist(),
        )
    else:
        for name in ("cpu", "gain"):
            # Positive saving means PEAK_RANK costs less than RANDOM_SCORED; positive gain means more utility.
            delta = np.array(
                [
                    (totals[base, "RANDOM_SCORED"][name] - totals[base, "PEAK_RANK"][name])
                    / counts[base, "PEAK_RANK"]
                    * (1 if name == "cpu" else -1)
                    for base in bases
                ]
            )
            rng = np.random.default_rng(2026100703)
            additional["PEAK_vs_SCORED_" + name] = dict(
                mean=float(delta.mean()),
                ci=np.quantile(
                    delta[rng.integers(32, size=(20000, 32))].mean(1), [0.025, 0.975]
                ).tolist(),
            )
    receipt = dict(
        status="independent_csv_aggregation_and_cluster_intervals_passed",
        base_arm_counts={f"{b}:{arm}": n for (b, arm), n in counts.items()},
        tests=tests,
        additional=additional,
        cpu_call_distribution={
            arm: dict(zip(("p50", "p90", "p99"), np.quantile(values, [0.5, 0.9, 0.99]).tolist()))
            for arm, values in timing.items()
        },
    )
    (a / "independent_verification.json").write_text(json.dumps(receipt, indent=2))
    print(
        json.dumps({"role": role, "status": receipt["status"], "additional": additional}),
        flush=True,
    )
