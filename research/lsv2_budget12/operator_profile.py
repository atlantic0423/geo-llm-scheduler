"""Describe B3L3 actions, using equal-run summaries and explicit selection-bias limits."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _summary(rows: list[dict], reward_tail: float) -> dict:
    """Measure action outcomes; empty means no complete proposal, not no computation."""
    t = np.array([r.get("charged_step_seconds", r["seconds"]) for r in rows])
    g = np.array([r["hv_gain"] for r in rows])
    empty = np.array([r["proposals"] == 0 for r in rows])
    positive = g > 1e-12
    out = dict(
        calls=len(rows),
        empty=float(empty.mean()),
        positive_hv=float(positive.mean()),
        nonempty_no_hv=float((~empty & ~positive).mean()),
        seconds_mean=float(t.mean()),
        seconds_median=float(np.median(t)),
        empty_seconds=float(t[empty].mean()) if empty.any() else None,
        nonempty_seconds=float(t[~empty].mean()) if (~empty).any() else None,
        archive_hit=float(np.mean([r["entered_archive"] > 0 for r in rows])),
        chain_changed=float(np.mean([r["chain"] != "stay" for r in rows])),
        time_sum=float(t.sum()),
    )
    if "q_reward" in rows[0]:
        q = np.array([r["q_reward"] for r in rows])
        pos = q > 0
        top = np.sort(np.maximum(q, 0))[-max(1, int(np.ceil(len(q) * 0.01))) :]
        out.update(
            negative=float((q < 0).mean()),
            reward_positive=float(pos.mean()),
            reward_mean=float(q.mean()),
            reward_median=float(np.median(q)),
            reward_p99=float(np.quantile(q, 0.99)),
            reward_max=float(q.max()),
            positive_reward_mean=float(q[pos].mean()) if pos.any() else None,
            all_negative=int(bool((q < 0).all())),
            tail_rate=float((q >= reward_tail).mean()),
            tail_events=int((q >= reward_tail).sum()),
            positive_mass_top1=float(top.sum() / q[pos].sum()) if pos.any() else None,
            positive_hv_but_negative=float((positive & (q < 0)).mean()),
            empty_reward_median=float(np.median(q[empty])) if empty.any() else None,
            best_set=float(
                np.mean(
                    [r["bandit"]["probability"] > 0.2 / len(r["available"]) + 1e-12 for r in rows]
                )
            ),
        )
    return out


def _csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    """Read sealed Q9/Q0/uniform B3L3 arms; export run, size, phase and segment profiles."""
    root = Path(sys.argv[1]) / "outputs/lsv2_budget12_20261028"
    source = root / "results/formal"
    out = root / "operator_profile_b3l3"
    out.mkdir(exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text())
    sealed = json.loads((root / "results/results_manifest.json").read_text())
    sealed = {r["path"]: r for r in sealed}
    old_rows = list(
        csv.DictReader((root / "analysis/operator_metrics.csv").open(encoding="utf-8-sig"))
    )
    old = {(r["key"], r["arm"], r["operator"]): r for r in old_rows}
    checked = 0
    hashes, totals, details, segments, phases = [], [], [], [], []
    for j in manifest["jobs"]:
        if j["arm"] not in ("Q1_NET_G9_B3L3", "Q1_NET_G0_B3L3", "UNIFORM_B3L3"):
            continue
        key = j["runs"][0]["key"]
        p = source / "results" / j["id"] / ("00_" + key)
        raw = (p / "lsv2_steps.json.gz").read_bytes()
        hashes.append(
            dict(
                path=(p / "lsv2_steps.json.gz").relative_to(root / "results").as_posix(),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
        assert hashes[-1]["sha256"] == sealed[hashes[-1]["path"]]["sha256"]
        assert len(raw) == sealed[hashes[-1]["path"]]["bytes"]
        rows = json.loads(gzip.decompress(raw))
        cfg = json.loads((p / "config.json").read_text())
        arm, n = j["arm"], cfg["spec"]["n"]
        qpos = [r["q_reward"] for r in rows if r.get("q_reward", 0) > 0]
        tail = float(np.quantile(qpos, 0.95)) if qpos else float("inf")
        total_time = sum(r.get("charged_step_seconds", r["seconds"]) for r in rows)
        total_ls = json.loads((p / "summary.json").read_text())["lsv2"]["ls_seconds"]
        totals.append(
            dict(
                arm=arm,
                key=key,
                n=n,
                calls=len(rows),
                time=total_time,
                positive_reward_p95=tail if qpos else None,
            )
        )
        grouped = defaultdict(list)
        for r in rows:
            grouped[r["operator"]].append(r)
        for op, rr in grouped.items():
            s = _summary(rr, tail)
            prior = old[key, arm, op]
            assert s["calls"] == int(prior["calls"])
            assert sum(r["proposals"] == 0 for r in rr) == int(prior["empty"])
            assert sum(r["hv_gain"] > 1e-12 for r in rr) == int(prior["positive"])
            checked += 1
            s.update(
                key=key,
                n=n,
                arm=arm,
                operator=op,
                call_share=len(rr) / len(rows),
                time_share=s["time_sum"] / total_time,
            )
            details.append(s)
            for seg in sorted({r["segment"] for r in rr}):
                ss = [r for r in rr if r["segment"] == seg]
                segments.append(
                    dict(key=key, n=n, arm=arm, operator=op, segment=seg, **_summary(ss, tail))
                )
            for phase in range(4):
                all_phase = [
                    r for r in rows if min(3, int(r["elapsed_ls"] / total_ls * 4)) == phase
                ]
                selected = [r for r in all_phase if r["operator"] == op]
                phases.append(
                    dict(
                        key=key,
                        n=n,
                        arm=arm,
                        operator=op,
                        phase=phase,
                        call_share=len(selected) / len(all_phase),
                        calls=len(selected),
                    )
                )
    _csv(out / "per_run.csv", details)
    _csv(out / "segments.csv", segments)
    _csv(out / "phases.csv", phases)
    _csv(out / "runs.csv", totals)
    aggregate = []
    for arm in sorted({r["arm"] for r in details}):
        for scope in [0, 20, 50, 100, 200]:
            for op in sorted({r["operator"] for r in details if r["arm"] == arm}):
                rr = [
                    r
                    for r in details
                    if r["arm"] == arm and r["operator"] == op and (scope == 0 or r["n"] == scope)
                ]
                a = dict(
                    arm=arm, n=scope, operator=op, runs=len(rr), calls=sum(r["calls"] for r in rr)
                )
                for field in rr[0]:
                    if field in ("key", "arm", "operator", "n", "calls"):
                        continue
                    vals = [r[field] for r in rr if r[field] is not None]
                    a[field] = float(np.mean(vals)) if vals else None
                a["all_negative_runs"] = sum(r.get("all_negative", 0) for r in rr)
                for phase in [0, 3]:
                    pr = [
                        r["call_share"]
                        for r in phases
                        if r["arm"] == arm
                        and r["operator"] == op
                        and r["phase"] == phase
                        and (scope == 0 or r["n"] == scope)
                    ]
                    a[f"phase{phase}_share"] = float(np.mean(pr))
                aggregate.append(a)
    _csv(out / "aggregate.csv", aggregate)
    (out / "source_manifest.json").write_text(json.dumps(hashes, indent=2))
    (out / "scope.json").write_text(
        json.dumps(
            dict(
                runs=len(totals),
                steps=sum(r["calls"] for r in totals),
                sources=len(hashes),
                source_hash_checks=len(hashes),
                independent_prior_operator_rows_checked=checked,
                summary="Equal-run means, with n=0 representing 16 equally weighted instances. Conditional empty/positive means exclude runs without that event; p99 is mean of per-run p99, not pooled p99.",
                tail="Within-run 95th percentile of strictly positive Q rewards; descriptive threshold only, no algorithm parameter.",
                selection="Uniform has no learned selector preference but its visited states are not Q counterfactual states.",
                causal=False,
            ),
            indent=2,
        )
    )
    for arm in ["UNIFORM_B3L3", "Q1_NET_G0_B3L3", "Q1_NET_G9_B3L3"]:
        print(arm)
        for a in aggregate:
            if a["arm"] == arm and a["n"] == 0:
                print(
                    json.dumps(
                        {
                            k: a.get(k)
                            for k in [
                                "operator",
                                "calls",
                                "empty",
                                "positive_hv",
                                "seconds_mean",
                                "empty_seconds",
                                "negative",
                                "reward_positive",
                                "positive_reward_mean",
                                "reward_p99",
                                "tail_rate",
                                "positive_mass_top1",
                                "all_negative_runs",
                                "call_share",
                                "phase0_share",
                                "phase3_share",
                            ]
                        },
                        ensure_ascii=False,
                    )
                )


if __name__ == "__main__":
    main()
