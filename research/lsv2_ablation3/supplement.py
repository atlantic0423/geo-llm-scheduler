"""Prespecified factorial summaries and frozen A3-rule replay; no solver changes."""

from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(sys.argv[1]).resolve() / "outputs/lsv2_ablation3_20261028"
BASE = ROOT.parent.parent
A = ROOT / "analysis"
CTRL = "Q1_NET_G9_B3L3"
FACTORS = ("P3", "A4", "P2M")


def read(path: Path) -> Any:
    """Read a recovered JSON artifact."""
    with (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(encoding="utf-8-sig") as f
    ):
        return json.load(f)


def csvread(path: Path) -> list[dict]:
    """Load a diagnostic table without changing its labels."""
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def csvwrite(name: str, rows: list[dict]) -> None:
    """Export the complete table with stable column order."""
    with (A / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def arm_for(removed: set[str]) -> str:
    """Map a deletion subset to its canonical frozen arm name."""
    return CTRL + ("_NO_" + "_".join(f for f in FACTORS if f in removed) if removed else "")


def selected(r: dict, rule: dict, budget: float) -> bool:
    """Apply a frozen predecision condition; rewards never enter selection."""
    name = rule["rule"]
    if name == "ALL":
        return True
    if name == "EARLY":
        return r["elapsed_ls"] / budget <= float(rule["cut"])
    if name == "FRAME0":
        return r["frame"] == 0
    raise ValueError(name)


def main() -> None:
    """Reconstruct exploratory factorial effects and frozen-rule diagnostics."""
    result = read(A / "analysis.json")
    metrics = csvread(A / "run_metrics.csv")
    parents = sorted({r["parent"] for r in metrics})
    by = {(r["parent"], r["arm"]): r for r in metrics}
    # Differences in differences, averaged over unused factors; parent is the unit.
    factor_rows = []
    for order in (1, 2, 3):
        for target in itertools.combinations(FACTORS, order):
            others = [f for f in FACTORS if f not in target]
            for p in parents:
                value = 0.0
                for bits in itertools.product((0, 1), repeat=3):
                    removed = {f for f, b in zip(FACTORS, bits) if b}
                    sign = np.prod([1 if f in removed else -1 for f in target])
                    value += sign * float(by[p, arm_for(removed)]["hv"]) / (2 ** len(others))
                factor_rows.append(
                    dict(
                        parent=p,
                        n=int(by[p, CTRL]["n"]),
                        order=order,
                        factors="+".join(target),
                        hv_contrast=value,
                    )
                )
    csvwrite("factorial_per_instance", factor_rows)
    factorial = {
        name: dict(
            mean=float(np.mean([r["hv_contrast"] for r in factor_rows if r["factors"] == name])),
            by_size={
                str(n): float(
                    np.mean(
                        [
                            r["hv_contrast"]
                            for r in factor_rows
                            if r["factors"] == name and r["n"] == n
                        ]
                    )
                )
                for n in (20, 50, 100, 200)
            },
        )
        for name in sorted({r["factors"] for r in factor_rows})
    }
    rules_path = ROOT / "a3_diagnostic/leave_instance_out.csv"
    rules = {r["key"]: r for r in csvread(rules_path) if r["arm"] == CTRL}
    manifest = read(ROOT / "results/formal/manifest.json")
    profiles, a3_rows, events, conditions = [], [], [], []
    first_compare = []
    for job in manifest["jobs"]:
        key = job["runs"][0]["key"]
        arm = job["arm"]
        folder = ROOT / "results/formal/results" / job["id"] / ("00_" + key)
        cfg = read(folder / "config.json")
        rr = read(folder / "lsv2_steps.json.gz")
        budget = cfg["spec"]["seconds"] * 0.2
        rule = rules[key]
        threshold = float(rule["large_reward_threshold"])
        positive = [r["q_reward"] for r in rr if r["q_reward"] > 0]
        descriptive_threshold = float(np.quantile(positive, 0.95)) if positive else float("inf")
        for op in sorted({r["operator"] for r in rr}):
            rows = [r for r in rr if r["operator"] == op]
            q = np.array([r["q_reward"] for r in rows])
            gain = np.array([r["hv_gain"] for r in rows])
            seconds = np.array([r.get("charged_step_seconds", r["seconds"]) for r in rows])
            empty = np.array([r["proposals"] == 0 for r in rows])
            profiles.append(
                dict(
                    key=key,
                    arm=arm,
                    n=cfg["spec"]["n"],
                    op=op,
                    calls=len(rows),
                    empty=int(empty.sum()),
                    empty_rate=float(empty.mean()),
                    positive_hv=int((gain > 1e-12).sum()),
                    negative_reward=int((q < 0).sum()),
                    negative_rate=float((q < 0).mean()),
                    seconds=float(seconds.sum()),
                    empty_seconds=float(seconds[empty].sum()),
                    time_mean=float(seconds.mean()),
                    q_mean=float(q.mean()),
                    q_max=float(q.max()),
                    tail=int((q >= descriptive_threshold).sum()),
                    gain=float(gain.sum()),
                    archive_hits=sum(r["entered_archive"] > 0 for r in rows),
                )
            )
        aa = [r for r in rr if r["operator"] == "A3"]
        ss = [r for r in aa if selected(r, rule, budget)]
        a3_rows.append(
            dict(
                key=key,
                arm=arm,
                n=cfg["spec"]["n"],
                rule=rule["rule"],
                cut=float(rule["cut"]),
                threshold=threshold,
                calls=len(aa),
                selected=len(ss),
                events=sum(r["q_reward"] >= threshold for r in aa),
                hits=sum(r["q_reward"] >= threshold for r in ss),
                positive_hv=sum(r["hv_gain"] > 1e-12 for r in aa),
            )
        )
        for i, r in enumerate(rr):
            if r["operator"] != "A3":
                continue
            prev = (
                rr[i - 1]["operator"]
                if i and rr[i - 1]["trajectory"] == r["trajectory"]
                else "START"
            )
            tail = r["q_reward"] >= descriptive_threshold
            frozen = r["q_reward"] >= threshold
            conditions.append(
                dict(
                    key=key,
                    arm=arm,
                    n=cfg["spec"]["n"],
                    frame0=r["frame"] == 0,
                    step=r["step"],
                    segment=r["segment"],
                    previous=prev,
                    progress=r["elapsed_ls"] / budget,
                    tail=tail,
                    frozen_tail=frozen,
                    positive_hv=r["hv_gain"] > 1e-12,
                )
            )
            if tail or frozen:
                events.append(
                    dict(
                        key=key,
                        arm=arm,
                        n=cfg["spec"]["n"],
                        frame=r["frame"],
                        step=r["step"],
                        segment=r["segment"],
                        previous=prev,
                        progress=r["elapsed_ls"] / budget,
                        elapsed_ls=r["elapsed_ls"],
                        gap=r["start_gap"],
                        q_reward=r["q_reward"],
                        hv_gain=r["hv_gain"],
                        descriptive_tail=tail,
                        frozen_tail=frozen,
                        selected=selected(r, rule, budget),
                        threshold=threshold,
                        descriptive_threshold=descriptive_threshold,
                    )
                )
        for opgroup in ("A3", "OTHER"):
            for first in (True, False):
                z = [
                    r
                    for r in rr
                    if (r["operator"] == "A3") == (opgroup == "A3") and (r["frame"] == 0) == first
                ]
                first_compare.append(
                    dict(
                        key=key,
                        arm=arm,
                        group=opgroup,
                        frame0=first,
                        calls=len(z),
                        tail=sum(r["q_reward"] >= descriptive_threshold for r in z),
                        frozen_tail=sum(r["q_reward"] >= threshold for r in z),
                    )
                )
    csvwrite("action_profile", profiles)
    csvwrite("a3_frozen_rule_per_run", a3_rows)
    csvwrite("a3_events", events)
    csvwrite("a3_first_frame_comparison", first_compare)
    condition_rows = []
    for arm in result["arms"]:
        rr = [r for r in conditions if r["arm"] == arm]
        for field in ("frame0", "step", "segment", "previous"):
            for value in sorted({r[field] for r in rr}, key=str):
                z = [r for r in rr if r[field] == value]
                condition_rows.append(
                    dict(
                        arm=arm,
                        condition=field,
                        value=value,
                        calls=len(z),
                        tail=sum(r["tail"] for r in z),
                        frozen_tail=sum(r["frozen_tail"] for r in z),
                        positive_hv=sum(r["positive_hv"] for r in z),
                        tail_instances=len({r["key"] for r in z if r["tail"]}),
                    )
                )
    csvwrite("a3_conditions", condition_rows)
    a3 = {}
    action = {}
    for arm in result["arms"]:
        z = [r for r in a3_rows if r["arm"] == arm]
        v = {
            k: sum(r[k] for r in z) for k in ("calls", "selected", "events", "hits", "positive_hv")
        }
        v.update(
            precision=v["hits"] / max(1, v["selected"]),
            recall=v["hits"] / max(1, v["events"]),
            selection_share=v["selected"] / max(1, v["calls"]),
            event_instances=sum(r["events"] > 0 for r in z),
            hit_instances=sum(r["hits"] > 0 for r in z),
        )
        a3[arm] = v
        z = [r for r in profiles if r["arm"] == arm]
        v = {
            k: sum(r[k] for r in z)
            for k in (
                "calls",
                "empty",
                "positive_hv",
                "seconds",
                "empty_seconds",
                "negative_reward",
                "gain",
                "tail",
            )
        }
        v.update(
            empty_rate=v["empty"] / v["calls"],
            negative_rate=v["negative_reward"] / v["calls"],
            positive_hv_rate=v["positive_hv"] / v["calls"],
            empty_time_share=v["empty_seconds"] / v["seconds"],
        )
        action[arm] = v
    payload = dict(
        factorial=factorial,
        a3_frozen_rules=a3,
        actions=action,
        frozen_rule_sha256=hashlib.sha256(rules_path.read_bytes()).hexdigest(),
        warning="Same dev instances and seed, correlated early paths. No independent confirmation, no matched-state causal comparison. Per-run descriptive-tail and frozen training-threshold events are different labels.",
    )
    (A / "supplement.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
