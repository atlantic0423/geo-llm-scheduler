"""Analyze frozen D7 eight-arm data with independent base-cluster inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.rl.state import decode

ARMS = ("BASE", "STATE6", "STATE84", "B3", "B10", "SEQ", "A3_TOU", "RPERM")
CONTRASTS = tuple((a, "BASE") for a in ARMS[1:])
PRIMARY = tuple((a + "-BASE", "hv_diff") for a in ARMS[1:])
SECONDARY = tuple((a + "-BASE", "igd_plus_gain") for a in ARMS[1:])
METRICS = (
    "hv_diff",
    "igd_plus_gain",
    "hv_pct",
    "igd_gain",
    "coverage_net",
    "flow_gain_pct",
    "bill_gain_pct",
    "exact_pct",
    "offspring_pct",
    "a8_cost_per_call_pct",
    "a8_cost_share_pp",
    "a8_proposals_per_call_diff",
)
SEED = 2026100907
SOURCE = "f01b1417cf6be1cb5e1f375dbc710eb575c06123"
CANONICAL = "6e4385dd639db34aec25f526943eda84bb66bb9961ca12252ef138d7888da114"


def nondominated(points: list) -> list[tuple[float, float]]:
    """Return unique strict two-objective minimization front, including equal-x ties."""
    result, best = [], math.inf
    for x, y in sorted(set(tuple(p) for p in points)):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Nonfinite objective")
        if y < best:
            result.append((x, y))
            best = y
    return result


def normalization(front: list) -> tuple[np.ndarray, np.ndarray]:
    """Use the common empirical front's ideal/nadir; leave a degenerate axis in units."""
    if not front:
        raise ValueError("Empty reference front")
    data = np.asarray(front)
    lo = data.min(axis=0)
    span = data.max(axis=0) - lo
    return lo, np.where(span <= 1e-12, 1.0, span)


def distance(points: list, front: list, plus: bool) -> float:
    """Compute IGD or IGD+ in bounded chunks, without clipping approximation points."""
    if not points or not front:
        raise ValueError("Empty approximation or reference")
    data, ref = np.asarray(points), np.asarray(front)
    sums = 0.0
    for start in range(0, len(ref), 128):
        diff = data[None, :, :] - ref[start : start + 128, None, :]
        if plus:
            diff = np.maximum(diff, 0)
        sums += np.sqrt(np.square(diff).sum(axis=2)).min(axis=1).sum()
    return float(sums / len(ref))


def coverage(a: list, b: list) -> float:
    """Fraction of B weakly dominated by A with the fixed normalized 1e-12 tolerance."""
    if not a or not b:
        raise ValueError("Coverage requires nonempty sets")
    return sum(any(all(x <= y + 1e-12 for x, y in zip(p, q)) for p in a) for q in b) / len(b)


def holm(pvalues: list[float]) -> list[float]:
    """Apply Holm step-down to a complete hypothesis family, preserving order."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("Invalid p-value")
    result = [0.0] * len(pvalues)
    previous = 0.0
    for rank, i in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        previous = max(previous, (len(pvalues) - rank) * pvalues[i])
        result[i] = min(previous, 1.0)
    return result


def cluster_summary(
    values: list,
    strata: list,
    rng: np.random.Generator,
    bootstrap: int = 20000,
    signflips: int = 100000,
) -> dict:
    """Resample independent base effects within size strata; test symmetric null signs."""
    if not values or len(values) != len(strata) or not all(map(math.isfinite, values)):
        raise ValueError("Invalid clusters")
    data = np.asarray(values)
    sums = np.zeros(bootstrap)
    for s in sorted(set(strata)):
        group = data[np.asarray(strata) == s]
        sums += group[rng.integers(len(group), size=(bootstrap, len(group)))].sum(axis=1)
    ci = np.quantile(sums / len(data), [0.025, 0.975])
    observed, extreme = abs(float(data.mean())), 0
    for start in range(0, signflips, 4096):
        signs = 2 * rng.integers(0, 2, size=(min(4096, signflips - start), len(data))) - 1
        extreme += int((np.abs((signs * data).mean(axis=1)) >= observed - 1e-14).sum())
    sd = float(data.std(ddof=1)) if len(data) > 1 else 0.0
    return dict(
        n_bases=len(data),
        mean=float(data.mean()),
        ci95=ci.tolist(),
        p_signflip=(extreme + 1) / (signflips + 1),
        positive_bases=int((data > 1e-12).sum()),
        negative_bases=int((data < -1e-12).sum()),
        standardized_paired_effect=float(data.mean()) / sd if sd else None,
    )


def base_values(rows: list[dict], field: str) -> tuple[list, list, list]:
    """Reduce all four paired H/T/seed observations to one equal-weight base effect."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["jobs"], r["base_seed"])].append(r)
    keys = sorted(groups)
    for k in keys:
        if {(r["tariff"], r["algorithm_seed"]) for r in groups[k]} != {
            (t, seed) for t in ("H", "T") for seed in (1101, 2202)
        } or len(groups[k]) != 4:
            raise ValueError("Incomplete or duplicated base cluster")
    keys = [k for k in keys if all(r[field] is not None for r in groups[k])]
    return (
        [float(np.mean([r[field] for r in groups[k]])) for k in keys],
        [k[0] for k in keys],
        keys,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write reproducible machine-readable analysis tables."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correct_families(statistics: dict) -> None:
    """Apply separate seven-test HV primary and IGD+ secondary Holm families."""
    for family, name in ((PRIMARY, "p_holm_primary_seven"), (SECONDARY, "p_holm_secondary_seven")):
        pvalues = [statistics[a][m]["p_signflip"] for a, m in family]
        for (a, m), adjusted in zip(family, holm(pvalues)):
            statistics[a][m][name] = adjusted


def analyze(raw: Path, out: Path) -> dict:
    """Read only hashed frozen runs and export prespecified and exploratory analyses."""
    if out.resolve().is_relative_to(raw.resolve()):
        raise ValueError("Analysis must stay outside immutable raw")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for node in (0, 1):
        folder = raw / f"formal_node{node}"
        m = json.loads((folder / "manifest.json").read_text())
        if m["source_commit"] != SOURCE or m["source_hash"] != CANONICAL or m["run_count"] != 512:
            raise ValueError("Unexpected source or matrix")
        for block in m["blocks"]:
            for order, key in enumerate(block["runs"]):
                spec = json.loads((folder / "specs" / (key + ".json")).read_text())
                run = folder / "runs" / key
                marker = json.loads((run / "complete.json").read_text())
                if any(
                    hashlib.sha256((run / n).read_bytes()).hexdigest() != h
                    for n, h in marker["files"].items()
                ):
                    raise ValueError("Changed artifact")
                with (run / "objectives.csv").open(newline="", encoding="utf-8") as f:
                    points = [
                        (float(r["flow_seconds"]), float(r["bill_cny"])) for r in csv.DictReader(f)
                    ]
                records.append(
                    dict(
                        **{
                            k: spec[k]
                            for k in (
                                "key",
                                "block",
                                "jobs",
                                "base_seed",
                                "tariff",
                                "algorithm_seed",
                                "arm",
                            )
                        },
                        node=node,
                        order=order,
                        points=points,
                        summary=json.loads((run / "summary.json").read_text()),
                        snapshots=json.loads((run / "anytime.json").read_text()),
                    )
                )
    if len(records) != 1024:
        raise ValueError("Incomplete matrix")
    groups = defaultdict(list)
    for r in records:
        groups[(r["jobs"], r["base_seed"], r["tariff"])].append(r)
    rows, snapshots, states, refs, normalized = [], [], [], {}, {}
    actions, budgets, qt = [], [], []
    for key, members in sorted(groups.items()):
        if {(r["arm"], r["algorithm_seed"]) for r in members} != {
            (a, s) for a in ARMS for s in (1101, 2202)
        } or len(members) != 16:
            raise ValueError("Incomplete group")
        union = nondominated([p for r in members for p in r["points"]])
        lo, span = normalization(union)

        def norm(points: list) -> list:
            return [tuple(map(float, p)) for p in (np.asarray(points) - lo) / span]

        front = norm(union)
        refs[str(key)] = dict(
            ideal=lo.tolist(), divisor=span.tolist(), raw_front=union, front=front
        )
        for r in members:
            s = r["summary"]
            p = norm(r["points"])
            normalized[r["key"]] = p
            causal = {
                k: v
                for k, v in s["light_operator_totals"].items()
                if k.startswith("state") and k.endswith(":A8")
            }
            calls = sum(v["calls"] for v in causal.values())
            proposals = s["counts"].get("proposals:A8", 0)
            cost = s["timings"].get("construction:A8", 0)
            row = {
                k: r[k]
                for k in (
                    "key",
                    "block",
                    "jobs",
                    "base_seed",
                    "tariff",
                    "algorithm_seed",
                    "arm",
                    "node",
                    "order",
                )
            }
            row.update(
                hv=hypervolume([q for q in p if max(q) <= 1.1], (1.1, 1.1)),
                igd_plus=distance(p, front, True),
                igd=distance(p, front, False),
                min_flow=min(q[0] for q in r["points"]),
                min_bill=min(q[1] for q in r["points"]),
                exact_rate=s["counts"]["exact"] / s["elapsed"],
                offspring_rate=s["offspring_count"] / s["elapsed"],
                elapsed=s["elapsed"],
                overshoot=s["time_overshoot_seconds"],
                export=s["verification_and_export_seconds"],
                rss=s["rss_peak_gib"],
                archive_size=s["archive_size"],
                trigger_rate=s["trigger_rate"],
                polish_cost_share=s["timings"].get("polish", 0) / s["elapsed"],
                polish_exact_share=s["counts"].get("polish_exact", 0) / s["counts"]["exact"],
                construction_cost_share=sum(
                    v for k, v in s["timings"].items() if k.startswith("construction:")
                )
                / s["elapsed"],
                state_visited=sum(
                    v > 0
                    for v in json.loads(
                        (
                            raw / f"formal_node{r['node']}" / "runs" / r["key"] / "qtable.json"
                        ).read_text()
                    )["visits"]
                ),
                a8_calls=calls,
                a8_attempts=s["counts"].get("construction_attempts:A8", 0),
                a8_proposals=proposals,
                a8_exact=s["counts"].get("exact:A8", 0),
                a8_construction_seconds=cost,
                a8_cost_share=cost / s["elapsed"],
                a8_cost_per_call=cost / calls if calls else None,
                a8_proposals_per_call=proposals / calls if calls else None,
                a8_archive_final=s["archive_final_by_origin"].get("A8", 0),
                a8_accepted=s["light_operator_totals"].get("8", {}).get("accepted", 0),
            )
            if r["arm"] == "BASE" and r["algorithm_seed"] == 1101:
                if not math.isclose(row["igd_plus"], igd_plus(p, front), abs_tol=1e-13):
                    raise ValueError("Independent IGD+ mismatch")
            rows.append(row)
            for state, v in causal.items():
                state_id = int(state.split(":")[0][5:])
                preference, condition, progress = decode(state_id, s["rl_state_policy"])
                states.append(
                    {
                        **{
                            k: row[k]
                            for k in ("jobs", "base_seed", "tariff", "algorithm_seed", "arm")
                        },
                        "state_id": state_id,
                        "coexistence": state_id % 2 if s["rl_state_policy"] == "compound" else -1,
                        "preference": preference,
                        "condition": condition,
                        "progress": progress,
                        **v,
                    }
                )
            identity = {
                k: row[k] for k in ("key", "jobs", "base_seed", "tariff", "algorithm_seed", "arm")
            }
            for name, value in s["light_operator_totals"].items():
                if not name.startswith("state"):
                    continue
                state_id, action = (
                    int(q) for q in name.replace("state", "").replace("A", "").split(":")
                )
                pref, cond, progress = decode(state_id, s["rl_state_policy"])
                actions.append(
                    {
                        **identity,
                        "state_id": state_id,
                        "action": action,
                        "preference": pref,
                        "condition": cond,
                        "progress": progress,
                        "coexistence": state_id % 2 if s["rl_state_policy"] == "compound" else -1,
                        **value,
                    }
                )
            for name, count in s["light_budget_histogram"].items():
                action, requested, effective = (int(q[1:]) for q in name.split(":"))
                budgets.append(
                    {
                        **identity,
                        "action": action,
                        "requested": requested,
                        "effective": effective,
                        "calls": count,
                    }
                )
            root = raw / f"formal_node{r['node']}" / "runs" / r["key"]
            qtable = json.loads((root / "qtable.json").read_text())
            for state_id, qvalues in enumerate(qtable["q"]):
                pref, cond, progress = decode(state_id, s["rl_state_policy"])
                qt.append(
                    {
                        **identity,
                        "state_id": state_id,
                        "preference": pref,
                        "condition": cond,
                        "progress": progress,
                        "coexistence": state_id % 2 if s["rl_state_policy"] == "compound" else -1,
                        "visits": qtable["visits"][state_id],
                        "selections": sum(qtable["selections"][state_id]),
                        "updates": sum(qtable["updates"][state_id]),
                        "q_max": max(qvalues),
                    }
                )
            for snap in r["snapshots"]:
                snapshots.append(
                    {
                        **{
                            k: row[k]
                            for k in (
                                "key",
                                "block",
                                "jobs",
                                "base_seed",
                                "tariff",
                                "algorithm_seed",
                                "arm",
                            )
                        },
                        "fraction": snap["target_seconds"] / s["time_budget_seconds"],
                        "delay": snap["elapsed"] - snap["target_seconds"],
                        "hv": hypervolume(
                            [q for q in norm(snap["objectives"]) if max(q) <= 1.1], (1.1, 1.1)
                        ),
                    }
                )
    blocks = defaultdict(dict)
    for r in rows:
        if r["arm"] in blocks[r["block"]]:
            raise ValueError("Duplicate arm")
        blocks[r["block"]][r["arm"]] = r
    paired = []
    for block, arms in sorted(blocks.items()):
        if set(arms) != set(ARMS):
            raise ValueError("Incomplete block")
        for a, b in CONTRASTS:
            x, y = arms[a], arms[b]
            if x["node"] != y["node"]:
                raise ValueError("Unpaired host")
            pair = {k: x[k] for k in ("jobs", "base_seed", "tariff", "algorithm_seed", "node")}

            def pct(field: str) -> float | None:
                return (
                    100 * (x[field] / y[field] - 1) if x[field] is not None and y[field] else None
                )

            ca, cb = (
                coverage(normalized[x["key"]], normalized[y["key"]]),
                coverage(normalized[y["key"]], normalized[x["key"]]),
            )
            pair.update(
                block=block,
                contrast=a + "-" + b,
                hv_diff=x["hv"] - y["hv"],
                igd_plus_gain=y["igd_plus"] - x["igd_plus"],
                hv_pct=pct("hv"),
                igd_gain=y["igd"] - x["igd"],
                coverage_ab=ca,
                coverage_ba=cb,
                coverage_net=ca - cb,
                flow_gain_pct=-pct("min_flow"),
                bill_gain_pct=-pct("min_bill"),
                exact_pct=pct("exact_rate"),
                offspring_pct=pct("offspring_rate"),
                a8_cost_per_call_pct=pct("a8_cost_per_call"),
                a8_cost_share_pp=100 * (x["a8_cost_share"] - y["a8_cost_share"]),
                a8_proposals_per_call_diff=x["a8_proposals_per_call"] - y["a8_proposals_per_call"],
            )
            paired.append(pair)
    statistics, bases = {}, []
    rng = np.random.default_rng(SEED)
    for a, b in CONTRASTS:
        name = a + "-" + b
        selected = [r for r in paired if r["contrast"] == name]
        statistics[name] = {}
        for metric in METRICS:
            values, strata, keys = base_values(selected, metric)
            statistics[name][metric] = cluster_summary(values, strata, rng) if values else None
            bases.extend(
                dict(contrast=name, metric=metric, jobs=k[0], base_seed=k[1], value=v)
                for k, v in zip(keys, values)
            )
    correct_families(statistics)

    def means(selection: list, fields: tuple | list) -> dict:
        return {k: float(np.mean([r[k] for r in selection if r[k] is not None])) for k in fields}

    report = dict(
        status="ANALYZED",
        runs=1024,
        blocks=128,
        units=32,
        source_commit=SOURCE,
        source_hash=CANONICAL,
        seed=SEED,
        bootstrap=20000,
        signflips=100000,
        primary_family=PRIMARY,
        secondary_family=SECONDARY,
        statistics=statistics,
        strata={
            f"{n}_{t}": {
                a + "-" + b: means(
                    [
                        r
                        for r in paired
                        if r["jobs"] == n and r["tariff"] == t and r["contrast"] == a + "-" + b
                    ],
                    METRICS,
                )
                for a, b in CONTRASTS
            }
            for n in (50, 100)
            for t in ("H", "T")
        },
        host={
            str(node): {
                a + "-" + b: means(
                    [r for r in paired if r["node"] == node and r["contrast"] == a + "-" + b],
                    METRICS,
                )
                for a, b in CONTRASTS
            }
            for node in (0, 1)
        },
        arm_means={a: means([r for r in rows if r["arm"] == a], list(rows[0])[9:]) for a in ARMS},
        max_snapshot_delay=max(r["delay"] for r in snapshots),
    )
    for name, table in (
        ("run_metrics", rows),
        ("paired_metrics", paired),
        ("base_metrics", bases),
        ("anytime_metrics", snapshots),
        ("a8_state_metrics", states),
        ("state_action_metrics", actions),
        ("budget_metrics", budgets),
        ("q_state_metrics", qt),
    ):
        write_csv(out / (name + ".csv"), table)
    (out / "references.json").write_text(json.dumps(refs), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Run postprocessing only, with explicit data and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.data, args.output)
    print(
        json.dumps(
            {
                a: {m: result["statistics"][a][m] for m in ("hv_diff", "igd_plus_gain")}
                for a in tuple(a + "-BASE" for a in ARMS[1:])
            }
        )
    )


if __name__ == "__main__":
    main()
