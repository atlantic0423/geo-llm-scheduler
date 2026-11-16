"""Audit 192 budget/A8 runs; eight primary HV contrasts, eight secondary contrasts.

Usage: python research/lsv2_budget12/analyze.py WORKSPACE
One seed per instance. Historical four-arm comparison is descriptive only.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

BASE = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else Path(__file__).resolve().parents[4]
)
DATA = BASE / "outputs/lsv2_budget12_20261028"
ROOT = DATA / "results/formal"
OUT = DATA / "analysis"
# The research script runs directly without requiring an editable package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from geo_llm_scheduler.domain.models import Genotype, Schedule  # noqa: E402
from geo_llm_scheduler.evaluation.exact import evaluate  # noqa: E402
from geo_llm_scheduler.experiments.lsv2_budget12 import ARMS, INTERNAL_ARMS  # noqa: E402
from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus  # noqa: E402
from geo_llm_scheduler.io.loaders import load_instance  # noqa: E402

SIZES = (20, 50, 100, 200)


def read(path: Path) -> Any:
    """Read a sealed JSON or gzip artifact."""
    with (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(encoding="utf-8-sig")
    ) as f:
        return json.load(f)


def nd(points: Iterable) -> list[tuple[float, float]]:
    """Return unique nondominated minimization points in increasing flow order."""
    best, result = math.inf, []
    for x, y in sorted(set(map(tuple, points))):
        assert math.isfinite(x) and math.isfinite(y)
        if y < best:
            result.append((x, y))
            best = y
    return result


def points(records: list[dict]) -> list[tuple[float, float]]:
    """Return objective pairs without silently accepting infeasible candidates."""
    assert records and all(r["evaluation"]["feasible"] for r in records)
    return [
        (r["evaluation"]["flow"], r["evaluation"]["tou"] + sum(r["evaluation"]["demand"]))
        for r in records
    ]


def hv(a: Iterable, ref: tuple[float, float] = (1.1, 1.1)) -> float:
    """HV with out-of-reference points discarded, never clipped into the box."""
    p = [tuple(x) for x in a if x[0] <= ref[0] and x[1] <= ref[1]]
    h = hypervolume(p, ref)
    f = nd(p)
    cross = sum(
        (f[i + 1][0] - x if i + 1 < len(f) else ref[0] - x) * (ref[1] - y)
        for i, (x, y) in enumerate(f)
    )
    assert math.isclose(h, cross, abs_tol=1e-11, rel_tol=1e-11)
    return h


def igdp(a: Iterable, r: Iterable) -> float:
    """Vectorized IGD+, checked against scalar implementation at reference subset."""
    a, r = np.asarray(a), np.asarray(r)
    if not len(r):
        return float("nan")
    z = np.maximum(a[None, :, :] - r[:, None, :], 0)
    v = float(np.linalg.norm(z, axis=2).min(axis=1).mean())
    small = r[:: max(1, len(r) // 7)]
    chk = np.linalg.norm(np.maximum(a[None, :, :] - small[:, None, :], 0), axis=2).min(1).mean()
    assert math.isclose(chk, igd_plus(list(map(tuple, a)), list(map(tuple, small))), abs_tol=1e-11)
    return v


def csvout(name: str, rows: list[dict]) -> None:
    """Write complete machine-readable analysis table."""
    with (OUT / (name + ".csv")).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def infer(d: list[float], sizes: list[int], seed: int) -> dict:
    """Exploratory paired-instance inference, four fixed instances in each size stratum."""
    a = np.asarray(d)
    sizes = np.asarray(sizes)
    assert len(a) == 16
    rng = np.random.default_rng(seed)
    boots = np.zeros(20000)
    for n in SIZES:
        z = a[sizes == n]
        assert len(z) == 4
        boots += z[rng.integers(4, size=(20000, 4))].mean(1) / 4
    signs = (
        2 * ((np.arange(65536, dtype=np.uint32)[:, None] >> np.arange(16)) & 1).astype(float) - 1
    )
    return dict(
        mean=float(a.mean()),
        ci95=np.quantile(boots, [0.025, 0.975]).tolist(),
        p=float((np.abs((signs * a).mean(1)) >= abs(a.mean()) - 1e-14).mean()),
        positive=int((a > 1e-12).sum()),
        negative=int((a < -1e-12).sum()),
        per_size={str(n): float(a[sizes == n].mean()) for n in SIZES},
    )


def main() -> None:
    """Audit, score and summarize sealed runs without altering solver artifacts."""
    OUT.mkdir(exist_ok=True)
    assert read(DATA / "RECOVERY_RECEIPT.json")["accepted"]
    manifest = read(ROOT / "manifest.json")
    jobs = manifest["jobs"]
    assert len(jobs) == 192
    specs = {k: read(ROOT / "specs" / f"{k}.json") for k in manifest["blocks"]}
    records, diag, ops, bystate, reward_arrays = {}, [], [], [], defaultdict(list)
    audit = dict(runs=0, exact_candidates=0, q_steps_replayed=0, initialization_groups=0)
    for job in jobs:
        key = job["runs"][0]["key"]
        s = specs[key]
        arm = job["arm"]
        path = ROOT / "results" / job["id"] / ("00_" + key)
        summary = read(path / "summary.json")
        cfg = read(path / "config.json")
        archive = read(path / "archive.json.gz")
        initial = read(path / "initial.json.gz")
        problem = load_instance(ROOT / s["input"])
        fp = points(archive)
        # Fresh independent exact of flow endpoint, middle and bill endpoint of every archive.
        ix = sorted(range(len(fp)), key=fp.__getitem__)
        for i in {ix[0], ix[len(ix) // 2], min(ix, key=lambda j: fp[j][1])}:
            rec = archive[i]
            g = Genotype(tuple(rec["genotype"]["ms"]), tuple(rec["genotype"]["os"]))
            sch = Schedule(
                *(
                    tuple(rec["schedule"][k])
                    for k in ("starts", "resource_wait", "intentional_wait")
                )
            )
            ev = evaluate(problem, g, sch)
            assert ev.feasible and np.allclose(ev.objectives, fp[i], rtol=1e-10, atol=1e-8)
            audit["exact_candidates"] += 1
        assert summary["termination"] == "time_budget" and summary["final_exact_verified"]
        assert cfg["commit"] == manifest["commit"]
        assert hashlib.sha256((ROOT / s["input"]).read_bytes()).hexdigest() == s["input_sha256"]
        steps = read(path / "lsv2_steps.json.gz")
        after = read(path / "q_after.json")
        assert read(path / "q_before.json") == dict(values={}, visits={})
        ls = summary["lsv2"]
        n = len(steps)
        if ls is None:
            records[key, arm] = dict(
                spec=s,
                summary=summary,
                front=nd(fp),
                initial=nd(points(initial)),
                anytime=read(path / "anytime.json"),
                events=read(path / "archive_events.json.gz"),
            )
            audit["runs"] += 1
            continue
        gains = np.array([r["hv_gain"] for r in steps])
        row = dict(
            key=key,
            parent=s["base_seed"],
            n=s["n"],
            arm=arm,
            steps=n,
            elapsed=summary["elapsed"],
            audit_seconds=summary["audit_seconds"],
            offspring=summary["offspring"],
            exact=summary["counts"]["exact"],
            ls_seconds=ls["ls_seconds"],
            ls_share=ls["ls_seconds"] / (summary["elapsed"] - ls["search_start"]),
            trajectories=ls["trajectories"],
            zero_gain_share=float((gains <= 1e-12).mean()),
            top1_gain_share=float(
                np.sort(gains)[-max(1, math.ceil(n * 0.01)) :].sum() / max(gains.sum(), 1e-30)
            ),
            q_states=0,
            q_pairs=0,
            observed_available_pairs=0,
            pairs_lt5_share=0.0,
            pairs_lt10_share=0.0,
            q_negative_share=0.0,
            q_reward_negative_share=0.0,
            q_reward_median=0.0,
            q_reward_p99=0.0,
            q_reward_max=0.0,
            q_layer_seconds_proxy=0.0,
            q_nonpositive_positive_target_share=0.0,
            q_zero_gain_positive_target_share=0.0,
            q_feature_high_share=0.0,
            q_unvisited_choice_share=0.0,
            q_exploit_choice_share=0.0,
        )
        if arm.startswith("Q"):
            q = {}
            visits = Counter()
            avail = defaultdict(set)
            unvisited = exploit = 0
            for step_index, r in enumerate(steps):
                terminal = (
                    step_index == len(steps) - 1
                    or steps[step_index + 1]["trajectory"] != r["trajectory"]
                )
                assert r["q_transition"]["terminal"] == terminal
                assert r["q_state"][1] == cfg["lsv2"]["trajectory_steps"] - r["step"]
                assert r["q_state"][2] == int("A8" in r["available"])
                if arm.endswith("NOA8"):
                    assert (
                        "A8" not in r["available"]
                        and "A8" not in r["q_transition"]["next_available"]
                    )
                if not terminal:
                    assert r["q_transition"]["next_state"] == steps[step_index + 1]["q_state"]
                    assert r["q_transition"]["next_available"] == steps[step_index + 1]["available"]
                state = ":".join(map(str, r["q_state"]))
                action = r["operator"]
                mask = r["available"]
                avail[state].update(mask)
                vals = q.get(state, {})
                top = max(vals.get(a, 0.0) for a in mask)
                best = [a for a in mask if vals.get(a, 0.0) == top]
                prob = 0.2 / len(mask) + (0.8 / len(best) if action in best else 0)
                assert math.isclose(prob, r["bandit"]["probability"], abs_tol=1e-12)
                unvisited += visits[state, action] == 0
                exploit += action in best
                transition = r["q_transition"]
                old = vals.get(action, 0.0)
                nxt = transition["next_state"]
                ns = ":".join(map(str, nxt)) if nxt else None
                future = max(
                    (q.get(ns, {}).get(a, 0.0) for a in transition["next_available"]), default=0.0
                )
                if transition["terminal"]:
                    assert future == 0
                reward = r["q_reward"]
                spec = cfg["qspec"]
                expected = (
                    r["hv_gain"] / r["charged_step_seconds"]
                    if "RATE" in arm
                    else (r["hv_gain"] - spec["rho"] * r["charged_step_seconds"]) / spec["h"]
                )
                target = reward + spec["gamma"] * future
                new = old + spec["alpha"] * (target - old)
                assert all(
                    math.isclose(x, y, rel_tol=1e-11, abs_tol=1e-11)
                    for x, y in [
                        (reward, expected),
                        (old, transition["q_old"]),
                        (future, transition["bootstrap"]),
                        (target, transition["td_target"]),
                        (new, transition["q_new"]),
                    ]
                )
                q.setdefault(state, {})[action] = new
                visits[state, action] += 1
            assert q == after["values"]
            assert dict(visits) == {
                (k, a): v for k, z in after["visits"].items() for a, v in z.items()
            }
            audit["q_steps_replayed"] += n
            rewards = np.array([r["q_reward"] for r in steps])
            reward_arrays[arm].extend(zip(gains.tolist(), rewards.tolist()))
            pairs = sum(len(v) for v in avail.values())
            row.update(
                q_states=len(q),
                q_pairs=len(visits),
                observed_available_pairs=pairs,
                pairs_lt5_share=sum(visits[k, a] < 5 for k, v in avail.items() for a in v) / pairs,
                pairs_lt10_share=sum(visits[k, a] < 10 for k, v in avail.items() for a in v)
                / pairs,
                q_negative_share=sum(x < 0 for z in q.values() for x in z.values()) / len(visits),
                q_reward_negative_share=float((rewards < 0).mean()),
                q_reward_median=float(np.median(rewards)),
                q_reward_p99=float(np.quantile(rewards, 0.99)),
                q_reward_max=float(rewards.max()),
                q_layer_seconds_proxy=sum(
                    max(0, r["charged_step_seconds"] - r["seconds"])
                    + r["bandit"]["state_update_select_seconds"]
                    + (r["q_update_seconds"] if r["q_transition"]["terminal"] else 0)
                    for r in steps
                ),
                q_nonpositive_positive_target_share=sum(
                    r["q_reward"] <= 0 and r["q_transition"]["td_target"] > 0 for r in steps
                )
                / n,
                q_zero_gain_positive_target_share=sum(
                    r["hv_gain"] <= 1e-12 and r["q_transition"]["td_target"] > 0 for r in steps
                )
                / n,
                q_feature_high_share=sum(r["q_state"][-1] == 1 for r in steps) / n
                if arm.startswith("Q2")
                else 0.0,
                q_unvisited_choice_share=unvisited / n,
                q_exploit_choice_share=exploit / n,
            )
            for state, z in q.items():
                bystate.append(
                    dict(
                        key=key,
                        parent=s["base_seed"],
                        n=s["n"],
                        arm=arm,
                        state=state,
                        visits=sum(after["visits"][state].values()),
                        visited_actions=len(z),
                        available_actions=len(avail[state]),
                        max_q=max(z.values()),
                        min_q=min(z.values()),
                    )
                )
        for op in sorted({r["operator"] for r in steps}):
            rr = [r for r in steps if r["operator"] == op]
            ops.append(
                dict(
                    key=key,
                    parent=s["base_seed"],
                    n=s["n"],
                    arm=arm,
                    operator=op,
                    calls=len(rr),
                    call_share=len(rr) / n,
                    seconds=sum(r["seconds"] for r in rr),
                    gain=sum(r["hv_gain"] for r in rr),
                    positive=sum(r["hv_gain"] > 1e-12 for r in rr),
                    empty=sum(r["proposals"] == 0 for r in rr),
                    exact=sum(r["exact"] for r in rr),
                    proposals=sum(r["proposals"] for r in rr),
                    archive_hits=sum(r["entered_archive"] > 0 for r in rr),
                )
            )
        row.update(
            ls_exact=sum(r["exact"] for r in steps),
            ls_proposals=sum(r["proposals"] for r in steps),
            ls_empty=sum(r["proposals"] == 0 for r in steps),
            ls_archive_hits=sum(r["entered_archive"] for r in steps),
            reflow=sum(r["reflow_replaced"] for r in steps),
            frames=ls["frames"],
        )
        diag.append(row)
        records[key, arm] = dict(
            spec=s,
            summary=summary,
            front=nd(fp),
            initial=nd(points(initial)),
            anytime=read(path / "anytime.json"),
            events=read(path / "archive_events.json.gz"),
        )
        audit["runs"] += 1
        if audit["runs"] % 32 == 0:
            print("audited", audit["runs"], flush=True)
    # Initialization equivalence applies within each policy, never across BILL4/hybrid.
    for key in specs:
        for arms in (INTERNAL_ARMS, ("EXTH_SPEA2", "EXTH_MOEAD")):
            fps = {
                (
                    records[key, a]["summary"]["initialization"]["initial_population_sha256"],
                    records[key, a]["summary"]["initialization"]["initial_archive_sha256"],
                )
                for a in arms
            }
            assert len(fps) == 1
            audit["initialization_groups"] += 1
    historic = ("BILL_RANDOM", "EXTH_NSGAII", "EXTH_RK_MOPSO", "EXTH_EPS_IBEA")
    roots = {
        a: BASE
        / (
            "outputs/lsv2_panel_20261026/results/formal"
            if a == "BILL_RANDOM"
            else "outputs/ext_hybrid_panel_20261026/results/formal"
        )
        for a in historic
    }
    seals = {
        r: {x["path"]: x for x in read(r.parents[1] / "formal_manifest.json")["files"]}
        for r in set(roots.values())
    }
    checked = {}

    def legacy(root: Path, path: Path) -> Any:
        rel = "formal/" + path.relative_to(root).as_posix()
        seal = seals[root][rel]
        assert (
            path.stat().st_size == seal["size"]
            and hashlib.sha256(path.read_bytes()).hexdigest() == seal["sha256"]
        )
        checked[str(path.relative_to(BASE))] = seal["sha256"]
        return read(path)

    for key, s in specs.items():
        for arm, root in roots.items():
            old = legacy(root, root / "specs" / f"{key}.json")
            assert (
                old["input_sha256"] == s["input_sha256"]
                and old["seconds"] == s["seconds"]
                and old["seed"] == s["seed"] == 16901
            )
            legacy(root, root / old["input"])
            path = root / "runs" / key / arm
            archive = legacy(root, path / "archive.json.gz")
            summary = legacy(root, path / "summary.json")
            cfg = legacy(root, path / "config.json")
            engine = cfg["engine"] if arm == "BILL_RANDOM" else cfg
            assert (
                engine["seconds"] == s["seconds"]
                and engine["seed"] == 16901
                and engine["population"] == 100
            )
            assert summary["final_exact_verified"] and summary["termination"] == "time_budget"
            assert (
                summary["initialization"]["initial_population_sha256"]
                == records[key, "Q1_NET_G9_B6L5" if arm == "BILL_RANDOM" else "EXTH_SPEA2"][
                    "summary"
                ]["initialization"]["initial_population_sha256"]
            )
            pp = points(archive)
            problem = load_instance(ROOT / s["input"])
            ix = sorted(range(len(pp)), key=pp.__getitem__)
            for i in {ix[0], ix[len(ix) // 2], min(ix, key=lambda i: pp[i][1])}:
                c = archive[i]
                g = Genotype(tuple(c["genotype"]["ms"]), tuple(c["genotype"]["os"]))
                sch = Schedule(
                    *(
                        tuple(c["schedule"][k])
                        for k in ("starts", "resource_wait", "intentional_wait")
                    )
                )
                ev = evaluate(problem, g, sch)
                assert ev.feasible and np.allclose(ev.objectives, pp[i], rtol=1e-10, atol=1e-8)
                audit["exact_candidates"] += 1
            records[key, arm] = dict(spec=s, summary=summary, front=nd(pp))
    metricrows = []
    broadrows = []
    fronts = {}
    curves = []
    runtime = []
    for key, s in specs.items():
        allarms = ARMS + historic
        ref = nd(p for a in ARMS for p in records[key, a]["front"])
        broadref = nd(p for a in allarms for p in records[key, a]["front"])
        u = np.array(ref)
        lo = u.min(0)
        span = np.maximum(u.max(0) - lo, 1e-12)
        bu = np.array(broadref)
        blo = bu.min(0)
        bspan = np.maximum(bu.max(0) - blo, 1e-12)

        def norm(p):
            return (np.array(p).reshape(-1, 2) - lo) / span

        def bnorm(p):
            return (np.array(p).reshape(-1, 2) - blo) / bspan

        un = norm(ref)
        bun = bnorm(broadref)
        fronts[s["base_seed"]] = dict(
            n=s["n"],
            lo=lo.tolist(),
            span=span.tolist(),
            reference=ref,
            broad_lo=blo.tolist(),
            broad_span=bspan.tolist(),
            broad_reference=broadref,
            runs={a: records[key, a]["front"] for a in allarms},
        )
        cap = min(records[key, a]["summary"]["counts"]["exact"] for a in INTERNAL_ARMS)
        for a in allarms:
            rec = records[key, a]
            pp = rec["front"]
            bn = bnorm(pp)
            broadrows.append(
                dict(
                    parent=s["base_seed"],
                    n=s["n"],
                    arm=a,
                    hv=hv(bn),
                    igd_plus=igdp(bn, bun),
                    min_flow=min(p[0] for p in pp),
                    min_bill=min(p[1] for p in pp),
                )
            )
            if a not in ARMS:
                continue
            an = norm(pp)
            equal = norm(nd(rec["initial"] + [(f, b) for e, t, f, b in rec["events"] if e <= cap]))
            metricrows.append(
                dict(
                    parent=s["base_seed"],
                    n=s["n"],
                    arm=a,
                    hv=hv(an),
                    igd_plus=igdp(an, un),
                    hv_ref12=hv(an, (1.2, 1.2)),
                    hv_equal_exact=hv(equal),
                    common_exact=cap,
                    min_flow=min(p[0] for p in pp),
                    min_bill=min(p[1] for p in pp),
                    points=len(pp),
                    igd_flow=igdp(an, un[un[:, 0] < 1 / 3]),
                    igd_middle=igdp(an, un[(un[:, 0] >= 1 / 3) & (un[:, 0] <= 2 / 3)]),
                    igd_bill=igdp(an, un[un[:, 0] > 2 / 3]),
                )
            )
            # Reconstruct only candidates actually observed by each wall-clock point.
            # Initial population is not backdated to fractions before initialization.
            initready = rec["anytime"][0]["elapsed"]
            for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
                sec = s["seconds"] * frac
                raw = ([*rec["initial"]] if sec >= initready else []) + [
                    (f, b) for e, t, f, b in rec["events"] if t <= sec
                ]
                value = hv(norm(nd(raw))) if raw else 0.0
                curves.append(dict(parent=s["base_seed"], n=s["n"], arm=a, fraction=frac, hv=value))
            summary = rec["summary"]
            runtime.append(
                dict(
                    parent=s["base_seed"],
                    n=s["n"],
                    arm=a,
                    budget=s["seconds"],
                    elapsed=summary["elapsed"],
                    overshoot=summary["elapsed"] - s["seconds"],
                    offspring=summary["offspring"],
                    generations=summary["offspring"] / 100,
                    exact=summary["counts"]["exact"],
                    initial_seconds=initready,
                    final_exact_verified=summary["final_exact_verified"],
                    audit_seconds=summary["audit_seconds"],
                )
            )
    csvout("run_metrics", metricrows)
    csvout("baseline_metrics", broadrows)
    csvout("runtime", runtime)
    csvout("run_diagnostics", diag)
    csvout("operator_metrics", ops)
    csvout("state_coverage", bystate)
    csvout("anytime", curves)
    (OUT / "fronts.json").write_text(json.dumps(fronts), encoding="utf-8")
    by = {(r["parent"], r["arm"]): r for r in metricrows}
    parents = sorted(fronts)
    sizes = [fronts[p]["n"] for p in parents]
    q6 = "Q1_NET_G9_B6L5"
    q3 = "Q1_NET_G9_B3L3"
    q1 = "Q1_NET_G0_B1L1"
    primary = [
        (q3, q6),
        (q1, q6),
        ("UNIFORM_B3L3", "UNIFORM_B6L5"),
        ("UNIFORM_B1L1", "UNIFORM_B6L5"),
        (q6 + "_NOA8", q6),
        ("UNIFORM_B6L5_NOA8", "UNIFORM_B6L5"),
        (q6, "Q1_NET_G0_B6L5"),
        (q3, "Q1_NET_G0_B3L3"),
    ]
    secondary = [(a, "UNIFORM_" + a.split("_", 3)[3]) for a in INTERNAL_ARMS if a.startswith("Q1_")]
    secondary += [("Q1_NET_G0_B3L3", "Q1_NET_G0_B6L5"), (q1, "Q1_NET_G0_B6L5")]
    tests = []
    for i, (t, c) in enumerate(primary + secondary):
        d = [by[p, t]["hv"] - by[p, c]["hv"] for p in parents]
        dd = [by[p, t]["igd_plus"] - by[p, c]["igd_plus"] for p in parents]
        tests.append(
            dict(
                treat=t,
                ctrl=c,
                family="primary" if i < 8 else "secondary",
                **infer(d, sizes, 2026102800 + i),
                igd_diff=float(np.mean(dd)),
                igd_wins=sum(x < -1e-12 for x in dd),
                hv_ref12_diff=float(
                    np.mean([by[p, t]["hv_ref12"] - by[p, c]["hv_ref12"] for p in parents])
                ),
                equal_exact_diff=float(
                    np.mean(
                        [by[p, t]["hv_equal_exact"] - by[p, c]["hv_equal_exact"] for p in parents]
                    )
                ),
                per_instance=dict(zip(parents, d)),
            )
        )
    for family in ("primary", "secondary"):
        group = [x for x in tests if x["family"] == family]
        last = 0.0
        for rank, t in enumerate(sorted(group, key=lambda x: x["p"])):
            last = max(last, (len(group) - rank) * t["p"])
            t["p_holm"] = min(1.0, last)
    last = 0.0
    for rank, t in enumerate(sorted(tests, key=lambda x: x["p"])):
        last = max(last, (len(tests) - rank) * t["p"])
        t["p_holm_all16"] = min(1.0, last)
    summary = {}
    for arm in ARMS:
        rr = [r for r in metricrows if r["arm"] == arm]
        dd = [r for r in diag if r["arm"] == arm]
        rt = [r for r in runtime if r["arm"] == arm]
        cols = (
            "hv",
            "igd_plus",
            "hv_ref12",
            "hv_equal_exact",
            "igd_flow",
            "igd_middle",
            "igd_bill",
        )
        summary[arm] = dict(
            metrics={k: float(np.nanmean([r[k] for r in rr])) for k in cols},
            by_size={
                str(n): {k: float(np.nanmean([r[k] for r in rr if r["n"] == n])) for k in cols}
                for n in SIZES
            },
            runtime={
                k: float(np.mean([r[k] for r in rt]))
                for k in ("elapsed", "overshoot", "generations", "exact", "initial_seconds")
            },
            diagnostic={
                k: float(np.mean([r[k] for r in dd]))
                for k in dd[0]
                if k not in ("key", "parent", "n", "arm")
            }
            if dd
            else {},
        )
    result = dict(
        audit=audit,
        arms=summary,
        tests=tests,
        primary_family_size=8,
        secondary_family_size=8,
        historical_checked_files=checked,
        source_commit=manifest["commit"],
        primary_reference="12 current arms, per-instance final nondominated union, ref (1.1,1.1)",
        broad_reference="12 current + 4 historical arms; descriptive only",
        validation="ANALYZED; full SHA recovery and deterministic TD/metric verification, no solver rerun",
    )
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                audit=audit,
                ranking=sorted(
                    [
                        (a, summary[a]["metrics"]["hv"], summary[a]["metrics"]["igd_plus"])
                        for a in ARMS
                    ],
                    key=lambda x: -x[1],
                ),
                tests=tests,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
