"""Frozen parent-cluster quality and mechanism analysis of all 23 RL23 arms."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from geo_llm_scheduler.experiments.metrics import hypervolume, igd_plus
from geo_llm_scheduler.experiments.rl23_support import ARMS

SIZES = (20, 50, 100, 200)
ROOT_SEED = 2026102101
BASELINE = "STRICT_TWO_FOCUS"


def nondominated(points: list) -> list[tuple[float, float]]:
    """Return exact unique minimization points, without an arbitrary epsilon filter."""
    result: list[tuple[float, float]] = []
    best = math.inf
    for x, y in sorted(set(map(tuple, points))):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Nonfinite objective")
        if y < best:
            result.append((x, y))
            best = y
    return result


def volume(points: list) -> float:
    """Compute union area in the frozen 1.1 reference rectangle."""
    return hypervolume([tuple(p) for p in points if max(p) <= 1.1], (1.1, 1.1))


def volume_audit(points: list) -> float:
    """Independently integrate horizontal strips rather than vertical rectangles."""
    front = nondominated([p for p in points if max(p) <= 1.1])
    return math.fsum(
        (1.1 - x) * ((front[i - 1][1] if i else 1.1) - y) for i, (x, y) in enumerate(front)
    )


def distance(points: list, reference: list, plus: bool = True) -> float:
    """Compute IGD+/IGD using bounded NumPy reference chunks."""
    a, r = np.asarray(points), np.asarray(reference)
    if not len(a) or not len(r):
        raise ValueError("Empty front")
    total = 0.0
    for i in range(0, len(r), 64):
        delta = a[None, :, :] - r[i : i + 64, None, :]
        if plus:
            delta = np.maximum(delta, 0)
        total += np.sqrt(np.square(delta).sum(axis=2)).min(axis=1).sum()
    return float(total / len(r))


def coverage(a: list, b: list) -> float:
    """Fraction of unique B points weakly dominated by A, using a 2-D sweep."""
    x, y = np.asarray(nondominated(a)), np.asarray(nondominated(b))
    indexes = np.searchsorted(x[:, 0], y[:, 0] + 1e-12, side="right") - 1
    return float(np.mean((indexes >= 0) & (x[np.maximum(indexes, 0), 1] <= y[:, 1] + 1e-12)))


def holm(values: list[float]) -> list[float]:
    """Adjust exactly one predeclared complete family and preserve its order."""
    answer = [0.0] * len(values)
    previous = 0.0
    for rank, i in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        previous = max(previous, (len(values) - rank) * values[i])
        answer[i] = min(1.0, previous)
    return answer


def inference(values: list[float], sizes: list[int], seed: int) -> dict:
    """Use 16 parents, stratified bootstrap and the frozen Monte Carlo sign-flip."""
    assert len(values) == 16 and all(sizes.count(n) == 4 for n in SIZES)
    a = np.asarray(values)
    rng = np.random.default_rng(seed)
    boots = np.zeros(20000)
    for n in SIZES:
        z = a[np.asarray(sizes) == n]
        boots += z[rng.integers(len(z), size=(20000, len(z)))].mean(1) / 4
    effect = float(a.mean())
    hits = 0
    for i in range(0, 100000, 4096):
        signs = rng.integers(0, 2, size=(min(4096, 100000 - i), 16)) * 2 - 1
        hits += int((np.abs((signs * a).mean(1)) >= abs(effect) - 1e-14).sum())
    exact = 0
    for i in range(0, 65536, 4096):
        signs = (
            2
            * ((np.arange(i, i + 4096, dtype=np.uint32)[:, None] >> np.arange(16)) & 1).astype(
                float
            )
            - 1
        )
        exact += int((np.abs((signs * a).mean(1)) >= abs(effect) - 1e-14).sum())
    sd = float(a.std(ddof=1))
    return dict(
        mean=effect,
        ci95=np.quantile(boots, [0.025, 0.975]).tolist(),
        p=(hits + 1) / 100001,
        p_exact_audit=exact / 65536,
        positive=int((a > 1e-12).sum()),
        negative=int((a < -1e-12).sum()),
        standardized_paired_effect=effect / sd if sd else None,
        n_parents=16,
        seed=seed,
        per_scale={str(n): float(a[np.asarray(sizes) == n].mean()) for n in SIZES},
    )


def stored_points(path: Path) -> tuple[list, int]:
    """Read saved feasible solutions, without claiming a fresh exact evaluation."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        records = json.load(f)
    assert records and all(r["evaluation"]["feasible"] for r in records)
    points = [
        (r["evaluation"]["flow"], r["evaluation"]["tou"] + sum(r["evaluation"]["demand"]))
        for r in records
    ]
    return nondominated(points), len(records)


def write_table(out: Path, name: str, rows: list[dict]) -> None:
    """Save complete scalar observations as UTF-8 CSV."""
    if not rows:
        raise ValueError("Missing table rows")
    with (out / (name + ".csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def step_diagnostics(path: Path, arm: str, q: dict) -> tuple[dict, list, list, list]:
    """Audit every logged step and summarize conditional, noncausal operator yields."""
    totals: Counter = Counter()
    actions: dict[int, Counter] = defaultdict(Counter)
    depths: dict[int, Counter] = defaultdict(Counter)
    cells: dict[tuple[int, int], Counter] = defaultdict(Counter)
    prefixes: dict[int, Counter] = defaultdict(Counter)
    previous_terminal = True
    expected_step = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            step, state, action = r["step"], r["state"], r["action"]
            assert step == (0 if previous_terminal else expected_step)
            assert 0 <= state < len(q["q"]) and action in r["available_actions"]
            assert r["action_seconds"] > 0 and r["seconds"] >= r["action_seconds"]
            assert len(r["candidate_objectives"]) == len(r["candidate_feasible"])
            assert len(r["candidate_objectives"]) == r["exact_evaluations"]
            assert math.isfinite(r["reward"]) and r["scalar_after"] <= r["scalar_before"] + 1e-9
            if arm == "STATE_TIME42":
                assert state % 2 == int(r["wall_progress"] >= 0.5)
            if arm == "STATE_TIME84":
                assert state % 2 == int(r["wall_progress"] >= 0.5)
            values = dict(
                steps=1,
                accepted=int(r["accepted"]),
                positive_quality=int(r["quality_reward"] > 0),
                negative_reward=int(r["reward"] < 0),
                zero_quality=int(r["quality_reward"] == 0),
                no_feasible=int(r["feasible"] == 0),
                empty_batch=int(r["exact_evaluations"] == 0),
                explore=int(r["explore"]),
                action_seconds=r["action_seconds"],
                logged_seconds=r["seconds"],
                exact=r["exact_evaluations"],
                scalar_delta=max(0.0, r["scalar_before"] - r["scalar_after"]),
                reward=r["reward"],
                quality_reward=r["quality_reward"],
                retained=r["archive_net_retained"],
                insertions=r["archive_insertions"],
                objective_gain=r["archive_hv_gain"],
            )
            for target in (totals, actions[action], depths[step + 1], cells[(state, action)]):
                target.update(values)
            totals["trajectories"] += int(step == 0)
            totals["terminal"] += int(r["terminal"])
            totals["late_steps"] += int(r["wall_progress"] >= 0.5)
            ideals, maximums = r["comparison_ideal"], r["comparison_maximum"]
            candidate_scores = [
                max(
                    w * abs((p[j] - ideals[j]) / max(maximums[j] - ideals[j], 1e-9))
                    for j, w in enumerate(r["weight"])
                )
                if feasible
                else math.inf
                for p, feasible in zip(r["candidate_objectives"], r["candidate_feasible"])
            ]
            full_best = min([r["scalar_before"], *candidate_scores])
            for k in (1, 2, 3, 6):
                if len(candidate_scores) < k:
                    continue
                prefix_best = min([r["scalar_before"], *candidate_scores[:k]])
                z = prefixes[k]
                z["eligible_steps"] += 1
                z["extra_candidate_win"] += int(
                    full_best < prefix_best
                    and not math.isclose(full_best, prefix_best, rel_tol=1e-9, abs_tol=1e-9)
                )
                z["full_scalar_gain"] += max(0.0, r["scalar_before"] - full_best)
                z["prefix_scalar_gain"] += max(0.0, r["scalar_before"] - prefix_best)
            previous_terminal = r["terminal"]
            expected_step = step + 1
    assert previous_terminal and totals["trajectories"] == totals["terminal"]
    assert totals["steps"] == int(np.asarray(q["selections"]).sum())
    for action in range(1, 9):
        assert actions[action]["steps"] == int(np.asarray(q["selections"])[:, action - 1].sum())
    count = max(1, totals["steps"])
    result = dict(
        rl_steps=totals["steps"],
        trajectories=totals["trajectories"],
        steps_per_trajectory=totals["steps"] / max(1, totals["trajectories"]),
        accepted_rate=totals["accepted"] / count,
        no_feasible_rate=totals["no_feasible"] / count,
        empty_batch_rate=totals["empty_batch"] / count,
        zero_quality_rate=totals["zero_quality"] / count,
        negative_reward_rate=totals["negative_reward"] / count,
        mean_quality_reward=totals["quality_reward"] / count,
        mean_reward=totals["reward"] / count,
        step_action_seconds=totals["action_seconds"],
        step_logged_seconds=totals["logged_seconds"],
        action_seconds_per_step=totals["action_seconds"] / count,
        mean_exact_per_step=totals["exact"] / count,
        retained_per_action_second=totals["retained"] / max(1e-12, totals["action_seconds"]),
        late_step_fraction=totals["late_steps"] / count,
        exploration_fraction=totals["explore"] / count,
    )
    action_rows = [
        dict(action=i, **dict(actions[i])) for i in sorted(actions) if actions[i]["steps"]
    ]
    depth_rows = [dict(depth=i, **dict(depths[i])) for i in sorted(depths)]
    prefix_rows = [dict(prefix=k, **dict(v)) for k, v in sorted(prefixes.items())]
    return result, action_rows, depth_rows, prefix_rows


def analyze(data: Path, out: Path) -> dict:
    """Recheck every recovered byte and execute the sealed comparison families."""
    out.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((data / "LOCAL_RECOVERY_ACCEPTANCE.json").read_text())
    assert receipt["accepted"] and receipt["formal_runs"] == 1472
    saved = json.loads((data / "SERVER_VALIDATION_AND_MANIFEST.json").read_text())
    for item in saved["files"]:
        p = data / "raw" / item["path"]
        with p.open("rb") as f:
            assert p.stat().st_size == item["size"]
            assert hashlib.file_digest(f, "sha256").hexdigest() == item["sha256"]
    raw = data / "raw/formal"
    m = json.loads((raw / "manifest.json").read_text())
    assert tuple(m["arms"]) == ARMS and m["runs"] == 1472
    specs = {k: json.loads((raw / "specs" / (k + ".json")).read_text()) for k in m["blocks"]}
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for key, spec in specs.items():
        groups[(spec["base_seed"], spec["variant"])].append(key)
    assert len(groups) == 32 and all(len(v) == 2 for v in groups.values())
    runs, actions, depths, prefixes, anytime, fronts, references = [], [], [], [], [], {}, {}
    checks = dict(raw_file_sha=len(saved["files"]), hv_dual=0, igd_plus_dual=0, step_streams=0)
    for index, ((parent, variant), keys) in enumerate(sorted(groups.items())):
        loaded = {}
        for key in sorted(keys):
            initial_hashes = set()
            for arm in ARMS:
                p = raw / "runs" / key / arm
                a, members = stored_points(p / "archive.json.gz")
                pop, _ = stored_points(p / "population.json.gz")
                initial, _ = stored_points(p / "initial.json.gz")
                loaded[(key, arm)] = a, pop, initial, members
                s = json.loads((p / "summary.json").read_text())
                initial_hashes.add(
                    (
                        s["initialization"]["initial_population_sha256"],
                        s["initialization"]["initial_archive_sha256"],
                    )
                )
            assert len(initial_hashes) == 1
        ref = nondominated([point for a, _, _, _ in loaded.values() for point in a])
        array = np.asarray(ref)
        lo, span = array.min(0), array.max(0) - array.min(0)
        span = np.where(span <= 1e-12, 1.0, span)

        def normalize(points: list) -> list:
            return ((np.asarray(points) - lo) / span).tolist()

        rn = normalize(ref)
        refid = f"b{parent}_v{variant}"
        references[refid] = dict(ideal=lo.tolist(), span=span.tolist(), front=ref)
        for (key, arm), (archive, pop, initial, member_count) in loaded.items():
            p, spec = raw / "runs" / key / arm, specs[key]
            s = json.loads((p / "summary.json").read_text())
            d = json.loads((p / "diagnostics.json").read_text())
            c, t = d["counts"], d["seconds"]
            q = json.loads((p / "qtable.json").read_text())
            assert s["source"] == m["source"] and s["seed"] == spec["seed"]
            assert len(q["q"]) == (84 if arm == "STATE_TIME84" else 6 if arm == "POOLED6" else 42)
            step, ars, drs, prs = step_diagnostics(p / "steps.jsonl.gz", arm, q)
            checks["step_streams"] += 1
            a, pn, ini = normalize(archive), normalize(pop), normalize(initial)
            score = volume(a)
            assert math.isclose(score, volume_audit(a), abs_tol=1e-11, rel_tol=1e-11)
            checks["hv_dual"] += 1
            sample = rn[:: max(1, len(rn) // 16)][:16]
            assert math.isclose(
                distance(a, sample), igd_plus(a, sample), abs_tol=1e-11, rel_tol=1e-11
            )
            checks["igd_plus_dual"] += 1
            initial_seconds, polish = t["initialization"], t.get("polish", 0.0)
            outer = sum(t.get(k + ":structural", 0.0) for k in ("ssgs", "exact", "archive"))
            ls = t.get("budget", 0.0) + sum(
                t.get(f"{k}:A{i}", 0.0)
                for i in range(1, 9)
                for k in ("construction", "ssgs", "exact", "archive")
            )
            production = s["production_seconds"]
            remaining = production - initial_seconds - polish - outer - ls
            assert remaining >= -1e-5
            snaps = json.loads((p / "anytime.json").read_text())
            initial_snap = [v for v in snaps if v["target"] == "initialization"]
            assert len(initial_snap) == 1
            init_archive_hv = volume(normalize(initial_snap[0]["objectives"]))
            assert score >= init_archive_hv - 1e-9
            bill = normalize(loaded[(key, "BILL_PERMUTED")][0])
            baseline = normalize(loaded[(key, BASELINE)][0])
            sc = s["structural_diagnostics"]
            visits = np.asarray(q["visits"], dtype=float)
            positive_visits = visits[visits > 0]
            probabilities = positive_visits / max(1.0, visits.sum())
            identity = dict(
                key=key,
                arm=arm,
                n=spec["n"],
                base_seed=parent,
                variant=variant,
                seed=spec["seed"],
                regions=spec["regions"],
                machines=spec["machines"],
                pressure=spec["pressure"],
            )
            row = dict(
                **identity,
                hv=score,
                igd_plus=distance(a, rn),
                igd=distance(a, rn, False),
                pop_hv=volume(pn),
                init_pop_hv=volume(ini),
                init_archive_hv=init_archive_hv,
                archive_gain=score - init_archive_hv,
                pop_gain=volume(pn) - volume(ini),
                hv_vs_bill_pct=100 * (score / volume(bill) - 1),
                hv_vs_baseline_pct=100 * (score / volume(baseline) - 1),
                coverage_to_bill=coverage(a, bill),
                coverage_from_bill=coverage(bill, a),
                min_flow=min(x for x, y in archive),
                min_bill=min(y for x, y in archive),
                archive_members=member_count,
                archive_nd=len(archive),
                population_nd=len(pop),
                population_unique_genotypes=sc["population_unique_genotypes"],
                search_unique_genotypes=sc["search_unique_structural_genotypes"],
                offspring=s["offspring"],
                generations=s["offspring"] / 100,
                exact=c["exact"],
                ssgs=c["ssgs"],
                production=production,
                initialization=initial_seconds,
                initialization_share=initial_seconds / production,
                polish=polish,
                polish_share=polish / production,
                ls_measured=ls,
                ls_measured_share=ls / production,
                outer_evaluation=outer,
                outer_evaluation_share=outer / production,
                unclassified=remaining,
                unclassified_share=remaining / production,
                step_action_share=step["step_action_seconds"] / production,
                ls_support_unclassified=max(0.0, step["step_action_seconds"] - ls),
                audit_seconds=s["audit_seconds"],
                overshoot=production - spec["seconds"],
                trigger_hits=c.get("trigger_hits", 0),
                trigger_rate=c.get("trigger_hits", 0) / max(1, c.get("trigger_attempts", 0)),
                polish_calls=c.get("polish_calls", 0),
                polish_success_rate=c.get("polish_improved_calls", 0)
                / max(1, c.get("polish_calls", 0)),
                polish_exact=c.get("polish_exact", 0),
                polish_seconds_per_call=polish / max(1, c.get("polish_calls", 0)),
                q_visited_states=len(positive_visits),
                q_states=len(q["q"]),
                q_active_state_entropy=float(-np.sum(probabilities * np.log(probabilities)))
                if len(probabilities)
                else 0.0,
                q_mean_visits_per_visited_state=float(positive_visits.mean())
                if len(positive_visits)
                else 0.0,
                q_unvisited_fraction=1 - len(positive_visits) / len(q["q"]),
                q_negative_fraction=float(np.mean(np.asarray(q["q"]) < 0)),
                cpu=s["cpu"],
                started=s["started"],
                completed=s["completed"],
                **step,
            )
            assert step["trajectories"] == c.get("trigger_hits", 0)
            runs.append(row)
            for ar in ars:
                i = ar["action"]
                ar.update(
                    {
                        k + "_seconds": t.get(f"{k}:A{i}", 0.0)
                        for k in ("construction", "ssgs", "exact", "archive")
                    }
                )
                actions.append(dict(**identity, **ar))
            depths.extend(dict(**identity, **x) for x in drs)
            prefixes.extend(dict(**identity, **x) for x in prs)
            fronts[key + "|" + arm] = dict(
                archive=a, population=pn, initial_population=ini, reference=refid
            )
            for snap in snaps:
                anytime.append(
                    dict(
                        **identity,
                        target=snap["target"],
                        elapsed=snap["elapsed"],
                        fraction=snap["elapsed"] / spec["seconds"],
                        hv=volume(normalize(snap["objectives"])),
                    )
                )
        print(json.dumps(dict(scored_inputs=index + 1, total=32)), flush=True)
    numeric = [
        k
        for k, v in runs[0].items()
        if isinstance(v, (int, float))
        and k
        not in (
            "n",
            "base_seed",
            "variant",
            "seed",
            "regions",
            "machines",
            "cpu",
            "started",
            "completed",
        )
    ]
    clusters: dict[tuple[int, str], list] = defaultdict(list)
    for r in runs:
        clusters[(r["base_seed"], r["arm"])].append(r)
    parents = []
    for (base, arm), items in sorted(clusters.items()):
        assert len(items) == 4 and {(r["variant"], r["seed"]) for r in items} == {
            (v, s) for v in (0, 1) for s in (16101, 26101)
        }
        parents.append(
            dict(
                base_seed=base,
                arm=arm,
                n=items[0]["n"],
                **{k: float(np.mean([r[k] for r in items])) for k in numeric},
            )
        )
    pairs = {(r["base_seed"], r["arm"]): r for r in parents}
    bases = sorted({r["base_seed"] for r in parents})
    comparisons = [("primary", arm, BASELINE) for arm in ARMS[6:22]]
    comparisons += [("polish", ARMS[i], ARMS[j]) for i in range(2, 6) for j in range(i + 1, 6)]
    comparisons += [("plain", arm, "BILL_PERMUTED") for arm in ARMS[2:6]]
    comparisons += [
        ("combined", "TIME_PENALTY_ACTIVE42", arm)
        for arm in ("REWARD_TIME_PENALTY", "STATE_ACTIVE42")
    ]
    statistics, contrasts = [], []
    for family, left, right in comparisons:
        contrast = left + "-" + right
        for metric in ("hv", "igd_plus"):
            vals = [
                (pairs[(b, left)][metric] - pairs[(b, right)][metric])
                * (1 if metric == "hv" else -1)
                for b in bases
            ]
            sizes = [pairs[(b, left)]["n"] for b in bases]
            seed = ROOT_SEED + int.from_bytes(
                hashlib.sha256((contrast + metric).encode()).digest()[:4], "big"
            )
            statistics.append(
                dict(
                    family=family,
                    contrast=contrast,
                    left=left,
                    right=right,
                    metric=metric,
                    **inference(vals, sizes, seed),
                )
            )
            contrasts.extend(
                dict(family=family, contrast=contrast, metric=metric, base_seed=b, n=n, effect=v)
                for b, n, v in zip(bases, sizes, vals)
            )
    for family, count in (("primary", 32), ("polish", 12), ("plain", 8), ("combined", 4)):
        items = [r for r in statistics if r["family"] == family]
        assert len(items) == count
        for row, p in zip(items, holm([r["p"] for r in items])):
            row["p_holm"] = p
    summaries = []
    for n in (0, *SIZES):
        for arm in ARMS:
            items = [r for r in parents if r["arm"] == arm and (not n or r["n"] == n)]
            summaries.append(
                dict(
                    n=n,
                    arm=arm,
                    parents=len(items),
                    **{k: float(np.mean([r[k] for r in items])) for k in numeric},
                )
            )
    for name, rows in [
        ("run_metrics", runs),
        ("parent_metrics", parents),
        ("arm_summary", summaries),
        ("action_metrics", actions),
        ("depth_metrics", depths),
        ("prefix_metrics", prefixes),
        ("anytime_metrics", anytime),
        ("parent_contrasts", contrasts),
    ]:
        write_table(out, name, rows)
    result = dict(
        formal_runs=1472,
        blocks=64,
        arms=23,
        independent_parents=16,
        root_seed=ROOT_SEED,
        bootstrap=20000,
        signflip=100000,
        statistics=statistics,
        checks=checks,
        raw_sha256=receipt["archive_sha256"],
        limitations=[
            "Development panel reused; four parents per size",
            "Empirical union is not true Pareto front",
            "Step/operator returns are conditional; not randomized action effects",
            "No fresh full exact re-evaluation in this analysis",
            "Measured LS subcomponents and step action durations overlap; never add the two",
            "TIME42 numeric state bit and wall_progress supersede legacy search_progress text",
        ],
    )
    (out / "statistics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "references.json").write_text(json.dumps(references), encoding="utf-8")
    (out / "normalized_fronts.json").write_text(json.dumps(fronts), encoding="utf-8")
    print(json.dumps(dict(completed=True, checks=checks, statistics=statistics)), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.data.resolve(), args.output.resolve())
