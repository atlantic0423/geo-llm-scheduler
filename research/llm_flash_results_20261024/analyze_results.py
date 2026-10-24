"""Frozen three-arm parent-level quality, online-policy and cost analysis."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from stats_core import (
    ROOT_SEED,
    SIZES,
    coverage,
    distance,
    holm,
    inference,
    nondominated,
    stored_points,
    volume,
    volume_audit,
    write_table,
)

from geo_llm_scheduler.experiments.metrics import igd_plus

ARMS = ("BILL_MOEAD", "MAIN_Q", "LLM_FLASH")


def json_read(path: Path) -> Any:
    """Read a complete UTF-8 JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def flow_window(points: list, limit: float) -> float | None:
    """Best bill among actual points satisfying a shared Flow limit, no interpolation."""
    values = [p[1] for p in points if p[0] <= limit]
    return min(values) if values else None


def steps(path: Path, arm: str) -> tuple[dict, list, dict]:
    """Read every LS step without equating main-Q insertion credit with LLM new objectives."""
    totals: Counter = Counter()
    actions: dict[int, Counter] = defaultdict(Counter)
    focuses: Counter = Counter()
    previous_terminal = True
    expected = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            assert row["step"] == (0 if previous_terminal else expected)
            assert row["action_seconds"] >= 0 and row["scalar_after"] <= row["scalar_before"] + 1e-8
            assert (
                row["exact_evaluations"]
                == len(row["candidate_objectives"])
                == len(row["candidate_feasible"])
            )
            values = dict(
                calls=1,
                action_s=row["action_seconds"],
                accepts=int(row["accepted"]),
                exact=row["exact_evaluations"],
                empty=int(row["exact_evaluations"] == 0),
                scalar_gain=max(0.0, row["scalar_before"] - row["scalar_after"]),
                new_objective_events=row.get("archive_new_objective_events", 0),
                dominated_members=row.get("dominated_archive_members", 0),
                main_archive_insertions=row.get("archive_insertions", 0),
                online_hv_gain=row.get("archive_hv_gain", 0.0),
            )
            totals.update(values)
            actions[row["action"]].update(values)
            totals["trajectories"] += int(row["step"] == 0)
            totals["terminals"] += int(row["terminal"])
            no_gain = row["terminal"] and row.get("stop_reason") == "no_gain_threshold"
            totals["no_gain_terminals"] += int(no_gain)
            totals["early_stops"] += int(no_gain and row["step"] < 4)
            totals["depth_total"] += (row["step"] + 1) * int(row["terminal"])
            if arm == "LLM_FLASH":
                focuses[(row["action"], row["requested_focus"], row["applied_focus"])] += 1
            previous_terminal = row["terminal"]
            expected = row["step"] + 1
    assert previous_terminal and totals["trajectories"] == totals["terminals"]
    result = dict(
        ls_steps=totals["calls"],
        ls_trajectories=totals["trajectories"],
        ls_mean_depth=totals["depth_total"] / max(1, totals["trajectories"]),
        ls_early_stop_rate=totals["early_stops"] / max(1, totals["trajectories"]),
        ls_no_gain_terminal_rate=totals["no_gain_terminals"] / max(1, totals["trajectories"]),
        ls_local_accept_rate=totals["accepts"] / max(1, totals["calls"]),
        ls_empty_rate=totals["empty"] / max(1, totals["calls"]),
        ls_step_wall=totals["action_s"],
    )
    return (
        result,
        [dict(action=i, **values) for i, values in sorted(actions.items())],
        dict(focuses),
    )


def analyze(data: Path, out: Path) -> dict:
    """Verify raw bytes, score all192 runs, then execute the four frozen tests."""
    out.mkdir(parents=True, exist_ok=True)
    assert json_read(data / "FORMAL_LOCAL_RECOVERY_ACCEPTANCE.json")["accepted"]
    receipt = json_read(data / "FORMAL_COMPLETE_MANIFEST.json")
    for row in receipt["files"]:
        path = data / "formal_raw" / row["path"]
        assert path.stat().st_size == row["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    raw = data / "formal_raw/formal"
    manifest = json_read(raw / "manifest.json")
    plan = json_read(Path(__file__).with_name("ANALYSIS_PLAN.json"))
    assert (
        tuple(manifest["arms"]) == ARMS
        and manifest["runs"] == 192
        and manifest["source"] == plan["source_hash"]
    )
    specs = {key: json_read(raw / "specs" / (key + ".json")) for key in manifest["blocks"]}
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for key, spec in specs.items():
        groups[(spec["base_seed"], spec["variant"])].append(key)
    assert len(groups) == 32 and all(len(keys) == 2 for keys in groups.values())
    runs, action_rows, focus_rows, call_rows, policy_rows, anytime = [], [], [], [], [], []
    references, fronts = {}, {}
    dual_hv = dual_igd = 0
    for index, ((parent, variant), keys) in enumerate(sorted(groups.items())):
        loaded = {}
        for key in keys:
            hashes = set()
            for arm in ARMS:
                path = raw / "runs" / key / arm
                archive, members = stored_points(path / "archive.json.gz")
                pop, _ = stored_points(path / "population.json.gz")
                initial, _ = stored_points(path / "initial.json.gz")
                loaded[(key, arm)] = (archive, pop, initial, members)
                summary = json_read(path / "summary.json")
                hashes.add(
                    (
                        summary["initialization"]["initial_population_sha256"],
                        summary["initialization"]["initial_archive_sha256"],
                    )
                )
            assert len(hashes) == 1
        reference = nondominated([point for a, _, _, _ in loaded.values() for point in a])
        array = np.asarray(reference)
        ideal = array.min(0)
        span = array.max(0) - ideal
        span = np.where(span <= 1e-12, 1.0, span)

        def normalize(points: list) -> list:
            return ((np.asarray(points) - ideal) / span).tolist()

        rn = normalize(reference)
        refid = f"b{parent}_v{variant}"
        references[refid] = dict(ideal=ideal.tolist(), span=span.tolist(), front=reference)
        common_fast = min(p[0] for p in reference)
        for (key, arm), (archive, pop, initial, members) in loaded.items():
            path = raw / "runs" / key / arm
            spec = specs[key]
            s = json_read(path / "summary.json")
            d = json_read(path / "diagnostics.json")
            c, t = d["counts"], d["seconds"]
            assert s["source"] == manifest["source"] and s["seed"] == spec["seed"]
            a, pn, ini = normalize(archive), normalize(pop), normalize(initial)
            hv = volume(a)
            assert math.isclose(hv, volume_audit(a), abs_tol=1e-10, rel_tol=1e-10)
            dual_hv += 1
            sample = rn[:: max(1, len(rn) // 16)][:16]
            assert math.isclose(
                distance(a, sample), igd_plus(a, sample), abs_tol=1e-10, rel_tol=1e-10
            )
            dual_igd += 1
            step, actions, focus = steps(path / "steps.jsonl.gz", arm)
            # The LLM checks its wall deadline before its first LS step: a
            # previously counted trigger can therefore yield no trajectory.
            empty_triggers = c.get("trigger_hits", 0) - step["ls_trajectories"]
            assert empty_triggers >= 0 and (arm == "LLM_FLASH" or empty_triggers == 0)
            assert empty_triggers <= 1
            if empty_triggers:
                assert s["production_seconds"] >= spec["seconds"]
            step["trigger_without_step"] = empty_triggers
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
            llm = s["initialization"].get("llm", {})
            initialization = t["initialization"]
            polish = t.get("polish", 0.0)
            outer = sum(t.get(f"{k}:structural", 0.0) for k in ("ssgs", "exact", "archive"))
            ls = t.get("budget", 0.0) + sum(
                t.get(f"{k}:A{i}", 0.0)
                for i in range(1, 9)
                for k in ("construction", "ssgs", "exact", "archive")
            )
            wait = t.get("llm_wait", 0.0)
            features = t.get("llm_features", 0.0)
            other = s["production_seconds"] - initialization - polish - outer - ls - wait - features
            assert other >= -1e-4
            baseline = loaded[(key, "MAIN_Q")][0]
            plain = loaded[(key, "BILL_MOEAD")][0]
            row = dict(
                **identity,
                hv=hv,
                igd_plus=distance(a, rn),
                igd=distance(a, rn, False),
                population_hv=volume(pn),
                initial_population_hv=volume(ini),
                archive_nd=len(archive),
                archive_members=members,
                population_nd=len(pop),
                min_flow=min(p[0] for p in archive),
                min_bill=min(p[1] for p in archive),
                hv_vs_main_pct=100 * (hv / volume(normalize(baseline)) - 1),
                hv_vs_bill_pct=100 * (hv / volume(normalize(plain)) - 1),
                coverage_to_main=coverage(a, normalize(baseline)),
                coverage_from_main=coverage(normalize(baseline), a),
                coverage_to_bill=coverage(a, normalize(plain)),
                coverage_from_bill=coverage(normalize(plain), a),
                offspring=s["offspring"],
                exact=c["exact"],
                ssgs=c["ssgs"],
                production=s["production_seconds"],
                requested=spec["seconds"],
                overshoot=s["production_seconds"] - spec["seconds"],
                audit=s["audit_seconds"],
                initialization=initialization,
                polish=polish,
                outer_evaluation=outer,
                ls_measured=ls,
                llm_wait=wait,
                llm_features=features,
                other=other,
                llm_requests=llm.get("requests", 0),
                llm_valid=llm.get("valid_requests", 0),
                trigger_rate=c.get("trigger_hits", 0) / max(1, c.get("trigger_attempts", 0)),
                **step,
            )
            for name in (
                "initialization",
                "polish",
                "outer_evaluation",
                "ls_measured",
                "llm_wait",
                "llm_features",
                "other",
                "ls_step_wall",
            ):
                row[name + "_share"] = row[name] / row["production"]
            for factor in (1.0, 1.1, 1.25, 1.5):
                name = "window_" + str(factor).replace(".", "_")
                value = flow_window(archive, common_fast * factor)
                row[name + "_bill"] = value
                base_value = flow_window(baseline, common_fast * factor)
                row[name + "_saving_vs_main_pct"] = (
                    100 * (base_value - value) / base_value
                    if base_value is not None and value is not None
                    else None
                )
            runs.append(row)
            action_rows.extend(dict(**identity, **ar) for ar in actions)
            focus_rows.extend(
                dict(
                    **identity,
                    action=i,
                    requested_focus=requested,
                    applied_focus=applied,
                    calls=count,
                )
                for (i, requested, applied), count in focus.items()
            )
            if arm == "LLM_FLASH":
                with gzip.open(path / "policy_calls.jsonl.gz", "rt", encoding="utf-8") as f:
                    calls = [json.loads(line) for line in f if line.strip()]
                assert (
                    len(calls) == llm["requests"] == 3
                    and sum(v["accepted"] for v in calls) == llm["valid_requests"] > 0
                )
                for call in calls:
                    response = call.get("response") or {}
                    response = response if isinstance(response, dict) else {}
                    usage = response.get("usage", {})
                    call_rows.append(
                        dict(
                            **identity,
                            update_index=call["request"]["update_index"],
                            accepted=call["accepted"],
                            wait_s=call["wait_s"],
                            elapsed=call["completed_elapsed_s"],
                            revision_before=call["policy_revision_before"],
                            revision_after=call.get(
                                "policy_revision_after", call["policy_revision_before"]
                            ),
                            returned_model=response.get("returned_model"),
                            finish_reason=response.get("finish_reason"),
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                            failure_reason=call.get("failure_reason"),
                        )
                    )
                    if call["accepted"]:
                        for policy in call["effective_policy"]:
                            policy_rows.append(
                                dict(
                                    **identity,
                                    update_index=call["request"]["update_index"],
                                    group=policy["group"],
                                    focus=policy["target_focus"],
                                    stop_after_no_gain=policy["stop_after_no_gain"],
                                    **policy["action_weights"],
                                )
                            )
            fronts[key + "|" + arm] = dict(
                archive=a, population=pn, initial_population=ini, reference=refid
            )
            for snap in json_read(path / "anytime.json"):
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
        and k not in ("n", "base_seed", "variant", "seed", "regions", "machines")
        and all(isinstance(r[k], (int, float)) for r in runs)
    ]
    clusters: dict[tuple[int, str], list] = defaultdict(list)
    for row in runs:
        clusters[(row["base_seed"], row["arm"])].append(row)
    parents = []
    for (base, arm), items in sorted(clusters.items()):
        assert len(items) == 4 and {(r["variant"], r["seed"]) for r in items} == {
            (v, s) for v in (0, 1) for s in (16401, 26401)
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
    statistics, contrasts = [], []
    for right in ("MAIN_Q", "BILL_MOEAD"):
        contrast = "LLM_FLASH-" + right
        for metric in ("hv", "igd_plus"):
            values = [
                (pairs[(b, "LLM_FLASH")][metric] - pairs[(b, right)][metric])
                * (1 if metric == "hv" else -1)
                for b in bases
            ]
            sizes = [pairs[(b, "LLM_FLASH")]["n"] for b in bases]
            seed = ROOT_SEED + int.from_bytes(
                hashlib.sha256((contrast + metric).encode()).digest()[:4], "big"
            )
            statistics.append(
                dict(
                    family="primary",
                    contrast=contrast,
                    metric=metric,
                    **inference(values, sizes, seed),
                )
            )
            contrasts.extend(
                dict(contrast=contrast, metric=metric, base_seed=b, n=n, effect=v)
                for b, n, v in zip(bases, sizes, values)
            )
    for row, p in zip(statistics, holm([r["p"] for r in statistics])):
        row["p_holm"] = p
    summaries = []
    for n in (0, *SIZES):
        for arm in ARMS:
            items = [r for r in parents if r["arm"] == arm and (n == 0 or r["n"] == n)]
            summaries.append(
                dict(
                    n=n,
                    arm=arm,
                    parents=len(items),
                    **{k: float(np.mean([r[k] for r in items])) for k in numeric},
                )
            )
    for name, values in [
        ("run_metrics", runs),
        ("parent_metrics", parents),
        ("arm_summary", summaries),
        ("action_metrics", action_rows),
        ("focus_metrics", focus_rows),
        ("policy_calls", call_rows),
        ("policies", policy_rows),
        ("anytime_metrics", anytime),
        ("parent_contrasts", contrasts),
        ("primary_statistics", statistics),
    ]:
        write_table(out, name, values)
    for name, value in [
        ("references", references),
        ("normalized_fronts", fronts),
        ("primary_statistics", statistics),
        ("arm_summary", summaries),
    ]:
        (out / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    result = dict(
        accepted=True,
        runs=len(runs),
        parents=len(bases),
        inputs=len(groups),
        raw_sha_verified=len(receipt["files"]),
        hv_dual=dual_hv,
        igd_plus_dual=dual_igd,
        primary_tests=4,
        seed=ROOT_SEED,
        model_requests=len(call_rows),
        valid_requests=sum(v["accepted"] for v in call_rows),
        models=sorted({r["returned_model"] for r in call_rows if r["returned_model"]}),
        prompt_tokens=sum(v["prompt_tokens"] for v in call_rows),
        completion_tokens=sum(v["completion_tokens"] for v in call_rows),
        raw_changed=False,
        statistics=statistics,
    )
    (out / "ANALYSIS_ACCEPTANCE.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.data, args.out)
