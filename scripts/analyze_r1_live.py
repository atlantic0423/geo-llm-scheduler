"""Replay sampled live opportunities independently, without changing the recorded search."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import r1_offline as r1

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.experiments.trans_runner import digest, write_json
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.numeric import TOL, close, less


def analyze_snapshot(problem: ProblemInstance, snapshot: dict) -> tuple[list[dict], list[dict]]:
    """Evaluate all regions of the preselected pairs against their own live snapshot.

    Each mixed candidate starts from the unchanged receiver. The ideal is updated
    hypothetically with that candidate, with the recorded post-replacement maximum
    frozen; all three scalar scores use the same context. No candidate is fed back.
    """
    donor = restore_candidate(snapshot["donor"])
    members = [restore_candidate(c) for c in snapshot["archive"]]
    front = [tuple(p) for p in snapshot["front"]]
    assert sorted(set(c.evaluation.objectives for c in members)) == sorted(front)
    recorded = snapshot["context"]
    frozen = NormalizationContext(tuple(recorded["ideal"]), tuple(recorded["maximum"]))
    for candidate in [donor] + [restore_candidate(r["candidate"]) for r in snapshot["receivers"]]:
        rebuilt = decode(problem, candidate.genotype)
        assert all(close(a, b, TOL.time) for a, b in zip(rebuilt.starts, candidate.schedule.starts))
        assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation
    common = dict(
        stage=snapshot["stage"],
        processed=snapshot["processed"],
        elapsed=snapshot["elapsed"],
        origin=donor.origin,
        birth=snapshot["subproblem"],
    )
    trials, pairs = [], []
    for receiver in snapshot["receivers"]:
        recipient = restore_candidate(receiver["candidate"])
        weight = tuple(receiver["weight"])
        tick = perf_counter()
        eligible = [
            r
            for r in range(len(problem.regions))
            if r1.region_jobs(problem, donor.genotype, r)
            and r1.region_jobs(problem, donor.genotype, r)
            == r1.region_jobs(problem, recipient.genotype, r)
        ]
        scan_seconds = perf_counter() - tick
        pair = dict(
            **common,
            slot=receiver["slot"],
            eligible=len(eligible),
            scan_seconds=scan_seconds,
            rescued=False,
            hv_positive=False,
            direction_improves=False,
            nonidentical=False,
            trial_seconds=0.0,
        )
        anchors = front + [recipient.evaluation.objectives, donor.evaluation.objectives]
        lo = tuple(min(p[k] for p in anchors) for k in range(2))
        hi = tuple(max(p[k] for p in anchors) for k in range(2))
        span = tuple(max(hi[k] - lo[k], 1e-9) for k in range(2))
        ref = tuple(hi[k] + 0.1 * span[k] for k in range(2))
        base_hv = r1.hypervolume(front, ref)
        for region in eligible:
            tick = perf_counter()
            genotype, predicted = r1.mix_region(problem, donor, recipient, region)
            construct_seconds = perf_counter() - tick
            tick = perf_counter()
            schedule = decode(problem, genotype)
            mixed = Candidate(
                genotype, schedule, evaluate(problem, genotype, schedule), "R1_REPLAY"
            )
            exact_seconds = perf_counter() - tick
            assert mixed.evaluation.feasible
            assert all(close(a, b, TOL.time) for a, b in zip(predicted.starts, schedule.starts))
            tick = perf_counter()
            ideal = (
                min(frozen.ideal[0], mixed.evaluation.objectives[0]),
                min(frozen.ideal[1], mixed.evaluation.objectives[1]),
            )
            context = NormalizationContext(ideal, frozen.maximum)
            whole = less(
                context.scalar(donor, weight), context.scalar(recipient, weight), TOL.scalar
            )
            improve = less(
                context.scalar(mixed, weight), context.scalar(recipient, weight), TOL.scalar
            )
            rescued = not whole and improve
            frozen_rescue = not less(
                frozen.scalar(donor, weight), frozen.scalar(recipient, weight), TOL.scalar
            ) and less(frozen.scalar(mixed, weight), frozen.scalar(recipient, weight), TOL.scalar)
            decision_seconds = perf_counter() - tick
            # Recreate the live archive and warm its identity cache outside insertion
            # timing. Setup is reported separately, not falsely charged as online work.
            tick = perf_counter()
            archive = Archive()
            archive.members = list(members)
            archive.identity_keys()
            archive_setup_seconds = perf_counter() - tick
            tick = perf_counter()
            inserted = archive.consider(mixed)
            archive_seconds = perf_counter() - tick
            gain = r1.hypervolume(front + [mixed.evaluation.objectives], ref) - base_hv
            scaled_gain = max(0.0, gain / (span[0] * span[1]))
            # Independent interval integration is audit work, excluded from the
            # measured prospective online cost; all regional windows are checked.
            r1._oracle_check(problem, mixed)
            nonidentical = genotype != recipient.genotype
            positive = inserted and scaled_gain > 1e-12
            total = construct_seconds + exact_seconds + decision_seconds + archive_seconds
            pair["rescued"] |= rescued
            pair["direction_improves"] |= improve
            pair["nonidentical"] |= nonidentical
            pair["hv_positive"] |= positive
            pair["trial_seconds"] += total
            trials.append(
                dict(
                    **common,
                    slot=receiver["slot"],
                    region=region,
                    whole_improves=whole,
                    block_improves=improve,
                    rescued=rescued,
                    frozen_rescued=frozen_rescue,
                    inserted=inserted,
                    hv_positive=positive,
                    scaled_hv_gain=scaled_gain,
                    relative_hv_gain=max(0.0, gain / base_hv) if base_hv > 0 else None,
                    nonidentical=nonidentical,
                    construct_seconds=construct_seconds,
                    exact_seconds=exact_seconds,
                    decision_seconds=decision_seconds,
                    archive_seconds=archive_seconds,
                    archive_setup_seconds=archive_setup_seconds,
                    total_seconds=total,
                    recipient=recipient.evaluation.objectives,
                    donor=donor.evaluation.objectives,
                    mixed=mixed.evaluation.objectives,
                    ideal_changed=ideal != frozen.ideal,
                )
            )
        pairs.append(pair)
    return pairs, trials


def analyze_run(task: tuple[str, str]) -> dict:
    """Verify source artifacts then write one run's complete staged replay evidence."""
    run, output = (Path(x) for x in task)
    key = run.parent.name
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for entry in json.loads((run / "manifest.json").read_text(encoding="utf-8")):
        path = run / entry["path"]
        assert path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"]
    instance = run.parents[2] / "instances" / cfg["instance"]
    assert digest(instance) == cfg["instance_sha256"]
    problem = load_instance(instance)
    with gzip.open(run / "regional_observation.json.gz", "rt", encoding="utf-8") as f:
        record = json.load(f)
    pairs: list[dict] = []
    trials: list[dict] = []
    for snapshot in record["snapshots"]:
        p, t = analyze_snapshot(problem, snapshot)
        pairs.extend(dict(key=key, n=len(problem.jobs), **x) for x in p)
        trials.extend(dict(key=key, n=len(problem.jobs), **x) for x in t)
    result = dict(
        key=key,
        n=len(problem.jobs),
        seed=cfg["seed"],
        seen=record["seen"],
        observer_seconds=record["observer_seconds"],
        samples=len(record["snapshots"]),
        pairs=pairs,
        trials=trials,
        input_sha256=cfg["instance_sha256"],
        source_commit=cfg["source"]["commit"],
        audit="passed",
    )
    write_json(output / (key + ".json"), result)
    return result


def summarize(results: list[dict]) -> dict:
    """Report pair denominators and instance-clustered descriptive rates per stage."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in results:
        for p in run["pairs"]:
            groups[(p["n"], p["stage"])].append(p)
    rows = []
    for (n, stage), pairs in sorted(groups.items()):
        eligible = sum(p["eligible"] > 0 for p in pairs)
        rescued = sum(p["rescued"] for p in pairs)
        positive = sum(p["hv_positive"] for p in pairs)
        by_instance: dict[str, list[dict]] = defaultdict(list)
        for p in pairs:
            by_instance[p["key"]].append(p)
        rates = [sum(p["rescued"] for p in ps) / len(ps) for ps in by_instance.values()]
        rows.append(
            dict(
                n=n,
                stage=stage,
                pairs=len(pairs),
                eligible_pairs=eligible,
                rescued_pairs=rescued,
                hv_positive_pairs=positive,
                rescue_unconditional=rescued / len(pairs),
                rescue_conditional=rescued / eligible if eligible else None,
                instance_rescue_mean=statistics.mean(rates),
                instance_rescue_median=statistics.median(rates),
                covered_instances=sum(any(p["rescued"] for p in ps) for ps in by_instance.values()),
                hv_covered_instances=sum(
                    any(p["hv_positive"] for p in ps) for ps in by_instance.values()
                ),
                prospective_seconds=sum(p["scan_seconds"] + p["trial_seconds"] for p in pairs),
            )
        )
    return dict(
        runs=len(results),
        strata=rows,
        trials=sum(len(r["trials"]) for r in results),
        observer_seconds=sum(r["observer_seconds"] for r in results),
        status="offline opportunities only; no online net-gain claim",
    )


def main() -> None:
    """Analyze an explicitly supplied recovered campaign; never launch search."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    runs = sorted(args.results.glob("formal_panel_shard*/runs/*/GS4_GUIDED_RAND_SWITCH_D4"))
    if len(runs) != 120:
        raise ValueError(f"Expected all 120 sealed runs, got {len(runs)}")
    args.out.mkdir(parents=True, exist_ok=True)
    per_run = args.out / "runs"
    per_run.mkdir(exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(analyze_run, [(str(r), str(per_run)) for r in runs]))
    summary = summarize(results)
    write_json(args.out / "analysis.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
