"""Read-only, paired regional-block replay; never modifies an input or solver.

Final population pairs are a proxy, not online offspring observations. Sampling
is uniform over eligible directed (donor slot, recipient slot, region) triples.
Every trial starts from its original recipient, without chaining accepted blocks.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geo_llm_scheduler.archive.pareto import Archive  # noqa: E402
from geo_llm_scheduler.domain.models import (  # noqa: E402
    Candidate,
    Genotype,
    ProblemInstance,
    Schedule,
)
from geo_llm_scheduler.evaluation.exact import evaluate  # noqa: E402
from geo_llm_scheduler.experiments.p1_observation import restore_candidate  # noqa: E402
from geo_llm_scheduler.io.loaders import load_instance  # noqa: E402
from geo_llm_scheduler.moead.core import (  # noqa: E402
    NormalizationContext,
    maximum,
    neighborhoods,
    weights,
)
from geo_llm_scheduler.scheduling.ssgs import decode  # noqa: E402
from geo_llm_scheduler.utils.numeric import TOL, close, less  # noqa: E402

ARM = "GS4_GUIDED_RAND_SWITCH_D4"


def region_jobs(problem: ProblemInstance, genotype: Genotype, region: int) -> frozenset[int]:
    """Return the region's job set; reject cross-region P/D assignments."""
    selected = set()
    for j in range(len(problem.jobs)):
        rp = problem.instances[genotype.ms[2 * j]].region
        rd = problem.instances[genotype.ms[2 * j + 1]].region
        if rp != rd:
            raise ValueError("P/D region mismatch")
        if rp == region:
            selected.add(j)
    return frozenset(selected)


def mix_region(
    problem: ProblemInstance, donor: Candidate, recipient: Candidate, region: int
) -> tuple[Genotype, Schedule]:
    """Copy a same-job-set region's MS and relative OS; predict inherited timing.

    Recipient OS positions belonging to the region receive the donor's regional
    subsequence. Other positions remain unchanged. Timing is a prediction to audit
    against SSGS, not an accepted structural evaluation shortcut.
    """
    jobs = region_jobs(problem, donor.genotype, region)
    if not jobs or jobs != region_jobs(problem, recipient.genotype, region):
        raise ValueError("Requires equal nonempty regional job sets")
    local_os = iter(j for j in donor.genotype.os if j in jobs)
    os = tuple(next(local_os) if j in jobs else j for j in recipient.genotype.os)
    ms = tuple(
        donor.genotype.ms[o] if o // 2 in jobs else m for o, m in enumerate(recipient.genotype.ms)
    )
    starts = tuple(
        donor.schedule.starts[o] if o // 2 in jobs else t
        for o, t in enumerate(recipient.schedule.starts)
    )
    return Genotype(ms, os), Schedule(starts)


def billing_oracle(
    problem: ProblemInstance, genotype: Genotype, schedule: Schedule
) -> tuple[float, tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Independent direct interval integration; no production billing helpers.

    Returns regional TOU and every fixed 900-second average, including idle tails.
    Supports both legacy and the input's explicit load-dependent power model.
    """
    horizon = schedule.horizon(problem)
    tou_regions = []
    all_windows = []
    for r, region in enumerate(problem.regions):
        intervals = {
            m: [
                (o, t, t + problem.profile(o).duration)
                for o, t in enumerate(schedule.starts)
                if genotype.ms[o] == m
            ]
            for m, machine in enumerate(problem.instances)
            if machine.region == r
        }
        points = sorted(
            {0.0, horizon}
            | {
                t
                for ops in intervals.values()
                for _, a, b in ops
                for t in (a, b)
                if 0 <= t <= horizon
            }
        )
        total = 0.0
        windows = [0.0] * math.ceil(horizon / 900)
        model = problem.power_model
        for a, b in zip(points, points[1:]):
            power = 0.0
            for m, ops in intervals.items():
                machine = problem.instances[m]
                active = [o for o, start, end in ops if start <= a < end]
                p = machine.idle_kw
                if active:
                    load = sum(
                        problem.profile(o).compute
                        * (model.kappa_decode if o % 2 else model.kappa_prefill)
                        for o in active
                    )
                    p += (machine.active_kw - machine.idle_kw) * (
                        model.beta + (1 - model.beta) * load
                    )
                power += p
            for tariff in region.tariffs:
                overlap = max(0.0, min(b, tariff.end) - max(a, tariff.start))
                total += power * overlap * tariff.price / 3600
            for k in range(int(a // 900), min(len(windows), math.ceil(b / 900))):
                overlap = max(0.0, min(b, (k + 1) * 900) - max(a, k * 900))
                windows[k] += power * overlap / 900
        tou_regions.append(total)
        all_windows.append(tuple(windows))
    return sum(tou_regions), tuple(tou_regions), tuple(all_windows)


def hypervolume(points: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    """Exact 2-D minimizing rectangle union; permit points below anchor minima."""
    result = 0.0
    best_y = ref[1]
    for x, y in sorted(points):
        if x < ref[0] and y < best_y:
            result += (ref[0] - x) * (best_y - y)
            best_y = y
    return result


def _read(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def _oracle_check(problem: ProblemInstance, candidate: Candidate) -> tuple[float, ...]:
    tou, regional_tou, windows = billing_oracle(problem, candidate.genotype, candidate.schedule)
    result = candidate.evaluation
    assert close(tou, result.tou, TOL.cost), "Independent TOU mismatch"
    assert len(windows) == len(result.windows)
    for r, (actual, expected) in enumerate(zip(windows, result.windows)):
        assert len(actual) == len(expected)
        assert all(close(a, b, TOL.power) for a, b in zip(actual, expected))
        demand = max(actual, default=0) * problem.regions[r].demand_rate
        assert close(demand, result.demand[r], TOL.cost), "Independent demand mismatch"
    return tuple(regional_tou[r] + result.demand[r] for r in range(len(problem.regions)))


def _trial_run(task: tuple[str, str, str, int, int]) -> dict:
    run_s, instances_s, out_s, samples, master_seed = task
    run, out = Path(run_s), Path(out_s)
    key = run.parent.name
    begin = perf_counter()
    config = _read(run / "config.json")
    algorithm = _read(run / "algorithm_config.json")
    assert isinstance(config, dict) and isinstance(algorithm, dict)
    instance_file = run.parents[2] / "instances" / config["instance"]
    assert _sha(instance_file) == config["instance_sha256"]
    base_file = Path(instances_s) / config["instance"]
    base_data, actual_data = _read(base_file), _read(instance_file)
    assert isinstance(base_data, dict) and isinstance(actual_data, dict)
    assert {k: v for k, v in base_data.items() if k != "power_model"} == {
        k: v for k, v in actual_data.items() if k != "power_model"
    }, "Recovered instance differs beyond declared power model"
    inputs = {"instance": _sha(instance_file), "base_instance": _sha(base_file)}
    manifest = _read(run / "manifest.json")
    assert isinstance(manifest, list)
    required = {
        "native_population.json.gz",
        "archive.json.gz",
        "checkpoints.jsonl",
        "algorithm_config.json",
        "config.json",
    }
    for item in manifest:
        if item["path"] in required:
            path = run / item["path"]
            assert path.stat().st_size == item["bytes"] and _sha(path) == item["sha256"]
            inputs[item["path"]] = item["sha256"]
    assert required <= inputs.keys()
    problem = load_instance(instance_file)
    raw = _read(run / "native_population.json.gz")
    raw_archive = _read(run / "archive.json.gz")
    assert isinstance(raw, list) and isinstance(raw_archive, list)
    population = [restore_candidate(c) for c in raw]
    final_archive = [restore_candidate(c) for c in raw_archive]
    snapshots = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    prefront = sorted(set(tuple(p) for p in snapshots[-1]["front"]))
    final_front = sorted(set(c.evaluation.objectives for c in final_archive))
    anchors = prefront + final_front + [c.evaluation.objectives for c in population]
    ideal = tuple(min(p[k] for p in anchors) for k in range(2))
    hi = tuple(max(p[k] for p in anchors) for k in range(2))
    ref = tuple(hi[k] + 0.1 * max(hi[k] - ideal[k], 1e-9) for k in range(2))
    pre_hv, final_hv = hypervolume(prefront, ref), hypervolume(final_front, ref)
    lambdas = weights(len(population))
    neighbor_map = neighborhoods(lambdas, algorithm["engine"]["neighborhood"])
    pop_max = maximum(population)
    # Audit every distinct population phenotype, preserving original slot weights.
    checked: dict[tuple, tuple[bool, tuple[float, ...]]] = {}
    valid = []
    original_ledgers = []
    max_parent_start_error = 0.0
    for c in population:
        identity = (c.genotype, c.schedule.starts)
        if identity not in checked:
            rebuilt = decode(problem, c.genotype)
            error = max(abs(a - b) for a, b in zip(rebuilt.starts, c.schedule.starts))
            max_parent_start_error = max(max_parent_start_error, error)
            reproducible = all(
                close(a, b, TOL.time) for a, b in zip(rebuilt.starts, c.schedule.starts)
            )
            exact = evaluate(problem, c.genotype, c.schedule)
            assert exact.feasible and exact == c.evaluation, "Stored population exact mismatch"
            checked[identity] = reproducible, _oracle_check(problem, c)
        ok, ledger = checked[identity]
        valid.append(ok)
        original_ledgers.append(ledger)
    region_sets = [
        [region_jobs(problem, c.genotype, r) for r in range(len(problem.regions))]
        for c in population
    ]
    fingerprints = [
        [
            tuple(
                (o, c.genotype.ms[o], c.schedule.starts[o])
                for o in range(problem.operation_count)
                if o // 2 in jobs
            )
            for jobs in sets
        ]
        for c, sets in zip(population, region_sets)
    ]
    started = perf_counter()
    pair_count = same_sets = 0
    eligible = []
    for i, neighbors in enumerate(neighbor_map):
        for j in neighbors:
            if i == j:
                continue
            pair_count += 1
            for r in range(len(problem.regions)):
                if region_sets[i][r] and region_sets[i][r] == region_sets[j][r]:
                    same_sets += 1
                    if valid[i] and valid[j] and fingerprints[i][r] != fingerprints[j][r]:
                        eligible.append((i, j, r))
    eligibility_seconds = perf_counter() - started
    seed = int.from_bytes(hashlib.sha256(f"{master_seed}:{key}".encode()).digest()[:8], "big")
    rng = random.Random(seed)
    selected = rng.sample(eligible, min(samples, len(eligible)))
    rows = []
    examples: list[dict] = []
    union_pre = list(prefront)
    union_final = list(final_front)
    for donor_i, target_i, r in selected:
        donor, recipient = population[donor_i], population[target_i]
        start = perf_counter()
        genotype, predicted = mix_region(problem, donor, recipient, r)
        construction_seconds = perf_counter() - start
        start = perf_counter()
        decoded = decode(problem, genotype)
        decode_seconds = perf_counter() - start
        err = max(abs(a - b) for a, b in zip(predicted.starts, decoded.starts))
        assert all(close(a, b, TOL.time) for a, b in zip(predicted.starts, decoded.starts)), (
            "Regional inheritance mismatch",
            key,
            donor_i,
            target_i,
            r,
            err,
        )
        start = perf_counter()
        result = evaluate(problem, genotype, decoded)
        exact_seconds = perf_counter() - start
        assert result.feasible
        mixed = Candidate(genotype, decoded, result, "R1_OFFLINE")
        regional_bills = _oracle_check(problem, mixed)
        expected_flow = sum(
            (donor if job in region_sets[donor_i][r] else recipient).schedule.end(
                problem, 2 * job + 1
            )
            - info.release
            for job, info in enumerate(problem.jobs)
        )
        assert close(expected_flow, result.flow, TOL.time)
        # Use one context for mixed, donor and recipient, with candidate-aware ideal.
        context = NormalizationContext(
            tuple(min(ideal[k], result.objectives[k]) for k in range(2)), pop_max
        )
        weight = lambdas[target_i]
        recipient_scalar = context.scalar(recipient, weight)
        whole = less(context.scalar(donor, weight), recipient_scalar, TOL.scalar)
        block = less(context.scalar(mixed, weight), recipient_scalar, TOL.scalar)
        start = perf_counter()
        archive = Archive()
        archive.members = list(final_archive)
        enters = archive.consider(mixed)
        archive_seconds = perf_counter() - start
        pre_delta = hypervolume(prefront + [result.objectives], ref) - pre_hv
        final_delta = hypervolume(final_front + [result.objectives], ref) - final_hv
        union_pre.append(result.objectives)
        union_final.append(result.objectives)
        old_ledger = original_ledgers[target_i]
        naive = sum(old_ledger) - old_ledger[r] + original_ledgers[donor_i][r]
        other_correction = sum(
            regional_bills[q] - old_ledger[q] for q in range(len(problem.regions)) if q != r
        )
        row = dict(
            donor=donor_i,
            recipient=target_i,
            region=r,
            whole_improves=whole,
            block_improves=block,
            rescued=block and not whole,
            enters_final_archive=enters,
            pre_hv_gain=pre_delta / pre_hv,
            final_hv_gain=final_delta / final_hv,
            horizon_changed=not close(
                decoded.horizon(problem), recipient.schedule.horizon(problem), TOL.time
            ),
            naive_bill_error=naive - result.bill,
            other_regions_bill_correction=other_correction,
            max_start_error=err,
            construction_seconds=construction_seconds,
            decode_seconds=decode_seconds,
            exact_seconds=exact_seconds,
            archive_seconds=archive_seconds,
        )
        rows.append(row)
        if row["rescued"] and len(examples) < 3:
            examples.append(
                dict(
                    **row,
                    donor_objectives=donor.evaluation.objectives,
                    recipient_objectives=recipient.evaluation.objectives,
                    mixed=asdict(mixed),
                    context=asdict(context),
                    weight=weight,
                )
            )
    summary = dict(
        key=key,
        n=len(problem.jobs),
        regions=len(problem.regions),
        seed=config["seed"],
        source=config["source"]["commit"],
        input_sha256=inputs,
        power_model=asdict(problem.power_model),
        sample_seed=seed,
        population=len(population),
        unique_phenotypes=len(checked),
        nonreproducible_slots=sum(not v for v in valid),
        max_parent_start_error=max_parent_start_error,
        directed_pairs=pair_count,
        pair_regions=pair_count * len(problem.regions),
        equal_nonempty_sets=same_sets,
        eligible_triples=len(eligible),
        sampled=len(rows),
        eligibility_seconds=eligibility_seconds,
        snapshot_elapsed=snapshots[-1]["elapsed"],
        context_policy="candidate-aware ideal from population + preterminal/final fronts; population maximum",
        hv_reference=ref,
        hv_anchor_min=ideal,
        pre_hv=pre_hv,
        final_hv=final_hv,
        union_pre_gain=(hypervolume(union_pre, ref) - pre_hv) / pre_hv,
        union_final_gain=(hypervolume(union_final, ref) - final_hv) / final_hv,
        elapsed_seconds=perf_counter() - begin,
    )
    _write(out / "runs" / (key + ".json"), dict(summary=summary, rows=rows, examples=examples))
    return summary


def main() -> None:
    """Replay the specified existing population files with bounded CPU workers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    runs = sorted(
        p.parent
        for p in args.results.glob(f"formal_panel_shard*/runs/*/{ARM}/native_population.json.gz")
    )
    if args.limit:
        runs = runs[: args.limit]
    assert runs
    source = _read(runs[0] / "config.json")
    assert isinstance(source, dict)
    repo = Path(__file__).resolve().parents[1]
    for name, expected in source["source"]["files"].items():
        canonical = (repo / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(canonical).hexdigest() == expected, (
            "Source differs from recovered run beyond Windows line endings",
            name,
        )
    assert source["source"]["commit"] == "3c3da11a601f0cc42fea9a1b0f75ff840d7a548c"
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "runs").mkdir()
    protocol = dict(
        level="offline development diagnostic",
        arm=ARM,
        runs=len(runs),
        samples_per_run=args.samples,
        master_seed=args.seed,
        sampling="uniform without replacement over eligible directed neighbor-pair-region triples",
        inferential_unit="instance; final-population proxy, not online child frequency",
        script_sha256=_sha(Path(__file__)),
        baseline_commit="3c3da11a601f0cc42fea9a1b0f75ff840d7a548c",
        source_validation="Package hashes verified after CRLF-to-LF normalization",
        results=str(args.results),
        instances=str(args.instances),
    )
    _write(args.out / "PROTOCOL.json", protocol)
    tasks = [(str(p), str(args.instances), str(args.out), args.samples, args.seed) for p in runs]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_trial_run, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            summary = future.result()
            print(
                json.dumps(
                    dict(
                        done=i,
                        total=len(runs),
                        key=summary["key"],
                        eligible=summary["eligible_triples"],
                        sampled=summary["sampled"],
                        seconds=summary["elapsed_seconds"],
                    )
                ),
                flush=True,
            )
    reports = [_read(p) for p in sorted((args.out / "runs").glob("*.json"))]
    groups = {}
    for size in sorted({r["summary"]["n"] for r in reports}):
        group = [r for r in reports if r["summary"]["n"] == size]
        rows = [row for r in group for row in r["rows"]]

        def rate(report: dict, field: str) -> float:
            return (
                sum(bool(x[field]) for x in report["rows"]) / len(report["rows"])
                if report["rows"]
                else 0.0
            )

        groups[str(size)] = dict(
            instances=len(group),
            sampled=len(rows),
            eligible_rate_median=statistics.median(
                r["summary"]["eligible_triples"] / r["summary"]["pair_regions"] for r in group
            ),
            rescued_rate_mean=statistics.mean(rate(r, "rescued") for r in group),
            rescued_instances=sum(any(row["rescued"] for row in r["rows"]) for r in group),
            block_improvement_rate_mean=statistics.mean(rate(r, "block_improves") for r in group),
            pre_hv_positive_instances=sum(r["summary"]["union_pre_gain"] > 1e-9 for r in group),
            final_hv_positive_instances=sum(r["summary"]["union_final_gain"] > 1e-9 for r in group),
            median_union_pre_gain=statistics.median(r["summary"]["union_pre_gain"] for r in group),
            median_union_final_gain=statistics.median(
                r["summary"]["union_final_gain"] for r in group
            ),
            horizon_changed=sum(row["horizon_changed"] for row in rows),
            naive_bill_wrong=sum(not close(row["naive_bill_error"], 0, TOL.cost) for row in rows),
            other_region_corrections=sum(
                not close(row["other_regions_bill_correction"], 0, TOL.cost) for row in rows
            ),
            cost_median_ms={
                k: 1000 * statistics.median(row[k] for row in rows) if rows else None
                for k in (
                    "construction_seconds",
                    "decode_seconds",
                    "exact_seconds",
                    "archive_seconds",
                )
            },
        )
    result = dict(
        status="completed_offline_proxy",
        by_size=groups,
        nonreproducible_slots=sum(r["summary"]["nonreproducible_slots"] for r in reports),
        samples=sum(len(r["rows"]) for r in reports),
    )
    _write(args.out / "SUMMARY.json", result)
    records = [
        dict(key=r["summary"]["key"], n=r["summary"]["n"], **row)
        for r in reports
        for row in r["rows"]
    ]
    if records:
        with (args.out / "trials.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    _write(
        args.out / "manifest.json",
        [
            {"path": str(p.relative_to(args.out)), "sha256": _sha(p), "bytes": p.stat().st_size}
            for p in sorted(args.out.rglob("*"))
            if p.is_file()
        ],
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
