"""Fixed real-input candidate/resource/RNG equivalence and cold/cached timing."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from random import Random
from time import perf_counter

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.common_refine import ReusedCommonPolish, _CalendarSearch
from geo_llm_scheduler.lsv2.flowsafe import FlowSafeBillMoves
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.lsv2.group_polish import ConcurrentRightShiftPolish, _Search
from geo_llm_scheduler.lsv2.polish_selection import PolishSelector, candidate_pool
from geo_llm_scheduler.moead.core import NormalizationContext, weights


def read(path: Path) -> dict | list:
    """Read a sealed local JSON or compressed result without modifying it."""
    with (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(encoding="utf-8")
    ) as source:
        return json.load(source)


def main() -> None:
    """Use evenly spaced existing C1 inputs, not a post-hoc successful subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-instance", type=int, default=12)
    parser.add_argument("--neighborhood-audit", action="store_true")
    args = parser.parse_args()
    manifest = read(args.historical / "manifest.json")
    rows = []
    resource_checks = 0
    exact_checks = 0
    selector_timings = []
    group_timings = []
    neighborhoods = []
    for job in manifest["jobs"]:
        if job["arm"] != "C1_COMMON":
            continue
        key = job["runs"][0]["key"]
        spec = read(args.historical / "specs" / (key + ".json"))
        problem = load_instance(args.historical / spec["input"])
        run = args.historical / "results" / job["id"] / ("00_" + key)
        data = [r for r in read(run / "rightshift_steps.json.gz") if "diagnostics" in r]
        indexes = sorted(
            {
                round(i * (len(data) - 1) / max(1, args.per_instance - 1))
                for i in range(args.per_instance)
            }
        )
        candidates = []
        for index in indexes:
            v = data[index]["input_schedule"]
            g = Genotype(tuple(v["ms"]), tuple(v["os"]))
            s = Schedule(tuple(v["starts"]))
            c = Candidate(g, s, evaluate(problem, g, s))
            assert c.evaluation.feasible
            candidates.append(c)
            prepared = FlowSafeBillMoves(right_only=True)._prepare(problem, c)
            old, cal = _Search(problem, c, prepared), _CalendarSearch(problem, c, prepared, False)
            for group in old.groups(score_order=False):
                points = [0.0, *old._shifts(group)]
                for delta in points + [(x + y) / 2 for x, y in zip(points, points[1:])]:
                    assert old._fits(group, delta) == cal._fits(group, delta), (
                        key,
                        index,
                        group,
                        delta,
                    )
                    resource_checks += 1
            for repeat in range(3):
                operators = {
                    "old": ReusedCommonPolish(),
                    "calendar": ReusedCommonPolish(resource_calendar=True),
                }
                names = ("old", "calendar") if repeat % 2 == 0 else ("calendar", "old")
                for cached in (False, True):
                    batches = {}
                    states = {}
                    for name in names:
                        rng = Random(16901 + repeat)
                        tick = perf_counter()
                        batch = operators[name].propose(problem, c, 3, Config(), rng)
                        seconds = perf_counter() - tick
                        batches[name] = batch
                        states[name] = rng.getstate()
                        rows.append(
                            dict(
                                key=key,
                                index=index,
                                repeat=repeat,
                                cached=cached,
                                method=name,
                                seconds=seconds,
                                diagnostics=batch.diagnostics,
                            )
                        )
                    assert batches["old"].proposals == batches["calendar"].proposals
                    assert states["old"] == states["calendar"]
                    assert (
                        batches["old"].diagnostics["incremental_deltas"]
                        == batches["calendar"].diagnostics["incremental_deltas"]
                    )
                    if repeat == 0 and not cached:
                        for p in batches["calendar"].proposals:
                            ev = evaluate(problem, p.genotype, p.schedule)
                            assert (
                                ev.feasible
                                and ev.flow == c.evaluation.flow
                                and ev.bill < c.evaluation.bill
                            )
                            exact_checks += 1
                contexts = {
                    "old": _Search(problem, c, prepared),
                    "calendar": _CalendarSearch(problem, c, prepared, False),
                }
                for name in names:
                    ctx = contexts[name]
                    groups = ctx.groups(score_order=False)
                    tick = perf_counter()
                    for group in groups:
                        ctx.common(group)
                    group_timings.append(
                        dict(
                            key=key,
                            index=index,
                            repeat=repeat,
                            method=name,
                            seconds=perf_counter() - tick,
                        )
                    )
                assert contexts["old"].pool == contexts["calendar"].pool
            if args.neighborhood_audit:
                for name, operator in (
                    ("common", ReusedCommonPolish()),
                    ("peak", ReusedCommonPolish(peak_switch=True)),
                    ("coordinated", ConcurrentRightShiftPolish(coordinated=True)),
                ):
                    tick = perf_counter()
                    batch = operator.propose(problem, c, 3, Config(), Random(16901))
                    seconds = perf_counter() - tick
                    bills = []
                    best_evaluation = c.evaluation
                    best_proposal = None
                    for proposal in batch.proposals:
                        ev = evaluate(problem, proposal.genotype, proposal.schedule)
                        assert ev.feasible and ev.flow == c.evaluation.flow
                        bills.append(ev.bill)
                        if ev.bill < best_evaluation.bill:
                            best_evaluation, best_proposal = ev, proposal
                        exact_checks += 1
                    neighborhoods.append(
                        dict(
                            key=key,
                            index=index,
                            method=name,
                            seconds=seconds,
                            input_bill=c.evaluation.bill,
                            best_bill=min([c.evaluation.bill, *bills]),
                            proposals=len(bills),
                            tou_saving=c.evaluation.tou - best_evaluation.tou,
                            demand_saving=sum(c.evaluation.demand) - sum(best_evaluation.demand),
                            moves=[]
                            if best_proposal is None
                            else [
                                dict(
                                    operation=o,
                                    machine=g.ms[o],
                                    old=s.starts[o],
                                    new=best_proposal.schedule.starts[o],
                                )
                                for o in best_proposal.selected
                            ],
                        )
                    )
        frame = FrozenFrame.of([c.evaluation.objectives for c in candidates])
        context = NormalizationContext(frame.ideal, frame.nadir)
        for full in (True, False):
            selector = PolishSelector()
            rng = Random(3)
            tick = perf_counter()
            selected = [
                selector.choose(
                    "pool_random",
                    candidates,
                    candidates[0],
                    frame,
                    context,
                    weights(100),
                    rng,
                    diagnose_directions=full,
                ).candidate
                for _ in range(100)
            ]
            duration = perf_counter() - tick
            if full:
                expected = selected
            else:
                assert selected == expected
            selector_timings.append(
                dict(
                    key=key,
                    full_directions=full,
                    seconds=duration,
                    calls=100,
                    raw_input_records=len(candidates),
                    pool_size=len(candidate_pool(candidates, candidates[0])),
                )
            )
        atomic_json(args.out, dict(accepted=False, rows=rows))
        print(key, flush=True)
    totals = {
        name: {
            kind: sum(r["seconds"] for r in rows if r["method"] == name and r["cached"] == cached)
            for kind, cached in (("cold", False), ("cached", True))
        }
        for name in ("old", "calendar")
    }
    actual = [r for r in rows if r["method"] == "calendar" and not r["cached"] and r["repeat"] == 0]
    diagnostics = {
        k: sum(r["diagnostics"].get(k, 0) for r in actual)
        for k in (
            "candidate_positions",
            "resource_rejected",
            "calendar_rejected",
            "calendar_pairs",
            "reference_resource_checks",
        )
    }
    result = dict(
        accepted=True,
        inputs=len(actual),
        resource_checks=resource_checks,
        fresh_exact=exact_checks,
        seconds=totals,
        resource=diagnostics,
        selection=selector_timings,
        group_seconds={
            name: sum(r["seconds"] for r in group_timings if r["method"] == name)
            for name in ("old", "calendar")
        },
        group_rows=group_timings,
        neighborhoods=neighborhoods,
        rows=rows,
    )
    atomic_json(args.out, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("rows", "selection", "group_rows", "neighborhoods")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
