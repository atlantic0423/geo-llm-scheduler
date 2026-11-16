"""Offline external-polish repetition and first-move opportunity diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from time import perf_counter

from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.lsv2.billing import PowerProfile, merge
from geo_llm_scheduler.lsv2.flowsafe import prefill_times
from geo_llm_scheduler.lsv2.polish_selection import physical_fingerprint
from geo_llm_scheduler.scheduling.resources import _fits_intervals
from geo_llm_scheduler.scheduling.timing import legal_bounds
from geo_llm_scheduler.utils.numeric import TOL, close, less


def repetition_summary(rows: Sequence[dict], run_seconds: float | None = None) -> dict:
    """Count same-scope re-calls after a completed empty call within one run.

    Deadline/region skips are not calls, interrupted evaluation is not failure,
    and a failed partial scan says nothing about an untested full target pool.
    Historical failures do not influence the running selector.
    """
    failed: set[tuple[str, str]] = set()
    unique: set[str] = set()
    calls = repeats = uninstrumented = 0
    repeated_seconds = total_seconds = 0.0
    statuses: Counter = Counter()
    for row in rows:
        statuses[row["status"]] += 1
        if "attempts" not in row:
            continue
        calls += 1
        total_seconds += row["seconds"]
        if "physical_fingerprint" not in row:
            uninstrumented += 1
            continue
        fingerprint = row["physical_fingerprint"]
        key = fingerprint, row["target_scope"]
        unique.add(fingerprint)
        if key in failed:
            repeats += 1
            repeated_seconds += row["seconds"]
        if row.get("complete_empty", False):
            failed.add(key)
    return dict(
        calls=calls,
        uninstrumented_calls=uninstrumented,
        complete_input_diagnostics=uninstrumented == 0,
        unique_inputs=len(unique),
        after_empty_recalls=repeats,
        after_empty_recall_seconds=repeated_seconds,
        called_polish_seconds=total_seconds,
        repeat_fraction_of_called_polish=repeated_seconds / total_seconds if total_seconds else 0,
        repeat_fraction_of_run=(
            repeated_seconds / run_seconds if run_seconds is not None and run_seconds > 0 else None
        ),
        statuses=dict(statuses),
        note="Instance/run scoped; measured re-call cost, not a prediction of cache savings",
    )


def inspect_start(problem: ProblemInstance, row: dict) -> dict:
    """Exact-audit all first-move critical times from one recorded frozen input.

    This auxiliary scan is deliberately offline. It distinguishes no right slack,
    no feasible later time, unchanged activity unions, and TOU savings offset by
    Demand. It also tests tasks outside the ordinary partial pool. It does not
    claim to reproduce randomized greedy multi-move proposal trajectories.
    """
    started = perf_counter()
    snapshot = row["input_schedule"]
    genotype = Genotype(tuple(snapshot["ms"]), tuple(snapshot["os"]))
    schedule = Schedule(tuple(snapshot["starts"]))
    base = evaluate(problem, genotype, schedule)
    candidate = Candidate(genotype, schedule, base)
    if not base.feasible or physical_fingerprint(candidate) != row["physical_fingerprint"]:
        raise ValueError("Invalid right-shift input snapshot or fingerprint")
    if not (
        close(base.flow, row["start_objectives"][0], TOL.time)
        and close(base.bill, row["start_objectives"][1], TOL.cost)
    ):
        raise ValueError("Snapshot objectives disagree with the supplied problem")
    profile = PowerProfile(problem, genotype.ms, schedule.starts)
    bounds = {}
    for o in range(0, problem.operation_count, 2):
        lo, hi = legal_bounds(problem, genotype, schedule, o)
        lo = max(lo, schedule.starts[o])
        if less(lo, hi, TOL.time):
            bounds[o] = lo, hi
    ranked = sorted(bounds, key=lambda o: profile.delta_for({genotype.ms[o]: {o: None}}))
    partial = set(ranked[: max(20, math.ceil(0.2 * len(problem.jobs)))])
    attempts = []
    for o, (lo, hi) in bounds.items():
        m = genotype.ms[o]
        others = [(q, a, b) for q, (a, b) in profile.items[m].items() if q != o]
        for t in prefill_times(problem, m, o, lo, hi, others):
            if close(t, schedule.starts[o], TOL.time):
                continue
            if not _fits_intervals(problem, m, o, t, others):
                continue
            starts = list(schedule.starts)
            starts[o] = t
            result = evaluate(problem, genotype, Schedule(tuple(starts)))
            if not result.feasible:
                raise AssertionError("Critical-time feasibility disagrees with exact evaluator")
            tou = result.tou - base.tou
            demand = sum(result.demand) - sum(base.demand)
            bill = result.bill - base.bill
            delta = profile.delta_for({m: {o: (t, t + problem.profile(o).duration)}})
            if not close(delta, bill, TOL.cost):
                raise AssertionError("Incremental and exact bill delta disagree")
            union = merge([(a, b) for _, a, b in others] + [(t, t + problem.profile(o).duration)])
            attempts.append(
                dict(
                    operation=o,
                    start=t,
                    in_partial_pool=o in partial,
                    union_changed=union != profile.unions[m],
                    delta_tou=tou,
                    delta_demand=demand,
                    delta_bill=bill,
                    improves=less(result.bill, base.bill, TOL.cost),
                    tou_offset_by_demand=less(result.tou, base.tou, TOL.cost)
                    and demand > 0
                    and not less(result.bill, base.bill, TOL.cost),
                )
            )
    improving = [a for a in attempts if a["improves"]]
    return dict(
        physical_fingerprint=row["physical_fingerprint"],
        eligible=len(bounds),
        partial_targets=len(partial),
        feasible_later_times=len(attempts),
        improving_later_times=len(improving),
        improves_outside_partial=any(not a["in_partial_pool"] for a in improving),
        union_unchanged=sum(not a["union_changed"] for a in attempts),
        tou_offset_by_demand=sum(a["tou_offset_by_demand"] for a in attempts),
        first_move_class=(
            "no_right_slack"
            if not bounds
            else "no_feasible_later_time"
            if not attempts
            else "no_strict_bill_improvement"
            if not improving
            else "improving_first_move"
        ),
        attempts=attempts,
        offline_seconds=perf_counter() - started,
        scope="All eligible tasks, first-move critical times; not the complete greedy search",
    )
