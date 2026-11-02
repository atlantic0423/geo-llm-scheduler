"""Observable, unit-defined LLM snapshots; no held-out or end-of-run reference data."""

import hashlib
from statistics import mean, median

from geo_llm_scheduler.archive.pareto import dominates
from geo_llm_scheduler.diagnostics.workload import imbalance, region_loads
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.evaluation.exact import tied_peaks
from geo_llm_scheduler.experiments.llm_focus import job_scores
from geo_llm_scheduler.moead.core import NormalizationContext, weights
from geo_llm_scheduler.rl.state import PREFERENCE_LABELS, associate, preference
from geo_llm_scheduler.utils.numeric import TOL, close, identity


def _schedule_id(candidate: Candidate) -> str:
    return hashlib.sha256(repr(identity(candidate)).encode()).hexdigest()[:20]


def unique_objectives(members: list[Candidate]) -> list[Candidate]:
    """Deduplicate with existing unit tolerances, preserving deterministic input order."""
    result: list[Candidate] = []
    for candidate in members:
        if not any(
            close(candidate.evaluation.flow, c.evaluation.flow, TOL.time)
            and close(candidate.evaluation.bill, c.evaluation.bill, TOL.cost)
            for c in result
        ):
            result.append(candidate)
    return result


def objective_summary(members: list[Candidate]) -> dict:
    """Describe actual stored solutions, not an inferred true Pareto front."""
    return (
        dict(
            members=len(members),
            unique_objectives=len(unique_objectives(members)),
            flow_range_s=[
                min(c.evaluation.flow for c in members),
                max(c.evaluation.flow for c in members),
            ],
            bill_range_cny=[
                min(c.evaluation.bill for c in members),
                max(c.evaluation.bill for c in members),
            ],
        )
        if members
        else dict(members=0, unique_objectives=0, flow_range_s=None, bill_range_cny=None)
    )


def static_problem(problem: ProblemInstance) -> dict:
    """Expose true topology, homogeneous profile distributions and global tariffs."""

    def distribution(values: list[float]) -> dict:
        ordered = sorted(values)
        return (
            dict(
                min=ordered[0],
                median=median(ordered),
                p90=ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
                max=ordered[-1],
            )
            if ordered
            else dict(min=None, median=None, p90=None, max=None)
        )

    return dict(
        jobs=len(problem.jobs),
        geographic_regions=len(problem.regions),
        units=dict(
            time="s",
            bill="CNY",
            power="kW",
            compute="normalized share, capacity1",
            vram="effective schedulable GB",
            tou_rate="CNY/kWh",
            demand_rate="CNY/kW",
        ),
        job_distributions={
            "release_s": distribution([j.release for j in problem.jobs]),
            "prefill_duration_s": distribution([j.prefill.duration for j in problem.jobs]),
            "decode_duration_s": distribution([j.decode.duration for j in problem.jobs]),
            "compute": distribution(
                [problem.profile(o).compute for o in range(problem.operation_count)]
            ),
            "vram_gb": distribution(
                [problem.profile(o).vram for o in range(problem.operation_count)]
            ),
            "kv_if_separate_s": distribution([j.kv_delay for j in problem.jobs]),
        },
        regions=[
            dict(
                geographic_region_id=r,
                name=region.name,
                demand_rate_cny_per_kw=region.demand_rate,
                tou_segments=[
                    dict(start_s=t.start, end_s=t.end, price_cny_per_kwh=t.price)
                    for t in region.tariffs
                ],
                instances=[
                    dict(
                        id=m,
                        name=s.name,
                        compute_capacity=1,
                        vram_gb=s.vram,
                        idle_kw=s.idle_kw,
                        active_kw=s.active_kw,
                        phases=s.phases,
                    )
                    for m, s in enumerate(problem.instances)
                    if s.region == r
                ],
            )
            for r, region in enumerate(problem.regions)
        ],
    )


def preference_summaries(
    problem: ProblemInstance,
    population: list[Candidate],
    archive: list[Candidate],
    context: NormalizationContext,
) -> list[dict]:
    """Separate slot means from unique-ND angular coverage in one snapshot context."""
    lambdas = weights(len(population))
    unique = unique_objectives(population)
    nd = [c for c in unique if not any(dominates(other, c) for other in unique)]
    occupied = {associate(context.normalize(c), lambdas) for c in nd}
    archive_occupied = {
        associate(context.normalize(c), lambdas) for c in unique_objectives(archive)
    }
    summaries = []
    n = len(problem.jobs)
    for group_id, group in enumerate(PREFERENCE_LABELS):
        ids = [i for i in range(len(population)) if preference(i, len(population)) == group_id]
        members = [population[i] for i in ids]
        if not members:
            summaries.append(
                dict(group=group, direction_ids=ids, member_count=0, representative_points=[])
            )
            continue
        unique_group = unique_objectives(members)
        first = min(unique_group, key=lambda c: (c.evaluation.flow, c.evaluation.bill, identity(c)))
        last = min(unique_group, key=lambda c: (c.evaluation.bill, c.evaluation.flow, identity(c)))
        center = tuple(mean(context.normalize(c)[k] for c in unique_group) for k in (0, 1))
        middle = min(
            unique_group,
            key=lambda c: (
                sum((a - b) ** 2 for a, b in zip(context.normalize(c), center)),
                identity(c),
            ),
        )
        representatives = list(dict.fromkeys((first, middle, last)))
        loads = region_loads(problem, middle.genotype)
        regions = []
        for r in range(len(problem.regions)):
            values = middle.evaluation.windows[r]
            peaks = tied_peaks(values)
            regions.append(
                dict(
                    geographic_region_id=r,
                    workload_proxy_s=loads[r],
                    exact_demand_bill_cny=middle.evaluation.demand[r],
                    exact_peak_kw=max(values, default=0),
                    peak_windows=dict(
                        total_count=len(peaks),
                        sent_count=min(8, len(peaks)),
                        truncated=len(peaks) > 8,
                        items=[dict(start_s=900 * b, end_s=900 * (b + 1)) for b in peaks[:8]],
                    ),
                )
            )
        summaries.append(
            dict(
                group=group,
                direction_ids=ids,
                flow_weight_range=[
                    min(lambdas[i][0] for i in ids),
                    max(lambdas[i][0] for i in ids),
                ],
                member_count=len(members),
                representative_points=[
                    dict(
                        schedule_id=_schedule_id(c),
                        source="owned_population_slots",
                        total_flow_s=c.evaluation.flow,
                        bill_cny=c.evaluation.bill,
                    )
                    for c in representatives
                ],
                population_occupied_directions=len(occupied.intersection(ids)),
                population_total_directions=len(ids),
                archive_occupied_directions=len(archive_occupied.intersection(ids)),
                mean_job_flow_s=mean(c.evaluation.flow / n for c in members),
                mean_job_flow_wait_s=mean(
                    sum(job_scores(problem, c, "FLOW_WAIT")) / n for c in members
                ),
                mean_resource_wait_s=mean(sum(c.schedule.resource_wait) / n for c in members),
                mean_kv_delay_s=mean(sum(job_scores(problem, c, "KV_DELAY")) / n for c in members),
                mean_total_bill_cny=mean(c.evaluation.bill for c in members),
                mean_tou_bill_cny=mean(c.evaluation.tou for c in members),
                mean_demand_bill_cny=mean(sum(c.evaluation.demand) for c in members),
                region_load_imbalance_proxy=mean(
                    imbalance(region_loads(problem, c.genotype)) for c in members
                ),
                diagnostic_schedule_id=_schedule_id(middle),
                region_diagnostics=regions,
            )
        )
    return summaries
