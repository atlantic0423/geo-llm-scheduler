"""State severities implementing docs/specifications/state_severity.md exactly."""

from geo_llm_scheduler.diagnostics.workload import imbalance, region_loads
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.scheduling.resources import intervals
from geo_llm_scheduler.scheduling.timing import packing_moves
from geo_llm_scheduler.utils.numeric import EPS_RATIO


def severities(problem: ProblemInstance, candidate: Candidate) -> tuple[float, ...]:
    """Resource, KV, Region, TOU, Demand, Compressible in their fixed state order."""
    if not problem.jobs:
        return (0.0,) * 6
    g, s = candidate.genotype, candidate.schedule
    total = sum(problem.profile(o).duration for o in range(problem.operation_count))
    resource = min(1, sum(s.resource_wait) / (total + EPS_RATIO))
    kv = sum(j.kv_delay * (g.ms[2 * i] != g.ms[2 * i + 1]) for i, j in enumerate(problem.jobs)) / (
        candidate.evaluation.flow + EPS_RATIO
    )
    region = imbalance(region_loads(problem, g))
    weighted = energy = 0.0
    end = s.horizon(problem)
    for m, server in enumerate(problem.instances):
        tariffs = [t for t in problem.regions[server.region].tariffs if t.start < end and t.end > 0]
        prices = [t.price for t in tariffs]
        low, high = min(prices, default=0), max(prices, default=0)
        ops = intervals(problem, g, s, m)
        events = sorted({t for _, a, b in ops for t in (a, b)})
        for a, b in zip(events, events[1:]):
            if not any(start <= a < finish for _, start, finish in ops):
                continue
            energy += server.active_kw * (b - a)
            if high > low:
                for tariff in tariffs:
                    overlap = max(0, min(b, tariff.end) - max(a, tariff.start))
                    weighted += server.active_kw * overlap * (tariff.price - low) / (high - low)
    tou = weighted / energy if energy > 0 else 0.0
    rates = sum(r.demand_rate for r in problem.regions)
    demand = 0.0
    for r, averages in zip(problem.regions, candidate.evaluation.windows):
        peak = max(averages, default=0)
        if peak > 0:
            demand += r.demand_rate * (peak - sum(averages) / len(averages)) / peak
    demand = demand / rates if rates > 0 else 0.0
    gains = [0.0] * problem.operation_count
    for o, _, gain, _ in packing_moves(problem, g, s):
        gains[o] = max(gains[o], gain)
    compress = sum(gains) / (total + EPS_RATIO)
    return resource, kv, region, tou, demand, compress
