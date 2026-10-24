"""Fail-fast input and genotype validation before scheduling."""

import math
from collections import Counter

from geo_llm_scheduler.domain.models import Genotype, ProblemInstance


def validate_problem(problem: ProblemInstance) -> None:
    """Reject malformed, non-finite, unsupported or unschedulable instances."""
    if not problem.regions or not problem.instances:
        raise ValueError("At least one region and server required")
    for collection in (problem.jobs, problem.regions, problem.instances):
        if len({x.name for x in collection}) != len(collection):
            raise ValueError("Duplicate identifier")
    for m in problem.instances:
        if type(m.region) is not int or not 0 <= m.region < len(problem.regions):
            raise ValueError("Unknown server region")
        if not m.phases or any(s not in (0, 1) for s in m.phases):
            raise ValueError("Invalid phase eligibility")
        if not all(math.isfinite(x) for x in (m.vram, m.idle_kw, m.active_kw)):
            raise ValueError("Non-finite server profile")
        if m.vram <= 0 or m.idle_kw < 0 or m.active_kw < m.idle_kw:
            raise ValueError("Invalid server capacity/power")
    if any(not any(m.region == r for m in problem.instances) for r in range(len(problem.regions))):
        raise ValueError("Each region requires positive compute and VRAM capacity")
    for r in problem.regions:
        if not math.isfinite(r.demand_rate) or r.demand_rate < 0:
            raise ValueError("Invalid demand rate")
        if not r.tariffs or r.tariffs[0].start != 0:
            raise ValueError("Tariff must begin at zero")
        end = 0.0
        for t in r.tariffs:
            if not all(math.isfinite(v) for v in (t.start, t.end, t.price)):
                raise ValueError("Non-finite tariff")
            if t.start != end or t.end <= t.start or t.price < 0:
                raise ValueError("Tariff must be contiguous, positive length and nonnegative")
            end = t.end
    for i, job in enumerate(problem.jobs):
        if not all(math.isfinite(v) and v >= 0 for v in (job.release, job.kv_delay)):
            raise ValueError("Invalid release/KV delay")
        for p in (job.prefill, job.decode):
            if not all(math.isfinite(v) for v in (p.duration, p.compute, p.vram)):
                raise ValueError("Non-finite operation profile")
            if p.duration <= 0 or not 0 <= p.compute <= 1 or p.vram < 0:
                raise ValueError("Invalid operation profile")
        if not any(
            problem.eligible(2 * i, r) and problem.eligible(2 * i + 1, r)
            for r in range(len(problem.regions))
        ):
            raise ValueError("No same-region eligible P/D path")


def validate_genotype(problem: ProblemInstance, genotype: Genotype) -> None:
    """Validate repeated-job encoding and MS sole-source assignment."""
    n = len(problem.jobs)
    if any(type(i) is not int for i in (*genotype.ms, *genotype.os)):
        raise ValueError("MS/OS indices must be integers")
    if len(genotype.ms) != 2 * n or Counter(genotype.os) != Counter({i: 2 for i in range(n)}):
        raise ValueError("MS/OS dimensions or occurrences invalid")
    for o, m in enumerate(genotype.ms):
        if m not in problem.eligible(o):
            raise ValueError("Ineligible assignment")
    for i in range(n):
        if (
            problem.instances[genotype.ms[2 * i]].region
            != problem.instances[genotype.ms[2 * i + 1]].region
        ):
            raise ValueError("P/D must share region")
