"""A8 bounded singleton/support-group repair; no beam, helpers or backtracking."""

import math
import random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.evaluation.exact import power_segments, regional_windows, violations
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.scheduling.resources import fits, intervals
from geo_llm_scheduler.utils.numeric import TOL


def support_segments(
    problem: ProblemInstance,
    genotype: Genotype,
    schedule: Schedule,
    region: int,
    peaks: tuple[int, ...],
) -> list[tuple[int, frozenset[int], float]]:
    """Original peak elementary active sets and removable average-power increments."""
    result = []
    for m, server in enumerate(problem.instances):
        if server.region != region:
            continue
        ops = intervals(problem, genotype, schedule, m)
        for b in peaks:
            left, right = 900 * b, 900 * (b + 1)
            points = sorted(
                {left, right} | {t for _, a, e in ops for t in (a, e) if left < t < right}
            )
            for a, e in zip(points, points[1:]):
                active = frozenset(o for o, s, c in ops if s <= a < c)
                if active:
                    result.append((b, active, (server.active_kw - server.idle_kw) * (e - a) / 900))
    return result


def removal_potential(
    segments: list[tuple[int, frozenset[int], float]], members: frozenset[int], peak: int
) -> float:
    """Original-support area that disappears only when every contributor is removed."""
    return sum(value for b, active, value in segments if b == peak and active <= members)


def coalition(
    segments: list[tuple[int, frozenset[int], float]],
    peaks: tuple[int, ...],
    cap: int,
    rng: random.Random,
) -> frozenset[int]:
    """Accumulate positive marginal groups; tiny contributions may jointly pass epsilon."""
    chosen: frozenset[int] = frozenset()
    groups = list(dict.fromkeys(active for _, active, _ in segments))
    rng.shuffle(groups)
    while True:
        deficient = [b for b in peaks if removal_potential(segments, chosen, b) <= TOL.power]
        if not deficient:
            return chosen
        options = []
        for group in groups:
            extended = chosen | group
            if len(extended) > cap:
                continue
            gain = sum(
                max(
                    0,
                    removal_potential(segments, extended, b)
                    - removal_potential(segments, chosen, b),
                )
                for b in deficient
            )
            if gain > 0:
                options.append((gain, group))
        if not options:
            return chosen
        # Seeded shuffle determines ties; no dynamic support graph.
        chosen |= max(options, key=lambda item: item[0])[1]


def repair_bounds(
    problem: ProblemInstance,
    genotype: Genotype,
    starts: list[float],
    operation: int,
    horizon: float,
) -> tuple[float, float]:
    """Precedence bounds during whole-set removal, including an uninserted Decode."""
    job = problem.jobs[operation // 2]
    delay = job.kv_delay * (
        genotype.ms[2 * (operation // 2)] != genotype.ms[2 * (operation // 2) + 1]
    )
    duration = problem.profile(operation).duration
    if operation % 2:
        if starts[operation - 1] < 0:
            raise ValueError("Decode cannot be reinserted before Prefill")
        return starts[operation - 1] + job.prefill.duration + delay, horizon - duration
    upper = (
        starts[operation + 1] - duration - delay
        if starts[operation + 1] >= 0
        else horizon - duration - delay - job.decode.duration
    )
    return job.release, upper


def repair_times(
    problem: ProblemInstance,
    genotype: Genotype,
    original: Schedule,
    partial: Schedule,
    members: frozenset[int],
    operation: int,
    region: int,
    bounds: tuple[float, float],
) -> list[float]:
    """Finite legal boundaries/events plus adjacent critical-start midpoints."""
    lower, upper = bounds
    duration = problem.profile(operation).duration
    if upper < lower:
        return []
    events = {float(b * 900) for b in range(math.ceil((upper + duration) / 900) + 1)}
    events |= {
        t
        for tariff in problem.regions[region].tariffs
        for t in (tariff.start, tariff.end)
        if lower - duration <= t <= upper + duration
    }
    events |= {
        t
        for _, a, b in intervals(problem, genotype, partial, genotype.ms[operation])
        for t in (a, b)
    }
    events |= {t for o in members for t in (original.starts[o], original.end(problem, o))}
    values = sorted(
        {lower, upper, original.starts[operation]} | events | {t - duration for t in events}
    )
    values = [t for t in values if lower <= t <= upper]
    return sorted(set(values) | {(a + b) / 2 for a, b in zip(values, values[1:])})


def repair(
    problem: ProblemInstance,
    x: Candidate,
    members: frozenset[int],
    region: int,
    positions: int,
    rng: random.Random,
) -> Schedule | None:
    """One greedy path; first feasible position; any failure rolls back whole recipe."""
    g = x.genotype
    starts = list(x.schedule.starts)
    for o in members:
        starts[o] = -1.0
    pending = set(members)
    while pending:
        ready = sorted(o for o in pending if o % 2 == 0 or o - 1 not in pending)
        bounds = {
            o: repair_bounds(problem, g, starts, o, x.schedule.horizon(problem)) for o in ready
        }
        rng.shuffle(ready)
        o = min(ready, key=lambda k: bounds[k][1] - bounds[k][0])
        partial = Schedule(tuple(starts))
        times = repair_times(problem, g, x.schedule, partial, members, o, region, bounds[o])
        rng.shuffle(times)
        found = False
        for t in times[:positions]:
            if fits(problem, g, partial, o, t):
                starts[o] = t
                found = True
                break
        if not found:
            return None
        pending.remove(o)
    schedule = Schedule(tuple(starts))
    return None if violations(problem, g, schedule) else schedule


def peak_reduced(
    problem: ProblemInstance,
    genotype: Genotype,
    schedule: Schedule,
    region: int,
    original_peak: float,
) -> bool:
    """Check every target window, including newly created peaks and actual tail."""
    new_peak = max(
        regional_windows(
            power_segments(problem, genotype, schedule, region), schedule.horizon(problem)
        ),
        default=0.0,
    )
    return new_peak < original_peak - TOL.power


class PeakCoalition:
    """One target Region per invocation with independent same-incumbent recipes."""

    action = 8

    def propose(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        config: Config,
        rng: random.Random,
    ) -> ProposalBatch:
        """Return strict all-window peak reductions within 2B construction attempts."""
        result = ProposalBatch()
        if budget <= 0:
            return result
        g = incumbent.genotype
        regions = [
            r
            for r, region in enumerate(problem.regions)
            if region.demand_rate > 0 and any(problem.instances[m].region == r for m in g.ms)
        ]
        if not regions:
            return result
        weights = [
            problem.regions[r].demand_rate * max(incumbent.evaluation.windows[r], default=0)
            for r in regions
        ]
        if not any(weights):
            return result
        region = rng.choices(regions, weights=weights)[0]
        windows = incumbent.evaluation.windows[region]
        peak = max(windows)
        peaks = tuple(b for b, v in enumerate(windows) if peak - v <= TOL.power)
        segments = support_segments(problem, g, incumbent.schedule, region, peaks)
        singletons = [o for o, m in enumerate(g.ms) if problem.instances[m].region == region]
        rng.shuffle(singletons)
        singletons.sort(
            key=lambda o: -sum(removal_potential(segments, frozenset({o}), b) > 0 for b in peaks)
        )
        singletons = [
            o
            for o in singletons
            if any(removal_potential(segments, frozenset({o}), b) > 0 for b in peaks)
        ]
        seen = {tuple(round(t / TOL.time) for t in incumbent.schedule.starts)}
        singleton_success = False
        result.diagnostics = {"region": region, "peaks": peaks, "recipes": []}
        for attempt in range(config.a8_attempt_multiplier * budget):
            if len(result.proposals) >= budget:
                break
            if singletons and (attempt < config.a8_singleton_attempts or singleton_success):
                members = frozenset({singletons[attempt % len(singletons)]})
            else:
                members = coalition(segments, peaks, config.a8_member_cap, rng)
            result.attempts += 1
            result.diagnostics["recipes"].append(tuple(sorted(members)))
            if not members:
                continue
            schedule = repair(problem, incumbent, members, region, config.a8_position_limit, rng)
            if schedule is None:
                continue
            if not peak_reduced(problem, g, schedule, region, peak):
                continue
            key = tuple(round(t / TOL.time) for t in schedule.starts)
            if key in seen:
                continue
            seen.add(key)
            singleton_success |= len(members) == 1
            result.proposals.append(Proposal(g, schedule, tuple(sorted(members))))
        return result
