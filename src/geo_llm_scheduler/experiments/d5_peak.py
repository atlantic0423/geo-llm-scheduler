"""Opt-in D5 online A8; original D4 modules and default operators are untouched."""

from __future__ import annotations

import random
from dataclasses import dataclass

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance, Schedule
from geo_llm_scheduler.evaluation.exact import violations
from geo_llm_scheduler.experiments.d4_prototypes import ResearchPeak
from geo_llm_scheduler.experiments.d5_windows import InsertionWindows
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.operators.peak_coalition import (
    coalition,
    peak_reduced,
    removal_potential,
    repair,
    repair_bounds,
    repair_times,
    support_segments,
)
from geo_llm_scheduler.scheduling.resources import fits
from geo_llm_scheduler.utils.numeric import TOL


@dataclass(frozen=True)
class CertificateCache:
    """Invocation-local immutable original support and fixed-horizon hypotheses."""

    windows: tuple[float, ...]
    tails: frozenset[int]
    segments: tuple[tuple[int, frozenset[int], float], ...]


def certificate_cache(problem: ProblemInstance, x: Candidate, region: int) -> CertificateCache:
    """Build all-window active-union support once; no process-global cache."""
    horizon = x.schedule.horizon(problem)
    windows = x.evaluation.windows[region]
    tails = frozenset(
        o for o in range(problem.operation_count) if x.schedule.end(problem, o) == horizon
    )
    return CertificateCache(
        windows,
        tails,
        tuple(
            support_segments(problem, x.genotype, x.schedule, region, tuple(range(len(windows))))
        ),
    )


def cached_certificate(cache: CertificateCache, members: frozenset[int]) -> dict:
    """Return the same conservative necessary bound as the original D4 observer."""
    lower = list(cache.windows)
    for b, active, value in cache.segments:
        if active <= members:
            lower[b] -= value
    threshold = max(cache.windows, default=0.0) - TOL.power
    applicable = bool(cache.tails - members)
    return {
        "applicable": applicable,
        "lower_windows": tuple(lower),
        "threshold": threshold,
        "denied": applicable and max(lower, default=0.0) > threshold + TOL.power * 0.001,
    }


def position_repair_fast(
    problem: ProblemInstance,
    x: Candidate,
    members: frozenset[int],
    region: int,
    positions: int,
    rng: random.Random,
    policy: str,
    record: dict | None,
) -> Schedule | None:
    """Compare peak ranking to a scored random-order control on the same finite quota.

    Both score every resource-feasible time among the same sampled at-most-L positions.
    The control chooses its first feasible time and discards scores; peak ranking chooses
    the least fixed-horizon partial maximum. Uninserted members remain absent. This is
    a heuristic, not a guarantee about completed repair. No beam or extra restart.
    """
    if policy not in {"random_scored", "peak_rank"}:
        raise ValueError("Unknown position policy")
    record = {} if record is None else record
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
        scorer = InsertionWindows(problem, g, partial, region, x.schedule.horizon(problem))
        feasible: list[tuple[float, float]] = []
        for t in times[:positions]:
            record["position_checks"] = record.get("position_checks", 0) + 1
            if fits(problem, g, partial, o, t):
                windows = scorer.windows(o, t)
                record["position_scores"] = record.get("position_scores", 0) + 1
                feasible.append((t, max(windows, default=0.0)))
        if not feasible:
            record["stage"] = "resource_positions"
            return None
        # Stable seeded order breaks peak ties without extra draws.
        starts[o] = (
            min(feasible, key=lambda item: item[1]) if policy == "peak_rank" else feasible[0]
        )[0]
        pending.remove(o)
    schedule = Schedule(tuple(starts))
    if violations(problem, g, schedule):
        record["stage"] = "final_feasibility"
        return None
    return schedule


class OnlinePeak:
    """Default-disconnected operator with common per-recipe selection/repair streams."""

    action = 8

    def __init__(
        self, guard: str = "off", positions: str = "first", diagnose: bool = False
    ) -> None:
        """Select guard and finite position ranking, including the predeclared combination."""
        if guard not in {"off", "all", "single"} or positions not in {
            "first",
            "random_scored",
            "peak_rank",
        }:
            raise ValueError("Unknown prototype")
        self.guard, self.positions = guard, positions
        self.diagnose = diagnose

    def propose(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        config: Config,
        rng: random.Random,
    ) -> ProposalBatch:
        """Generate bounded timing candidates; skipped repair cannot shift later RNG."""
        x = incumbent
        result = ProposalBatch()
        if budget <= 0:
            return result
        g = x.genotype
        regions = [
            r
            for r, p in enumerate(problem.regions)
            if p.demand_rate > 0 and any(problem.instances[m].region == r for m in g.ms)
        ]
        weights = [
            problem.regions[r].demand_rate * max(x.evaluation.windows[r], default=0.0)
            for r in regions
        ]
        if not regions or not any(weights):
            return result
        region = rng.choices(regions, weights=weights)[0]
        windows = x.evaluation.windows[region]
        peak = max(windows)
        peaks = tuple(b for b, v in enumerate(windows) if peak - v <= TOL.power)
        support = support_segments(problem, g, x.schedule, region, peaks)
        singletons = [o for o, m in enumerate(g.ms) if problem.instances[m].region == region]
        rng.shuffle(singletons)
        singletons.sort(
            key=lambda o: -sum(removal_potential(support, frozenset({o}), b) > 0 for b in peaks)
        )
        singletons = [
            o
            for o in singletons
            if any(removal_potential(support, frozenset({o}), b) > 0 for b in peaks)
        ]
        cache: CertificateCache | None = None
        singleton_success = False
        seen = {tuple(round(t / TOL.time) for t in x.schedule.starts)}
        trace: list[dict] = []
        if self.diagnose:
            result.diagnostics = {"region": region, "trace": trace}
        for attempt in range(config.a8_attempt_multiplier * budget):
            if len(result.proposals) >= budget:
                break
            selection_rng = random.Random(rng.getrandbits(64))
            repair_rng = random.Random(rng.getrandbits(64))
            use_single = bool(
                singletons and (attempt < config.a8_singleton_attempts or singleton_success)
            )
            members = (
                frozenset({singletons[attempt % len(singletons)]})
                if use_single
                else coalition(support, peaks, config.a8_member_cap, selection_rng)
            )
            result.attempts += 1
            record: dict = {"members": tuple(sorted(members)), "stage": "empty_members"}
            if self.diagnose:
                trace.append(record)
            if not members:
                continue
            if self.guard == "all" or (self.guard == "single" and len(members) == 1):
                if cache is None:
                    cache = certificate_cache(problem, x, region)
                cert = cached_certificate(cache, members)
                if self.diagnose:
                    record["certificate"] = cert
                if cert["denied"]:
                    if self.diagnose:
                        record["stage"] = "certificate"
                    continue
            if self.positions == "first":
                schedule = repair(
                    problem,
                    x,
                    members,
                    region,
                    config.a8_position_limit,
                    repair_rng,
                    record if self.diagnose else None,
                )
            else:
                schedule = position_repair_fast(
                    problem,
                    x,
                    members,
                    region,
                    config.a8_position_limit,
                    repair_rng,
                    self.positions,
                    record if self.diagnose else None,
                )
            if schedule is None:
                continue
            if self.diagnose:
                record.update(stage="peak_gate", repaired_starts=schedule.starts)
            if not peak_reduced(problem, g, schedule, region, peak):
                continue
            key = tuple(round(t / TOL.time) for t in schedule.starts)
            if key in seen:
                if self.diagnose:
                    record["stage"] = "duplicate"
                continue
            seen.add(key)
            singleton_success |= len(members) == 1
            result.proposals.append(Proposal(g, schedule, tuple(sorted(members))))
            if self.diagnose:
                record["stage"] = "proposal"
        return result


class ResearchReference(ResearchPeak):
    """Typed engine adapter around the unchanged original D4 reference operator."""

    action = 8

    def propose(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        config: Config,
        rng: random.Random,
    ) -> ProposalBatch:
        """Delegate every draw and proposal to the original same-stream reference."""
        return super().propose(problem, incumbent, budget, config, rng)
