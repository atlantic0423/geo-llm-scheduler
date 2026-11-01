"""Research-only electricity-guided seeds; original initialization is untouched.

Prefix electricity scores integrate active *unions*, idle power over the current
prefix horizon, and every fixed 900-second window. They are greedy prefix scores,
not estimates of the future whole-schedule marginal bill. Final schedules always
use the original SSGS and independently repeated exact evaluator.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from time import perf_counter, process_time
from typing import Callable

from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.initial_bill_index import ResourceCalendar, TariffIndex
from geo_llm_scheduler.initialization.generators import generate
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.scheduling.resources import earliest
from geo_llm_scheduler.scheduling.ssgs import decode

FAMILIES = ("RESAMPLE", "TOU", "DEMAND", "BILL", "PORTFOLIO")
ARMS = tuple(f"{name}{share}" for name in FAMILIES for share in (10, 20))


def tariff_integral(problem: ProblemInstance, region: int, a: float, b: float) -> float:
    """Integrate CNY/kWh times seconds on a fully covered global interval."""
    if not 0 <= a <= b <= problem.regions[region].tariffs[-1].end:
        raise ValueError("Prefix time is outside declared tariffs")
    return math.fsum(
        max(0.0, min(b, t.end) - max(a, t.start)) * t.price for t in problem.regions[region].tariffs
    )


def merged(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the disjoint active union, counting overlap once."""
    out: list[tuple[float, float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


@dataclass
class PrefixPower:
    """Incremental active-union integrals; idle electricity is horizon dependent."""

    problem: ProblemInstance
    unions: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    active_tou: float = 0.0
    energy: dict[tuple[int, int], float] = field(default_factory=dict)
    horizon: float = 0.0
    index: TariffIndex | None = None

    def trial(self, additions: dict[int, list[tuple[float, float]]], horizon: float) -> tuple:
        """Score an uncommitted feasible prefix, returning updates for commit."""
        new_unions = {m: merged(self.unions.get(m, []) + items) for m, items in additions.items()}
        integral = self.index.integral if self.index is not None else tariff_integral
        delta_tou = 0.0
        delta_energy: dict[tuple[int, int], float] = {}
        for m, union in new_unions.items():
            server = self.problem.instances[m]
            for sign, items in ((1.0, union), (-1.0, self.unions.get(m, []))):
                for a, b in items:
                    power = sign * (server.active_kw - server.idle_kw)
                    delta_tou += power * integral(self.problem, server.region, a, b) / 3600
                    for w in range(int(a // 900), math.ceil(b / 900)):
                        key = (server.region, w)
                        overlap = max(0.0, min(b, (w + 1) * 900) - max(a, w * 900))
                        delta_energy[key] = delta_energy.get(key, 0.0) + power * overlap
        idle = (
            self.index.idle
            if self.index is not None
            else [
                sum(s.idle_kw for s in self.problem.instances if s.region == r)
                for r in range(len(self.problem.regions))
            ]
        )
        tou = (
            self.active_tou
            + delta_tou
            + math.fsum(
                power * integral(self.problem, r, 0.0, horizon) / 3600
                for r, power in enumerate(idle)
            )
        )
        demand = []
        for r, region in enumerate(self.problem.regions):
            windows = [
                (
                    self.energy.get((r, w), 0.0)
                    + delta_energy.get((r, w), 0.0)
                    + idle[r] * max(0.0, min(900.0, horizon - w * 900))
                )
                / 900
                for w in range(math.ceil(horizon / 900))
            ]
            demand.append(region.demand_rate * max(windows, default=0.0))
        return tou, math.fsum(demand), new_unions, delta_tou, delta_energy, horizon

    def commit(self, trial: tuple) -> None:
        """Apply exactly the selected trial, never another path's updates."""
        self.unions.update(trial[2])
        self.active_tou += trial[3]
        for key, value in trial[4].items():
            self.energy[key] = self.energy.get(key, 0.0) + value
        self.horizon = trial[5]


def guided_genotype(
    problem: ProblemInstance,
    family: str,
    rng: random.Random,
    flow_weight: float,
    *,
    accelerated: bool = False,
) -> tuple[Genotype, dict]:
    """Build an MS/OS seed by randomized, resource-feasible prefix path choice.

    TOU and DEMAND score their named component plus a fixed 10% normalized Flow
    term. BILL uses total prefix Bill and a declared diversified Flow weight.
    Randomized restricted lists contain only paths within 5% of the score range
    of the best path, so numerical ties do not force the same machine forever.
    """
    if family not in ("TOU", "DEMAND", "BILL") or not 0 <= flow_weight <= 1:
        raise ValueError("Unknown guide or invalid weight")
    n = len(problem.jobs)
    ms = [0] * (2 * n)
    starts = [-1.0] * (2 * n)
    order = list(range(n))
    style = rng.randrange(3)
    tie = [rng.random() for _ in order]
    if style == 0:
        order.sort(key=lambda j: (problem.jobs[j].release, tie[j]))
    elif style == 1:
        order.sort(
            key=lambda j: (
                problem.jobs[j].release,
                problem.jobs[j].prefill.duration + problem.jobs[j].decode.duration,
                tie[j],
            )
        )
    else:
        order.sort(
            key=lambda j: (
                problem.jobs[j].release,
                -problem.jobs[j].prefill.duration - problem.jobs[j].decode.duration,
                tie[j],
            )
        )
    os = tuple(j for j in order for _ in (0, 1))
    power = PrefixPower(problem, index=TariffIndex(problem) if accelerated else None)
    calendar = ResourceCalendar(problem) if accelerated else None
    scored = 0
    for job in order:
        p_op, d_op = 2 * job, 2 * job + 1
        rows = []
        p_cache: dict[int, float] = {}
        for p, d in paths(problem, job):
            ms[p_op], ms[d_op] = p, d
            g = Genotype(tuple(ms), os)
            if calendar is None:
                p_start = earliest(
                    problem, g, Schedule(tuple(starts)), p_op, problem.jobs[job].release
                )
            else:
                if p not in p_cache:
                    p_cache[p] = calendar.earliest(p, p_op, problem.jobs[job].release)
                p_start = p_cache[p]
            starts[p_op] = p_start
            d_est = (
                p_start + problem.jobs[job].prefill.duration + (p != d) * problem.jobs[job].kv_delay
            )
            d_start = (
                earliest(problem, g, Schedule(tuple(starts)), d_op, d_est)
                if calendar is None
                else calendar.earliest(
                    d,
                    d_op,
                    d_est,
                    (p_op, p_start, p_start + problem.jobs[job].prefill.duration)
                    if p == d
                    else None,
                )
            )
            starts[p_op] = -1.0
            d_end = d_start + problem.jobs[job].decode.duration
            additions: dict[int, list[tuple[float, float]]] = {}
            additions.setdefault(p, []).append(
                (p_start, p_start + problem.jobs[job].prefill.duration)
            )
            additions.setdefault(d, []).append((d_start, d_end))
            trial = power.trial(additions, max(power.horizon, d_end))
            cost = (
                trial[0]
                if family == "TOU"
                else trial[1]
                if family == "DEMAND"
                else trial[0] + trial[1]
            )
            rows.append((cost, d_end - problem.jobs[job].release, p, d, p_start, d_start, trial))
        costs = [row[0] for row in rows]
        flows = [row[1] for row in rows]
        lo_c, lo_f = min(costs), min(flows)
        range_c, range_f = max(costs) - lo_c, max(flows) - lo_f
        weight = flow_weight if family == "BILL" else 0.1
        scores = [
            (1 - weight) * (row[0] - lo_c) / (range_c or 1.0)
            + weight * (row[1] - lo_f) / (range_f or 1.0)
            for row in rows
        ]
        threshold = min(scores) + 0.05 * (max(scores) - min(scores)) + 1e-12
        selected = rng.choice(
            [row for row, score in zip(rows, scores, strict=True) if score <= threshold]
        )
        ms[p_op], ms[d_op] = selected[2], selected[3]
        starts[p_op], starts[d_op] = selected[4], selected[5]
        power.commit(selected[6])
        if calendar is not None:
            calendar.commit(selected[2], p_op, selected[4])
            calendar.commit(selected[3], d_op, selected[5])
        scored += len(rows)
    genotype = Genotype(tuple(ms), os)
    if decode(problem, genotype).starts != tuple(starts):
        raise AssertionError("Prefix construction did not replay through original SSGS")
    return genotype, {
        "path_scores": scored,
        "prefix_tou": power.trial({}, power.horizon)[0],
        "prefix_demand": power.trial({}, power.horizon)[1],
        "flow_weight": flow_weight,
        "order_style": style,
    }


def retain_original(parents: list[Candidate], share: int) -> list[Candidate]:
    """Reduce all ten original source quotas equally, without quality selection."""
    if share not in (10, 20) or len(parents) != 100:
        raise ValueError("Require P100 and share 10/20")
    quota = (100 - share) // 10
    counts = [0] * 10
    retained = []
    for candidate in parents:
        mode = int(candidate.origin.removeprefix("initial_mode_"))
        if not 0 <= mode < 10:
            raise ValueError("Unknown original source")
        if counts[mode] < quota:
            retained.append(candidate)
            counts[mode] += 1
    if counts != [quota] * 10:
        raise ValueError("Original realized source quotas are not ten balanced classes")
    return retained


def seed_bank(
    problem: ProblemInstance,
    parents: list[Candidate],
    rng: random.Random,
    trials: int,
    progress: Callable[[dict], None] | None = None,
    *,
    families: tuple[str, ...] = FAMILIES[:-1],
    size: int = 20,
    accelerated: bool = False,
    evaluator: Callable[..., Candidate] | None = None,
    verify_trials: bool = True,
) -> tuple[dict, dict]:
    """Build 20 distinct seeds per family with a fixed exact tournament budget.

    RESAMPLE uses the original random generator and the same tournament count,
    controlling the cost of extra sampling/selection. No fallback on duplicates:
    bounded exhaustion is a recorded failure. Each evaluated trial and every bank
    winner receive independent exact verification; all counts/costs are retained.
    """
    if trials not in ((1, 2, 4) if accelerated else (1, 4)):
        raise ValueError("Reference allows 1/4; accelerated research allows 1/2/4 trials")
    if (
        not families
        or len(set(families)) != len(families)
        or any(f not in FAMILIES[:-1] for f in families)
        or size not in (10, 20)
    ):
        raise ValueError("Require distinct base families and size 10/20")
    bank: dict[str, list[Candidate]] = {}
    observations = []
    exact_count = 0
    started = perf_counter()
    cpu_started = process_time()
    verification_seconds = 0.0
    verification_cpu_seconds = 0.0
    seen = {p.genotype for p in parents}
    for family in families:
        members = []
        for slot in range(size):
            weight = (0.0, 0.1, 0.25, 0.5)[slot % 4]
            pool: list[Candidate] = []
            for attempt in range(100):
                if len(pool) == trials:
                    break
                if family == "RESAMPLE":
                    g = generate(problem, 7, rng, 0.1)
                    evidence: dict = {}
                else:
                    g, evidence = guided_genotype(
                        problem, family, rng, weight, accelerated=accelerated
                    )
                if g in seen:
                    continue
                seen.add(g)
                if evaluator is None:
                    schedule = decode(problem, g)
                    result = evaluate(problem, g, schedule)
                    candidate = Candidate(g, schedule, result, f"bill_seed_{family}")
                else:
                    candidate = evaluator(g, origin=f"bill_seed_{family}")
                    schedule, result = candidate.schedule, candidate.evaluation
                repeated = result
                if verify_trials:
                    tick = perf_counter()
                    cpu_tick = process_time()
                    repeated = evaluate(problem, g, schedule)
                    verification_seconds += perf_counter() - tick
                    verification_cpu_seconds += process_time() - cpu_tick
                if not result.feasible or repeated != result:
                    raise AssertionError("New seed failed independent exact verification")
                if evidence and (
                    not math.isclose(
                        evidence["prefix_tou"], result.tou, rel_tol=1e-10, abs_tol=1e-8
                    )
                    or not math.isclose(
                        evidence["prefix_demand"], sum(result.demand), rel_tol=1e-10, abs_tol=1e-8
                    )
                ):
                    raise AssertionError("Incremental prefix power disagrees with full exact")
                pool.append(candidate)
                exact_count += 1
                observations.append(
                    {
                        "family": family,
                        "slot": slot,
                        "attempt": attempt,
                        "flow": result.flow,
                        "bill": result.bill,
                        "tou": result.tou,
                        "demand": sum(result.demand),
                        **evidence,
                    }
                )
                if progress:
                    progress(
                        {
                            "family": family,
                            "slot": slot,
                            "exact": exact_count,
                            "elapsed": perf_counter() - started,
                        }
                    )
            if len(pool) != trials:
                raise ValueError("Unique seed trials unavailable within frozen attempt limit")
            costs = [
                c.evaluation.tou
                if family == "TOU"
                else sum(c.evaluation.demand)
                if family == "DEMAND"
                else c.evaluation.bill
                for c in pool
            ]
            flows = [c.evaluation.flow for c in pool]
            w = weight if family == "BILL" else 0.1
            scores = [
                (1 - w) * (c - min(costs)) / (max(costs) - min(costs) or 1)
                + w * (f - min(flows)) / (max(flows) - min(flows) or 1)
                for c, f in zip(costs, flows, strict=True)
            ]
            winner = pool[
                min(
                    range(trials),
                    key=lambda i: (scores[i], pool[i].genotype.ms, pool[i].genotype.os),
                )
            ]
            members.append(winner)
        bank[family] = members
    # Shared construction costs are reported once, not charged once per reused arm.
    return bank, {
        "exact": exact_count,
        "independent_exact_verified": exact_count if verify_trials else 0,
        "elapsed": perf_counter() - started,
        "verification_seconds": verification_seconds,
        "production_seconds": perf_counter() - started - verification_seconds,
        "production_cpu_seconds": process_time() - cpu_started - verification_cpu_seconds,
        "verification_cpu_seconds": verification_cpu_seconds,
        "trials_per_slot": trials,
        "rng_state_sha256": hashlib.sha256(repr(rng.getstate()).encode()).hexdigest(),
        "observations": observations,
    }


def assemble(parents: list[Candidate], bank: dict, arm: str) -> list[Candidate]:
    """Combine fixed old quotas and electricity seeds; final size is exactly 100."""
    if arm not in ARMS:
        raise ValueError("Unknown mixture arm")
    share = int(arm[-2:])
    family = arm[:-2]
    retained = retain_original(parents, share)
    if family == "PORTFOLIO":
        new = [bank[("TOU", "DEMAND", "BILL")[i % 3]][i // 3] for i in range(share)]
    else:
        new = bank[family][:share]
    result = retained + new
    if len(result) != 100 or len({c.genotype for c in result}) != 100:
        raise AssertionError("Mixture size, provenance or uniqueness violated")
    return result
