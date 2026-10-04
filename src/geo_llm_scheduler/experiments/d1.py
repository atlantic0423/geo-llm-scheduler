"""Offline D1 hypotheses: bounded residual-wait transfer after structural rebuild.

This module never updates an online population, archive, controller or RNG stream.
Positive residual waiting is one limited intent representation, not a claim that
all useful packing, tariff or peak structure can be inherited.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from time import perf_counter

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.io.validation import validate_genotype
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.operators.active_pack import ActivePack
from geo_llm_scheduler.scheduling.resources import earliest, est_for
from geo_llm_scheduler.scheduling.ssgs import decode, refresh_diagnostics
from geo_llm_scheduler.utils.numeric import TOL, less
from geo_llm_scheduler.utils.rng import RNGManager

ARMS = ("ZERO", "TRUE", "SHUFFLED", "FRESH_A7")
SCALES = (0.25, 0.5, 1.0)


def extract_intent(problem: ProblemInstance, source: Candidate) -> tuple[float, ...]:
    """Recompute positive residual waits against the frozen source profile."""
    if not source.evaluation.feasible:
        raise ValueError("Intent requires a feasible source")
    return refresh_diagnostics(problem, source.genotype, source.schedule).intentional_wait


def shuffle_intent(intent: tuple[float, ...], rng: random.Random) -> tuple[float, ...]:
    """Permute within P and D separately, retaining phase-specific histograms."""
    values = list(intent)
    for phase in (0, 1):
        indices = list(range(phase, len(values), 2))
        shuffled = [intent[o] for o in indices]
        rng.shuffle(shuffled)
        for o, value in zip(indices, shuffled):
            values[o] = value
    return tuple(values)


def project_intent(
    problem: ProblemInstance,
    genotype: Genotype,
    rebuilt: Schedule,
    intent: tuple[float, ...],
    scale: float,
    cap_seconds: float = 900.0,
) -> Schedule | None:
    """Serially project bounded relative waits on a new, already rebuilt structure.

    No source absolute starts enter this function. New release/precedence/KV and
    both resource capacities are used at every placement. Reject projections
    extending the rebuilt batch end by more than the prespecified intent cap.
    """
    validate_genotype(problem, genotype)
    if len(intent) != problem.operation_count or not 0 <= scale <= 1:
        raise ValueError("Invalid intent dimension or scale")
    if not math.isfinite(cap_seconds) or cap_seconds <= 0:
        raise ValueError("A positive finite cap is required")
    if any(not math.isfinite(w) or w < 0 for w in intent):
        raise ValueError("Intent waits must be finite and nonnegative")
    if scale == 0 or not any(intent):
        return rebuilt
    starts = [-1.0] * problem.operation_count
    seen = [0] * len(problem.jobs)
    for job in genotype.os:
        o = 2 * job + seen[job]
        seen[job] += 1
        partial = Schedule(tuple(starts))
        desired = est_for(problem, genotype, partial, o) + scale * min(intent[o], cap_seconds)
        starts[o] = earliest(problem, genotype, partial, o, desired)
    proposal = Schedule(tuple(starts))
    if less(rebuilt.horizon(problem) + cap_seconds, proposal.horizon(problem), TOL.time):
        return None
    return proposal


def perturb_structure(
    problem: ProblemInstance, genotype: Genotype, kind: str, jobs: int, rng: random.Random
) -> Genotype | None:
    """Apply reproducible OS, within-region instance or joint region perturbations.

    Return None if no actual structural change is available; never label an
    unchanged genotype as evidence of transfer across structural changes.
    """
    if kind not in ("OS", "INSTANCE", "REGION") or jobs < 1:
        raise ValueError("Unknown structure perturbation or invalid size")
    ms, order = list(genotype.ms), list(genotype.os)
    for job in rng.sample(range(len(problem.jobs)), min(jobs, len(problem.jobs))):
        if kind == "OS":
            positions = [i for i, j in enumerate(order) if j == job]
            others = [i for i, j in enumerate(order) if j != job]
            if others:
                a, b = rng.choice(positions), rng.choice(others)
                order[a], order[b] = order[b], order[a]
        else:
            region = problem.instances[ms[2 * job]].region
            if kind == "INSTANCE":
                o = 2 * job + rng.randrange(2)
                choices = [m for m in problem.eligible(o, region) if m != ms[o]]
                if choices:
                    ms[o] = rng.choice(choices)
            else:
                path_choices = [
                    path
                    for path in paths(problem, job)
                    if problem.instances[path[0]].region != region
                ]
                if path_choices:
                    ms[2 * job : 2 * job + 2] = rng.choice(path_choices)
    target = Genotype(tuple(ms), tuple(order))
    validate_genotype(problem, target)
    return None if target == genotype else target


def paired_probe(
    problem: ProblemInstance,
    source: Candidate,
    target: Genotype,
    weight: tuple[float, float],
    context: NormalizationContext,
    seed: int,
    quota_seconds: float,
    cap_seconds: float = 900.0,
) -> dict:
    """Compare four paired treatments under a common time and exact-evaluation cap.

    Each treatment has the same positive wall-clock quota and at most three
    distinct additional exact evaluations. Atomic construction/evaluation can
    overshoot; late candidates are recorded but excluded from the quota result.
    All completed feasible proposals share one batch normalization context.
    The zero-intent rebuilt solution is always available as a fallback.
    """
    if target == source.genotype or not math.isfinite(quota_seconds) or quota_seconds <= 0:
        raise ValueError("A changed structure and positive finite quota are required")
    start = perf_counter()
    if evaluate(problem, source.genotype, source.schedule) != source.evaluation:
        raise ValueError("Source failed exact re-evaluation")
    rebuilt = decode(problem, target)
    base = Candidate(target, rebuilt, evaluate(problem, target, rebuilt), "D1:ZERO")
    if not base.evaluation.feasible:
        raise ValueError("Structural rebuild failed exact feasibility")
    # Source SSGS is diagnostic only; inherited intent is not an SSGS-start delta.
    source_rebuilt = decode(problem, source.genotype)
    source_reset = evaluate(problem, source.genotype, source_rebuilt)
    shared_seconds = perf_counter() - start
    records: dict[str, dict] = {}
    completed: dict[str, list[tuple[Candidate, bool]]] = {}
    streams = RNGManager(seed)
    order = list(ARMS)
    streams.stream("D1:arm_order").shuffle(order)
    for arm in order:
        begin = perf_counter()
        timings: dict[str, float] = {}
        candidates: list[tuple[Candidate, bool]] = []
        attempts = failures = duplicates = 0
        intent_nonzero = intent_total = 0.0
        seen = {rebuilt.starts}
        if arm in ("TRUE", "SHUFFLED"):
            tick = perf_counter()
            intent = extract_intent(problem, source)
            if arm == "SHUFFLED":
                intent = shuffle_intent(intent, streams.stream("D1:shuffle"))
            intent_nonzero = float(sum(w > TOL.time for w in intent))
            intent_total = sum(intent)
            timings["extraction_seconds"] = perf_counter() - tick
            for scale in SCALES:
                if perf_counter() - begin >= quota_seconds:
                    break
                attempts += 1
                tick = perf_counter()
                schedule = project_intent(problem, target, rebuilt, intent, scale, cap_seconds)
                timings["projection_seconds"] = timings.get("projection_seconds", 0.0) + (
                    perf_counter() - tick
                )
                if schedule is None:
                    failures += 1
                    continue
                if schedule.starts in seen:
                    duplicates += 1
                    continue
                seen.add(schedule.starts)
                tick = perf_counter()
                evaluation = evaluate(problem, target, schedule)
                timings["exact_seconds"] = timings.get("exact_seconds", 0.0) + perf_counter() - tick
                candidates.append(
                    (
                        Candidate(target, schedule, evaluation, f"D1:{arm}"),
                        perf_counter() - begin <= quota_seconds,
                    )
                )
        elif arm == "FRESH_A7":
            tick = perf_counter()
            batch = ActivePack().propose(problem, base, 3, Config(), streams.stream("D1:A7"))
            timings["construction_seconds"] = perf_counter() - tick
            attempts = batch.attempts
            for proposal in batch.proposals:
                if perf_counter() - begin >= quota_seconds:
                    break
                assert proposal.schedule is not None
                if proposal.schedule.starts in seen:
                    duplicates += 1
                    continue
                seen.add(proposal.schedule.starts)
                tick = perf_counter()
                evaluation = evaluate(problem, target, proposal.schedule)
                timings["exact_seconds"] = timings.get("exact_seconds", 0.0) + perf_counter() - tick
                candidates.append(
                    (
                        Candidate(target, proposal.schedule, evaluation, "D1:FRESH_A7"),
                        perf_counter() - begin <= quota_seconds,
                    )
                )
        elapsed = perf_counter() - begin
        completed[arm] = candidates
        records[arm] = {
            "seconds": elapsed,
            "quota_seconds": quota_seconds,
            "overshoot_seconds": max(0.0, elapsed - quota_seconds),
            "attempts": attempts,
            "failed_projection": failures,
            "duplicates": duplicates,
            "exact": len(candidates),
            "feasible": sum(c.evaluation.feasible for c, _ in candidates),
            "on_time_exact": sum(on_time for _, on_time in candidates),
            "intent_nonzero": intent_nonzero,
            "intent_total_seconds": intent_total,
            **timings,
        }
    all_feasible = [base] + [
        c for values in completed.values() for c, _ in values if c.evaluation.feasible
    ]
    shared = NormalizationContext(
        tuple(
            min(context.ideal[m], *(c.evaluation.objectives[m] for c in all_feasible))
            for m in range(2)
        ),  # type: ignore[arg-type]
        context.maximum,
    )
    base_scalar = shared.scalar(base, weight)
    for arm in ARMS:
        eligible = [base] + [
            c for c, on_time in completed[arm] if on_time and c.evaluation.feasible
        ]
        best = min(eligible, key=lambda c: (shared.scalar(c, weight), c.schedule.starts))
        records[arm].update(
            objectives=best.evaluation.objectives,
            scalar=shared.scalar(best, weight),
            scalar_gain=base_scalar - shared.scalar(best, weight),
            base_is_best=best is base,
            proposals=[
                {
                    "objectives": c.evaluation.objectives,
                    "feasible": c.evaluation.feasible,
                    "on_time": on_time,
                    "scalar": shared.scalar(c, weight),
                }
                for c, on_time in completed[arm]
            ],
        )
    return {
        "seed": seed,
        "weight": weight,
        "context": asdict(shared),
        "source_objectives": source.evaluation.objectives,
        "source_reset_objectives": source_reset.objectives,
        "base_objectives": base.evaluation.objectives,
        "base_scalar": base_scalar,
        "shared_seconds": shared_seconds,
        "shared_ssgs": 2,
        "shared_exact": 3,
        "ms_distance": sum(a != b for a, b in zip(source.genotype.ms, target.ms)),
        "os_distance": sum(a != b for a, b in zip(source.genotype.os, target.os)),
        "arm_order": order,
        "arms": records,
    }
