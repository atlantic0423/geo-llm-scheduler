"""Offline, prespecified D1 routing comparisons; never change the online engine."""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from time import perf_counter

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.d1 import SCALES, extract_intent, project_intent
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.active_pack import ActivePack
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.numeric import TOL
from geo_llm_scheduler.utils.rng import RNGManager

GROUPS = ("ALWAYS_A7", "CONDITIONAL", "RANDOM")


def gate_transfer(source: Genotype, target: Genotype, intent: tuple[float, ...]) -> bool:
    """Require positive residual wait, identical actual MS and genuinely changed OS."""
    if len(intent) != len(source.ms) or any(not math.isfinite(w) or w < 0 for w in intent):
        raise ValueError("Invalid residual-wait vector")
    if len(target.ms) != len(source.ms) or len(target.os) != len(source.os):
        raise ValueError("Source and target dimensions differ")
    return any(w > TOL.time for w in intent) and target.ms == source.ms and target.os != source.os


def matched_random_routes(
    source: Genotype, targets: list[Genotype], intent: tuple[float, ...], rng: random.Random
) -> tuple[bool, ...]:
    """Randomize exactly the gated transfer count within this source's target pool.

    The positive-wait guard remains in both routing policies. Targets with changed
    assignments remain eligible for random selection, isolating the OS/MS gate.
    No candidate objectives enter allocation; targets must be genuine changes.
    """
    if any(target == source for target in targets):
        raise ValueError("Unchanged targets cannot enter a routing pool")
    count = sum(gate_transfer(source, target, intent) for target in targets)
    selected = set(rng.sample(range(len(targets)), count))
    return tuple(i in selected for i in range(len(targets)))


def routing_probe(
    problem: ProblemInstance,
    source: Candidate,
    target: Genotype,
    weight: tuple[float, float],
    context: NormalizationContext,
    seed: int,
    random_transfer: bool,
    quota_seconds: float,
    cap_seconds: float = 900.0,
) -> dict:
    """Execute all three actual policies on a paired target, charging their guards.

    Source exact validation and target SSGS/exact are shared, explicitly charged
    common work. Freeze the source-population context, extended only by the base,
    before generating proposals. Every policy executes only its preselected route,
    has at most three additional exact evaluations and retains the common base.
    Atomic late work is recorded and charged, but cannot supply a retained result.
    A7 uses an identical named RNG in every policy, so A7 fallback is truly paired.
    """
    if target == source.genotype or not math.isfinite(quota_seconds) or quota_seconds <= 0:
        raise ValueError("A changed target and positive finite quota are required")
    if not math.isfinite(cap_seconds) or cap_seconds <= 0:
        raise ValueError("A positive finite intent cap is required")
    if len(weight) != 2 or any(not math.isfinite(w) or w < 0 for w in weight):
        raise ValueError("Invalid subproblem weight")
    tick = perf_counter()
    exact_source = evaluate(problem, source.genotype, source.schedule)
    if not exact_source.feasible or exact_source != source.evaluation:
        raise ValueError("Source failed exact validation")
    rebuilt = decode(problem, target)
    base = Candidate(target, rebuilt, evaluate(problem, target, rebuilt), "D1H:BASE")
    if not base.evaluation.feasible:
        raise ValueError("Rebuilt target is infeasible")
    frozen = NormalizationContext(
        (min(context.ideal[0], base.evaluation.flow), min(context.ideal[1], base.evaluation.bill)),
        context.maximum,
    )
    base_scalar = frozen.scalar(base, weight)
    common_seconds = perf_counter() - tick
    order = list(GROUPS)
    RNGManager(seed).stream("D1H:policy_order").shuffle(order)
    records = {}
    for group in order:
        begin = perf_counter()
        extract_seconds = projection_seconds = construction_seconds = exact_seconds = 0.0
        nonzero = 0
        intent: tuple[float, ...] = ()
        transfer = False
        if group != "ALWAYS_A7":
            tick = perf_counter()
            intent = extract_intent(problem, source)
            nonzero = sum(w > TOL.time for w in intent)
            transfer = (
                gate_transfer(source.genotype, target, intent)
                if group == "CONDITIONAL"
                else random_transfer and nonzero > 0
            )
            extract_seconds = perf_counter() - tick
        route = "TRUE" if transfer else "A7"
        proposals: list[tuple[Candidate, bool]] = []
        attempts = failures = duplicates = 0
        seen = {base.schedule.starts}
        if transfer:
            for scale in SCALES:
                if perf_counter() - begin >= quota_seconds:
                    break
                attempts += 1
                tick = perf_counter()
                schedule = project_intent(problem, target, rebuilt, intent, scale, cap_seconds)
                projection_seconds += perf_counter() - tick
                if schedule is None:
                    failures += 1
                    continue
                if schedule.starts in seen:
                    duplicates += 1
                    continue
                seen.add(schedule.starts)
                tick = perf_counter()
                evaluation = evaluate(problem, target, schedule)
                exact_seconds += perf_counter() - tick
                proposals.append(
                    (
                        Candidate(target, schedule, evaluation, "D1H:TRUE"),
                        perf_counter() - begin <= quota_seconds,
                    )
                )
        elif perf_counter() - begin < quota_seconds:
            tick = perf_counter()
            batch = ActivePack().propose(
                problem, base, 3, Config(), RNGManager(seed).stream("D1:A7")
            )
            construction_seconds = perf_counter() - tick
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
                exact_seconds += perf_counter() - tick
                proposals.append(
                    (
                        Candidate(target, proposal.schedule, evaluation, "D1H:A7"),
                        perf_counter() - begin <= quota_seconds,
                    )
                )
        eligible = [base] + [c for c, on_time in proposals if on_time and c.evaluation.feasible]
        best = min(eligible, key=lambda c: (frozen.scalar(c, weight), c.schedule.starts))
        seconds = perf_counter() - begin
        records[group] = {
            "route": route,
            "intent_nonzero": nonzero,
            "seconds": seconds,
            "quota_seconds": quota_seconds,
            "overshoot_seconds": max(0.0, seconds - quota_seconds),
            "extraction_seconds": extract_seconds,
            "projection_seconds": projection_seconds,
            "construction_seconds": construction_seconds,
            "exact_seconds": exact_seconds,
            "attempts": attempts,
            "failed_projection": failures,
            "duplicates": duplicates,
            "exact": len(proposals),
            "feasible": sum(c.evaluation.feasible for c, _ in proposals),
            "on_time_exact": sum(on_time for _, on_time in proposals),
            "objectives": best.evaluation.objectives,
            "starts": best.schedule.starts,
            "scalar": frozen.scalar(best, weight),
            "relative_gain": (base_scalar - frozen.scalar(best, weight)) / max(base_scalar, 1e-12),
            "base_is_best": best is base,
            "proposals": [
                {
                    "starts": c.schedule.starts,
                    "objectives": c.evaluation.objectives,
                    "feasible": c.evaluation.feasible,
                    "on_time": on_time,
                    "scalar": frozen.scalar(c, weight),
                    "below_frozen_ideal": any(
                        c.evaluation.objectives[m] < frozen.ideal[m] for m in range(2)
                    ),
                }
                for c, on_time in proposals
            ],
        }
    return {
        "seed": seed,
        "weight": weight,
        "context": asdict(frozen),
        "policy_order": order,
        "random_transfer": random_transfer,
        "target": asdict(target),
        "source_objectives": source.evaluation.objectives,
        "base_objectives": base.evaluation.objectives,
        "base_starts": base.schedule.starts,
        "base_scalar": base_scalar,
        "shared_seconds": common_seconds,
        "shared_exact": 2,
        "shared_ssgs": 1,
        "ms_distance": sum(a != b for a, b in zip(source.genotype.ms, target.ms)),
        "os_distance": sum(a != b for a, b in zip(source.genotype.os, target.os)),
        "groups": records,
    }
