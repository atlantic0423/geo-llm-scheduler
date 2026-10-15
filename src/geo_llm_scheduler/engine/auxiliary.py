"""Bounded non-elite trajectories with online Q and whole-history output protection."""

from __future__ import annotations

from dataclasses import asdict, replace
from time import perf_counter
from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.diagnostics.severity import severities
from geo_llm_scheduler.domain.models import Candidate, Genotype
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.macrosearch.distance import trajectory_signal
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.macrosearch.trajectory_budget import ARM_SPECS, allocate
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.active_pack import ActivePack
from geo_llm_scheduler.operators.base import Operator
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.operators.right_shift import PolishStats, polish
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.rl.controller import Controller, reward
from geo_llm_scheduler.rl.state import decode, extract
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.utils.numeric import TOL, identity, less, schedule_identity
from geo_llm_scheduler.utils.rng import RNGManager


def best_history(
    history: list[Candidate], context: NormalizationContext, weight: tuple[float, float]
) -> Candidate:
    """Rescore all raw historical objectives under one output-time comparison context."""
    if not history:
        raise ValueError("History must include the feasible starting solution")
    return min(
        history, key=lambda candidate: (context.scalar(candidate, weight), identity(candidate))
    )


def improve_auxiliary(
    child: Candidate,
    index: int,
    stagnant_count: int,
    progress: float,
    weights: tuple[tuple[float, float], ...],
    maximum: tuple[float, float],
    gateway: EvaluationGateway,
    controller: Controller,
    config: Config,
    streams: RNGManager,
    *,
    observer: Callable[[str, dict], None] | None = None,
    operator_overrides: dict[int, Operator] | None = None,
) -> tuple[Candidate, list[dict]]:
    """Spend bounded breadth/depth credits and learn from the actual exploration position."""
    spec = ARM_SPECS[config.auxiliary_arm]
    problem = gateway.problem
    weight = weights[index]
    current = child
    positions = [child]
    history = [child]
    visited = {schedule_identity(child)}
    records: list[dict] = []
    spent = 0
    previous_action = 1
    operators: dict[int, Operator] = {a: StructuralOperator(a) for a in range(1, 7)}
    operators.update({7: ActivePack(), 8: PeakCoalition()})
    if operator_overrides is not None:
        if set(operator_overrides) - {8}:
            raise ValueError("Research overrides are limited to A8")
        operators.update(operator_overrides)
    values = severities(problem, current)
    for step in range(spec.steps):
        start = perf_counter()
        state_values = values
        state = extract(index, state_values, stagnant_count, config)
        action, explore = controller.select(state, progress, streams.stream("qlearning"))
        budget_start = perf_counter()
        signal, parts = trajectory_signal(
            problem, positions, spec.metric, spec.window, spec.signal, previous_action
        )
        budget = allocate(
            spec,
            signal,
            step,
            spec.credit - spent,
            config.auxiliary_cutoffs,
            config.auxiliary_random_probabilities,
            streams.stream("budget"),
        )
        budget_seconds = perf_counter() - budget_start
        gateway.seconds["budget"] += budget_seconds
        context = NormalizationContext(gateway.ideal, maximum)
        construct = perf_counter()
        batch = operators[action].propose(
            problem, current, budget, config, streams.stream(f"A{action}")
        )
        construction_seconds = perf_counter() - construct
        gateway.seconds[f"construction:A{action}"] += construction_seconds
        gateway.counts[f"construction_attempts:A{action}"] += batch.attempts
        gateway.counts[f"proposals:A{action}"] += len(batch.proposals)
        archive_before = {identity(candidate) for candidate in gateway.archive.members}
        insertions = gateway.archive.insertions

        def observe_target(target: Genotype) -> None:
            if observer is not None:
                observer(
                    "structure",
                    {
                        "source": current,
                        "target": target,
                        "path": f"A{action}",
                        "step": step,
                        "context": context,
                        "weight": weight,
                    },
                )

        result = execute(
            batch,
            current,
            budget,
            weight,
            context,
            gateway,
            f"A{action}",
            structural_observer=observe_target if observer is not None else None,
            aid=spec.aid or config.auxiliary_random_aid,
        )
        spent += result.exact_evaluations
        history.extend(item.candidate for item in result.feasible_scored)
        feasible = [
            item
            for item in result.feasible_scored
            if schedule_identity(item.candidate) != schedule_identity(current)
        ]
        next_candidate = (
            min(feasible, key=lambda item: (item.score, identity(item.candidate))).candidate
            if feasible
            else None
        )
        before = result.context.scalar(current, weight)
        prior = current
        if spec.walk:
            if next_candidate is not None:
                current = next_candidate
        elif result.best is not None and less(
            result.context.scalar(result.best, weight), before, TOL.scalar
        ):
            current = result.best
        if current is not prior and action in (7, 8):
            current = replace(
                current, schedule=refresh_diagnostics(problem, current.genotype, current.schedule)
            )
        after = result.context.scalar(current, weight)
        scalar_reward = reward(before, after, result.feasible_count)
        values = severities(problem, current)
        next_state = extract(index, values, stagnant_count, config)
        terminal = step + 1 == spec.steps or spent >= spec.credit
        controller.update(state, action, scalar_reward, next_state, terminal=terminal)
        preference, condition, stage = decode(state, config.rl_state_policy)
        key = schedule_identity(current)
        moved = key != schedule_identity(prior)
        revisit = moved and key in visited
        visited.add(key)
        positions.append(current)
        archive_after = {identity(candidate) for candidate in gateway.archive.members}
        row = {
            "step": step,
            "state": state,
            "next_state": next_state,
            "action": action,
            "explore": explore,
            "available_actions": controller.available_actions(state),
            "mask_policy": config.action_mask_policy,
            "preference": preference,
            "dominant_condition": condition,
            "search_progress": stage,
            "progress": progress,
            "severities": state_values,
            "reward": scalar_reward,
            "budget": budget,
            "budget_details": {"signal": signal, **asdict(parts), "credit_used": spent},
            "sequential": result.sequential,
            "effective": result.effective,
            "proposals": len(batch.proposals),
            "exact_evaluations": result.exact_evaluations,
            "attempts": result.construction_attempts,
            "feasible": result.feasible_count,
            "archive_insertions": gateway.archive.insertions - insertions,
            "archive_net_retained": len(archive_after - archive_before),
            "scalar_before": before,
            "scalar_after": after,
            "accepted": moved,
            "delta_flow": current.evaluation.flow - prior.evaluation.flow,
            "delta_bill": current.evaluation.bill - prior.evaluation.bill,
            "region_change": sum(
                problem.instances[a].region != problem.instances[b].region
                for a, b in zip(prior.genotype.ms, current.genotype.ms)
            )
            / max(1, problem.operation_count),
            "seconds": perf_counter() - start,
            "construction_seconds": construction_seconds,
            "budget_seconds": budget_seconds,
            "construction": batch.diagnostics,
            "operator_instrumentation": batch.instrumentation,
            "terminal": terminal,
            "revisit": revisit,
            "regression": moved and less(before, after, TOL.scalar),
            "position_objectives": current.evaluation.objectives,
            "historical_objectives": [candidate.evaluation.objectives for candidate in history],
        }
        records.append(row)
        previous_action = action
        if terminal:
            break
    output_context = NormalizationContext(gateway.ideal, maximum)
    history_tick = perf_counter()
    selected = best_history(history, output_context, weight)
    output = selected if spec.best else current
    if output.origin in ("A7", "A8"):
        output = replace(
            output, schedule=refresh_diagnostics(problem, output.genotype, output.schedule)
        )
    gateway.seconds["auxiliary_history"] += perf_counter() - history_tick
    gateway.counts["auxiliary_episodes"] += 1
    gateway.counts["auxiliary_steps"] += len(records)
    gateway.counts["auxiliary_exact"] += spent
    if observer is not None:
        observer(
            "auxiliary_episode",
            {
                "positions": positions,
                "history": history,
                "selected": selected,
                "output": output,
                "context": output_context,
                "weight": weight,
                "arm": config.auxiliary_arm,
                "records": records,
            },
        )
    if config.polish:
        start = perf_counter()
        gateway.counts["polish_calls"] += 1
        stats = PolishStats()
        output = polish(problem, output, gateway, stats)
        gateway.counts["polish_candidates"] += stats.candidates
        gateway.counts["polish_exact"] += stats.exact
        gateway.counts["polish_accepted"] += stats.accepted
        gateway.seconds["polish"] += perf_counter() - start
    return output, records
