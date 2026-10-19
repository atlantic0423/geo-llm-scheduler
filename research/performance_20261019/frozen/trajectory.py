"""Same-offspring RL trajectory followed by one separately-accounted polish."""

from dataclasses import replace
from time import perf_counter
from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.diagnostics.severity import severities
from geo_llm_scheduler.domain.models import Candidate, Genotype
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.macrosearch.adaptive import AdaptiveBudget
from geo_llm_scheduler.macrosearch.budget import choose_budget
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.active_pack import ActivePack
from geo_llm_scheduler.operators.base import Operator
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.operators.right_shift import PolishStats, polish
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.rl.controller import Controller, reward
from geo_llm_scheduler.rl.state import decode, extract
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.utils.numeric import TOL, identity, less
from geo_llm_scheduler.utils.rng import RNGManager


def improve(
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
    population: list[Candidate] | None = None,
    adaptive: AdaptiveBudget | None = None,
    *,
    observer: Callable[[str, dict], None] | None = None,
    operator_overrides: dict[int, Operator] | None = None,
) -> tuple[Candidate, list[dict]]:
    """Run L_RL decisions; each batch restarts from that step's frozen incumbent."""
    problem = gateway.problem
    current = child
    records = []
    operators: dict[int, Operator] = {a: StructuralOperator(a) for a in range(1, 7)}
    operators.update({7: ActivePack(), 8: PeakCoalition()})
    if operator_overrides is not None:
        if set(operator_overrides) - {8}:
            raise ValueError("Research overrides are limited to A8")
        operators.update(operator_overrides)
    values = severities(problem, current)
    for step in range(config.rl_steps):
        start = perf_counter()
        state_values = values
        state = extract(index, state_values, stagnant_count, config)
        available_actions = controller.available_actions(state)
        action, explore = controller.select(state, progress, streams.stream("qlearning"))
        context = NormalizationContext(gateway.ideal, maximum)
        budget_start = perf_counter()
        budget_details: dict[str, float | int] = {}
        if config.budget_policy in ("coverage_v2", "severity_v2", "sequential"):
            if adaptive is None or population is None:
                raise ValueError("E15 budget policy needs per-run controller and population")
            budget, budget_details = adaptive.choose(
                action,
                state_values,
                stagnant_count,
                index,
                population,
                gateway.archive.members,
                weights,
                context,
                config,
            )
        else:
            budget = choose_budget(
                action,
                state_values,
                stagnant_count,
                index,
                gateway.archive.members,
                weights,
                context,
                config,
                streams.stream("budget"),
            )
        budget_seconds = perf_counter() - budget_start
        gateway.seconds["budget"] += budget_seconds
        if adaptive is not None and config.budget_policy == "coverage_v2":
            adaptive.cpu_seconds += budget_seconds
        construction_start = perf_counter()
        batch = operators[action].propose(
            problem, current, budget, config, streams.stream(f"A{action}")
        )
        construction_seconds = perf_counter() - construction_start
        gateway.seconds[f"construction:A{action}"] += construction_seconds
        gateway.counts[f"construction_attempts:A{action}"] += batch.attempts
        gateway.counts[f"proposals:A{action}"] += len(batch.proposals)
        insertions = gateway.archive.insertions
        archive_before = {identity(candidate) for candidate in gateway.archive.members}

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
                        "weight": weights[index],
                    },
                )

        result = execute(
            batch,
            current,
            budget,
            weights[index],
            context,
            gateway,
            f"A{action}",
            sequential=config.budget_policy == "sequential",
            structural_observer=observe_target if observer is not None else None,
        )
        archive_after = {identity(candidate) for candidate in gateway.archive.members}
        before = result.context.scalar(current, weights[index])
        prior = current
        if result.best is not None and less(
            result.context.scalar(result.best, weights[index]), before, TOL.scalar
        ):
            current = result.best
            if action in (7, 8):
                current = replace(
                    current,
                    schedule=refresh_diagnostics(problem, current.genotype, current.schedule),
                )
        after = result.context.scalar(current, weights[index])
        signal = reward(before, after, result.feasible_count)
        if config.polish and config.polish_mode == "step":
            polish_start = perf_counter()
            gateway.counts["polish_calls"] += 1
            stats = PolishStats()
            current = polish(problem, current, gateway, stats)
            gateway.counts["polish_candidates"] += stats.candidates
            gateway.counts["polish_exact"] += stats.exact
            gateway.counts["polish_accepted"] += stats.accepted
            gateway.seconds["polish"] += perf_counter() - polish_start
        values = severities(problem, current)
        next_state = extract(index, values, stagnant_count, config)
        # Each short trajectory ends an episode; keep the run-level Q table.
        # Polish is outside the episode and never contributes bootstrap/reward.
        controller.update(state, action, signal, next_state, terminal=step == config.rl_steps - 1)
        preference_label, condition_label, progress_label = decode(state, config.rl_state_policy)
        records.append(
            {
                "step": step,
                "state": state,
                "available_actions": available_actions,
                "mask_policy": config.action_mask_policy,
                "preference": preference_label,
                "dominant_condition": condition_label,
                "search_progress": progress_label,
                "progress": progress,
                "severities": state_values,
                "next_state": next_state,
                "action": action,
                "explore": explore,
                "reward": signal,
                "budget": budget,
                "budget_details": budget_details,
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
                "accepted": current is not prior,
                "delta_flow": current.evaluation.flow - prior.evaluation.flow,
                "delta_bill": current.evaluation.bill - prior.evaluation.bill,
                "seconds": perf_counter() - start,
                "construction_seconds": construction_seconds,
                "budget_seconds": budget_seconds,
                "construction": batch.diagnostics,
                "operator_instrumentation": batch.instrumentation,
            }
        )
    if config.polish and config.polish_mode == "trajectory":
        start = perf_counter()
        gateway.counts["polish_calls"] += 1
        stats = PolishStats()
        current = polish(problem, current, gateway, stats)
        gateway.counts["polish_candidates"] += stats.candidates
        gateway.counts["polish_exact"] += stats.exact
        gateway.counts["polish_accepted"] += stats.accepted
        gateway.seconds["polish"] += perf_counter() - start
    return current, records
