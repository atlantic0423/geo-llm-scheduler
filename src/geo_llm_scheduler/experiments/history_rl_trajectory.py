"""Same-child transitions with explicit history and no unnecessary severity probes."""

from dataclasses import replace
from time import perf_counter
from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments.history_rl import HistoryPolicy
from geo_llm_scheduler.experiments.metrics import hypervolume
from geo_llm_scheduler.macrosearch.adaptive import AdaptiveBudget
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.active_pack import ActivePack
from geo_llm_scheduler.operators.base import Operator
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.rl.controller import Controller, reward
from geo_llm_scheduler.rl.state import PREFERENCE_LABELS, preference
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.utils.numeric import TOL, less
from geo_llm_scheduler.utils.rng import RNGManager


def archive_value(members: list[Candidate], context: NormalizationContext | None) -> float:
    """Exact 2-D union in initialization-frozen coordinates, excluding points beyond 1.1."""
    if context is None:
        raise ValueError("Missing initialization-frozen frontier reference")
    points = [context.normalize(c) for c in members]
    return hypervolume([p for p in points if p[0] <= 1.1 and p[1] <= 1.1], (1.1, 1.1))


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
    policy: HistoryPolicy,
) -> tuple[Candidate, list[dict]]:
    """Reset history at START; terminalize L5 or two consecutive unproductive steps."""
    if config.budget_policy != "fixed" or config.polish:
        raise ValueError("History protocol requires fixed budget and episode-external Polish")
    current = child
    learner = policy.learner(controller, config)
    operators: dict[int, Operator] = {a: StructuralOperator(a) for a in range(1, 7)}
    operators.update({7: ActivePack(), 8: PeakCoalition()})
    if operator_overrides is not None:
        if set(operator_overrides) - {8}:
            raise ValueError("Research overrides are limited to A8")
        operators.update(operator_overrides)
    encoding_start = perf_counter()
    state = policy.state(index, config.population)
    state_seconds = perf_counter() - encoding_start
    previous_action = previous_outcome = failures = 0
    previous_reward = 0.0
    records = []
    for step in range(config.rl_steps):
        start = perf_counter()
        action, explore = learner.select(state, progress, streams.stream("qlearning"))
        context = NormalizationContext(gateway.ideal, maximum)
        # Match the fixed-budget reference's named-stream lifecycle without a draw.
        streams.stream("budget")
        budget = config.fixed_budget
        construction_start = perf_counter()
        batch = operators[action].propose(
            gateway.problem, current, budget, config, streams.stream(f"A{action}")
        )
        construction_seconds = perf_counter() - construction_start
        gateway.seconds[f"construction:A{action}"] += construction_seconds
        gateway.counts[f"construction_attempts:A{action}"] += batch.attempts
        gateway.counts[f"proposals:A{action}"] += len(batch.proposals)
        insertions = gateway.archive.insertions
        archive_before = gateway.archive.identity_keys()
        track_objectives = "STOP2" in policy.arm
        objectives_before = (
            {c.evaluation.objectives for c in gateway.archive.members}
            if track_objectives
            else set()
        )
        track_hv = "FRONTIER" in policy.arm
        fixed_context = policy.initial_context
        if track_hv and fixed_context is None:
            raise ValueError("Missing initialization-frozen frontier reference")
        hv_before = archive_value(gateway.archive.members, fixed_context) if track_hv else 0.0

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
            structural_observer=observe_target if observer is not None else None,
        )
        prior = current
        before = result.context.scalar(current, weights[index])
        if result.best is not None and less(
            result.context.scalar(result.best, weights[index]), before, TOL.scalar
        ):
            current = result.best
            if action in (7, 8):
                current = replace(
                    current,
                    schedule=refresh_diagnostics(
                        gateway.problem, current.genotype, current.schedule
                    ),
                )
        after = result.context.scalar(current, weights[index])
        quality = reward(before, after, result.feasible_count)
        hv_gain = (
            max(0.0, archive_value(gateway.archive.members, fixed_context) - hv_before)
            if track_hv
            else 0.0
        )
        outcome = policy.outcome(quality, result.feasible_count, hv_gain)
        signal = policy.signal(quality, result.feasible_count, hv_gain)
        new_objectives = (
            {c.evaluation.objectives for c in gateway.archive.members} - objectives_before
            if track_objectives
            else set()
        )
        failures = failures + 1 if current is prior and not new_objectives else 0
        early_stop = track_objectives and failures >= 2
        terminal = early_stop or step == config.rl_steps - 1
        encoding_start = perf_counter()
        next_state = policy.state(index, config.population, action, outcome)
        state_seconds += perf_counter() - encoding_start
        gateway.seconds["history_state"] += state_seconds
        learner.update(state, action, signal, next_state, terminal=terminal)
        seconds = perf_counter() - start
        records.append(
            dict(
                step=step,
                state=state,
                next_state=next_state,
                action=action,
                explore=explore,
                previous_action=previous_action,
                previous_outcome=previous_outcome,
                previous_reward=previous_reward,
                outcome=outcome,
                reward=signal,
                quality_reward=quality,
                preference=PREFERENCE_LABELS[preference(index, config.population)],
                dominant_condition="History",
                search_progress="History",
                progress=progress,
                terminal=terminal,
                stop_reason="two_unproductive" if early_stop else "horizon" if terminal else None,
                available_actions=learner.available_actions(state),
                budget=budget,
                exact_evaluations=result.exact_evaluations,
                feasible=result.feasible_count,
                effective=result.effective,
                proposals=len(batch.proposals),
                attempts=result.construction_attempts,
                candidate_objectives=[c.evaluation.objectives for c in result.evaluated],
                candidate_feasible=[c.evaluation.feasible for c in result.evaluated],
                comparison_ideal=result.context.ideal,
                comparison_maximum=result.context.maximum,
                weight=weights[index],
                archive_hv_gain=hv_gain,
                archive_insertions=gateway.archive.insertions - insertions,
                archive_net_retained=len(gateway.archive.identity_keys() - archive_before),
                scalar_before=before,
                scalar_after=after,
                accepted=current is not prior,
                delta_flow=current.evaluation.flow - prior.evaluation.flow,
                delta_bill=current.evaluation.bill - prior.evaluation.bill,
                action_seconds=seconds,
                seconds=seconds,
                construction_seconds=construction_seconds,
                state_seconds=state_seconds,
                construction=batch.diagnostics,
                operator_instrumentation=batch.instrumentation,
            )
        )
        if observer is not None:
            observer("rl_step", records[-1])
        if terminal:
            break
        state, previous_action, previous_outcome, previous_reward = (
            next_state,
            action,
            outcome,
            signal,
        )
        state_seconds = 0.0
    return current, records
