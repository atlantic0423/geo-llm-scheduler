"""Same-offspring RL trajectory followed by one separately-accounted polish."""

from dataclasses import replace
from time import perf_counter

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.diagnostics.severity import severities
from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.macrosearch.budget import choose_budget
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.active_pack import ActivePack
from geo_llm_scheduler.operators.base import Operator
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.operators.right_shift import PolishStats, polish
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.rl.controller import Controller, reward
from geo_llm_scheduler.rl.state import extract
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
) -> tuple[Candidate, list[dict]]:
    """Run L_RL decisions; each batch restarts from that step's frozen incumbent."""
    problem = gateway.problem
    current = child
    records = []
    operators: dict[int, Operator] = {a: StructuralOperator(a) for a in range(1, 7)}
    operators.update({7: ActivePack(), 8: PeakCoalition()})
    values = severities(problem, current)
    for step in range(config.rl_steps):
        start = perf_counter()
        state = extract(index, values, stagnant_count, config)
        action, explore = controller.select(state, progress, streams.stream("qlearning"))
        context = NormalizationContext(gateway.ideal, maximum)
        budget = choose_budget(
            action,
            values,
            stagnant_count,
            index,
            gateway.archive.members,
            weights,
            context,
            config,
            streams.stream("budget"),
        )
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
        result = execute(batch, current, budget, weights[index], context, gateway, f"A{action}")
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
        values = severities(problem, current)
        next_state = extract(index, values, stagnant_count, config)
        # Each short trajectory ends an episode; keep the run-level Q table.
        # Polish is outside the episode and never contributes bootstrap/reward.
        controller.update(state, action, signal, next_state, terminal=step == config.rl_steps - 1)
        records.append(
            {
                "step": step,
                "state": state,
                "next_state": next_state,
                "action": action,
                "explore": explore,
                "reward": signal,
                "budget": budget,
                "effective": result.effective,
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
                "construction": batch.diagnostics,
            }
        )
    if config.polish:
        start = perf_counter()
        gateway.counts["polish_calls"] += 1
        stats = PolishStats()
        current = polish(problem, current, gateway, stats)
        gateway.counts["polish_candidates"] += stats.candidates
        gateway.counts["polish_exact"] += stats.exact
        gateway.counts["polish_accepted"] += stats.accepted
        gateway.seconds["polish"] += perf_counter() - start
    return current, records
