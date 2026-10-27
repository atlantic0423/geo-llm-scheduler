"""Research-only paired continuations from trusted LSV2 decision snapshots."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.lsv2.frontier import REF, SeedSampler
from geo_llm_scheduler.lsv2.search import LSV2Search, segment_of
from geo_llm_scheduler.scheduling.timing import legal_bounds
from geo_llm_scheduler.utils.numeric import identity
from geo_llm_scheduler.utils.rng import RNGManager


class ActionUnavailableError(ValueError):
    """A requested action is excluded by the original action set or A8 gate."""


@dataclass
class ReplayContext:
    """Isolated mutable search state, with a fixed external evaluation frame."""

    snapshot: dict[str, Any]
    search: LSV2Search
    gateway: EvaluationGateway
    current: Candidate
    candidates: list[Candidate]


def restore(snapshot: dict[str, Any]) -> ReplayContext:
    """Deep-clone a trusted snapshot without mutating its baseline objects."""
    s = copy.deepcopy(snapshot)
    archive = Archive()
    archive.members = s["archive"]
    archive.attempts = s["archive_attempts"]
    archive.insertions = s["archive_insertions"]
    archive.peak_size = s["archive_peak"]
    gateway = EvaluationGateway(
        s["problem"], archive, decode_cache_size=s["engine"].decode_cache_size
    )
    gateway.ideal = s["ideal"]
    gateway._decoded = s["cache"]
    search = LSV2Search(replace(s["config"], selector="bandit"), s["engine"], s["operators"])
    search.gate = s["gate"]
    search.frame_id = s["frame_id"]
    search.ls_seconds = s["ls_seconds"]
    candidates: list[Candidate] = []
    gateway.candidate_sink = candidates.append
    return ReplayContext(s, search, gateway, s["current"], candidates)


def fingerprint(context: ReplayContext) -> str:
    """Hash solution-level state for paired path and clone-isolation checks."""
    data = (
        identity(context.current),
        sorted(identity(c) for c in context.gateway.archive.members),
        [identity(c) for c in context.snapshot["population"]],
        context.gateway.ideal,
    )
    return hashlib.sha256(repr(data).encode()).hexdigest()


def force_step(context: ReplayContext, action: str, seed: int) -> dict[str, Any]:
    """Execute the original one-step transition; unavailable actions raise ValueError."""
    s, search, gateway = context.snapshot, context.search, context.gateway
    segment = segment_of(s["frame"].normalize(context.current.evaluation.objectives))
    gate = copy.deepcopy(search.gate)
    if action not in search.config.operators or (action == "A8" and not gate.available(segment)):
        raise ActionUnavailableError("Action unavailable in this branch")

    def choose(segment: int, operators: Sequence[str], rng: Any) -> tuple[str, dict]:
        assert action in operators
        return action, {"rule": "paired_forced"}

    search.bandit.choose = choose  # type: ignore[method-assign]
    before = list(gateway.archive.members)
    old_current = identity(context.current)
    old_population = [identity(c) for c in s["population"]]
    tick = perf_counter()
    context.current, _ = search._step(
        s["population"], gateway, RNGManager(seed), s["frame"], context.current, s["box"], s["info"]
    )
    wall = perf_counter() - tick
    row = dict(search.steps[-1])
    old_keys = {identity(c) for c in before}
    new = [c for c in gateway.archive.members if identity(c) not in old_keys]
    points = {c.evaluation.objectives for c in before}
    row.update(
        charged_seconds=wall,
        current_changed=identity(context.current) != old_current,
        population_changed=sum(a != identity(b) for a, b in zip(old_population, s["population"])),
        archive_added=len(new),
        added_equal_objective=sum(c.evaluation.objectives in points for c in new),
        added_outside_reference=sum(
            any(x > REF for x in s["frame"].normalize(c.evaluation.objectives)) for c in new
        ),
        state_sha256=fingerprint(context),
    )
    return row


def audit_candidates(context: ReplayContext) -> int:
    """Recompute all generated candidates independently, outside search timing."""
    for c in context.candidates:
        result = evaluate(context.gateway.problem, c.genotype, c.schedule)
        assert result.feasible == c.evaluation.feasible
        assert all(
            math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)
            for a, b in zip(result.objectives, c.evaluation.objectives)
        )
    return len(context.candidates)


def opportunity_features(snapshot: dict[str, Any]) -> dict[str, float]:
    """Measure pre-action wait and precedence-window proxies, not exact move feasibility."""
    tick = perf_counter()
    p, c = snapshot["problem"], snapshot["current"]
    work = sum(p.profile(o).duration for o in range(p.operation_count))
    widths = [
        max(0.0, hi - lo)
        for o in range(0, p.operation_count, 2)
        for lo, hi in [legal_bounds(p, c.genotype, c.schedule, o)]
    ]
    norm = snapshot["frame"].normalize(c.evaluation.objectives)
    return dict(
        normalized_flow=norm[0],
        normalized_bill=norm[1],
        resource_wait_fraction=sum(c.schedule.resource_wait) / max(work, 1e-12),
        prefill_window_fraction=sum(w > 1e-9 for w in widths) / max(1, len(widths)),
        prefill_window_width_per_work=sum(widths) / max(work, 1e-12),
        seconds=perf_counter() - tick,
    )


def pair(
    snapshot: dict[str, Any], first: str, second: str, seed: int, follow_seed: int
) -> dict[str, Any]:
    """Compare two original transitions with a paired second-action random stream."""
    context = restore(snapshot)
    initial = snapshot["frame"].hv([c.evaluation.objectives for c in snapshot["archive"]])
    one = force_step(context, first, seed)
    try:
        two = force_step(context, second, follow_seed)
    except ActionUnavailableError:
        return dict(
            first=first,
            second=second,
            eligible=False,
            first_step=one,
            fresh_exact=audit_candidates(context),
        )
    final = snapshot["frame"].hv([c.evaluation.objectives for c in context.gateway.archive.members])
    return dict(
        first=first,
        second=second,
        eligible=True,
        first_step=one,
        second_step=two,
        terminal_gain=final - initial,
        initial_hv=initial,
        terminal_objectives=[c.evaluation.objectives for c in context.gateway.archive.members],
        charged_seconds=one["charged_seconds"] + two["charged_seconds"],
        state_sha256=fingerprint(context),
        fresh_exact=audit_candidates(context),
    )


def continue_search(
    snapshot: dict[str, Any],
    first: str,
    seed: int,
    follow_seed: int,
    seconds: float,
    *,
    offspring_limit: int | None = None,
) -> dict[str, Any]:
    """Apply a prefix then resume the original engine under explicitly reset control history.

    Prefix time is charged. Clone and independent exact audit are outside timing.
    ``offspring_limit`` is a deterministic engine-boundary test seam, not the formal budget.
    """
    if seconds <= 0:
        raise ValueError("Positive total continuation time required")
    context = restore(snapshot)
    s = context.snapshot
    initial = s["frame"].hv([c.evaluation.objectives for c in context.gateway.archive.members])
    one = None if first == "NOOP" else force_step(context, first, seed)
    charged = 0.0 if one is None else one["charged_seconds"]
    cfg = replace(s["engine"], seed=follow_seed, seconds=max(1e-12, seconds - charged))
    assert cfg.method != "full" and cfg.replacement_policy == "permuted"
    search = LSV2Search(
        replace(s["config"], selector="uniform"), replace(cfg, seconds=seconds), s["operators"]
    )
    search.gate = context.search.gate
    search.frame = s["frame"]
    search.frame_id = s["frame_id"]
    search.ls_seconds = charged
    observed: dict[str, Any] = {}

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        gateway.archive = context.gateway.archive
        gateway.ideal = context.gateway.ideal
        gateway._decoded = context.gateway._decoded
        context.gateway = gateway
        return s["population"]

    def interleave(
        population: list[Candidate], gateway: EvaluationGateway, streams: RNGManager, elapsed: Any
    ) -> None:
        if offspring_limit is None:
            if search.sampler is None:
                search.search_start = 0.0
                search.sampler = SeedSampler(
                    search.config.seed_rule,
                    streams.stream("lsv2:seed"),
                    discount=search.config.seed_discount,
                    shares=search.config.flow_end_shares,
                )
            search(population, gateway, streams, lambda: elapsed() + charged)

    class LimitReached(Exception):
        pass

    def observer(event: str, data: dict) -> None:
        if event == "candidate":
            context.candidates.append(data["candidate"])
        elif event == "offspring":
            observed.update(offspring=data["processed"], population=data["population"])
            if offspring_limit is not None and data["processed"] >= offspring_limit:
                raise LimitReached

    tick = perf_counter()
    reason = "prefix_exhausted"
    if charged < seconds:
        try:
            result = run(
                s["problem"],
                cfg,
                initializer=initialize,
                observer=observer,
                retain_trace=False,
                interleave=interleave,
            )
            reason = result.termination_reason
        except LimitReached:
            reason = "test_offspring_limit"
    total = charged + perf_counter() - tick
    final = s["frame"].hv([c.evaluation.objectives for c in context.gateway.archive.members])
    return dict(
        first=first,
        first_step=one,
        terminal_gain=final - initial,
        initial_hv=initial,
        terminal_objectives=[c.evaluation.objectives for c in context.gateway.archive.members],
        charged_seconds=total,
        requested_seconds=seconds,
        overshoot=max(0.0, total - seconds),
        termination=reason,
        offspring=observed.get("offspring", 0),
        tail_ls_steps=len(search.steps),
        tail_trajectories=search.trajectories,
        tail_exact=context.gateway.counts["exact"],
        state_sha256=fingerprint(context),
        fresh_exact=audit_candidates(context),
    )
