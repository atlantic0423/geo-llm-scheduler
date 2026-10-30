"""Concurrent Prefill right shift using event-cover groups and capacity-first search.

All scoring is incremental, all emitted proposals still require the exact gateway.
Groups are natural active sets, not arbitrary subsets or synchronized batches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance, Schedule
from geo_llm_scheduler.lsv2.billing import WINDOW, PowerProfile, merge
from geo_llm_scheduler.lsv2.flowsafe import prefill_times
from geo_llm_scheduler.lsv2.polish import FastRightShiftPolish
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.scheduling.resources import ResourceView, _fits_intervals
from geo_llm_scheduler.scheduling.timing import legal_bounds
from geo_llm_scheduler.utils.numeric import TOL, close, less

Intervals = list[tuple[int, float, float]]


@dataclass(frozen=True)
class ConcurrentRightShiftPolish:
    """Extend the old single-target proposals by common or coordinated group moves.

    The old partial-target path runs unchanged. Natural cover groups are fully
    scanned, without a group-size threshold. Incremental ranking emits at most
    ``budget`` complete proposals; none is exact-evaluated inside this generator.
    """

    coordinated: bool = False
    all_targets: bool = False

    def propose(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        config: Config,
        rng: Random,
    ) -> ProposalBatch:
        """Return right-only, Flow-invariant timing candidates from one input."""
        single = FastRightShiftPolish(self.all_targets).propose(
            problem, incumbent, budget, config, rng
        )
        context = _Search(problem, incumbent)
        for proposal in single.proposals:
            assert proposal.schedule is not None
            context.offer(
                dict((o, proposal.schedule.starts[o]) for o in proposal.selected), "single"
            )
        groups = context.groups()
        for group in groups:
            context.common(group)
            if self.coordinated:
                order = sorted(group, key=lambda o: (context.upper[o] - context.starts[o], o))
                orders = [tuple(order)]
                if budget > 1:
                    orders.append(tuple(reversed(order)))
                for _ in range(max(0, budget - 2)):
                    shuffled = list(group)
                    rng.shuffle(shuffled)
                    orders.append(tuple(shuffled))
                for sequence in dict.fromkeys(orders):
                    context.insert(group, sequence)
        ranked = sorted(context.pool.items(), key=lambda item: (item[1][0], item[0]))[:budget]
        batch = ProposalBatch(attempts=single.attempts + context.stats["bill_checks"])
        for starts, (_, source) in ranked:
            selected = tuple(o for o, (a, b) in enumerate(zip(starts, context.starts)) if a != b)
            batch.proposals.append(
                Proposal(incumbent.genotype, Schedule(starts), selected=selected)
            )
        batch.diagnostics = dict(
            mode="coordinated" if self.coordinated else "common",
            single=single.diagnostics,
            groups=len(groups),
            eligible=len(context.upper),
            proposals=len(batch.proposals),
            proposal_sources=[source for _, (_, source) in ranked],
            incremental_deltas=[delta for _, (delta, _) in ranked],
            **context.stats,
        )
        return batch


class _Search:
    def __init__(self, problem: ProblemInstance, incumbent: Candidate) -> None:
        self.problem, self.incumbent = problem, incumbent
        self.starts, self.ms = incumbent.schedule.starts, incumbent.genotype.ms
        self.base = PowerProfile(problem, self.ms, self.starts)
        self.view = ResourceView(problem, incumbent.genotype, incumbent.schedule)
        self.upper = {
            o: legal_bounds(problem, incumbent.genotype, incumbent.schedule, o)[1]
            for o in range(0, problem.operation_count, 2)
        }
        self.upper = {o: hi for o, hi in self.upper.items() if less(self.starts[o], hi, TOL.time)}
        self.pool: dict[tuple[float, ...], tuple[float, str]] = {}
        self.stats = dict(
            candidate_positions=0,
            resource_rejected=0,
            union_unchanged=0,
            bill_checks=0,
            nonimproving=0,
            improving=0,
            insert_failed=0,
            group_decode_blocked=0,
            group_slack_blocked=0,
            duplicate_candidates=0,
        )

    def groups(self) -> list[tuple[int, ...]]:
        groups: set[tuple[int, ...]] = set()
        for m, items in enumerate(self.view.items):
            # Event sweep makes each distinct simultaneous cover set only once.
            events: dict[float, tuple[list[int], list[int]]] = {}
            for o, a, b in items:
                if a < b:
                    events.setdefault(a, ([], []))[1].append(o)
                    events.setdefault(b, ([], []))[0].append(o)
            active: set[int] = set()
            for t in sorted(events):
                ends, begins = events[t]
                active.difference_update(ends)
                active.update(begins)
                if len(active) < 2:
                    continue
                if any(o % 2 for o in active):
                    self.stats["group_decode_blocked"] += 1
                elif not active <= self.upper.keys():
                    self.stats["group_slack_blocked"] += 1
                else:
                    groups.add(tuple(sorted(active)))
        # Whole-group removal potential, not the sum of exclusive single scores.
        return sorted(
            groups, key=lambda g: (self.base.delta_for({self.ms[g[0]]: {o: None for o in g}}), g)
        )

    def offer(self, edits: dict[int, float], source: str) -> None:
        edits = {o: t for o, t in edits.items() if t != self.starts[o]}
        if not edits:
            return
        starts = list(self.starts)
        changes: dict[int, dict[int, tuple[float, float]]] = {}
        for o, t in edits.items():
            starts[o] = t
            changes.setdefault(self.ms[o], {})[o] = (t, t + self.problem.profile(o).duration)
        key = tuple(starts)
        if key in self.pool:
            self.stats["duplicate_candidates"] += 1
            return
        if all(
            merge([changes[m].get(o, (a, b)) for o, a, b in self.view.items[m]])
            == self.base.unions[m]
            for m in changes
        ):
            self.stats["union_unchanged"] += 1
            return
        self.stats["bill_checks"] += 1
        delta = self.base.delta_for(changes)
        if delta < -TOL.cost:
            self.stats["improving"] += 1
            self.pool[key] = (delta, source)
        else:
            self.stats["nonimproving"] += 1

    def common(self, group: tuple[int, ...]) -> None:
        m = self.ms[group[0]]
        others = self.view.intervals(m, frozenset(group))
        upper = min(self.upper[o] - self.starts[o] for o in group)
        events = {t for _, a, b in others for t in (a, b)}
        events.update(
            t
            for tariff in self.problem.regions[self.problem.instances[m].region].tariffs
            for t in (tariff.start, tariff.end)
        )
        low = min(self.starts[o] for o in group)
        high = max(self.starts[o] + self.problem.profile(o).duration for o in group) + upper
        events.update(WINDOW * i for i in range(int(low // WINDOW), math.ceil(high / WINDOW) + 1))
        shifts = {upper}
        for o in group:
            shifts.update(
                t - v
                for t in events
                for v in (self.starts[o], self.starts[o] + self.problem.profile(o).duration)
            )
        for delta in sorted(d for d in shifts if TOL.time < d <= upper):
            self.stats["candidate_positions"] += 1
            moved = [
                (
                    o,
                    self.starts[o] + delta,
                    self.starts[o] + delta + self.problem.profile(o).duration,
                )
                for o in group
            ]
            merged = sorted(others + moved)
            if not all(
                _fits_intervals(self.problem, m, o, t, [v for v in merged if v[0] != o])
                for o, t, _ in moved
            ):
                self.stats["resource_rejected"] += 1
                continue
            self.offer({o: t for o, t, _ in moved}, "common")

    def insert(self, group: tuple[int, ...], order: tuple[int, ...]) -> None:
        m = self.ms[group[0]]
        others = self.view.intervals(m, frozenset(group))
        profile = self.base.copy()
        profile.apply({m: {o: None for o in group}})
        edits: dict[int, float] = {}
        for o in order:
            best: tuple[float, float] | None = None
            p = self.problem.profile(o).duration
            for t in prefill_times(self.problem, m, o, self.starts[o], self.upper[o], others):
                self.stats["candidate_positions"] += 1
                if not _fits_intervals(self.problem, m, o, t, others):
                    self.stats["resource_rejected"] += 1
                    continue
                self.stats["bill_checks"] += 1
                delta = profile.delta_for({m: {o: (t, t + p)}})
                if (
                    best is None
                    or delta < best[0] - TOL.cost
                    or (close(delta, best[0], TOL.cost) and t < best[1])
                ):
                    best = delta, t
            if best is None:
                self.stats["insert_failed"] += 1
                return
            t = best[1]
            edits[o] = t
            profile.apply({m: {o: (t, t + p)}})
            others = sorted(others + [(o, t, t + p)])
        self.offer(edits, "coordinated")
