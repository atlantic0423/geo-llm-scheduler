"""Opt-in equivalent COMMON reuse and a separate demand-envelope extension."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from random import Random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance, Schedule
from geo_llm_scheduler.lsv2.billing import difference, merge
from geo_llm_scheduler.lsv2.flowsafe import FlowSafeBillMoves, _PreparedFlow
from geo_llm_scheduler.lsv2.group_polish import _Search
from geo_llm_scheduler.lsv2.polish_selection import PhysicalKey, physical_key
from geo_llm_scheduler.lsv2.shift_calendar import ShiftCalendar
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.utils.numeric import TOL


def _peak_switches(left: list[float], right: list[float]) -> list[float]:
    """Interior upper-envelope knots of window lines on the unit interval."""
    # Equal slopes retain only the highest intercept. Build the upper hull in
    # slope order, O(W log W), rather than enumerating every pair of windows.
    by_slope: dict[float, float] = {}
    for a, b in zip(left, right):
        slope = b - a
        by_slope[slope] = max(a, by_slope.get(slope, -float("inf")))
    hull: list[tuple[float, float, float]] = []
    for slope, intercept in sorted(by_slope.items()):
        crossing = -float("inf")
        while hull:
            old_slope, old_intercept, begin = hull[-1]
            crossing = (old_intercept - intercept) / (slope - old_slope)
            if crossing > begin:
                break
            hull.pop()
        hull.append((slope, intercept, crossing if hull else -float("inf")))
    return [start for _, _, start in hull if 0 < start < 1]


class _PeakSearch(_Search):
    def common(self, group: tuple[int, ...]) -> None:
        super().common(group)
        self._peaks(group)

    def _peaks(self, group: tuple[int, ...]) -> None:
        machine = self.ms[group[0]]
        server = self.problem.instances[machine]
        if self.problem.regions[server.region].demand_rate == 0:
            return
        # Original resource/union/window event alignments partition the
        # displacement axis into intervals with affine window powers.
        knots = [0.0, *self._shifts(group)]
        windows: dict[float, list[float]] = {}
        for low, high in zip(knots, knots[1:]):
            if high - low <= TOL.time:
                continue
            self.stats["peak_intervals"] += 1
            # Resource activity order is constant on each OPEN event interval.
            if not self._fits(group, (low + high) / 2):
                self.stats["peak_infeasible_intervals"] += 1
                continue
            for delta in (low, high):
                if delta in windows:
                    continue
                edits = {
                    o: (
                        self.starts[o] + delta,
                        self.starts[o] + delta + self.problem.profile(o).duration,
                    )
                    for o in group
                }
                union = merge([edits.get(o, (a, b)) for o, a, b in self.view.items[machine]])
                values = list(self.base.windows[server.region])
                for a, b, sign in difference(self.base.unions[machine], union):
                    self.base._add_segment(
                        server.region, a, b, sign * (server.active_kw - server.idle_kw), values
                    )
                windows[delta] = values
                self.stats["peak_window_evaluations"] += 1
            for fraction in _peak_switches(windows[low], windows[high]):
                delta = low + fraction * (high - low)
                if not low + TOL.time < delta < high - TOL.time:
                    continue
                self.stats["peak_positions"] += 1
                self.stats["candidate_positions"] += 1
                if not self._fits(group, delta):
                    self.stats["resource_rejected"] += 1
                    continue
                self.offer({o: self.starts[o] + delta for o in group}, "peak_switch")


class _CalendarSearch(_PeakSearch):
    def __init__(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        prepared: _PreparedFlow,
        peak_switch: bool,
    ) -> None:
        super().__init__(problem, incumbent, prepared)
        self.peak_switch = peak_switch
        self.calendars: dict[tuple[int, ...], ShiftCalendar] = {}
        self.stats.update(
            calendar_builds=0, calendar_pairs=0, calendar_rejected=0, reference_resource_checks=0
        )

    def common(self, group: tuple[int, ...]) -> None:
        for delta in self._shifts(group):
            self.stats["candidate_positions"] += 1
            if not self._fits(group, delta):
                self.stats["resource_rejected"] += 1
                continue
            self.offer({o: self.starts[o] + delta for o in group}, "common")
        if self.peak_switch:
            self._peaks(group)

    def _fits(self, group: tuple[int, ...], delta: float) -> bool:
        if group not in self.calendars:
            machine = self.ms[group[0]]
            moving = [
                (o, self.starts[o], self.starts[o] + self.problem.profile(o).duration)
                for o in group
            ]
            calendar = ShiftCalendar(
                self.problem,
                machine,
                moving,
                self.view.intervals(machine, frozenset(group)),
                min(self.upper[o] - self.starts[o] for o in group),
            )
            self.calendars[group] = calendar
            self.stats["calendar_builds"] += 1
            self.stats["calendar_pairs"] += calendar.pairs
        if self.calendars[group].rejects(delta):
            self.stats["calendar_rejected"] += 1
            return False
        self.stats["reference_resource_checks"] += 1
        return super()._fits(group, delta)


@dataclass
class _Entry:
    prepared: _PreparedFlow
    group_pool: dict[tuple[float, ...], tuple[float, str]]
    stats: dict[str, int]
    groups: int


@dataclass
class ReusedCommonPolish:
    """Reuse only current-pool physical inputs; still run exact/archive on outputs.

    Cache scope includes the immutable instance identity, search budget, target
    mode, peak-extension mode and operator configuration. OS is intentionally
    excluded: all timing proposals receive the CURRENT genotype. No unsuccessful
    status is inferred from a deadline or from Archive rejection.
    """

    peak_switch: bool = False
    all_targets: bool = False
    resource_calendar: bool = False
    _entries: dict[PhysicalKey, _Entry] = field(default_factory=dict, init=False)
    _scope: tuple | None = field(default=None, init=False)

    def retain(self, candidates: Sequence[Candidate]) -> None:
        """Discard entries absent from the current population-plus-endpoint pool."""
        keys = {physical_key(c) for c in candidates}
        self._entries = {k: v for k, v in self._entries.items() if k in keys}

    def known_empty(
        self, problem: ProblemInstance, candidate: Candidate, budget: int, config: Config
    ) -> bool:
        """Recognize only a complete no-first-move and empty-group certificate.

        Scope must match the immutable problem identity and every search setting.
        Partial execution, Archive rejection and zero HV are never certificates.
        """
        scope = (
            problem,
            budget,
            self.all_targets,
            self.peak_switch,
            config,
            self.resource_calendar,
        )
        if self._scope is None or self._scope[0] is not problem or self._scope[1:] != scope[1:]:
            return False
        entry = self._entries.get(physical_key(candidate))
        return (
            entry is not None and not entry.group_pool and entry.prepared.empty_attempts is not None
        )

    def propose(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        config: Config,
        rng: Random,
    ) -> ProposalBatch:
        """Preserve COMMON proposals/RNG, or extend positions only when opted in."""
        if type(budget) is not int or budget < 1:
            raise ValueError("Polish candidate budget must be a positive integer")
        scope = (
            problem,
            budget,
            self.all_targets,
            self.peak_switch,
            config,
            self.resource_calendar,
        )
        # Identity protects against equal-looking instances with changed profiles.
        if self._scope is None or self._scope[0] is not problem or self._scope[1:] != scope[1:]:
            self._entries.clear()
            self._scope = scope
        key = physical_key(incumbent)
        entry = self._entries.get(key)
        hit = entry is not None
        single_op = FlowSafeBillMoves(right_only=True, all_targets=self.all_targets)
        if entry is None:
            prepared = single_op._prepare(problem, incumbent)
            ctx = (
                _CalendarSearch(problem, incumbent, prepared, self.peak_switch)
                if self.resource_calendar
                else (_PeakSearch if self.peak_switch else _Search)(problem, incumbent, prepared)
            )
            ctx.stats.update(
                peak_intervals=0,
                peak_infeasible_intervals=0,
                peak_window_evaluations=0,
                peak_positions=0,
            )
            groups = ctx.groups(score_order=False)
            for group in groups:
                ctx.common(group)
            # Top B of the group-only pool suffice for the top B of its union
            # with any single-branch candidates; no candidate-score approximation.
            pool = dict(sorted(ctx.pool.items(), key=lambda item: (item[1][0], item[0]))[:budget])
            entry = _Entry(prepared, pool, ctx.stats, len(groups))
            self._entries[key] = entry
        single = single_op._propose_prepared(
            problem, incumbent, budget, rng, entry.prepared, reuse_empty=True
        )
        # Single-first preserves duplicate-source labels and current genotype.
        ctx = _Search(problem, incumbent, entry.prepared)
        for proposal in single.proposals:
            assert proposal.schedule is not None
            ctx.offer({o: proposal.schedule.starts[o] for o in proposal.selected}, "single")
        for starts, value in entry.group_pool.items():
            ctx.pool.setdefault(starts, value)
        ranked = sorted(ctx.pool.items(), key=lambda item: (item[1][0], item[0]))[:budget]
        stats = {k: 0 if hit else v for k, v in entry.stats.items()}
        for name in (
            "bill_checks",
            "improving",
            "nonimproving",
            "union_unchanged",
            "duplicate_candidates",
        ):
            stats[name] += ctx.stats[name]
        batch = ProposalBatch(attempts=single.attempts + stats["bill_checks"])
        for starts, _ in ranked:
            selected = tuple(o for o, (a, b) in enumerate(zip(starts, key[1])) if a != b)
            batch.proposals.append(
                Proposal(incumbent.genotype, Schedule(starts), selected=selected)
            )
        batch.diagnostics = dict(
            mode="common_peak" if self.peak_switch else "common_fast",
            single=single.diagnostics,
            groups=entry.groups,
            eligible=len(ctx.upper),
            proposals=len(batch.proposals),
            proposal_sources=[v[1] for _, v in ranked],
            incremental_deltas=[v[0] for _, v in ranked],
            group_cache_hit=hit,
            cached_inputs=len(self._entries),
            reused_group_work=entry.stats if hit else {},
            **stats,
        )
        return batch
