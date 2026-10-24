"""Direct full-eligible Job/phase/path/position control; exact remains authoritative."""

from __future__ import annotations

import math
import random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.evaluation.exact import violations
from geo_llm_scheduler.experiments.jev_decisions import DecisionDeadline, DecisionSession
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.operators.peak_coalition import (
    peak_reduced,
    removal_potential,
    repair_bounds,
    repair_times,
    support_segments,
)
from geo_llm_scheduler.operators.structural import operation_order, with_order
from geo_llm_scheduler.scheduling.resources import _fits_intervals, est_for, intervals
from geo_llm_scheduler.scheduling.timing import move, packing_moves
from geo_llm_scheduler.utils.numeric import TOL
from geo_llm_scheduler.utils.rng import RNGManager


class DirectCandidates:
    """Opt-in research generator; never truncates Job eligibility by heuristic score."""

    def __init__(self, action: int, session: DecisionSession, state: dict) -> None:
        if action not in range(1, 9):
            raise ValueError("Action must be A1-A8")
        self.action, self.session, self.state = action, session, state
        self._streams: RNGManager | None = None

    def _pick(self, kind: str, values: list, rng: random.Random, **extra: object):
        options = {
            str(i): value if isinstance(value, dict) else {"value": value}
            for i, value in enumerate(values)
        }
        choice_rng = self._streams.stream(kind) if self._streams is not None else rng
        key = self.session.choose(kind, options, dict(self.state, **extra), choice_rng)
        return values[int(key)]

    def _job(self, groups: dict[int, list], rng: random.Random) -> int:
        jobs = sorted(j for j, values in groups.items() if values)
        row = self._pick(
            "job", [dict(job_id=j, eligible_options=len(groups[j])) for j in jobs], rng
        )
        return row["job_id"]

    def propose(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        config: Config,
        rng: random.Random,
    ) -> ProposalBatch:
        """Generate same-incumbent proposals with explicit original hard constraints."""
        if budget <= 0:
            return ProposalBatch()
        self._streams = RNGManager(rng.getrandbits(256))
        try:
            if self.action <= 3:
                return self._assign(problem, incumbent, budget, rng)
            if self.action in (4, 5):
                return self._insert(problem, incumbent, budget, rng)
            if self.action == 6:
                return self._destroy(problem, incumbent, budget, config, rng)
            if self.action == 7:
                return self._pack(problem, incumbent, budget, rng)
            return self._peak(problem, incumbent, budget, config, rng)
        except DecisionDeadline:
            # No unvalued proposal is called an evaluated/accepted candidate.
            return ProposalBatch(diagnostics={"deadline_during_construction": True})

    def _assign(
        self, p: ProblemInstance, x: Candidate, budget: int, rng: random.Random
    ) -> ProposalBatch:
        g = x.genotype
        groups: dict[int, list[dict]] = {}
        for j in range(len(p.jobs)):
            region = p.instances[g.ms[2 * j]].region
            options: list[dict] = []
            if self.action == 1:
                for o in (2 * j, 2 * j + 1):
                    options.extend(
                        dict(operation=o, machine=m) for m in p.eligible(o, region) if m != g.ms[o]
                    )
            else:
                options = [
                    dict(region=p.instances[a].region, prefill_instance=a, decode_instance=b)
                    for a, b in paths(p, j)
                    if (a, b) != g.ms[2 * j : 2 * j + 2]
                    and ((p.instances[a].region == region) == (self.action == 2))
                ]
            groups[j] = options
        out = ProposalBatch()
        while any(groups.values()) and len(out.proposals) < budget:
            j = self._job(groups, rng)
            parent = dict(selected_job=j)
            if self.action == 1:
                operations = sorted({v["operation"] for v in groups[j]})
                o = self._pick("phase", operations, rng, **parent)
                options = [v for v in groups[j] if v["operation"] == o]
                item = self._pick(
                    "path",
                    options,
                    rng,
                    selected_operation=o,
                    selected_region=p.instances[g.ms[o]].region,
                    **parent,
                )
                ms = list(g.ms)
                ms[o] = item["machine"]
            else:
                regions = sorted({v["region"] for v in groups[j]})
                r = self._pick("region", regions, rng, **parent)
                item = self._pick(
                    "path",
                    [v for v in groups[j] if v["region"] == r],
                    rng,
                    selected_region=r,
                    **parent,
                )
                ms = list(g.ms)
                ms[2 * j : 2 * j + 2] = item["prefill_instance"], item["decode_instance"]
            groups[j].remove(item)
            out.attempts += 1
            out.proposals.append(Proposal(Genotype(tuple(ms), g.os)))
        return out

    def _insert(
        self, p: ProblemInstance, x: Candidate, budget: int, rng: random.Random
    ) -> ProposalBatch:
        order = operation_order(x.genotype)
        indices = {o: i for i, o in enumerate(order)}
        waits = [
            max(0.0, x.schedule.starts[o] - est_for(p, x.genotype, x.schedule, o))
            for o in range(p.operation_count)
        ]
        used: set[tuple] = set()
        out = ProposalBatch()
        # Build just counts/ranges: never materialize the quadratic A5 pair space.
        for _ in range(budget):
            groups: dict[int, list] = {}
            for j in range(len(p.jobs)):
                if self.action == 4:
                    ops = []
                    for o in (2 * j, 2 * j + 1):
                        lower = indices[o - 1] + 1 if o % 2 else 0
                        if waits[o] > TOL.time and any(
                            (o, q) not in used for q in range(lower, indices[o])
                        ):
                            ops.append(o)
                    groups[j] = ops
                else:
                    a, b = indices[2 * j], indices[2 * j + 1]
                    qs = (
                        [q for q in range(a) if any((j, q, d) not in used for d in range(q + 1, b))]
                        if waits[2 * j] + waits[2 * j + 1] > TOL.time
                        else []
                    )
                    groups[j] = qs
            if not any(groups.values()):
                break
            j = self._job(groups, rng)
            parent = dict(selected_job=j)
            if self.action == 4:
                o = self._pick("phase", groups[j], rng, **parent)
                lower = indices[o - 1] + 1 if o % 2 else 0
                q = self._pick(
                    "position",
                    [q for q in range(lower, indices[o]) if (o, q) not in used],
                    rng,
                    position_kind="single_operation_earlier_OS",
                    selected_operation=o,
                    **parent,
                )
                used.add((o, q))
                new = order.copy()
                new.pop(indices[o])
                new.insert(q, o)
            else:
                q = self._pick(
                    "position", groups[j], rng, position_kind="joint_Prefill_OS", **parent
                )
                d = self._pick(
                    "position",
                    [d for d in range(q + 1, indices[2 * j + 1]) if (j, q, d) not in used],
                    rng,
                    position_kind="joint_Decode_OS_after_selected_Prefill",
                    selected_prefill_position=q,
                    **parent,
                )
                used.add((j, q, d))
                new = [o for o in order if o // 2 != j]
                new.insert(q, 2 * j)
                new.insert(d, 2 * j + 1)
            out.attempts += 1
            out.proposals.append(Proposal(with_order(x.genotype, new)))
        # Different selected operations can occasionally encode the same OS.
        out.proposals = list(dict.fromkeys(out.proposals))
        return out

    def _destroy(
        self, p: ProblemInstance, x: Candidate, budget: int, config: Config, rng: random.Random
    ) -> ProposalBatch:
        n = len(p.jobs)
        out = ProposalBatch()
        seen = {x.genotype}
        for _ in range(config.random_attempt_multiplier * budget):
            if not n or len(out.proposals) >= budget:
                break
            selected: list[int] = []
            count = min(n, max(2, math.ceil(config.a6_destroy_ratio * n)))
            for _ in range(count):
                j = self._pick(
                    "job",
                    [dict(job_id=j) for j in range(n) if j not in selected],
                    rng,
                    selected_destroy_jobs=selected.copy(),
                )
                selected.append(j["job_id"])
            order = [o for o in operation_order(x.genotype) if o // 2 not in selected]
            ms = list(x.genotype.ms)
            # Keep original legal random repair; it is not mislabeled Jev repair.
            for j in selected:
                region = rng.choice([r for r in range(len(p.regions)) if paths(p, j, r)])
                ms[2 * j : 2 * j + 2] = rng.choice(paths(p, j, region))
                a = rng.randrange(len(order) + 1)
                order.insert(a, 2 * j)
                order.insert(rng.randrange(a + 1, len(order) + 1), 2 * j + 1)
            g = Genotype(tuple(ms), tuple(o // 2 for o in order))
            out.attempts += 1
            if g not in seen:
                seen.add(g)
                out.proposals.append(
                    Proposal(g, selected=tuple(2 * j + s for j in selected for s in (0, 1)))
                )
        return out

    def _pack(
        self, p: ProblemInstance, x: Candidate, budget: int, rng: random.Random
    ) -> ProposalBatch:
        # Same finite legal generator; deliberately bypass original 2-representative/2B prescreen.
        groups: dict[int, list] = {j: [] for j in range(len(p.jobs))}
        raw = packing_moves(p, x.genotype, x.schedule)
        for o, t, gain, delta in raw:
            groups[o // 2].append(
                dict(operation=o, start_s=t, active_union_gain_s=gain, flow_delta_s=delta)
            )
        out = ProposalBatch(attempts=len(raw))
        while any(groups.values()) and len(out.proposals) < budget:
            j = self._job(groups, rng)
            o = self._pick(
                "phase", sorted({v["operation"] for v in groups[j]}), rng, selected_job=j
            )
            item = self._pick(
                "position",
                [v for v in groups[j] if v["operation"] == o],
                rng,
                selected_job=j,
                selected_operation=o,
                position_kind="legal_positive_packing_critical_start",
            )
            groups[j].remove(item)
            schedule = move(p, x.genotype, x.schedule, o, item["start_s"])
            assert schedule is not None
            out.proposals.append(Proposal(x.genotype, schedule, (o,)))
        return out

    def _coalition(
        self, segments: list, peaks: tuple[int, ...], cap: int, rng: random.Random, region: int
    ) -> frozenset[int]:
        chosen: frozenset[int] = frozenset()
        groups = list(dict.fromkeys(active for _, active, _ in segments))
        while True:
            deficient = [b for b in peaks if removal_potential(segments, chosen, b) <= TOL.power]
            if not deficient:
                return chosen
            options = []
            for group in groups:
                extended = chosen | group
                gain = sum(
                    max(
                        0.0,
                        removal_potential(segments, extended, b)
                        - removal_potential(segments, chosen, b),
                    )
                    for b in deficient
                )
                if len(extended) <= cap and gain > 0:
                    options.append(dict(members=sorted(group), marginal_removal_proxy_kw=gain))
            if not options:
                return chosen
            item = self._pick(
                "support",
                options,
                rng,
                selected_region=region,
                original_tied_peaks=peaks,
                selected_members=sorted(chosen),
            )
            chosen |= frozenset(item["members"])

    def _repair(
        self,
        p: ProblemInstance,
        x: Candidate,
        members: frozenset[int],
        region: int,
        positions: int,
        rng: random.Random,
    ) -> Schedule | None:
        starts = list(x.schedule.starts)
        for o in members:
            starts[o] = -1.0
        pending = set(members)
        while pending:
            ready = sorted(o for o in pending if o % 2 == 0 or o - 1 not in pending)
            bounds = {
                o: repair_bounds(p, x.genotype, starts, o, x.schedule.horizon(p)) for o in ready
            }
            rng.shuffle(ready)
            o = min(ready, key=lambda k: bounds[k][1] - bounds[k][0])
            partial = Schedule(tuple(starts))
            times = repair_times(p, x.genotype, x.schedule, partial, members, o, region, bounds[o])
            others = intervals(p, x.genotype, partial, x.genotype.ms[o], frozenset({o}))
            for _ in range(min(positions, len(times))):
                t = self._pick(
                    "position",
                    times,
                    rng,
                    selected_job=o // 2,
                    selected_operation=o,
                    selected_region=region,
                    selected_members=sorted(members),
                    partial_solution=dict(starts_s=starts.copy()),
                    position_kind="A8_precedence_bounded_resource_not_yet_checked",
                )
                times.remove(t)
                if _fits_intervals(p, x.genotype.ms[o], o, t, others):
                    starts[o] = t
                    break
            else:
                return None
            pending.remove(o)
        result = Schedule(tuple(starts))
        return None if violations(p, x.genotype, result) else result

    def _peak(
        self, p: ProblemInstance, x: Candidate, budget: int, cfg: Config, rng: random.Random
    ) -> ProposalBatch:
        regions = [
            r
            for r, data in enumerate(p.regions)
            if data.demand_rate > 0
            and max(x.evaluation.windows[r], default=0) > 0
            and any(p.instances[m].region == r for m in x.genotype.ms)
        ]
        out = ProposalBatch()
        if not regions:
            return out
        region = self._pick("region", regions, rng)
        windows = x.evaluation.windows[region]
        peak = max(windows)
        peaks = tuple(b for b, v in enumerate(windows) if peak - v <= TOL.power)
        segments = support_segments(p, x.genotype, x.schedule, region, peaks)
        singles = [
            o
            for o, m in enumerate(x.genotype.ms)
            if p.instances[m].region == region
            and any(removal_potential(segments, frozenset({o}), b) > 0 for b in peaks)
        ]
        success = False
        seen = {x.schedule.starts}
        for attempt in range(cfg.a8_attempt_multiplier * budget):
            if len(out.proposals) >= budget:
                break
            if singles and (attempt < cfg.a8_singleton_attempts or success):
                groups: dict[int, list] = {}
                for o in singles:
                    groups.setdefault(o // 2, []).append(o)
                j = self._job(groups, rng)
                o = self._pick("phase", groups[j], rng, selected_job=j, selected_region=region)
                members = frozenset({o})
            else:
                members = self._coalition(segments, peaks, cfg.a8_member_cap, rng, region)
            out.attempts += 1
            if not members:
                continue
            schedule = self._repair(p, x, members, region, cfg.a8_position_limit, rng)
            if schedule is None or not peak_reduced(p, x.genotype, schedule, region, peak):
                continue
            if schedule.starts in seen:
                continue
            seen.add(schedule.starts)
            success |= len(members) == 1
            out.proposals.append(Proposal(x.genotype, schedule, tuple(sorted(members))))
        out.diagnostics = dict(region=region, peaks=peaks, full_window_gate=True)
        return out
