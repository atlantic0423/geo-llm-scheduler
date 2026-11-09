"""Prototype 2: targeted destroy plus greedy best-position reconstruction (replaces A3/A6).

Destroy picks about 8% of Jobs by a mode score (Flow wait, KV, TOU exposure, peak
window share, Region load, or a rank mix), plus a random share for diversity.
Reconstruction works in schedule space: every kept operation stays frozen, and each
removed Job is placed, one at a time, on the best of a few legal P/D paths and
resource-feasible slots, scored by a Tchebycheff of its Flow and incremental Bill.
The resulting timetable is committed as a genotype (MS from the placement, OS by
start time); the gateway then SSGS-decodes and exact-evaluates it as usual. SSGS
starts are never later than the timetable, so decoded Flow cannot exceed it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.lsv2.billing import WINDOW, PowerProfile
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.scheduling.resources import ResourceCalendar

MODES = ("FLOW", "KV", "TOU", "PEAK", "REGION", "MIXED")


def job_scores(problem: ProblemInstance, candidate: Candidate, mode: str) -> list[float]:
    """Per-Job destroy priority for one mode; larger means more worth rebuilding."""
    if mode not in MODES:
        raise ValueError("Unknown destroy mode")
    g, s = candidate.genotype, candidate.schedule
    n = len(problem.jobs)
    if mode == "MIXED":
        parts = [job_scores(problem, candidate, m) for m in ("FLOW", "TOU", "PEAK")]
        ranks = []
        for part in parts:
            order = sorted(range(n), key=lambda j: part[j])
            rank = [0.0] * n
            for r, j in enumerate(order):
                rank[j] = r / max(1, n - 1)
            ranks.append(rank)
        return [sum(r[j] for r in ranks) / len(ranks) for j in range(n)]
    out = []
    windows = candidate.evaluation.windows
    peaks = [max(range(len(w)), key=w.__getitem__) if w else -1 for w in windows]
    loads = [0.0] * len(problem.regions)
    for o in range(problem.operation_count):
        loads[problem.instances[g.ms[o]].region] += problem.profile(o).duration
    for j, job in enumerate(problem.jobs):
        P, D = 2 * j, 2 * j + 1
        sep = g.ms[P] != g.ms[D]
        if mode == "FLOW":
            pf = s.starts[P] - job.release
            df = s.starts[D] - (s.end(problem, P) + job.kv_delay * sep)
            out.append(max(0.0, pf) + max(0.0, df))
        elif mode == "KV":
            out.append(job.kv_delay * sep)
        elif mode == "TOU":
            value = 0.0
            for o in (P, D):
                server = problem.instances[g.ms[o]]
                a, b = s.starts[o], s.end(problem, o)
                for t in problem.regions[server.region].tariffs:
                    value += (
                        (server.active_kw - server.idle_kw)
                        * max(0.0, min(b, t.end) - max(a, t.start))
                        * t.price
                    )
            out.append(value)
        elif mode == "PEAK":
            value = 0.0
            for o in (P, D):
                server = problem.instances[g.ms[o]]
                w = peaks[server.region]
                if w < 0:
                    continue
                a, b = s.starts[o], s.end(problem, o)
                overlap = max(0.0, min(b, (w + 1) * WINDOW) - max(a, w * WINDOW))
                value += (
                    problem.regions[server.region].demand_rate
                    * (server.active_kw - server.idle_kw)
                    * overlap
                    / WINDOW
                )
            out.append(value)
        else:  # REGION
            out.append(loads[problem.instances[g.ms[P]].region])
    return out


@dataclass
class TargetedDestroyRepair:
    """IG-style destroy-repair with objective-aware schedule-space reinsertion."""

    mode: str = "MIXED"
    weight: tuple[float, float] = (0.5, 0.5)
    scale: tuple[float, float] | None = None
    destroy_ratio: float = 0.08
    min_destroy: int = 2
    random_share: float = 0.2
    paths_k: int = 4
    slots_k: int = 3
    second_best: float = 0.15
    action: int = 102

    def _slots(
        self, cal: ResourceCalendar, machine: int, operation: int, est: float
    ) -> list[float]:
        """First few distinct feasible starts: from EST and from later completion events."""
        found: list[float] = []
        candidates = [est] + sorted({b for _, _, b in cal.items[machine] if b > est})
        for e in candidates:
            try:
                t = cal.earliest(machine, operation, e)
            except ValueError:
                continue
            if not found or all(abs(t - x) > 1e-9 for x in found):
                found.append(t)
            if len(found) >= self.slots_k:
                break
        return found

    def propose(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        config: Config,
        rng: random.Random,
    ) -> ProposalBatch:
        """Return up to ``budget`` distinct structural proposals from one frozen incumbent."""
        n = len(problem.jobs)
        scores = job_scores(problem, incumbent, self.mode)
        d = min(n, max(self.min_destroy, round(self.destroy_ratio * n)))
        d_rand = max(0, min(d - 1, round(self.random_share * d)))
        ranked = sorted(range(n), key=lambda j: -scores[j])
        fs, bs = self.scale or incumbent.evaluation.objectives
        fs, bs = max(fs, 1e-12), max(bs, 1e-12)
        wf, wb = self.weight
        batch = ProposalBatch()
        seen = set()
        for _ in range(max(1, budget)):
            core = ranked[: d - d_rand]
            rest = ranked[d - d_rand :]
            removed = core + (rng.sample(rest, d_rand) if d_rand and rest else [])
            removed_ops = {o for j in removed for o in (2 * j, 2 * j + 1)}
            ms = list(incumbent.genotype.ms)
            starts = [
                -1.0 if o in removed_ops else t for o, t in enumerate(incumbent.schedule.starts)
            ]
            cal = ResourceCalendar(problem)
            for o, t in enumerate(starts):
                if t >= 0:
                    cal.commit(ms[o], o, t)
            profile = PowerProfile(problem, ms, starts)
            order = sorted(removed, key=lambda j: -scores[j])
            for j in order:
                P, D = 2 * j, 2 * j + 1
                job = problem.jobs[j]
                load = [0.0] * len(problem.instances)
                for m, items in enumerate(cal.items):
                    load[m] = sum(
                        problem.profile(o).duration * problem.profile(o).compute
                        for o, _, _ in items
                    )
                options = paths(problem, j)
                current = (incumbent.genotype.ms[P], incumbent.genotype.ms[D])
                options.sort(
                    key=lambda pd: load[pd[0]] + load[pd[1]] + job.kv_delay * (pd[0] != pd[1])
                )
                chosen = options[: self.paths_k]
                if current in options and current not in chosen:
                    chosen.append(current)
                if self.mode == "REGION":
                    here = problem.instances[current[0]].region
                    away = [pd for pd in options if problem.instances[pd[0]].region != here]
                    for pd in away[:2]:
                        if pd not in chosen:
                            chosen.append(pd)
                pP, pD = problem.profile(P).duration, problem.profile(D).duration
                evaluated = []
                for mp, md in chosen:
                    for tp in self._slots(cal, mp, P, job.release):
                        est_d = tp + pP + job.kv_delay * (mp != md)
                        temp = (P, tp, tp + pP) if md == mp else None
                        try:
                            td = cal.earliest(md, D, est_d, temporary=temp)
                        except ValueError:
                            continue
                        flow = td + pD - job.release
                        changes = {mp: {P: (tp, tp + pP)}}
                        changes.setdefault(md, {})[D] = (td, td + pD)
                        end = td + pD
                        if end > profile.horizon:
                            longer = profile.with_horizon(end)
                            bill = longer.bill() - profile.bill() + longer.delta_for(changes)
                        else:
                            bill = profile.delta_for(changes)
                        score = max(wf * flow / fs, wb * bill / bs)
                        evaluated.append((score, bill, flow, mp, md, tp, td))
                if not evaluated:
                    raise ValueError("Removed Job has no feasible reinsertion")
                evaluated.sort()
                pick = (
                    evaluated[1]
                    if len(evaluated) > 1 and rng.random() < self.second_best
                    else evaluated[0]
                )
                _, _, _, mp, md, tp, td = pick
                end = td + pD
                if end > profile.horizon:
                    profile = profile.with_horizon(end)
                changes = {mp: {P: (tp, tp + pP)}}
                changes.setdefault(md, {})[D] = (td, td + pD)
                profile.apply(changes)
                cal.commit(mp, P, tp)
                cal.commit(md, D, td)
                ms[P], ms[D] = mp, md
                starts[P], starts[D] = tp, td
            ops = sorted(range(problem.operation_count), key=lambda o: (starts[o], o))
            genotype = Genotype(tuple(ms), tuple(o // 2 for o in ops))
            if genotype not in seen and genotype != incumbent.genotype:
                seen.add(genotype)
                batch.proposals.append(Proposal(genotype))
            batch.attempts += 1
        batch.diagnostics = dict(
            mode=self.mode, destroyed=d, random=d_rand, proposals=len(batch.proposals)
        )
        return batch


__all__ = ["MODES", "TargetedDestroyRepair", "job_scores"]
