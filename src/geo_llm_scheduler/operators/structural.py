"""A1-A6 current-specification structural neighborhoods."""

import math
import random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.diagnostics.workload import imbalance, machine_loads, region_loads
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.scheduling.resources import est_for
from geo_llm_scheduler.utils.numeric import EPS_RATIO


def operation_order(genotype: Genotype) -> list[int]:
    """Expand repeated jobs into explicit P/D operation identities."""
    seen: set[int] = set()
    result = []
    for j in genotype.os:
        result.append(2 * j + (j in seen))
        seen.add(j)
    return result


def with_order(genotype: Genotype, order: list[int]) -> Genotype:
    """Fold legal explicit operations back into repeated-job OS."""
    return Genotype(genotype.ms, tuple(o // 2 for o in order))


def round_robin(groups: list[list[Genotype]], cap: int) -> list[Genotype]:
    """Interleave targets so a single operation cannot monopolize the pool."""
    result = []
    for k in range(max(map(len, groups), default=0)):
        for group in groups:
            if k < len(group) and group[k] not in result:
                result.append(group[k])
                if len(result) == cap:
                    return result
    return result


class StructuralOperator:
    """A1-A5 rank a 2B pool; A6 samples direct distinct random repairs."""

    def __init__(self, action: int):
        if action not in range(1, 7):
            raise ValueError("Structural action must be A1-A6")
        self.action = action

    def propose(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        config: Config,
        rng: random.Random,
    ) -> ProposalBatch:
        """Generate bounded same-incumbent structural candidates."""
        if budget <= 0:
            return ProposalBatch()
        if self.action == 6:
            return self._lns(problem, incumbent, budget, config, rng)
        candidates = (
            self._assignments(problem, incumbent, config)
            if self.action <= 3
            else self._insertions(problem, incumbent, budget, rng)
        )
        unique = list(dict.fromkeys(g for g in candidates if g != incumbent.genotype))[: 2 * budget]
        selected = rng.sample(unique, min(budget, len(unique)))
        return ProposalBatch([Proposal(g) for g in selected], len(candidates))

    def _assignments(self, p: ProblemInstance, x: Candidate, config: Config) -> list[Genotype]:
        g = x.genotype
        wait = x.schedule.resource_wait
        loads = machine_loads(p, g)
        ranked: list[tuple[tuple[float, ...], Genotype]] = []
        resource = min(
            1,
            sum(wait) / (sum(p.profile(o).duration for o in range(p.operation_count)) + EPS_RATIO),
        )
        kv = sum(j.kv_delay * (g.ms[2 * i] != g.ms[2 * i + 1]) for i, j in enumerate(p.jobs)) / (
            x.evaluation.flow + EPS_RATIO
        )
        prefer_resource = (
            resource / config.severity_thresholds[0] >= kv / config.severity_thresholds[1]
        )
        for i, job in enumerate(p.jobs):
            region = p.instances[g.ms[2 * i]].region
            if self.action == 1:
                for o in (2 * i, 2 * i + 1):
                    for m in p.eligible(o, region):
                        if m == g.ms[o]:
                            continue
                        ms = list(g.ms)
                        ms[o] = m
                        ranked.append(((-wait[o], loads[m], o, m), Genotype(tuple(ms), g.os)))
                continue
            for a, b in paths(p, i):
                if (a, b) == g.ms[2 * i : 2 * i + 2]:
                    continue
                same = p.instances[a].region == region
                if same != (self.action == 2):
                    continue
                ms = list(g.ms)
                ms[2 * i : 2 * i + 2] = a, b
                new = Genotype(tuple(ms), g.os)
                newloads = machine_loads(p, new)
                pathload = max(newloads[a], newloads[b])
                newkv = job.kv_delay * (a != b)
                if self.action == 2:
                    key = (
                        (-wait[2 * i] - wait[2 * i + 1], pathload, newkv, i, a, b)
                        if prefer_resource
                        else (
                            -job.kv_delay * (g.ms[2 * i] != g.ms[2 * i + 1]),
                            newkv,
                            pathload,
                            i,
                            a,
                            b,
                        )
                    )
                else:
                    key = (imbalance(region_loads(p, new)), pathload, newkv, i, a, b)
                ranked.append((key, new))
        ranked.sort(key=lambda item: item[0])
        return [g for _, g in ranked]

    def _insertions(
        self, p: ProblemInstance, x: Candidate, budget: int, rng: random.Random
    ) -> list[Genotype]:
        g = x.genotype
        order = operation_order(g)
        waits = [
            x.schedule.starts[o] - est_for(p, g, x.schedule, o) for o in range(p.operation_count)
        ]
        groups = []
        if self.action == 4:
            targets = sorted(range(p.operation_count), key=lambda o: (-waits[o], o))
            for o in targets:
                old = order.index(o)
                lower = order.index(o - 1) + 1 if o % 2 else 0
                if waits[o] <= 0 or lower >= old:
                    continue
                group = []
                for pos in range(lower, old):
                    new = order.copy()
                    new.pop(old)
                    new.insert(pos, o)
                    group.append(with_order(g, new))
                groups.append(group)
                if len(groups) == budget:
                    break
        else:
            targets = sorted(range(len(p.jobs)), key=lambda i: (-min(waits[2 * i : 2 * i + 2]), i))
            for i in targets:
                oldp, oldd = order.index(2 * i), order.index(2 * i + 1)
                if min(waits[2 * i : 2 * i + 2]) <= 0:
                    continue
                base = [o for o in order if o // 2 != i]
                group = []
                for a in range(oldp):
                    for b in range(a + 1, oldd):
                        new = base.copy()
                        new.insert(a, 2 * i)
                        new.insert(b, 2 * i + 1)
                        group.append(with_order(g, new))
                if group:
                    rng.shuffle(group)
                    groups.append(group)
                if len(groups) == budget:
                    break
        return round_robin(groups, 2 * budget)

    def _lns(
        self, p: ProblemInstance, x: Candidate, budget: int, config: Config, rng: random.Random
    ) -> ProposalBatch:
        n = len(p.jobs)
        if not n:
            return ProposalBatch()
        result = ProposalBatch()
        seen = {x.genotype}
        for _ in range(config.random_attempt_multiplier * budget):
            result.attempts += 1
            selected = rng.sample(range(n), min(n, max(2, math.ceil(config.a6_destroy_ratio * n))))
            order = [o for o in operation_order(x.genotype) if o // 2 not in selected]
            ms = list(x.genotype.ms)
            for job in selected:
                regions = [r for r in range(len(p.regions)) if paths(p, job, r)]
                region = rng.choice(regions)
                ms[2 * job : 2 * job + 2] = rng.choice(paths(p, job, region))
                a = rng.randrange(len(order) + 1)
                order.insert(a, 2 * job)
                b = rng.randrange(a + 1, len(order) + 1)
                order.insert(b, 2 * job + 1)
            g = Genotype(tuple(ms), tuple(o // 2 for o in order))
            if g not in seen:
                seen.add(g)
                result.proposals.append(
                    Proposal(g, selected=tuple(2 * j + s for j in selected for s in (0, 1)))
                )
            if len(result.proposals) == budget:
                break
        return result
