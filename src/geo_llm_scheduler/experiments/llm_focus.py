"""Study-only A3 ranking and A6 destruction; original repair and exact remain intact."""

import math
import random
from dataclasses import replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.diagnostics.workload import region_loads
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.evaluation.exact import tied_peaks
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.operators.structural import StructuralOperator, operation_order
from geo_llm_scheduler.scheduling.resources import est_for


def job_scores(problem: ProblemInstance, current: Candidate, focus: str) -> list[float]:
    """Return nonnegative current-incumbent proxies, never marginal cost claims."""
    scores = []
    loads = region_loads(problem, current.genotype) if focus == "REGION_LOAD" else ()
    for i, job in enumerate(problem.jobs):
        ops = (2 * i, 2 * i + 1)
        region = problem.instances[current.genotype.ms[2 * i]].region
        if focus == "FLOW_WAIT":
            value = sum(
                max(
                    0.0,
                    current.schedule.starts[o]
                    - est_for(problem, current.genotype, current.schedule, o),
                )
                for o in ops
            )
        elif focus == "RESOURCE_WAIT":
            value = sum(max(0.0, current.schedule.resource_wait[o]) for o in ops)
        elif focus == "KV_DELAY":
            value = job.kv_delay * (current.genotype.ms[2 * i] != current.genotype.ms[2 * i + 1])
        elif focus == "REGION_LOAD":
            value = loads[region]
        elif focus == "TOU_EXPOSURE":
            value = (
                sum(
                    max(
                        0.0,
                        min(current.schedule.end(problem, o), t.end)
                        - max(current.schedule.starts[o], t.start),
                    )
                    * t.price
                    for o in ops
                    for t in problem.regions[region].tariffs
                )
                / 3600
            )
        elif focus == "DEMAND_PEAK":
            # P precedes D, so intervals cannot overlap: sum is their union exposure.
            value = sum(
                max(
                    0.0,
                    min(current.schedule.end(problem, o), (b + 1) * 900)
                    - max(current.schedule.starts[o], b * 900),
                )
                for b in tied_peaks(current.evaluation.windows[region])
                for o in ops
            )
        elif focus == "MIXED":
            value = 0.0
        else:
            raise ValueError("Unsupported focus")
        scores.append(max(0.0, value))
    return scores


def sample_destroy(
    scores: list[float], count: int, rng: random.Random, uniform_mix: float = 0.2
) -> list[int]:
    """Sample without replacement, recomputing mixed probabilities after each draw."""
    if (
        not 0 <= uniform_mix <= 1
        or not 0 <= count <= len(scores)
        or any(not math.isfinite(v) or v < 0 for v in scores)
    ):
        raise ValueError("Invalid destruction scores or parameters")
    if not scores or max(scores) == min(scores):
        return rng.sample(range(len(scores)), count)
    low, high = min(scores), max(scores)
    values = [(v - low) / (high - low) for v in scores]
    remaining = list(range(len(scores)))
    chosen = []
    for _ in range(count):
        total = sum(values[i] for i in remaining)
        probs = [
            (1 - uniform_mix) * values[i] / total + uniform_mix / len(remaining)
            if total
            else 1 / len(remaining)
            for i in remaining
        ]
        target = rng.choices(remaining, weights=probs, k=1)[0]
        chosen.append(target)
        remaining.remove(target)
    return chosen


class FocusStructural(StructuralOperator):
    """Explicit research generator; no modification of historical A3 or random A6."""

    def __init__(self, action: int, focus: str, uniform_mix: float = 0.2) -> None:
        super().__init__(action)
        if action not in (3, 6):
            raise ValueError("Focus is limited to A3 and A6")
        self.focus = focus
        self.uniform_mix = uniform_mix

    def _assignments(
        self, p: ProblemInstance, x: Candidate, config: Config
    ) -> tuple[list[Genotype], int, dict]:
        selected_config = (
            replace(config, a3_region_policy="tariff") if self.focus == "TOU_EXPOSURE" else config
        )
        if self.focus == "REGION_LOAD":
            selected_config = replace(config, a3_region_policy="load")
        candidates, attempts, diagnostics = super()._assignments(p, x, selected_config)
        if self.focus not in ("MIXED", "REGION_LOAD", "TOU_EXPOSURE"):
            scores = job_scores(p, x, self.focus)

            def priority(g: Genotype) -> float:
                changed = next(
                    i
                    for i in range(len(p.jobs))
                    if g.ms[2 * i : 2 * i + 2] != x.genotype.ms[2 * i : 2 * i + 2]
                )
                return -scores[changed]

            candidates.sort(key=priority)
        diagnostics.update(requested_focus=self.focus, applied_focus=self.focus)
        return candidates, attempts, diagnostics

    def _lns(
        self, p: ProblemInstance, x: Candidate, budget: int, config: Config, rng: random.Random
    ) -> ProposalBatch:
        if self.focus == "MIXED":
            result = super()._lns(p, x, budget, config, rng)
            result.diagnostics.update(requested_focus=self.focus, applied_focus="MIXED")
            return result
        scores = job_scores(p, x, self.focus)
        n = len(p.jobs)
        result = ProposalBatch(
            diagnostics={
                "requested_focus": self.focus,
                "applied_focus": self.focus,
                "destroy_uniform_mix": self.uniform_mix,
            }
        )
        if not n:
            return result
        seen = {x.genotype}
        count = min(n, max(2, math.ceil(config.a6_destroy_ratio * n)))
        for _ in range(config.random_attempt_multiplier * budget):
            result.attempts += 1
            selected = sample_destroy(scores, count, rng, self.uniform_mix)
            order = [o for o in operation_order(x.genotype) if o // 2 not in selected]
            ms = list(x.genotype.ms)
            for job in selected:
                region = rng.choice([r for r in range(len(p.regions)) if paths(p, job, r)])
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
