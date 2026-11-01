"""Prospective bounded neighbor panels; observations never consume solver RNG."""

from __future__ import annotations

import math
from dataclasses import asdict
from time import perf_counter

from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, neighborhoods, weights
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.operators.structural import operation_order
from geo_llm_scheduler.rl.state import preference
from geo_llm_scheduler.scheduling.resources import est_for
from geo_llm_scheduler.utils.numeric import EPS_NORM, TOL
from geo_llm_scheduler.utils.rng import RNGManager

FRACTIONS = (0.1, 0.5, 0.9)
CONTROL_NAMES = (
    "flow",
    "bill",
    "scalar",
    "ms_distance",
    "os_distance",
    "weight_distance",
    "stagnation",
    "history",
    "phase",
    "preference",
    "jobs",
    "mate_flow",
    "mate_bill",
    "mate_scalar",
    "objective_distance",
)
OPPORTUNITY_NAMES = (
    "flow_wait",
    "positive_wait",
    "a4_positions",
    "a5_pairs",
    "prefill_slack",
    "assignment_options",
    "region_options",
    "active_gap",
    "peak_ties",
    "demand_fraction",
)


def opportunity_features(problem: ProblemInstance, source: Candidate) -> tuple[float, ...]:
    """Compute pre-action proxies without generating or evaluating any move.

    Legal order positions and assignment counts are opportunity proxies, not
    certificates of feasible improvement. Time slack is precedence slack, not
    a full cumulative-resource feasibility search.
    """
    n = problem.operation_count
    if not n:
        return (0.0,) * len(OPPORTUNITY_NAMES)
    duration = sum(problem.profile(o).duration for o in range(n))
    waits = [
        max(0.0, source.schedule.starts[o] - est_for(problem, source.genotype, source.schedule, o))
        for o in range(n)
    ]
    position = {o: i for i, o in enumerate(operation_order(source.genotype))}
    a4 = a5 = 0
    for o in range(n):
        lower = position[o - 1] + 1 if o % 2 else 0
        if waits[o] > TOL.time:
            a4 += max(0, position[o] - lower)
    slack = 0.0
    assignment = regional = 0
    for j, job in enumerate(problem.jobs):
        p, d = 2 * j, 2 * j + 1
        if waits[p] + waits[d] > TOL.time:
            prefill_position, decode_position = position[p], position[d]
            a5 += max(
                0,
                prefill_position * (decode_position - 1)
                - prefill_position * (prefill_position - 1) // 2,
            )
        slack += max(
            0.0,
            source.schedule.starts[d]
            - source.schedule.end(problem, p)
            - job.kv_delay * (source.genotype.ms[p] != source.genotype.ms[d]),
        )
        region = problem.instances[source.genotype.ms[p]].region
        for o in (p, d):
            assignment += max(0, len(problem.eligible(o, region)) - 1)
        regional += sum(problem.instances[a].region != region for a, _ in paths(problem, j))
    intervals: dict[int, list[tuple[float, float]]] = {}
    for o, m in enumerate(source.genotype.ms):
        intervals.setdefault(m, []).append(
            (source.schedule.starts[o], source.schedule.end(problem, o))
        )
    gap = span = 0.0
    for items in intervals.values():
        items.sort()
        left, right = items[0]
        union = 0.0
        first = left
        for a, b in items[1:]:
            if a > right:
                union += right - left
                left, right = a, b
            else:
                right = max(right, b)
        union += right - left
        width = max(b for _, b in items) - first
        span += width
        gap += max(0.0, width - union)
    peak_ties = sum(
        sum(abs(v - max(windows)) <= TOL.power for v in windows)
        for windows in source.evaluation.windows
        if windows
    )
    values = (
        sum(waits) / max(duration, EPS_NORM),
        sum(w > TOL.time for w in waits) / n,
        a4 / n**2,
        a5 / n**3,
        slack / max(duration, EPS_NORM),
        assignment / max(1, n * len(problem.instances)),
        regional / max(1, len(problem.jobs) * len(problem.instances) ** 2),
        gap / max(span, EPS_NORM),
        peak_ties / max(1, sum(map(len, source.evaluation.windows))),
        sum(source.evaluation.demand) / max(source.evaluation.bill, EPS_NORM),
    )
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite opportunity feature")
    return values


class OpportunityObserver:
    """Capture nine prospective panels and causal own-slot improvement history."""

    def __init__(
        self,
        problem: ProblemInstance,
        seconds: float,
        seed: int,
        neighborhood: int,
        pool_size: int = 6,
    ) -> None:
        """Initialize independent sampling streams, bounded history and storage."""
        if seconds <= 0 or pool_size < 2:
            raise ValueError("Positive duration and at least two source slots required")
        self.problem, self.seconds = problem, seconds
        self.seed, self.neighborhood, self.pool_size = seed, neighborhood, pool_size
        self.panels: list[dict] = []
        self.done: set[int] = set()
        self.history: list[float] = []
        self.calls: list[int] = []
        self.lambdas: tuple[tuple[float, float], ...] = ()
        self.neighbors: tuple[tuple[int, ...], ...] = ()
        self.previous: tuple[int, Candidate] | None = None
        self.observer_seconds = 0.0

    def __call__(self, event: str, values: dict) -> None:
        """Observe pre-offspring pools and only then update causal past history."""
        if event not in ("initialization", "before_offspring", "offspring"):
            return
        tick = perf_counter()
        population = values["population"]
        if not self.history:
            self.history = [0.0] * len(population)
            self.calls = [0] * len(population)
            self.lambdas = weights(len(population))
            self.neighbors = neighborhoods(self.lambdas, self.neighborhood)
        context = NormalizationContext(values["gateway"].ideal, maximum(population))
        if event == "offspring" and self.previous is not None:
            i, old = self.previous
            before, after = (
                context.scalar(old, self.lambdas[i]),
                context.scalar(population[i], self.lambdas[i]),
            )
            gain = (before - after) / max(abs(before), EPS_NORM)
            self.history[i] = 0.8 * self.history[i] + 0.2 * gain
            self.calls[i] += 1
            self.previous = None
        if event == "before_offspring":
            i = values["subproblem"]
            self.previous = (i, population[i])
            lambdas, neighbors = self.lambdas, self.neighbors
            for stage, fraction in enumerate(FRACTIONS):
                if stage in self.done or values["elapsed"] < fraction * self.seconds:
                    continue
                rng = RNGManager(self.seed).stream(f"D2:panel:{stage}")
                for pref in range(3):
                    choices = [
                        j for j in range(len(population)) if preference(j, len(population)) == pref
                    ]
                    caller = rng.choice(choices)
                    indices = rng.sample(
                        list(neighbors[caller]), min(self.pool_size, len(neighbors[caller]))
                    )
                    self.panels.append(
                        {
                            "panel": f"s{stage}_p{pref}",
                            "stage": stage,
                            "preference": pref,
                            "elapsed": values["elapsed"],
                            "caller": caller,
                            "weight": lambdas[caller],
                            "context": asdict(context),
                            "mate": population[caller],
                            "generation": values["generation"],
                            "sources": [
                                {
                                    "slot": j,
                                    "candidate": population[j],
                                    "history": self.history[j],
                                    "history_calls": self.calls[j],
                                    "stagnation": values["stagnation"][j],
                                    "weight": lambdas[j],
                                }
                                for j in indices
                            ],
                        }
                    )
                self.done.add(stage)
        self.observer_seconds += perf_counter() - tick

    def export(self) -> dict:
        """Serialize panels whose source/mate values precede all offline labels."""
        return {
            "fractions": FRACTIONS,
            "pool_size": self.pool_size,
            "observer_seconds": self.observer_seconds,
            "panels": [
                {
                    **p,
                    "mate": solution(p["mate"]),
                    "sources": [{**s, "candidate": solution(s["candidate"])} for s in p["sources"]],
                }
                for p in self.panels
            ],
        }


def control_features(
    problem: ProblemInstance, source: Candidate, mate: Candidate, panel: dict, item: dict
) -> tuple[float, ...]:
    """Return matched quality, distance, stage and causal history controls."""
    context = NormalizationContext(
        tuple(panel["context"]["ideal"]), tuple(panel["context"]["maximum"])
    )
    raw = source.evaluation.objectives
    normalized = [
        (raw[m] - context.ideal[m]) / max(context.maximum[m] - context.ideal[m], EPS_NORM)
        for m in range(2)
    ]
    mate_normalized = [
        (mate.evaluation.objectives[m] - context.ideal[m])
        / max(context.maximum[m] - context.ideal[m], EPS_NORM)
        for m in range(2)
    ]
    n = max(1, problem.operation_count)
    return (
        *normalized,
        context.scalar(source, tuple(panel["weight"])),
        sum(a != b for a, b in zip(source.genotype.ms, mate.genotype.ms)) / n,
        sum(a != b for a, b in zip(source.genotype.os, mate.genotype.os)) / n,
        abs(item["weight"][0] - panel["weight"][0]),
        math.log1p(item["stagnation"]),
        item["history"],
        float(panel["stage"]),
        float(panel["preference"]),
        len(problem.jobs) / 100,
        *mate_normalized,
        context.scalar(mate, tuple(panel["weight"])),
        math.dist(normalized, mate_normalized),
    )
