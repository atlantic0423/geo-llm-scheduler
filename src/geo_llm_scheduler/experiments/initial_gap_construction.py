"""Online G1 construction from the exact prefix of the initial candidate stream."""

import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from time import perf_counter

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments.initial_bill_benchmark import traced_original
from geo_llm_scheduler.experiments.initial_bill_mixture import guided_genotype, retain_original
from geo_llm_scheduler.experiments.runner import digest, solution
from geo_llm_scheduler.initialization.retention import retain_pool
from geo_llm_scheduler.lsv2.frontier import FrozenFrame, nondominated
from geo_llm_scheduler.utils.numeric import EPS_NORM, TOL, close
from geo_llm_scheduler.utils.rng import RNGManager

Point = tuple[float, float]
Edge = tuple[Point, Point]
CONSTRUCTIONS = ("bill4", "g1")
_ROUNDS = 10
_TRIALS = 4


def _normalized(frame: FrozenFrame, point: Point) -> Point:
    def axis(k: int) -> float:
        span = frame.nadir[k] - frame.ideal[k]
        return (point[k] - frame.ideal[k]) / span if span > EPS_NORM else 0.0

    return axis(0), axis(1)


def _direction(
    front: list[Point], frame: FrozenFrame, used: list[float], tried: set[Edge]
) -> tuple[str, float, Edge | None]:
    if not used:
        return "bill_anchor", 0.0, None
    positions = [_normalized(frame, point) for point in front]
    best_area = 0.0
    edge: Edge | None = None
    flow_weight = 0.0
    for i, pair in enumerate(zip(front, front[1:])):
        if pair in tried:
            continue
        a, b = positions[i], positions[i + 1]
        area = (b[0] - a[0]) * (a[1] - b[1])
        if area > best_area:
            # Both midpoint coordinates have a common factor of one half.
            f, c = a[0] + b[0], a[1] + b[1]
            if f + c > EPS_NORM:
                best_area, edge, flow_weight = area, pair, c / (f + c)
    if edge is not None:
        return "gap", flow_weight, edge
    grid = (i / (_ROUNDS - 1) for i in range(_ROUNDS))
    available = [w for w in grid if not any(close(w, p, TOL.scalar) for p in used)]
    return "grid", max(available, key=lambda w: min(abs(w - p) for p in used)), None


def gap_initializer(
    problem: ProblemInstance, config: Config, evidence: dict, *, retention: str = "global_pool"
) -> Callable[[EvaluationGateway, RNGManager], list[Candidate]]:
    """Build 100 original + 10×4 candidates and apply explicit final retention.

    Freeze each round's objective frame before its four constructions. Each
    completed feasible candidate goes through the run gateway; frozen-frame
    Tchebycheff scores are diagnostic only. Gap memory and the BILL RNG are local
    to one initialization. Bill trials use the same bank seed as historical BILL4.
    Legacy keeps nine originals per mode plus each round's weighted-sum winner,
    using the historical four-trial min-max/tie rule with that round's direction.
    Retention never feeds back into construction, Archive updates or RNG draws.
    """
    if config.population != 100:
        raise ValueError("The G1 panel initializer requires population 100")
    if retention not in ("legacy", "global_pool"):
        raise ValueError("Unknown G1 retention")

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        start = perf_counter()
        pool = traced_original(problem, config, evaluator=gateway.evaluate)
        bank_seed = config.seed + 100000
        rng = random.Random(bank_seed)
        seen = {c.genotype for c in pool}
        tried: set[Edge] = set()
        used: list[float] = []
        rounds = []
        for slot in range(_ROUNDS):
            front = nondominated([c.evaluation.objectives for c in pool])
            frame = FrozenFrame.of(front)
            kind, weight, edge = _direction(front, frame, used, tried)
            used.append(weight)
            if edge is not None:
                tried.add(edge)
            points = []
            scores = []
            for attempt in range(1, 101):  # Historical BILL4 per-slot uniqueness limit.
                genotype, prefix = guided_genotype(
                    problem, "BILL", rng, weight, accelerated=True, tchebycheff=True
                )
                if genotype in seen:
                    continue
                seen.add(genotype)
                c = gateway.evaluate(genotype, origin="bill_seed_G1")
                ev = c.evaluation
                if not ev.feasible or not all(
                    math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8)
                    for actual, expected in (
                        (prefix["prefix_tou"], ev.tou),
                        (prefix["prefix_demand"], sum(ev.demand)),
                    )
                ):
                    raise AssertionError("G1 prefix is infeasible or disagrees with exact billing")
                f, b = _normalized(frame, ev.objectives)
                scores.append(max(weight * f, (1 - weight) * b))
                pool.append(c)
                points.append(ev.objectives)
                if len(points) == _TRIALS:
                    break
            else:
                raise ValueError("Four distinct G1 candidates unavailable within attempt limit")
            rounds.append(
                dict(
                    slot=slot,
                    kind=kind,
                    flow_weight=weight,
                    edge=edge,
                    ideal=frame.ideal,
                    nadir=frame.nadir,
                    front_size=len(front),
                    attempts=attempt,
                    points=points,
                    scores=scores,
                )
            )
        tick = perf_counter()
        if retention == "global_pool":
            members = retain_pool(pool, config.population)
        else:
            members = retain_original(pool[:100], 10)
            for slot, w in enumerate(used):
                candidates = pool[100 + 4 * slot : 104 + 4 * slot]
                costs = [c.evaluation.bill for c in candidates]
                flows = [c.evaluation.flow for c in candidates]
                legacy_scores = [
                    (1 - w) * (c - min(costs)) / (max(costs) - min(costs) or 1)
                    + w * (f - min(flows)) / (max(flows) - min(flows) or 1)
                    for c, f in zip(costs, flows, strict=True)
                ]
                winner = min(
                    range(_TRIALS),
                    key=lambda i: (
                        legacy_scores[i],
                        candidates[i].genotype.ms,
                        candidates[i].genotype.os,
                    ),
                )
                members.append(candidates[winner])
        selection_seconds = perf_counter() - tick
        evidence.update(
            policy="g1",
            initial_retention=retention,
            retention_protocol="pareto-maxmin-physical/v1"
            if retention == "global_pool"
            else "original90-round_winner10/v1",
            bank_seed=bank_seed,
            construction_protocol="bill-gap-tchebycheff/v1",
            trials=_TRIALS,
            rounds=rounds,
            config=asdict(config),
            original_parent_count=100,
            original_best_flow=min(c.evaluation.flow for c in pool[:100]),
            initial_population_sha256=digest([solution(c) for c in members]),
            initial_archive_sha256=digest([solution(c) for c in gateway.archive.members]),
            retention=dict(
                pool_count=len(pool),
                selected_count=len(members),
                physical_unique=len({(c.genotype.ms, c.schedule.starts) for c in members}),
                selected_origins=dict(Counter(c.origin for c in members)),
                selection_seconds=selection_seconds,
            ),
        )
        evidence["initialization_seconds"] = perf_counter() - start
        return members

    return initialize
