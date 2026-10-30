"""LSV2 research arm: a time-share LS scheduler over the Archive front (default off).

The engine calls :class:`LSV2Search` through its opt-in ``interleave`` seam before
every offspring slot. Whenever cumulative LS wall time is below the configured
share of search time (production time after initialization), it runs one LS
trajectory:

1. Freeze a normalization frame from the Archive front (refreshed every
   ``frame_steps`` LS steps, only at trajectory boundaries). HV uses reference 1.1.
2. Sample the seed from the Archive front by a declared rule (``gap`` scores with
   stagnation discount, or the ``flow_end`` 80/15/5 reference rule).
3. Each step: the current solution's segment (Flow end / middle / Bill end, by
   normalized Flow thirds) restricts a stateless sliding-window UCB bandit to the
   arms ``segment x operator``; A8 is gated by its recent empty-call rate.
4. All proposals go through the normal gateway (SSGS for structural, exact for
   all, every feasible candidate tries the Archive). Reward = Archive HV gain in
   the frozen frame / actual step seconds, so locally rejected candidates that
   reach the Archive are credited.
5. Chain acceptance: continue from a candidate dominating the current solution,
   else from the new Archive member that most reduces the gap around the seed,
   else stay. Accepted timing proposals get refreshed resource-wait diagnostics.
6. Archive entrants of the step try to replace MOEA/D members in the
   neighborhood of their associated direction, each neighbor by its own weight.

Post-episode fast right shift is opt-in; the original arm has no Polish.
All randomness uses named streams.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from random import Random
from time import perf_counter

from geo_llm_scheduler.archive.pareto import dominates
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.lsv2.common_refine import ReusedCommonPolish
from geo_llm_scheduler.lsv2.destroy_repair import TargetedDestroyRepair
from geo_llm_scheduler.lsv2.flowsafe import FlowSafeBillMoves
from geo_llm_scheduler.lsv2.frontier import REF, FrozenFrame, SeedSampler, gap_scores, nondominated
from geo_llm_scheduler.lsv2.group_polish import ConcurrentRightShiftPolish
from geo_llm_scheduler.lsv2.polish import FastRightShiftPolish
from geo_llm_scheduler.lsv2.polish_selection import (
    POOL_POLICIES,
    SEGMENTS,
    PolishSelector,
    physical_fingerprint,
    segment_of,
)
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import (
    NormalizationContext,
    maximum,
    neighborhoods,
    replace_neighbors,
    weights,
)
from geo_llm_scheduler.operators.active_pack import ActivePack
from geo_llm_scheduler.operators.base import Operator, ProposalBatch
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.rl.state import associate
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.utils.numeric import TOL, close, identity, less
from geo_llm_scheduler.utils.rng import RNGManager

OPERATORS = ("A1", "A2", "A3", "A4", "A5", "A7", "A8", "P2_FLOW", "P2_MIXED", "P3")
Point = tuple[float, float]
Box = tuple[float, float, float, float]


def _chain_next(
    current: Candidate,
    feasible: Sequence[Candidate],
    entrants: Sequence[Candidate],
    frame: FrozenFrame,
    local_gain: Callable[[Candidate], float],
) -> tuple[Candidate, str]:
    """Resolve score ties against fixed extrema, then canonical phenotype identity.

    Eligibility retains exact dominance and strictly positive gap gain. Only the
    ranking of eligible candidates uses the contract's dimensionless tolerance.
    Anchoring each tie set at its extremum avoids non-transitive pairwise sorting.
    """
    dominating = [c for c in feasible if dominates(c, current)]
    scored = [(local_gain(c), c) for c in (dominating or entrants)]
    if not dominating:
        scored = [(gain, c) for gain, c in scored if gain > 0]
    if not scored:
        return current, "stay"
    best_gain = max(gain for gain, _ in scored)
    tied = [c for gain, c in scored if close(gain, best_gain, TOL.scalar)]
    if dominating:
        totals = [(sum(frame.normalize(c.evaluation.objectives)), c) for c in tied]
        best_total = min(total for total, _ in totals)
        tied = [c for total, c in totals if close(total, best_total, TOL.scalar)]
    return min(tied, key=identity), "dominate" if dominating else "gap"


@dataclass(frozen=True)
class LSV2Config:
    """Declared LSV2 research parameters (working values for one development batch)."""

    time_share: float = 0.20
    seed_rule: str = "gap"
    seed_discount: float = 0.5
    end_bonus: float = 1.0
    flow_end_shares: tuple[float, float, float] = (0.8, 0.15, 0.05)
    trajectory_steps: int = 5
    budget: int = 6
    frame_steps: int = 50
    bandit_window: int = 100
    bandit_exploration: float = 0.5
    selector: str = "bandit"
    a8_gate_window: int = 8
    a8_gate_min_calls: int = 4
    a8_gate_empty_rate: float = 0.75
    a8_probe_interval: int = 20
    operators: tuple[str, ...] = OPERATORS
    polish_region: str = "off"
    polish_targets: str = "partial"
    polish_budget: int = 3
    polish_selection: str = "endpoint"
    polish_diagnostics: bool = False
    polish_method: str = "single"

    def __post_init__(self) -> None:
        if not 0 < self.time_share < 1:
            raise ValueError("LS time share must lie in (0, 1)")
        if self.seed_rule not in ("gap", "flow_end"):
            raise ValueError("Unknown LSV2 seed rule")
        if self.selector not in ("bandit", "uniform"):
            raise ValueError("Unknown LSV2 operator selector")
        if not self.operators or set(self.operators) - set(OPERATORS):
            raise ValueError("Unknown LSV2 operator")
        if len(set(self.operators)) != len(self.operators):
            raise ValueError("LSV2 operators must be unique")
        if self.polish_region not in ("off", "all", "flow_end", "middle", "bill_end"):
            raise ValueError("Unknown fast right-shift region")
        if self.polish_targets not in ("partial", "all"):
            raise ValueError("Unknown fast right-shift target pool")
        if self.polish_method not in (
            "single",
            "common",
            "coordinated",
            "common_fast",
            "common_peak",
        ):
            raise ValueError("Unknown external right-shift method")
        if self.polish_selection not in ("endpoint", *POOL_POLICIES):
            raise ValueError("Unknown external right-shift selection policy")
        if self.polish_selection != "endpoint" and self.polish_region != "all":
            raise ValueError("Shared-pool selection requires all-region external right shift")
        if self.polish_region != "off" and "P3" in self.operators:
            raise ValueError("External right shift requires removing P3 from the action pool")
        counts = (
            self.trajectory_steps,
            self.budget,
            self.frame_steps,
            self.bandit_window,
            self.a8_gate_window,
            self.a8_gate_min_calls,
            self.a8_probe_interval,
            self.polish_budget,
        )
        if any(type(v) is not int or v < 1 for v in counts):
            raise ValueError("LSV2 counts must be positive integers")
        if not 0 <= self.seed_discount <= 1 or not 0 <= self.a8_gate_empty_rate <= 1:
            raise ValueError("LSV2 rates must lie in [0, 1]")
        if self.end_bonus < 0 or self.bandit_exploration < 0:
            raise ValueError("LSV2 bonuses must be nonnegative")
        if len(self.flow_end_shares) != 3 or min(self.flow_end_shares) < 0:
            raise ValueError("Three nonnegative Flow-end shares required")


def local_box(front_norm: Sequence[Point], point: Point) -> Box:
    """Rectangle between a point's front neighbours; open sides extend to the reference.

    The left neighbour is the front point with the largest Flow below the point's
    Flow, the right one the point with the largest Bill below the point's Bill.
    Returns ``(x_lo, x_hi, y_lo, y_hi)`` in normalized coordinates.
    """
    left = [q for q in front_norm if q[0] < point[0]]
    right = [q for q in front_norm if q[1] < point[1]]
    x_lo, y_hi = max(left) if left else (-math.inf, REF)
    r = max(right, key=lambda q: (q[1], -q[0])) if right else (REF, -math.inf)
    return x_lo, min(r[0], REF), r[1], min(y_hi, REF)


def box_hv(points_norm: Sequence[Point], box: Box) -> float:
    """Area of the box attained (weakly dominated) by any point; 0 for an empty box."""
    x_lo, x_hi, y_lo, y_hi = box
    if not (x_lo < x_hi and y_lo < y_hi):
        return 0.0
    clipped = [
        (max(x, x_lo), max(y, y_lo)) for x, y in points_norm if max(x, x_lo) < x_hi and y < y_hi
    ]
    area, best = 0.0, y_hi
    for x, y in nondominated(clipped):
        if y < best:
            area += (x_hi - x) * (best - y)
            best = y
    return area


class SlidingWindowBandit:
    """Stateless sliding-window UCB over ``segment x operator`` arms.

    The window keeps the last ``window`` pulls of all arms. Within a decision the
    candidate arms are those of the current segment: an arm without a pull in the
    window is chosen first (uniformly), otherwise the score is the window mean
    reward normalized by the best candidate mean, plus
    ``exploration * sqrt(2 ln N / n)``. Ties are broken by the supplied RNG.
    """

    def __init__(self, window: int, exploration: float) -> None:
        self.window: deque[tuple[tuple[int, str], float]] = deque(maxlen=window)
        self.exploration = exploration
        self.pulls: dict[tuple[int, str], int] = {}
        self.reward_sum: dict[tuple[int, str], float] = {}

    def _window_stats(self) -> dict[tuple[int, str], tuple[int, float]]:
        stats: dict[tuple[int, str], tuple[int, float]] = {}
        for arm, value in self.window:
            n, total = stats.get(arm, (0, 0.0))
            stats[arm] = (n + 1, total + value)
        return stats

    def choose(self, segment: int, operators: Sequence[str], rng: Random) -> tuple[str, dict]:
        """Return the chosen operator and the decision's window scores."""
        if not operators:
            raise ValueError("No available LSV2 operator")
        stats = self._window_stats()
        arms = [(segment, op) for op in operators]
        unplayed = [arm for arm in arms if arm not in stats]
        if unplayed:
            arm = unplayed[rng.randrange(len(unplayed))]
            return arm[1], {"rule": "unplayed", "candidates": len(arms), "unplayed": len(unplayed)}
        means = {arm: stats[arm][1] / stats[arm][0] for arm in arms}
        scale = max(means.values())
        total = sum(stats[arm][0] for arm in arms)
        scores = {
            arm: (means[arm] / scale if scale > 0 else 0.0)
            + self.exploration * math.sqrt(2 * math.log(total) / stats[arm][0])
            for arm in arms
        }
        top = max(scores.values())
        best = [arm for arm in arms if scores[arm] == top]
        arm = best[rng.randrange(len(best))]
        return arm[1], {
            "rule": "ucb",
            "candidates": len(arms),
            "score": scores[arm],
            "mean": means[arm],
            "window_pulls": stats[arm][0],
        }

    def update(self, segment: int, operator: str, value: float) -> None:
        """Record one pull's reward in the window and in the run totals."""
        arm = (segment, operator)
        self.window.append((arm, value))
        self.pulls[arm] = self.pulls.get(arm, 0) + 1
        self.reward_sum[arm] = self.reward_sum.get(arm, 0.0) + value

    def summary(self) -> dict:
        """JSON-ready run totals and final window means per arm."""
        stats = self._window_stats()
        return {
            "arms": [
                {
                    "segment": SEGMENTS[s],
                    "operator": op,
                    "pulls": self.pulls[(s, op)],
                    "reward_sum": self.reward_sum[(s, op)],
                    "window_pulls": stats.get((s, op), (0, 0.0))[0],
                    "window_mean": (
                        stats[(s, op)][1] / stats[(s, op)][0] if (s, op) in stats else None
                    ),
                }
                for s, op in sorted(self.pulls)
            ]
        }


@dataclass
class A8Gate:
    """Per-segment A8 availability from its recent empty-call (no proposal) rate.

    A8 is unavailable in a segment when at least ``min_calls`` of its last
    ``window`` calls there exist and their empty share is at least ``empty_rate``.
    While unavailable it is re-offered once every ``probe_interval`` LS steps of
    that segment, so the gate can reopen.
    """

    window: int
    min_calls: int
    empty_rate: float
    probe_interval: int
    history: dict[int, deque[bool]] = field(default_factory=dict)
    skipped: dict[int, int] = field(default_factory=dict)

    def available(self, segment: int) -> bool:
        """Decide availability for one LS step of this segment (counts skipped steps)."""
        recent = self.history.get(segment, ())
        if len(recent) < self.min_calls or sum(recent) / len(recent) < self.empty_rate:
            return True
        self.skipped[segment] = self.skipped.get(segment, 0) + 1
        if self.skipped[segment] >= self.probe_interval:
            self.skipped[segment] = 0
            return True
        return False

    def record(self, segment: int, empty: bool) -> None:
        """Append one A8 call outcome for the segment."""
        self.history.setdefault(segment, deque(maxlen=self.window)).append(empty)


def _with_diagnostics(problem: ProblemInstance, candidate: Candidate) -> Candidate:
    """Retained-timing candidates lack resource waits; recompute them before reuse."""
    if candidate.schedule.resource_wait:
        return candidate
    schedule = refresh_diagnostics(problem, candidate.genotype, candidate.schedule)
    return replace(candidate, schedule=schedule)


class LSV2Search:
    """Engine ``interleave`` callback implementing the LSV2 research arm.

    ``operator_config`` supplies operator parameters (A3 tariff policy, A8 limits,
    severity thresholds); ``engine_config`` supplies population/neighborhood,
    replacement cap and the wall-clock budget. ``timer`` measures LS seconds and
    can be injected for deterministic tests.
    """

    def __init__(
        self,
        config: LSV2Config,
        engine_config: Config,
        operator_config: Config,
        *,
        timer: Callable[[], float] = perf_counter,
        retain_steps: bool = True,
    ) -> None:
        self.config = config
        self.engine_config = engine_config
        self.operator_config = operator_config
        self.timer = timer
        self.retain_steps = retain_steps
        self.lambdas = weights(engine_config.population)
        self.neighbors = neighborhoods(self.lambdas, engine_config.neighborhood)
        self.bandit = SlidingWindowBandit(config.bandit_window, config.bandit_exploration)
        self.gate = A8Gate(
            config.a8_gate_window,
            config.a8_gate_min_calls,
            config.a8_gate_empty_rate,
            config.a8_probe_interval,
        )
        self.sampler: SeedSampler | None = None
        self.frame: FrozenFrame | None = None
        self.frame_id = -1
        self.steps_since_frame = 0
        self.search_start: float | None = None
        self.ls_seconds = 0.0
        self.trajectories = 0
        self.steps: list[dict] = []
        self.polish_steps: list[dict] = []
        self.polish_seconds = 0.0
        self.polish_selector = PolishSelector()
        self.reused_polish = ReusedCommonPolish()
        self.counts: dict[str, int] = {}
        self._structural: dict[int, Operator] = {a: StructuralOperator(a) for a in range(1, 6)}
        self._timing: dict[str, Operator] = {
            "A7": ActivePack(),
            "A8": PeakCoalition(),
            "P3": FlowSafeBillMoves(),
        }

    # ---------------- scheduling ----------------
    def __call__(
        self,
        population: list[Candidate],
        gateway: EvaluationGateway,
        streams: RNGManager,
        budget_elapsed: Callable[[], float],
    ) -> None:
        """Run LS trajectories while LS time is below its share of search time."""
        now = budget_elapsed()
        if self.search_start is None:
            self.search_start = now
            self.sampler = SeedSampler(
                self.config.seed_rule,
                streams.stream("lsv2:seed"),
                discount=self.config.seed_discount,
                shares=self.config.flow_end_shares,
            )
        while not self._deadline(now) and self.ls_seconds < self.config.time_share * (
            now - self.search_start
        ):
            started = self.timer()
            self._trajectory(population, gateway, streams, budget_elapsed)
            self.ls_seconds += self.timer() - started
            now = budget_elapsed()

    def _deadline(self, elapsed: float) -> bool:
        seconds = self.engine_config.seconds
        return seconds is not None and elapsed >= seconds

    def _count(self, name: str, value: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + value

    # ---------------- one trajectory ----------------
    def _trajectory(
        self,
        population: list[Candidate],
        gateway: EvaluationGateway,
        streams: RNGManager,
        budget_elapsed: Callable[[], float],
    ) -> None:
        assert self.sampler is not None
        problem = gateway.problem
        by_objective: dict[Point, Candidate] = {}
        for member in gateway.archive.members:
            by_objective.setdefault(member.evaluation.objectives, member)
        front = nondominated(list(by_objective))
        if self.frame is None or self.steps_since_frame >= self.config.frame_steps:
            self.frame = FrozenFrame.of(front)
            self.frame_id += 1
            self.steps_since_frame = 0
        frame = self.frame
        index = self.sampler.pick(frame, front)
        seed_point = front[index]
        front_norm = [frame.normalize(p) for p in front]
        seed_gap = gap_scores(front_norm, self.config.end_bonus)[index]
        box = local_box(front_norm, front_norm[index])
        current = _with_diagnostics(problem, by_objective[seed_point])
        trajectory = self.trajectories
        self.trajectories += 1
        total_gain = 0.0
        executed = 0
        for step in range(self.config.trajectory_steps):
            if self._deadline(budget_elapsed()):
                break
            current, gain = self._step(
                population,
                gateway,
                streams,
                frame,
                current,
                box,
                dict(trajectory=trajectory, step=step, seed=seed_point, seed_gap=seed_gap),
            )
            total_gain += gain
            executed += 1
            self.steps_since_frame += 1
        self.sampler.report(seed_point, total_gain > 0)
        self._finish_episode()
        if self.config.polish_region != "off" and executed:
            self._polish(
                current,
                frame,
                trajectory,
                population,
                gateway,
                streams,
                budget_elapsed,
            )

    def _finish_episode(self) -> None:
        """Close controller credit before any external polish (base has no pending TD)."""

    def _polish(
        self,
        current: Candidate,
        frame: FrozenFrame,
        trajectory: int,
        population: list[Candidate],
        gateway: EvaluationGateway,
        streams: RNGManager,
        budget_elapsed: Callable[[], float],
    ) -> None:
        """Select and polish one input after terminal Q credit, within shared LS time."""
        started = self.timer()
        service_started: float | None = None
        segment = SEGMENTS[segment_of(frame.normalize(current.evaluation.objectives))]
        row: dict = dict(
            trajectory=trajectory,
            frame=self.frame_id,
            segment=segment,
            start_objectives=list(current.evaluation.objectives),
            status="region_skip",
            proposals=0,
            exact=0,
            accepted=False,
            hv_gain=0.0,
            bill_gain=0.0,
        )
        try:
            if self.config.polish_region not in ("all", segment):
                return
            elapsed = budget_elapsed()
            if self._deadline(elapsed):
                row["status"] = "deadline_skip"
                return
            if self.config.polish_diagnostics or self.config.polish_selection != "endpoint":
                row.update(
                    selection=self.config.polish_selection,
                    target_scope=self.config.polish_targets,
                    endpoint_objectives=list(current.evaluation.objectives),
                )
                if self.config.polish_selection != "endpoint":
                    selected = self.polish_selector.choose(
                        self.config.polish_selection,
                        population,
                        current,
                        frame,
                        NormalizationContext(gateway.ideal, maximum(population)),
                        self.lambdas,
                        streams.stream("lsv2:rightshift_selection"),
                    )
                    current = selected.candidate
                    segment = SEGMENTS[selected.region]
                    row.update(
                        pool_size=selected.pool_size,
                        region_sizes=selected.region_sizes,
                        direction_sizes=selected.direction_sizes,
                        direction=selected.direction,
                        source=selected.source,
                        segment=segment,
                        start_objectives=list(current.evaluation.objectives),
                    )
                else:
                    row.update(pool_size=1, source="endpoint")
                selected_at = self.timer()
                row["seconds_selection"] = selected_at - started
                if self.config.polish_diagnostics:
                    row.update(
                        physical_fingerprint=physical_fingerprint(current),
                        fingerprint_version="physical-v1/instance-scoped",
                        input_schedule=dict(
                            ms=current.genotype.ms,
                            os=current.genotype.os,
                            starts=current.schedule.starts,
                        ),
                        input_tou=current.evaluation.tou,
                        input_demand=sum(current.evaluation.demand),
                    )
                diagnosed_at = self.timer()
                row["seconds_input_diagnostics"] = diagnosed_at - selected_at
                if self._deadline(budget_elapsed()):
                    row["status"] = "deadline_after_selection"
                    return
                service_started = diagnosed_at
            # The outer scheduler admits the whole Q+polish episode under one
            # cumulative share. Rechecking that share here would starve polish:
            # a short Q trajectory normally spends the small admission deficit.
            # Its full cost is charged once and delays the next admission.
            before = list(gateway.archive.members)
            before_ids = {id(c) for c in before}
            hv_before = frame.hv([c.evaluation.objectives for c in before])
            polish: FastRightShiftPolish | ConcurrentRightShiftPolish | ReusedCommonPolish
            if self.config.polish_method == "single":
                polish = FastRightShiftPolish(self.config.polish_targets == "all")
            elif self.config.polish_method in ("common_fast", "common_peak"):
                self.reused_polish.peak_switch = self.config.polish_method == "common_peak"
                self.reused_polish.all_targets = self.config.polish_targets == "all"
                self.reused_polish.retain([*population, current])
                polish = self.reused_polish
            else:
                polish = ConcurrentRightShiftPolish(
                    coordinated=self.config.polish_method == "coordinated",
                    all_targets=self.config.polish_targets == "all",
                )
            batch = polish.propose(
                gateway.problem,
                current,
                self.config.polish_budget,
                self.operator_config,
                streams.stream("lsv2:fast_rightshift"),
            )
            constructed = self.timer()
            row.update(
                status="empty" if not batch.proposals else "evaluated",
                proposals=len(batch.proposals),
                attempts=batch.attempts,
                diagnostics=batch.diagnostics,
                seconds_construct=constructed - (service_started or started),
            )
            improving = []
            exact_start = gateway.counts["exact"]
            for proposal in batch.proposals:
                if self._deadline(budget_elapsed()):
                    row["status"] = "deadline_partial"
                    break
                candidate = gateway.evaluate(
                    proposal.genotype, proposal.schedule, "LSV2:RIGHTSHIFT"
                )
                if not candidate.evaluation.feasible:
                    raise AssertionError("Fast right shift generated an infeasible schedule")
                if (
                    candidate.genotype != current.genotype
                    or candidate.schedule.starts[1::2] != current.schedule.starts[1::2]
                    or any(
                        less(a, b, TOL.time)
                        for a, b in zip(
                            candidate.schedule.starts[::2], current.schedule.starts[::2]
                        )
                    )
                    or not close(candidate.evaluation.flow, current.evaluation.flow, TOL.time)
                ):
                    raise AssertionError("Fast right shift violated frozen timing semantics")
                if less(candidate.evaluation.bill, current.evaluation.bill, TOL.cost):
                    improving.append(candidate)
            evaluated = self.timer()
            entrants = [c for c in gateway.archive.members if id(c) not in before_ids]
            # Refresh diagnostics for every admitted timing candidate, including non-best ones.
            refreshed = {id(c): _with_diagnostics(gateway.problem, c) for c in entrants}
            gateway.archive.members[:] = [refreshed.get(id(c), c) for c in gateway.archive.members]
            replaced = self.reflow(population, gateway, list(refreshed.values()))
            best = min(improving, key=lambda c: (c.evaluation.bill, identity(c)), default=current)
            row.update(
                exact=gateway.counts["exact"] - exact_start,
                accepted=best is not current,
                next_objectives=list(best.evaluation.objectives),
                bill_gain=current.evaluation.bill - best.evaluation.bill,
                hv_gain=frame.hv([c.evaluation.objectives for c in gateway.archive.members])
                - hv_before,
                entered_archive=len(entrants),
                reflow_replaced=replaced,
                seconds_evaluate=evaluated - constructed,
                seconds_bookkeeping=self.timer() - evaluated,
            )
        finally:
            finished = self.timer()
            row["seconds"] = finished - started
            if service_started is not None:
                service_seconds = finished - service_started
                region = SEGMENTS.index(segment)
                self.polish_selector.charge(region, service_seconds)
                row.update(
                    seconds_service=service_seconds,
                    region_service_totals=tuple(self.polish_selector.region_seconds),
                    complete_empty=row["status"] == "empty",
                    evaluation_complete=row["status"] in ("empty", "evaluated"),
                )
            self.polish_seconds += row["seconds"]
            self._count("polish:" + row["status"])
            if self.retain_steps:
                self.polish_steps.append(row)

    def _propose(
        self,
        name: str,
        current: Candidate,
        frame: FrozenFrame,
        rng: Random,
        problem: ProblemInstance,
    ) -> ProposalBatch:
        cfg = self.operator_config
        budget = self.config.budget
        if name.startswith("P2_"):
            span = (frame.nadir[0] - frame.ideal[0], frame.nadir[1] - frame.ideal[1])
            operator: Operator = TargetedDestroyRepair(
                mode=name.split("_", 1)[1],
                weight=frame.weight_for(current.evaluation.objectives),
                scale=span,
            )
            try:
                return operator.propose(problem, current, budget, cfg, rng)
            except ValueError as exc:
                if "no feasible reinsertion" not in str(exc):
                    raise
                self._count("p2_reinsertion_failure")
                return ProposalBatch()
        if name in self._timing:
            return self._timing[name].propose(problem, current, budget, cfg, rng)
        return self._structural[int(name[1:])].propose(problem, current, budget, cfg, rng)

    def _step(
        self,
        population: list[Candidate],
        gateway: EvaluationGateway,
        streams: RNGManager,
        frame: FrozenFrame,
        current: Candidate,
        box: Box,
        info: dict,
    ) -> tuple[Candidate, float]:
        problem = gateway.problem
        t0 = self.timer()
        archive = gateway.archive
        members_before = list(archive.members)
        before_ids = {id(m) for m in members_before}
        points_before = [m.evaluation.objectives for m in members_before]
        hv_before = frame.hv(points_before)
        norm_before = [frame.normalize(p) for p in points_before]
        box_before = box_hv(norm_before, box)
        current_norm = frame.normalize(current.evaluation.objectives)
        segment = segment_of(current_norm)
        front_norm = [frame.normalize(p) for p in nondominated(points_before)]
        on_front = front_norm.index(current_norm) if current_norm in front_norm else None
        current_gap = (
            gap_scores(front_norm, self.config.end_bonus)[on_front]
            if on_front is not None
            else None
        )
        available = [
            op for op in self.config.operators if op != "A8" or self.gate.available(segment)
        ]
        operator, decision = self._choose_operator(segment, available, streams, current, info)
        exact0, ssgs0 = gateway.counts["exact"], gateway.counts["ssgs"]
        t1 = self.timer()
        batch = self._propose(
            operator, current, frame, streams.stream("lsv2:op:" + operator), problem
        )
        t2 = self.timer()
        context = NormalizationContext(gateway.ideal, maximum(population))
        result = execute(
            batch,
            current,
            self.config.budget,
            frame.weight_for(current.evaluation.objectives),
            context,
            gateway,
            "LSV2:" + operator,
        )
        t3 = self.timer()
        hv_after = frame.hv([m.evaluation.objectives for m in archive.members])
        gain = hv_after - hv_before
        entrants = [m for m in archive.members if id(m) not in before_ids]
        feasible = [s.candidate for s in result.feasible_scored]

        def local_gain(c: Candidate) -> float:
            return box_hv(norm_before + [frame.normalize(c.evaluation.objectives)], box) - (
                box_before
            )

        nxt, decision_rule = _chain_next(current, feasible, entrants, frame, local_gain)
        if nxt is not current:
            nxt = _with_diagnostics(problem, nxt)
        replaced = self.reflow(population, gateway, entrants)
        t4 = self.timer()
        seconds = t4 - t0
        reward = gain / seconds if seconds > 0 else 0.0
        self.bandit.update(segment, operator, reward)
        empty = not batch.proposals
        if operator == "A8":
            self.gate.record(segment, empty)
        self._count("steps")
        self._count("steps:" + operator)
        self._count("empty:" + operator, int(empty))
        self._count("decision:" + decision_rule)
        self._count("reflow_replaced", replaced)
        if self.retain_steps:
            self.steps.append(
                dict(
                    trajectory=info["trajectory"],
                    step=info["step"],
                    frame=self.frame_id,
                    elapsed_ls=self.ls_seconds,
                    seed_objectives=list(info["seed"]),
                    seed_gap=info["seed_gap"],
                    start_objectives=list(current.evaluation.objectives),
                    start_normalized=list(current_norm),
                    segment=SEGMENTS[segment],
                    start_gap=current_gap,
                    operator=operator,
                    available=available,
                    bandit=decision,
                    proposals=len(batch.proposals),
                    attempts=batch.attempts,
                    exact=gateway.counts["exact"] - exact0,
                    ssgs=gateway.counts["ssgs"] - ssgs0,
                    feasible=len(feasible),
                    entered_archive=len(entrants),
                    hv_gain=gain,
                    reward=reward,
                    chain=decision_rule,
                    next_objectives=list(nxt.evaluation.objectives),
                    reflow_replaced=replaced,
                    seconds=seconds,
                    seconds_select=t1 - t0,
                    seconds_construct=t2 - t1,
                    seconds_evaluate=t3 - t2,
                    seconds_bookkeeping=t4 - t3,
                )
            )
        return nxt, gain

    def _choose_operator(
        self,
        segment: int,
        available: Sequence[str],
        streams: RNGManager,
        current: Candidate,
        info: dict,
    ) -> tuple[str, dict]:
        """Selector seam; default keeps the original draw order and bandit behavior."""
        if self.config.selector == "uniform":
            # Matched control: same arms and gate, uniform choice; the bandit only logs.
            pick = streams.stream("lsv2:uniform")
            operator = available[pick.randrange(len(available))]
            decision: dict = {"rule": "uniform", "candidates": len(available)}
        else:
            operator, decision = self.bandit.choose(
                segment, available, streams.stream("lsv2:bandit")
            )
        return operator, decision

    def reflow(
        self, population: list[Candidate], gateway: EvaluationGateway, members: list[Candidate]
    ) -> int:
        """Offer LS Archive entrants to their associated direction's MOEA/D neighborhood.

        Each member is associated in the current population context; at most the
        engine replacement cap of neighbors is replaced, each compared by its own
        weight, exactly as offspring replacement does.
        """
        replaced = 0
        for member in members:
            ctx = NormalizationContext(gateway.ideal, maximum(population))
            direction = associate(ctx.normalize(member), self.lambdas)
            replaced += replace_neighbors(
                population,
                member,
                self.neighbors[direction],
                self.lambdas,
                ctx,
                self.engine_config.replacement_cap,
            )
        return replaced

    def summary(self) -> dict:
        """JSON-ready configuration, accounting and bandit choice distribution."""
        return dict(
            config=asdict(self.config),
            ls_seconds=self.ls_seconds,
            polish_seconds=self.polish_seconds,
            search_start=self.search_start,
            trajectories=self.trajectories,
            frames=self.frame_id + 1,
            counts=dict(sorted(self.counts.items())),
            bandit=self.bandit.summary(),
            a8_gate=dict(
                history={SEGMENTS[s]: list(h) for s, h in self.gate.history.items()},
                skipped={SEGMENTS[s]: v for s, v in self.gate.skipped.items()},
            ),
        )


__all__ = [
    "A8Gate",
    "LSV2Config",
    "LSV2Search",
    "OPERATORS",
    "SEGMENTS",
    "SlidingWindowBandit",
    "box_hv",
    "local_box",
    "segment_of",
]
