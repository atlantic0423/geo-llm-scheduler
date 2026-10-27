"""Opt-in small-state Q screen; configurable short episodes, frozen evaluation tables.

Selection, opportunity measurement and update overhead remain charged search work.
Pending transitions resolve at the next *actual* decision, so A8's stateful gate
is queried only once. A new trajectory never supplies a terminal bootstrap.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from random import Random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.lsv2.search import OPERATORS, Box, LSV2Config, LSV2Search
from geo_llm_scheduler.scheduling.timing import legal_bounds
from geo_llm_scheduler.utils.numeric import identity
from geo_llm_scheduler.utils.rng import RNGManager

DESIGNS = ("Q1_RATE", "Q1_NET", "Q2_P_NET", "Q2_W_NET", "Q3_NET")
State = tuple[int, ...]


@dataclass(frozen=True)
class QSpec:
    """Training-fold calibration and predeclared learning parameters."""

    design: str
    gamma: float
    rho: float = 0.0
    h: float = 1.0
    cutoff: float = 0.0
    alpha: float = 0.1
    epsilon: float = 0.2

    def __post_init__(self) -> None:
        if self.design not in DESIGNS or self.gamma not in (0.0, 0.9):
            raise ValueError("Unknown short-episode design/discount")
        if (
            not all(
                math.isfinite(x) for x in (self.rho, self.h, self.cutoff, self.alpha, self.epsilon)
            )
            or self.h <= 0
            or not 0 < self.alpha <= 1
            or not 0 <= self.epsilon <= 1
        ):
            raise ValueError("Invalid calibration or learning parameters")


@dataclass
class QTable:
    """Sparse shared table; reads and frozen evaluation never allocate entries."""

    values: dict[str, dict[str, float]] = field(default_factory=dict)
    visits: dict[str, dict[str, int]] = field(default_factory=dict)

    @staticmethod
    def key(state: State) -> str:
        """Canonical JSON-friendly discrete state key."""
        return ":".join(map(str, state))

    def choose(
        self, state: State, available: Sequence[str], epsilon: float, rng: Random
    ) -> tuple[str, float]:
        """Epsilon-greedy with uniform exact ties and the full marginal propensity."""
        if not available or not 0 <= epsilon <= 1:
            raise ValueError("Invalid action mask or epsilon")
        row = self.values.get(self.key(state), {})
        top = max(row.get(a, 0.0) for a in available)
        best = [a for a in available if row.get(a, 0.0) == top]
        pool = list(available) if rng.random() < epsilon else best
        action = pool[rng.randrange(len(pool))]
        probability = epsilon / len(available) + (
            (1 - epsilon) / len(best) if action in best else 0.0
        )
        return action, probability

    def update(
        self,
        state: State,
        action: str,
        reward: float,
        next_state: State | None,
        available: Sequence[str],
        spec: QSpec,
        *,
        training: bool,
    ) -> dict:
        """Masked one-step TD; None means terminal, including shortened trajectories."""
        if action not in OPERATORS or not math.isfinite(reward):
            raise ValueError("Invalid action or reward")
        if next_state is not None and not available:
            raise ValueError("Nonterminal transition needs actual next mask")
        row = self.values.get(self.key(state), {})
        old = row.get(action, 0.0)
        future = (
            0.0
            if next_state is None
            else max(self.values.get(self.key(next_state), {}).get(a, 0.0) for a in available)
        )
        target = reward + spec.gamma * future
        new = old + spec.alpha * (target - old) if training else old
        if not math.isfinite(new) or not math.isfinite(target):
            raise ValueError("Nonfinite Q target")
        if training:
            self.values.setdefault(self.key(state), {})[action] = new
            counts = self.visits.setdefault(self.key(state), {})
            counts[action] = counts.get(action, 0) + 1
        return dict(
            terminal=next_state is None,
            next_state=next_state,
            next_available=list(available),
            q_old=old,
            q_new=new,
            td_target=target,
            bootstrap=future,
            training=training,
        )


def encode_state(
    spec: QSpec,
    segment: int,
    step: int,
    available: Sequence[str],
    previous_event: int,
    feature: float = 0.0,
    *,
    trajectory_steps: int = 5,
) -> State:
    """Encode only information known before action; opportunities split at train median."""
    if (
        trajectory_steps < 1
        or segment not in range(3)
        or step not in range(trajectory_steps)
        or previous_event not in range(6)
    ):
        raise ValueError("State outside declared cardinality")
    gate = int("A8" in available)
    base = (previous_event if spec.design == "Q3_NET" else segment, trajectory_steps - step, gate)
    return base + (int(feature > spec.cutoff),) if spec.design.startswith("Q2_") else base


def event_of(action: str, current_changed: bool, shared_changed: bool) -> int:
    """Encode the previous observed transition, never its reward or the next outcome."""
    if current_changed:
        return 3 if action == "P3" else 2 if action in ("A7", "A8") else 1
    return 4 if shared_changed else 5


class QLearningSearch(LSV2Search):
    """LSV2 with a shared Q table; gate, proposals, acceptance and reflow are inherited."""

    def __init__(
        self,
        engine: Config,
        operators: Config,
        spec: QSpec,
        table: QTable,
        *,
        training: bool,
        lsv2: LSV2Config | None = None,
    ) -> None:
        super().__init__(lsv2 or LSV2Config(), engine, operators, retain_steps=True)
        self.spec, self.table, self.training = spec, table, training
        self._pending: dict | None = None
        self._event = 0
        self._feature = 0.0
        self._feature_seconds = 0.0
        self._work: float | None = None
        self._identities: dict[int, tuple[Candidate, tuple]] = {}

    def _opportunity(self, problem: ProblemInstance, current: Candidate) -> float:
        # Same arithmetic as the diagnostic reference, computing only the selected proxy.
        if self.spec.design == "Q2_W_NET":
            if self._work is None:
                self._work = sum(
                    problem.profile(o).duration for o in range(problem.operation_count)
                )
            return sum(current.schedule.resource_wait) / max(self._work, 1e-12)
        positive = sum(
            max(0.0, hi - lo) > 1e-9
            for o in range(0, problem.operation_count, 2)
            for lo, hi in [legal_bounds(problem, current.genotype, current.schedule, o)]
        )
        return positive / max(1, len(problem.jobs))

    def _identity_snapshot(
        self, current: Candidate, population: list[Candidate], archive: list[Candidate]
    ) -> tuple[tuple, list[tuple], set[tuple]]:
        fresh: dict[int, tuple[Candidate, tuple]] = {}

        def key(c: Candidate) -> tuple:
            cached = fresh.get(id(c)) or self._identities.get(id(c))
            value = cached[1] if cached is not None and cached[0] is c else identity(c)
            fresh[id(c)] = (c, value)
            return value

        result = (key(current), [key(c) for c in population], {key(c) for c in archive})
        # Strong references prevent id reuse; prune to current live state every snapshot.
        self._identities = fresh
        return result

    def _settle(self, state: State | None, available: Sequence[str]) -> None:
        if self._pending is not None:
            tick = self.timer()
            row = self._pending
            row["q_transition"] = self.table.update(
                tuple(row["q_state"]),
                row["operator"],
                row["q_reward"],
                state,
                available,
                self.spec,
                training=self.training,
            )
            row["q_update_seconds"] = self.timer() - tick
            self._pending = None

    def _choose_operator(
        self,
        segment: int,
        available: Sequence[str],
        streams: RNGManager,
        current: Candidate,
        info: dict,
    ) -> tuple[str, dict]:
        tick = self.timer()
        state = encode_state(
            self.spec,
            segment,
            info["step"],
            available,
            self._event,
            self._feature,
            trajectory_steps=self.config.trajectory_steps,
        )
        self._settle(state, available)
        action, probability = self.table.choose(
            state,
            available,
            self.spec.epsilon if self.training else 0.0,
            streams.stream("lsv2:qpolicy"),
        )
        return action, dict(
            rule="qlearning",
            state=state,
            probability=probability,
            feature=self._feature,
            previous_event=self._event,
            state_update_select_seconds=self.timer() - tick,
        )

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
        tick = self.timer()
        if self.spec.design.startswith("Q2_"):
            self._feature = self._opportunity(gateway.problem, current)
        self._feature_seconds = self.timer() - tick
        before = (
            self._identity_snapshot(current, population, gateway.archive.members)
            if self.spec.design == "Q3_NET"
            else None
        )
        nxt, gain = super()._step(population, gateway, streams, frame, current, box, info)
        row = self.steps[-1]
        if before is not None:
            after = self._identity_snapshot(nxt, population, gateway.archive.members)
            changed = before[0] != after[0]
            shared = before[1:] != after[1:]
            self._event = event_of(row["operator"], changed, shared)
            row.update(current_changed=changed, shared_changed=shared, next_event=self._event)
        # Includes state/identity measurement, previous TD update and inherited bookkeeping.
        # Current transition's future update is charged by its next decision/episode end.
        seconds = self.timer() - tick
        reward = (
            gain / seconds
            if self.spec.design == "Q1_RATE"
            else (gain - self.spec.rho * seconds) / self.spec.h
        )
        row.update(
            q_state=row["bandit"]["state"],
            q_reward=reward,
            charged_step_seconds=seconds,
            feature_seconds=self._feature_seconds,
        )
        self._pending = row
        return nxt, gain

    def _trajectory(
        self,
        population: list[Candidate],
        gateway: EvaluationGateway,
        streams: RNGManager,
        budget_elapsed: Callable[[], float],
    ) -> None:
        assert self._pending is None
        self._event = 0
        super()._trajectory(population, gateway, streams, budget_elapsed)
        self._settle(None, ())


class FixedA7Search(LSV2Search):
    """Matched reference selecting A7 with all other LSV2 mechanisms unchanged."""

    def _choose_operator(
        self,
        segment: int,
        available: Sequence[str],
        streams: RNGManager,
        current: Candidate,
        info: dict,
    ) -> tuple[str, dict]:
        if "A7" not in available:
            raise ValueError("A7 missing from matched action set")
        return "A7", dict(rule="fixed_a7", probability=1.0)
