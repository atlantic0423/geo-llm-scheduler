"""Explicit research-only state, credit and action-eligibility policies."""

import math
from dataclasses import replace
from typing import Any

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.experiments.metrics import hypervolume
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.rl.controller import Controller
from geo_llm_scheduler.rl.state import extract, state_count
from geo_llm_scheduler.scheduling.resources import est_for
from geo_llm_scheduler.utils.numeric import TOL

ARMS = (
    "BILL_BIRTH",
    "BILL_PERMUTED",
    "STRICT_REFERENCE",
    "STRICT_NO_POLISH",
    "STRICT_TWO_FOCUS",
    "STRICT_SPARSE_SCAN",
    "REWARD_TIME_PENALTY",
    "REWARD_BOUNDED_RATE",
    "REWARD_ARCHIVE_TIME",
    "TIME_DISCOUNT_Q",
    "STATE_TIME42",
    "STATE_TIME84",
    "STATE_ACTIVE42",
    "MATCHED_RANDOM",
    "CONTEXTUAL_BANDIT",
    "POOLED6",
    "TRAJECTORY_L1",
    "TRAJECTORY_L3",
    "TRAJECTORY_STOP2",
    "MACRO_B1",
    "MACRO_B3",
    "LEGAL_OPPORTUNITY_MASK",
    "TIME_PENALTY_ACTIVE42",
)


def study_config(arm: str, seed: int, seconds: float) -> Config:
    """Return only declared base-engine changes; research credit is separately sealed."""
    from geo_llm_scheduler.experiments.right_shift_policies import policy_config

    if arm not in ARMS:
        raise ValueError("Unknown RL23 arm")
    cfg = replace(
        policy_config("BILL_MOEAD" if arm.startswith("BILL_") else "TWO_FOCUS", seed, seconds),
        trigger_mode="strict",
        trigger_delta=0.0,
    )
    if arm == "BILL_PERMUTED":
        cfg = replace(cfg, replacement_policy="permuted")
    if arm in ("MATCHED_RANDOM", "CONTEXTUAL_BANDIT"):
        cfg = replace(cfg, controller="random" if arm == "MATCHED_RANDOM" else "bandit")
    if arm == "POOLED6":
        cfg = replace(cfg, rl_state_policy="pooled")
    if arm in ("TRAJECTORY_L1", "TRAJECTORY_L3"):
        cfg = replace(cfg, rl_steps=1 if arm.endswith("L1") else 3)
    if arm in ("MACRO_B1", "MACRO_B3"):
        cfg = replace(cfg, fixed_budget=1 if arm.endswith("B1") else 3)
    return cfg


class StudyController(Controller):
    """Use the original selector/update with an explicit necessary-condition mask."""

    def __init__(self, config: Config, rows: int) -> None:
        super().__init__(config)
        self.q = [[0.0] * 8 for _ in range(rows)]
        self.visits = [0] * rows
        self.updates = [[0] * 8 for _ in range(rows)]
        self.selections = [[0] * 8 for _ in range(rows)]
        self.allowed: dict[int, tuple[int, ...]] = {}

    def available_actions(self, state: int) -> tuple[int, ...]:
        """Apply a cached necessary-condition mask, including for bootstrap."""
        return self.allowed.get(state, super().available_actions(state))


class StudyPolicy:
    """One immutable treatment definition and one run-local learner, never global state."""

    def __init__(self, arm: str, reference_seconds: float = 1.0) -> None:
        if arm not in ARMS or not math.isfinite(reference_seconds) or reference_seconds <= 0:
            raise ValueError("Invalid RL23 policy or shared time reference")
        self.arm = arm
        self.reference_seconds = reference_seconds
        self.controller: StudyController | None = None
        self.initial_context: NormalizationContext | None = None
        self._mask_candidate: Candidate | None = None
        self._mask_problem: ProblemInstance | None = None
        self._mask_enabled: tuple[int, ...] = ()
        self._mask_actions: tuple[int, ...] = ()

    def learner(self, exported: Controller, config: Config) -> StudyController:
        """Share actual learned arrays with the engine's serialized controller."""
        if self.controller is None:
            rows = 84 if self.arm == "STATE_TIME84" else state_count(config.rl_state_policy)
            self.controller = StudyController(config, rows)
            for name in ("q", "visits", "updates", "selections"):
                setattr(exported, name, getattr(self.controller, name))
        return self.controller

    def state(
        self,
        index: int,
        values: tuple[float, ...],
        stagnant: int,
        config: Config,
        current: Candidate,
        context: NormalizationContext,
        weight: tuple[float, float],
        progress: float,
    ) -> int:
        """Encode real elapsed stage or the actual active Tchebycheff objective."""
        base = extract(index, values, stagnant, config)
        if self.arm == "STATE_TIME42":
            return 2 * (base // 2) + int(progress >= 0.5)
        if self.arm == "STATE_TIME84":
            return 2 * base + int(progress >= 0.5)
        if self.arm in ("STATE_ACTIVE42", "TIME_PENALTY_ACTIVE42"):
            flow, bill = [abs(v) * w for v, w in zip(context.normalize(current), weight)]
            active = 0 if flow > 1.1 * bill else 2 if bill > 1.1 * flow else 1
            return active * 14 + base % 14
        return base

    def mask(
        self, learner: StudyController, state: int, problem: ProblemInstance, current: Candidate
    ) -> None:
        """Only exclude A4/A5 when every operation lacks required positive Flow wait."""
        if self.arm != "LEGAL_OPPORTUNITY_MASK":
            return
        if (
            current is not self._mask_candidate
            or problem is not self._mask_problem
            or learner.config.enabled_operators != self._mask_enabled
        ):
            has_wait = any(
                current.schedule.starts[o] - est_for(problem, current.genotype, current.schedule, o)
                > TOL.time
                for o in range(problem.operation_count)
            )
            self._mask_actions = tuple(
                a for a in learner.config.enabled_operators if has_wait or a not in (4, 5)
            )
            self._mask_candidate = current
            self._mask_problem = problem
            self._mask_enabled = learner.config.enabled_operators
        learner.allowed[state] = self._mask_actions

    def archive_value(self, members: list[Candidate]) -> float:
        """Measure exact 2-D rectangle union in the fixed initialization context."""
        if self.initial_context is None:
            raise ValueError("Online reward requires initialization-only reference")
        points = [self.initial_context.normalize(c) for c in members]
        return hypervolume([p for p in points if p[0] <= 1.1 and p[1] <= 1.1], (1.1, 1.1))

    def signal(self, quality: float, duration: float, archive_gain: float = 0.0) -> float:
        """Use fixed lambda=.25 and common per-size pilot median; never per-action scaling."""
        cost = max(0.0, duration) / self.reference_seconds
        if self.arm in ("REWARD_TIME_PENALTY", "TIME_PENALTY_ACTIVE42"):
            return quality - 0.25 * cost
        if self.arm == "REWARD_BOUNDED_RATE":
            return quality / max(cost, 0.1) if quality >= 0 else quality - 0.25 * cost
        if self.arm == "REWARD_ARCHIVE_TIME":
            return (
                quality if quality < 0 else min(10.0, 100 * max(0.0, archive_gain))
            ) - 0.25 * cost
        return quality

    def update(
        self,
        learner: StudyController,
        state: int,
        action: int,
        signal: float,
        next_state: int,
        duration: float,
        terminal: bool,
    ) -> None:
        """Keep terminal future zero; time-discount changes only nonterminal bootstrap."""
        if self.arm != "TIME_DISCOUNT_Q":
            learner.update(state, action, signal, next_state, terminal=terminal)
            return
        future = (
            0.0
            if terminal
            else max(learner.q[next_state][a - 1] for a in learner.available_actions(next_state))
        )
        discount = learner.config.gamma ** (max(0.0, duration) / self.reference_seconds)
        old = learner.q[state][action - 1]
        learner.q[state][action - 1] += learner.config.alpha * (signal + discount * future - old)
        learner.updates[state][action - 1] += 1

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Candidate, list[dict]]:
        """Invoke the explicit research trajectory seam without touching engine defaults."""
        from geo_llm_scheduler.experiments.rl23_trajectory import improve

        return improve(*args, study=self, **kwargs)
