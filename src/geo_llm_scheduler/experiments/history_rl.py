"""History-state study, isolated from the default and frozen RL23 policies."""

from dataclasses import replace
from typing import Any, Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import RunResult, run
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.right_shift_policies import PolishPostprocess
from geo_llm_scheduler.experiments.rl23 import run_policy as reference_run
from geo_llm_scheduler.experiments.rl23_support import StudyController
from geo_llm_scheduler.experiments.rl23_support import study_config as reference_config
from geo_llm_scheduler.moead.core import NormalizationContext, maximum
from geo_llm_scheduler.rl.controller import Controller
from geo_llm_scheduler.rl.state import preference
from geo_llm_scheduler.utils.rng import RNGManager

CONTROLS = ("BILL_PERMUTED", "STRICT_TWO_FOCUS", "MATCHED_RANDOM", "MACRO_B3")
ARMS = (
    CONTROLS[:3]
    + (
        "HISTORY_ACTION",
        "HISTORY_RESULT",
        "PREF_HISTORY",
        "PREF_HISTORY_BANDIT",
        "PREF_HISTORY_STOP2",
        "PREF_HISTORY_FRONTIER",
        "PREF_HISTORY_FRONTIER_STOP2",
    )
    + CONTROLS[3:]
    + ("PREF_HISTORY_B3",)
)


def table_rows(arm: str) -> int:
    """Return declared rows, including a distinct episode-start state."""
    if arm not in ARMS:
        raise ValueError("Unknown history study arm")
    if arm in CONTROLS:
        return 42
    return 9 if arm == "HISTORY_ACTION" else 25 if arm == "HISTORY_RESULT" else 75


def study_config(arm: str, seed: int, seconds: float) -> Config:
    """Keep strict trigger, BILL4, P100, B6/L5 and permuted replacement fixed."""
    table_rows(arm)
    if arm in CONTROLS:
        return reference_config(arm, seed, seconds)
    config = reference_config("STRICT_TWO_FOCUS", seed, seconds)
    if arm == "PREF_HISTORY_BANDIT":
        config = replace(config, controller="bandit")
    if arm == "PREF_HISTORY_B3":
        config = replace(config, fixed_budget=3)
    return config


class HistoryPolicy:
    """One run-shared table; history is an explicit argument local to each child."""

    def __init__(self, arm: str) -> None:
        if arm in CONTROLS:
            raise ValueError("Control arms must use their unchanged reference path")
        self.rows = table_rows(arm)
        self.arm = arm
        self.controller: StudyController | None = None
        self.initial_context: NormalizationContext | None = None

    def learner(self, exported: Controller, config: Config) -> StudyController:
        """Expose the actual arrays used for learning to the standard serializer."""
        if self.controller is None:
            self.controller = StudyController(config, self.rows)
            for name in ("q", "visits", "updates", "selections"):
                setattr(exported, name, getattr(self.controller, name))
        return self.controller

    def state(self, index: int, count: int, action: int = 0, outcome: int = 0) -> int:
        """Encode START or the previous executed action and its observed outcome."""
        if count <= 0 or index not in range(count) or action not in range(9):
            raise ValueError("Invalid history state component")
        if outcome not in range(3) or (action == 0 and outcome != 0):
            raise ValueError("Invalid outcome or nonempty episode start")
        if self.arm == "HISTORY_ACTION":
            return action
        history = 0 if action == 0 else 1 + 3 * (action - 1) + outcome
        return history if self.arm == "HISTORY_RESULT" else 25 * preference(index, count) + history

    def outcome(self, quality: float, feasible: int, hv_gain: float = 0.0) -> int:
        """Separate no-feasible, feasible-zero, and productive previous actions."""
        if feasible < 0:
            raise ValueError("Negative feasible count")
        if feasible == 0:
            return 0
        return 2 if quality > 0 or ("FRONTIER" in self.arm and hv_gain > 1e-12) else 1

    def signal(self, quality: float, feasible: int, hv_gain: float = 0.0) -> float:
        """Frontier variants use bounded binary credit; no uncalibrated time subtraction."""
        if "FRONTIER" not in self.arm:
            return quality
        return -1.0 if feasible == 0 else float(quality > 0) + float(hv_gain > 1e-12)

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Candidate, list[dict]]:
        """Invoke the study-only trajectory; default engine behavior is untouched."""
        from geo_llm_scheduler.experiments.history_rl_trajectory import improve

        return improve(*args, policy=self, **kwargs)


def run_policy(
    problem: ProblemInstance,
    arm: str,
    config: Config,
    evidence: dict,
    *,
    observer: Callable[[str, dict], None] | None = None,
    parameters: dict | None = None,
) -> RunResult:
    """Run matched controls verbatim or a sealed history-state treatment."""
    expected = study_config(arm, config.seed, config.seconds or 1.0)
    if replace(config, generations=expected.generations, seconds=expected.seconds) != expected:
        raise ValueError("Undeclared history-study configuration")
    if arm in CONTROLS:
        return reference_run(
            problem, arm, config, evidence, observer=observer, parameters=parameters
        )
    policy = HistoryPolicy(arm)
    factory = charged_initializer(problem, config, "FULL_CURRENT_I4", evidence)

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        members = factory(gateway, streams)
        policy.initial_context = NormalizationContext(gateway.ideal, maximum(members))
        if observer is not None:
            observer("initial_ready", {"population": members, "gateway": gateway})
        return members

    def observe(event: str, data: dict) -> None:
        if event == "initialization":
            policy.learner(data["controller"], config)
        if observer is not None:
            observer(event, data)

    return run(
        problem,
        config,
        initializer=initialize,
        observer=observe,
        retain_trace=False,
        trajectory_override=policy,
        offspring_transform=PolishPostprocess("TWO_FOCUS"),
    )
