"""Controlled, research-only invocation policies on the unchanged BILL4 main algorithm."""

from dataclasses import replace
from time import perf_counter
from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import RunResult, run
from geo_llm_scheduler.experiments.component_random import arm_config as component_config
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.operators.polish_policies import policy_polish
from geo_llm_scheduler.operators.right_shift import PolishStats
from geo_llm_scheduler.utils.rng import RNGManager

ARMS = ("REFERENCE_REUSE", "TWO_FOCUS", "SPARSE_SCAN", "NO_POLISH", "BILL_MOEAD")


def policy_config(arm: str, seed: int, seconds: float) -> Config:
    """Keep A3/permuted/B6/L5/Q42 and BILL4 settings; place Polish outside Q credit."""
    if arm not in ARMS:
        raise ValueError("Unknown right-shift policy arm")
    original = component_config(
        "BILL_MOEAD" if arm == "BILL_MOEAD" else "FULL_CURRENT_I4", seed, seconds
    )
    return replace(original, polish=False)


class PolishPostprocess:
    """Apply exactly one declared, charged invocation policy with a separate RNG stream."""

    def __init__(self, arm: str, *, reuse: bool = True) -> None:
        if arm not in ARMS:
            raise ValueError("Unknown right-shift policy arm")
        self.arm = arm
        self.reuse = reuse

    def __call__(
        self, child: Candidate, entered_ls: bool, gateway: EvaluationGateway, streams: RNGManager
    ) -> Candidate:
        """Sample 5% conditional on LS, 10% conditional on no LS; never add them to15%."""
        if self.arm in ("NO_POLISH", "BILL_MOEAD"):
            return child
        group = "ls" if entered_ls else "no_ls"
        gateway.counts["polish_eligible:" + group] += 1
        if self.arm == "SPARSE_SCAN":
            chance = 0.05 if entered_ls else 0.10
            gateway.counts["polish_sampling_draws:" + group] += 1
            invoke = streams.stream("polish_sampling").random() < chance
            policy = "sequential_scan"
        else:
            invoke = entered_ls
            policy = "two_focus" if self.arm == "TWO_FOCUS" else "reference_reuse"
        if not invoke:
            return child
        started = perf_counter()
        gateway.counts["polish_calls"] += 1
        gateway.counts["polish_calls:" + group] += 1
        stats = PolishStats()
        result = policy_polish(gateway.problem, child, gateway, policy, stats, reuse=self.reuse)
        gateway.counts["polish_candidates"] += stats.candidates
        gateway.counts["polish_exact"] += stats.exact
        gateway.counts["polish_accepted"] += stats.accepted
        gateway.counts["polish_improved_calls"] += int(result is not child)
        gateway.seconds["polish"] += perf_counter() - started
        return result


def run_policy(
    problem: ProblemInstance,
    arm: str,
    config: Config,
    evidence: dict,
    *,
    observer: Callable[[str, dict], None] | None = None,
) -> RunResult:
    """Charge the identical BILL4 initializer and invoke only the declared treatment."""
    expected = policy_config(arm, config.seed, config.seconds or 1.0)
    if replace(config, generations=expected.generations, seconds=expected.seconds) != expected:
        raise ValueError("Policy study configuration differs from its declared treatment")
    factory = charged_initializer(
        problem, config, "BILL_MOEAD" if arm == "BILL_MOEAD" else "FULL_CURRENT_I4", evidence
    )

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        members = factory(gateway, streams)
        if observer is not None:
            observer("initial_ready", {"population": members, "gateway": gateway})
        return members

    return run(
        problem,
        config,
        initializer=initialize,
        observer=observer,
        retain_trace=False,
        offspring_transform=PolishPostprocess(arm),
    )
