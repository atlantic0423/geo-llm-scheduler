"""Twenty-three frozen research arms on the original component33 development panel."""

from dataclasses import replace
from typing import Callable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import RunResult, run
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.right_shift_policies import PolishPostprocess
from geo_llm_scheduler.experiments.rl23_support import ARMS, StudyPolicy, study_config
from geo_llm_scheduler.moead.core import NormalizationContext, maximum
from geo_llm_scheduler.utils.rng import RNGManager


def run_policy(
    problem: ProblemInstance,
    arm: str,
    config: Config,
    evidence: dict,
    *,
    observer: Callable[[str, dict], None] | None = None,
    parameters: dict | None = None,
    initial_order: Callable[[list[Candidate]], list[Candidate]] | None = None,
) -> RunResult:
    """Charge common BILL4 initialization; keep study policies outside engine defaults.

    ``initial_order`` optionally permutes the finished initial members onto
    subproblem slots. It runs inside the charged clock after the unchanged
    initializer, so the initial member set and every RNG draw stay identical.
    Omitting it preserves the original random slot correspondence.
    """
    expected = study_config(arm, config.seed, config.seconds or 1.0)
    # decode_cache_size is a semantics-preserving engineering option, not a study parameter.
    declared = replace(
        config,
        generations=expected.generations,
        seconds=expected.seconds,
        decode_cache_size=expected.decode_cache_size,
    )
    if declared != expected:
        raise ValueError("Undeclared RL23 base configuration")
    parameters = parameters or {"time_reference_seconds": {str(n): 1.0 for n in (20, 50, 100, 200)}}
    reference = parameters["time_reference_seconds"].get(str(len(problem.jobs)), 1.0)
    study = StudyPolicy(arm, float(reference))
    factory = charged_initializer(
        problem, config, "BILL_MOEAD" if arm.startswith("BILL_") else "FULL_CURRENT_I4", evidence
    )

    def initialize(gateway: EvaluationGateway, streams: RNGManager) -> list[Candidate]:
        members = factory(gateway, streams)
        if initial_order is not None:
            ordered = initial_order(members)
            if len(ordered) != len(members) or {id(c) for c in ordered} != {id(c) for c in members}:
                raise ValueError("Initial slot order must permute the same members")
            members = ordered
        study.initial_context = NormalizationContext(gateway.ideal, maximum(members))
        if observer is not None:
            observer("initial_ready", {"population": members, "gateway": gateway})
        return members

    policy = {
        "BILL_BIRTH": "BILL_MOEAD",
        "BILL_PERMUTED": "BILL_MOEAD",
        "STRICT_NO_POLISH": "NO_POLISH",
        "STRICT_REFERENCE": "REFERENCE_REUSE",
        "STRICT_SPARSE_SCAN": "SPARSE_SCAN",
    }.get(arm, "TWO_FOCUS")

    def observe(event: str, data: dict) -> None:
        # Allocate the declared table even if a strict gate never admits LS.
        if event == "initialization":
            study.learner(data["controller"], config)
        if observer is not None:
            observer(event, data)

    return run(
        problem,
        config,
        initializer=initialize,
        observer=observe,
        retain_trace=False,
        trajectory_override=study,
        offspring_transform=PolishPostprocess(policy),
    )


__all__ = ["ARMS", "run_policy", "study_config"]
