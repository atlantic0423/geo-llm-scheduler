"""Check shared optimizations through baseline and adaptive-budget run paths."""

from unittest.mock import patch

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments.classic_baselines import run_classic
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.nsga2 import run_nsga2
from geo_llm_scheduler.experiments.runner import deterministic_trace
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.utils.rng import RNGManager
from research.performance_reuse_20261020.reference_runtime import reference_runtime


@pytest.mark.parametrize(
    "variant",
    (
        "MOEAD",
        "NSGAII",
        "SPEA2",
        "EPS_IBEA",
        "RK_MOPSO",
        "fixed",
        "static",
        "random",
        "severity",
        "coverage",
        "severity_v2",
        "coverage_v2",
        "sequential",
    ),
)
def test_variant_complete_candidate_archive_q_and_rng_match_frozen_runtime(variant):
    problem = case_problem(8, 3, 6, 20261020, "demand", 1)
    baseline = variant in ("MOEAD", "NSGAII", "SPEA2", "EPS_IBEA", "RK_MOPSO")
    config = Config(
        population=8,
        neighborhood=4,
        generations=2,
        seed=1520,
        method="plain" if baseline else "full",
        trigger_mode="always",
        rl_steps=2,
        budget_policy="fixed" if baseline else variant,
        polish=not baseline,
    )

    def execute():
        candidates, managers = [], []
        original_evaluate = EvaluationGateway.evaluate
        original_init = RNGManager.__init__

        def initialize_rng(self, seed):
            original_init(self, seed)
            managers.append(self)

        def evaluate(self, *args, **kwargs):
            candidate = original_evaluate(self, *args, **kwargs)
            candidates.append(candidate)
            return candidate

        def initialize(gateway, streams):
            return [
                gateway.evaluate(g, origin="differential_initial")
                for g in initial_genotypes(problem, config, streams.stream("initialization"))
            ]

        with (
            patch.object(RNGManager, "__init__", initialize_rng),
            patch.object(EvaluationGateway, "evaluate", evaluate),
        ):
            if variant == "NSGAII":
                result = run_nsga2(problem, config, initializer=initialize)
            elif variant in ("SPEA2", "EPS_IBEA", "RK_MOPSO"):
                result = run_classic(problem, config, variant, initialize)
            else:
                result = run(problem, config, initializer=initialize)
        return (
            candidates,
            result.population,
            result.archive.members,
            result.archive.attempts,
            result.archive.insertions,
            dict(result.gateway.counts),
            result.controller.q,
            result.controller.updates,
            deterministic_trace(result.trace),
            result.offspring_count,
            [manager.snapshot() for manager in managers],
        )

    optimized = execute()
    with reference_runtime():
        original = execute()
    assert optimized == original
