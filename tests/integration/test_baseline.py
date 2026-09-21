"""Baseline integration: every exact feasible candidate independently visits Archive."""

import random
from dataclasses import replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.io.validation import validate_genotype
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, weights
from geo_llm_scheduler.moead.variation import crossover, reproduce


def test_baseline_replay_archive_dispatch(problem):
    config = Config(population=10, neighborhood=3, generations=2)
    a, b = run(problem, config), run(problem, config)
    assert [{k: v for k, v in row.items() if k != "elapsed"} for row in a.trace] == [
        {k: v for k, v in row.items() if k != "elapsed"} for row in b.trace
    ]
    assert a.gateway.counts["exact"] == 30
    assert a.archive.attempts == a.gateway.counts["feasible"]
    assert all(c.evaluation.feasible for c in a.population)
    assert weights(3) == ((0, 1), (0.5, 0.5), (1, 0))
    context = NormalizationContext(a.gateway.ideal, maximum(a.population))
    assert all(context.scalar(c, (0.5, 0.5)) >= 0 for c in a.population)


def test_job_linked_variation(problem):
    a, b = Genotype((0, 1, 2, 2), (0, 1, 0, 1)), Genotype((2, 2, 0, 0), (1, 0, 1, 0))
    for seed in range(50):
        child = crossover(a, b, random.Random(seed))
        validate_genotype(problem, child)
        for j in range(2):
            assert child.ms[2 * j : 2 * j + 2] in (a.ms[2 * j : 2 * j + 2], b.ms[2 * j : 2 * j + 2])
        validate_genotype(
            problem,
            reproduce(
                problem, a, b, replace(Config(), mutation_probability=1), random.Random(seed)
            ),
        )
