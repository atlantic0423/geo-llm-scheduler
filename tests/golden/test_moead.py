"""Numerical MOEA/D counterexamples that detect lambda and ideal mistakes."""

from dataclasses import replace

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, replace_neighbors


def candidate(flow, bill, t):
    return Candidate(
        Genotype((0, 0), (0, 0)), Schedule((t, t + 1)), EvaluationResult(True, flow, bill, (), ())
    )


def test_neighbor_own_lambda_and_cap():
    a, b, y = candidate(10, 100, 0), candidate(100, 10, 1), candidate(20, 20, 2)
    population = [a, b]
    ctx = NormalizationContext((0, 0), (100, 100))
    assert replace_neighbors(population, y, (0, 1), ((1, 0), (0, 1)), ctx, 2) == 0
    population = [a, b]
    assert replace_neighbors(population, y, (0, 1), ((0, 1), (1, 0)), ctx, 1) == 1
    assert population == [y, b]
    assert maximum(population) == (100, 20)


def test_archive_independent_and_phenotype_aware():
    archive = Archive()
    a = candidate(10, 100, 0)
    b = candidate(100, 10, 1)
    assert archive.consider(a) and archive.consider(b)
    assert archive.consider(replace(a, schedule=Schedule((0.5, 1.5))))
    assert not archive.consider(a)
    assert not archive.consider(candidate(101, 101, 2))
    assert archive.attempts == 5
