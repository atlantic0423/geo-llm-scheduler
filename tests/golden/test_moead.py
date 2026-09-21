"""Numerical MOEA/D counterexamples that detect lambda and ideal mistakes."""

from dataclasses import replace

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.moead import core
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, replace_neighbors
from geo_llm_scheduler.utils.numeric import Tolerances


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
    assert archive.insertions == 3
    assert archive.peak_size == 3


def test_archive_insertions_are_not_final_contributions():
    dominated_first = Archive()
    dominated_first.consider(candidate(20, 20, 0))
    dominated_first.consider(candidate(10, 10, 1))
    dominant_first = Archive()
    dominant_first.consider(candidate(10, 10, 1))
    dominant_first.consider(candidate(20, 20, 0))
    assert len(dominated_first.members) == len(dominant_first.members) == 1
    assert dominated_first.insertions == 2
    assert dominant_first.insertions == 1


def test_scalar_comparison_does_not_use_cost_tolerance(monkeypatch):
    monkeypatch.setattr(core, "TOL", Tolerances(cost=1.0, scalar=1e-9))
    population = [candidate(20, 20, 0)]
    better = candidate(19, 20, 1)
    context = NormalizationContext((0, 0), (100, 100))
    assert core.replace_neighbors(population, better, (0,), ((1, 0),), context, 1) == 1
