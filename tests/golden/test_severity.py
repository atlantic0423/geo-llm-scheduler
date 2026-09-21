"""Independent experimental severity conventions, including zero cases."""

from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.diagnostics.severity import severities
from geo_llm_scheduler.domain.models import Genotype, Schedule, Tariff
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics


def test_flat_tou_and_region(problem):
    gateway = EvaluationGateway(problem, Archive())
    x = gateway.evaluate(Genotype((0, 0, 0, 0), (0, 1, 0, 1)))
    values = severities(problem, x)
    assert values[:5] == pytest.approx((0, 0, 1, 0, 0))
    assert all(0 <= v <= 1 for v in values)


def test_active_energy_weighted_tou(problem):
    p = replace(
        problem,
        regions=(
            replace(problem.regions[0], tariffs=(Tariff(0, 150, 0.1), Tariff(150, 10000, 0.3))),
            problem.regions[1],
        ),
    )
    gateway = EvaluationGateway(p, Archive())
    x = gateway.evaluate(Genotype((0, 0, 0, 0), (0, 1, 0, 1)))
    assert severities(p, x)[3] == pytest.approx(0.5)


def test_timing_diagnostics_separates_intent(problem):
    gateway = EvaluationGateway(problem, Archive())
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    s = refresh_diagnostics(problem, g, Schedule((50, 200, 50, 200)))
    x = gateway.evaluate(g, s)
    assert severities(problem, x)[0] == 0


def test_empty_batch(problem):
    p = replace(problem, jobs=())
    x = EvaluationGateway(p, Archive()).evaluate(Genotype((), ()), Schedule(()))
    assert severities(p, x) == (0,) * 6
