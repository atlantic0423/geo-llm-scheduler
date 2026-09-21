"""Accepted A7 is followed by full diagnosis without structural decoding."""

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.trajectory import improve
from geo_llm_scheduler.rl.controller import Controller
from geo_llm_scheduler.scheduling.ssgs import refresh_diagnostics
from geo_llm_scheduler.utils.rng import RNGManager


def test_accepted_timing_refresh(problem):
    gateway = EvaluationGateway(problem, Archive())
    g = Genotype((0, 1, 0, 1), (0, 1, 0, 1))
    s = refresh_diagnostics(problem, g, Schedule((0, 500, 200, 700)))
    x = gateway.evaluate(g, s)
    config = Config(
        population=2,
        neighborhood=2,
        rl_steps=2,
        enabled_operators=(7,),
        fixed_budget=10,
        polish=False,
    )
    y, records = improve(
        x,
        0,
        0,
        0,
        ((0, 1), (1, 0)),
        (2000, 1000),
        gateway,
        Controller(config),
        config,
        RNGManager(1),
    )
    assert any(r["accepted"] for r in records)
    assert gateway.counts["ssgs"] == 0
    assert y.genotype == g
    assert y.schedule == refresh_diagnostics(problem, g, y.schedule)
    assert gateway.archive.attempts == gateway.counts["feasible"]
