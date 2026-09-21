"""Hand-selected directions and quality bounds for the selective search gate."""

import random
from dataclasses import replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.engine.trigger import triggered
from geo_llm_scheduler.moead.core import NormalizationContext, weights


def point(flow, bill):
    return Candidate(
        Genotype((), ()), Schedule(()), EvaluationResult(True, flow, bill, (), ()), "test"
    )


def test_direction_quality_and_zero_vector():
    config = Config(population=6, neighborhood=2)
    w = weights(6)
    context = NormalizationContext((0, 0), (10, 10))
    incumbent = point(5, 5)
    # High Flow/low bill points align with the low Flow-weight endpoint.
    assert triggered(point(10, 0), incumbent, 0, w, context, config, random.Random(0))
    assert not triggered(point(10, 0), incumbent, 5, w, context, config, random.Random(0))
    assert not triggered(point(5, 10), incumbent, 0, w, context, config, random.Random(0))
    assert triggered(point(0, 0), incumbent, 5, w, context, config, random.Random(0))
    strict = replace(config, trigger_mode="strict")
    assert not triggered(point(10, 0), incumbent, 1, w, context, strict, random.Random(0))
    assert triggered(point(10, 0), incumbent, 0, w, context, strict, random.Random(0))
    fixed = replace(config, trigger_mode="fixed", fixed_ls_probability=0)
    assert not triggered(incumbent, incumbent, 0, w, context, fixed, random.Random(0))
    assert triggered(
        incumbent,
        incumbent,
        0,
        w,
        context,
        replace(fixed, fixed_ls_probability=1),
        random.Random(0),
    )
