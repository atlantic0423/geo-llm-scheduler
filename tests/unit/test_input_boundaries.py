"""Malformed input fails at the boundary, before objective evaluation."""

from dataclasses import replace

import pytest

from geo_llm_scheduler.domain.models import Genotype, Tariff
from geo_llm_scheduler.io.validation import validate_genotype, validate_problem


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "duplicate",
        "region",
        "phase",
        "nan",
        "power",
        "rate",
        "tariff_start",
        "tariff_nan",
        "gap",
        "release",
        "profile_nan",
        "duration",
        "eligible",
        "zero_region",
    ],
)
def test_invalid_problem(problem, case):
    p = problem
    m = p.instances[0]
    r = p.regions[0]
    j = p.jobs[0]
    if case == "empty":
        p = replace(p, regions=())
    elif case == "duplicate":
        p = replace(p, jobs=(j, j))
    elif case == "region":
        p = replace(p, instances=(replace(m, region=9), *p.instances[1:]))
    elif case == "phase":
        p = replace(p, instances=(replace(m, phases=(2,)), *p.instances[1:]))
    elif case == "nan":
        p = replace(p, instances=(replace(m, vram=float("nan")), *p.instances[1:]))
    elif case == "power":
        p = replace(p, instances=(replace(m, active_kw=0), *p.instances[1:]))
    elif case == "rate":
        p = replace(p, regions=(replace(r, demand_rate=-1), p.regions[1]))
    elif case == "tariff_start":
        p = replace(p, regions=(replace(r, tariffs=(Tariff(1, 10, 0.1),)), p.regions[1]))
    elif case == "tariff_nan":
        p = replace(p, regions=(replace(r, tariffs=(Tariff(0, 10, float("nan")),)), p.regions[1]))
    elif case == "gap":
        p = replace(
            p, regions=(replace(r, tariffs=(Tariff(0, 5, 0.1), Tariff(6, 10, 0.1))), p.regions[1])
        )
    elif case == "release":
        p = replace(p, jobs=(replace(j, release=-1),))
    elif case == "profile_nan":
        p = replace(p, jobs=(replace(j, prefill=replace(j.prefill, compute=float("nan"))),))
    elif case == "duration":
        p = replace(p, jobs=(replace(j, prefill=replace(j.prefill, duration=0)),))
    elif case == "eligible":
        p = replace(p, jobs=(replace(j, prefill=replace(j.prefill, vram=100)),))
    elif case == "zero_region":
        p = replace(p, instances=p.instances[:2])
    with pytest.raises(ValueError):
        validate_problem(p)


def test_integer_and_same_region_assignments(problem):
    for ms in ((0.0, 0, 0, 0), (0, 2, 0, 0), (99, 0, 0, 0)):
        with pytest.raises(ValueError):
            validate_genotype(problem, Genotype(ms, (0, 1, 0, 1)))
