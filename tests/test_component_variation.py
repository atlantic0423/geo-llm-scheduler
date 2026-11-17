"""Structural invariants and independent exact checks for v4 operators."""

import random
from collections import Counter

import pytest

from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.moead.component_variation import (
    CROSSOVERS,
    ORDER_MUTATIONS,
    WITHIN_MUTATIONS,
    cross,
    mutate_order,
    mutate_within,
)
from geo_llm_scheduler.moead.variation import crossover
from geo_llm_scheduler.scheduling.ssgs import decode


@pytest.mark.parametrize("rule", CROSSOVERS)
def test_cross_exact_and_pair_inheritance(problem, rule):
    a = Genotype((0, 1, 1, 0), (0, 1, 0, 1))
    b = Genotype((2, 2, 0, 1), (1, 1, 0, 0))
    for seed in range(30):
        g = cross(a, b, rule, random.Random(seed))
        assert g == cross(a, b, rule, random.Random(seed))
        assert Counter(g.os) == Counter(a.os)
        for j in range(2):
            assert g.ms[2 * j : 2 * j + 2] in (a.ms[2 * j : 2 * j + 2], b.ms[2 * j : 2 * j + 2])
        assert evaluate(problem, g, decode(problem, g)).feasible


def test_default_crossover_preserves_random_stream():
    a = Genotype((0, 0, 1, 1), (0, 0, 1, 1))
    b = Genotype((1, 1, 0, 0), (1, 0, 1, 0))
    for seed in range(30):
        r, s = random.Random(seed), random.Random(seed)
        assert cross(a, b, "job_linked", r) == crossover(a, b, s)
        assert r.getstate() == s.getstate()


@pytest.mark.parametrize("rule", ORDER_MUTATIONS)
def test_order_conserves_jobs_and_exact(problem, rule):
    g = Genotype((0, 1, 1, 0), (0, 0, 1, 1))
    for seed in range(30):
        changed = mutate_order(g, rule, random.Random(seed))
        assert changed.ms == g.ms
        assert Counter(changed.os) == Counter(g.os)
        assert evaluate(problem, changed, decode(problem, changed)).feasible
    assert mutate_order(Genotype((), ()), rule, random.Random(1)) == Genotype((), ())


@pytest.mark.parametrize("rule", WITHIN_MUTATIONS)
def test_within_preserves_regions_and_exact(problem, rule):
    g = Genotype((0, 0, 1, 1), (0, 1, 0, 1))
    for seed in range(30):
        changed = mutate_within(problem, g, rule, random.Random(seed))
        assert changed.os == g.os
        assert all(problem.instances[m].region == 0 for m in changed.ms)
        assert evaluate(problem, changed, decode(problem, changed)).feasible


def test_invalid_rules_rejected(problem):
    g = Genotype((0, 0, 1, 1), (0, 1, 0, 1))
    with pytest.raises(ValueError):
        cross(g, g, "unknown", random.Random(1))
    with pytest.raises(ValueError):
        mutate_order(g, "unknown", random.Random(1))
    with pytest.raises(ValueError):
        mutate_within(problem, g, "unknown", random.Random(1))
