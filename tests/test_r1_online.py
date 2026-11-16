"""Online R1 mixer: spliced members are genuine exact candidates; archive mode is passive."""

import random

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import (
    Genotype,
    Job,
    PowerModel,
    ProblemInstance,
    Profile,
    Region,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.r1_online import R1Options, RegionalBlockMixer, mix_genotype
from geo_llm_scheduler.moead.core import weights
from geo_llm_scheduler.utils.rng import RNGManager


def _problem() -> ProblemInstance:
    tariffs = (Tariff(0, 700, 0.2), Tariff(700, 2000, 0.9), Tariff(2000, 100000, 0.4))
    return ProblemInstance(
        tuple(
            Job(str(i), 0 if i < 4 else 650, Profile(150 + 40 * i, 0.6, 2), Profile(300, 0.5, 2))
            for i in range(8)
        ),
        tuple(Region(str(r), tariffs, 2 + r) for r in range(2)),
        tuple(ServingInstance(str(m), m // 2, 8, 10 + m, 100) for m in range(4)),
        PowerModel(0.4, 0.8, 0.6),
    )


@pytest.mark.parametrize("mode", ["population", "archive"])
def test_mixer_keeps_exact_members_and_respects_mode(mode):
    p = _problem()
    rng = random.Random(4)
    gateway = EvaluationGateway(p, Archive(), None, 0)
    regions = [j % 2 for j in range(8)]

    def member():
        os = [j for j in range(8) for _ in range(2)]
        rng.shuffle(os)
        ms = tuple(2 * regions[j] + rng.randrange(2) for j in range(8) for _ in range(2))
        return gateway.evaluate(Genotype(ms, tuple(os)))

    population = [member() for _ in range(6)]
    lambdas = weights(6)
    mixer = RegionalBlockMixer(p, R1Options(mode=mode, probability=1.0, recipients=5))
    streams = RNGManager(7)
    for _ in range(30):
        before = list(population)
        child = member()
        mixer(population, child, tuple(range(6)), lambdas, gateway, streams)
        for c in population:
            assert evaluate(p, c.genotype, c.schedule) == c.evaluation
        if mode == "archive":
            assert population == before
    assert mixer.counts["predicted"] > 0 and mixer.counts["exact"] <= mixer.counts["predicted"]
    if mode == "population":
        assert mixer.counts["accepted"] == mixer.counts["improved"] > 0
    g = mix_genotype(population[0].genotype, population[1].genotype, frozenset({0, 2}))
    assert sorted(g.os) == sorted(population[1].genotype.os)
