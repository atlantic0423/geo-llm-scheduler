"""Counterexamples for bounded D1 projection and paired cost accounting."""

from dataclasses import replace

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, Job, Schedule
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.d1 import (
    ARMS,
    extract_intent,
    paired_probe,
    perturb_structure,
    project_intent,
    shuffle_intent,
)
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.rng import RNGManager


def test_projection_new_release_kv_assignment_and_capacity(problem):
    source_g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    source_s = Schedule((300, 500, 600, 800))
    source = Candidate(source_g, source_s, evaluate(problem, source_g, source_s))
    assert source.evaluation.feasible
    intent = extract_intent(problem, source)
    # New structure has a different KV path and region; old absolute starts
    # would violate this later release. All placements use the new problem.
    later = replace(problem, jobs=(replace(problem.jobs[0], release=1000), problem.jobs[1]))
    target = Genotype((0, 1, 2, 2), (1, 0, 1, 0))
    rebuilt = decode(later, target)
    projected = project_intent(later, target, rebuilt, intent, 0.5)
    assert projected is not None
    assert evaluate(later, target, projected).feasible
    assert projected.starts[0] >= 1000
    assert projected.starts[1] >= projected.end(later, 0) + 30
    assert source.schedule == source_s and source.genotype == source_g
    assert project_intent(later, target, rebuilt, intent, 0) == rebuilt


def test_serial_projection_resource_conflict_and_horizon_cap(problem):
    demanding = replace(
        problem,
        jobs=tuple(
            Job(
                j.name,
                j.release,
                replace(j.prefill, compute=1),
                replace(j.decode, compute=1),
                j.kv_delay,
            )
            for j in problem.jobs
        ),
    )
    genotype = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    rebuilt = decode(demanding, genotype)
    intent = (20.0,) * 4
    projected = project_intent(demanding, genotype, rebuilt, intent, 1, 900)
    assert projected is not None and evaluate(demanding, genotype, projected).feasible
    # Four serial delays can collectively exceed the independently bounded
    # batch extension. Such a recipe must fail, rather than propagate forever.
    assert project_intent(demanding, genotype, rebuilt, (900.0,) * 4, 1, 900) is None
    for values, scale, cap in (
        ((-1.0,) * 4, 1, 900),
        ((float("nan"),) * 4, 1, 900),
        ((1.0,), 1, 900),
        (intent, 2, 900),
        (intent, 1, float("inf")),
    ):
        with pytest.raises(ValueError):
            project_intent(demanding, genotype, rebuilt, values, scale, cap)


def test_phase_shuffle_preserves_histogram_and_explicit_rng():
    values = (1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0)
    streams = RNGManager(17)
    unrelated = RNGManager(17)
    unrelated.stream("search").random()
    result = shuffle_intent(values, streams.stream("D1:shuffle"))
    assert result == shuffle_intent(values, unrelated.stream("D1:shuffle"))
    assert sorted(result[::2]) == sorted(values[::2])
    assert sorted(result[1::2]) == sorted(values[1::2])


@pytest.mark.parametrize("kind", ["OS", "INSTANCE", "REGION"])
def test_changed_structures_are_legal_and_replayable(problem, kind):
    original = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    first = perturb_structure(problem, original, kind, 1, RNGManager(37).stream("structure"))
    second = perturb_structure(problem, original, kind, 1, RNGManager(37).stream("structure"))
    assert first == second and first is not None and first != original
    assert evaluate(problem, first, decode(problem, first)).feasible
    no_alternative = replace(
        problem, instances=(problem.instances[0],), regions=(problem.regions[0],)
    )
    if kind != "OS":
        assert (
            perturb_structure(no_alternative, original, kind, 1, RNGManager(37).stream("s")) is None
        )
    with pytest.raises(ValueError):
        perturb_structure(problem, original, "unknown", 1, RNGManager(37).stream("s"))


def test_paired_probe_quota_fallback_exact_and_fixed_batch_context(problem):
    genotype = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    schedule = Schedule((300, 500, 600, 800))
    source = Candidate(genotype, schedule, evaluate(problem, genotype, schedule))
    target = Genotype((0, 1, 2, 2), (1, 0, 1, 0))
    context = NormalizationContext((0, 0), (10000, 1000))
    result = paired_probe(problem, source, target, (0.5, 0.5), context, 17, 30)
    assert tuple(result["arms"]) != () and set(result["arms"]) == set(ARMS)
    assert result["shared_ssgs"] == 2 and result["shared_exact"] == 3
    assert result["ms_distance"] > 0 and result["os_distance"] > 0
    for stats in result["arms"].values():
        assert stats["exact"] <= 3 and stats["on_time_exact"] <= stats["exact"]
        assert stats["feasible"] == stats["exact"]
        assert stats["scalar"] <= result["base_scalar"]
        assert stats["seconds"] >= 0
    # A tiny quota cannot admit even one full proposal. Late work is charged
    # and reported, never credited as a feasible within-budget improvement.
    tiny = paired_probe(problem, source, target, (0.5, 0.5), context, 17, 1e-12)
    assert all(r["base_is_best"] for r in tiny["arms"].values())
    assert tiny["arms"]["FRESH_A7"]["overshoot_seconds"] > 0
    with pytest.raises(ValueError):
        paired_probe(problem, source, genotype, (0.5, 0.5), context, 17, 30)
    with pytest.raises(ValueError):
        paired_probe(
            problem,
            replace(source, evaluation=evaluate(problem, genotype, decode(problem, genotype))),
            target,
            (0.5, 0.5),
            context,
            17,
            30,
        )


def test_observation_rng_preserves_fixed_generation_engine_semantics(problem):
    config = Config(population=4, neighborhood=2, generations=2, seconds=9999, seed=7)
    original = run(problem, config, retain_trace=False)
    observed_rng = RNGManager(config.seed).stream("D1:observation")

    def observe(event, data):
        if event == "offspring":
            observed_rng.random()

    observed = run(problem, config, retain_trace=False, observer=observe)
    assert observed.population == original.population
    assert observed.archive.members == original.archive.members
    assert observed.controller.q == original.controller.q
    assert observed.gateway.counts == original.gateway.counts
