"""Causal interventions, unchanged RNG replay and independent exact evidence."""

from dataclasses import replace

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.initial_frontier import (
    METHODS,
    SCENARIOS,
    diagnostic_problem,
    evaluate_population,
    sample_population,
)
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.utils.rng import RNGManager


def test_mixed_replay_and_labels(problem):
    cfg = replace(Config(), population=20, neighborhood=10, seed=1101)
    trace = sample_population(problem, cfg, "MIXED")
    assert trace.genotypes == initial_genotypes(
        problem, cfg, RNGManager(cfg.seed).stream("initialization")
    )
    assert trace.original_replay_verified and len(trace.modes) == 20
    assert set(trace.modes) == set(range(10))
    pop, archive, evidence = evaluate_population(problem, trace)
    assert evidence["independent_exact_verified"] == 20
    assert len(pop) == 20 and archive.members
    assert all(c.origin == f"initial_mode_{m}" for c, m in zip(pop, trace.modes, strict=True))


@pytest.mark.parametrize("method", METHODS[1:])
def test_pure_controls_and_determinism(problem, method):
    cfg = replace(Config(), population=10, neighborhood=5, seed=27)
    a = sample_population(problem, cfg, method)
    b = sample_population(problem, cfg, method)
    assert a.genotypes == b.genotypes and len(set(a.genotypes)) == 10
    assert set(a.modes) == {METHODS.index(method) - 1}
    assert not a.original_replay_verified


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_validity_and_tariff_pair(scenario):
    h = diagnostic_problem(12, 1405001, "H", scenario)
    t = diagnostic_problem(12, 1405001, "T", scenario)
    assert h.jobs == t.jobs and h.instances == t.instances
    assert h.regions != t.regions
    assert h.regions[0].tariffs[-1].end >= max(j.release for j in h.jobs) + sum(
        j.prefill.duration + j.decode.duration + j.kv_delay for j in h.jobs
    )
    cfg = replace(Config(), population=10, neighborhood=5)
    _, _, evidence = evaluate_population(t, sample_population(t, cfg, "MIXED"))
    assert evidence["independent_exact_verified"] == 10


def test_each_stress_changes_only_declared_job_field():
    base = diagnostic_problem(12, 1405001, "H", "mixed")
    for scenario in SCENARIOS[1:]:
        p = diagnostic_problem(12, 1405001, "H", scenario)
        assert p.instances == base.instances
        for b, j in zip(base.jobs, p.jobs, strict=True):
            assert j.name == b.name
            assert j.release == (0 if scenario == "burst" else b.release)
            assert j.kv_delay == (4 * b.kv_delay if scenario == "kv_delay" else b.kv_delay)
            for bp, jp in zip((b.prefill, b.decode), (j.prefill, j.decode), strict=True):
                assert jp.duration == bp.duration
                assert jp.compute == (
                    min(0.95, 1.5 * bp.compute) if scenario == "compute_tight" else bp.compute
                )
                assert jp.vram == (min(15.5, 2 * bp.vram) if scenario == "vram_tight" else bp.vram)
        assert [r.demand_rate for r in p.regions] == [
            r.demand_rate * (5 if scenario == "demand_high" else 1) for r in base.regions
        ]


def test_insufficient_pure_population_has_no_silent_fallback(problem):
    cfg = replace(
        Config(),
        population=20,
        neighborhood=10,
        initialization_attempts=5,
        initialization_perturbation=0,
    )
    trace = sample_population(problem, cfg, "FCFS_LOAD")
    assert len(trace.genotypes) == 1 and trace.attempts == 5 and trace.duplicates == 4
    assert trace.modes == [0]


def test_unknown_conditions_rejected(problem):
    with pytest.raises(ValueError):
        diagnostic_problem(12, 1, "X", "mixed")
    with pytest.raises(ValueError):
        sample_population(problem, Config(), "POLISHED")
