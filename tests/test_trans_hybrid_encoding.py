"""Regression for exact genotype and timing round-trip into continuous keys."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_lsv2_fast_rightshift import synthetic

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import ext_hybrid
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.trans_common import SearchContext
from geo_llm_scheduler.experiments.trans_hybrid import HybridSearchContext
from geo_llm_scheduler.experiments.trans_hybrid_encoding import encode_schedule
from geo_llm_scheduler.experiments.trans_runner import (
    ALGORITHMS,
    AVAILABLE_ALGORITHMS,
    HYBRID,
    run_case,
)
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.scheduling.timing import legal_bounds, move
from geo_llm_scheduler.utils.numeric import TOL, close
from geo_llm_scheduler.utils.rng import RNGManager


def test_key_round_trip_for_initial_and_representable_shifted_schedule():
    problem, fixture = synthetic(4)
    schedule = decode(problem, fixture.genotype)
    initial = replace(
        fixture, schedule=schedule, evaluation=evaluate(problem, fixture.genotype, schedule)
    )
    ctx = SearchContext(problem, 1, 30)
    keys = encode_schedule(problem, initial)
    assert keys is not None
    replay = ctx.evaluate(keys)
    assert replay.candidate.genotype == initial.genotype
    assert replay.candidate.schedule.starts == initial.schedule.starts
    assert replay.candidate.evaluation == initial.evaluation

    # Draw an actual nonzero waiting-key phenotype and ensure it is replayable.
    richer = case_problem(8, 2, 2, 123, "arrival", 0)
    ctx = SearchContext(richer, 42, 30)
    for _ in range(100):
        keys2 = ctx.rng.random(ctx.dimensions)
        shifted = ctx.evaluate(keys2).candidate
        if shifted.schedule.starts != decode(richer, shifted.genotype).starts:
            restored = encode_schedule(richer, shifted)
            assert restored is not None
            roundtrip = ctx.evaluate(restored).candidate
            assert roundtrip.genotype == shifted.genotype
            assert all(
                close(a, b, TOL.time)
                for a, b in zip(roundtrip.schedule.starts, shifted.schedule.starts, strict=True)
            )
            break
    else:
        raise AssertionError("Fixture did not provide a feasible movable prefill")


def test_unrepresentable_decode_delay_is_not_silently_inserted():
    problem, fixture = synthetic(4)
    schedule = decode(problem, fixture.genotype)
    op = 1
    _, upper = legal_bounds(problem, fixture.genotype, schedule, op)
    assert np.isinf(upper)
    delayed = move(problem, fixture.genotype, schedule, op, schedule.starts[op] + 1)
    assert delayed is not None
    candidate = replace(
        fixture, schedule=delayed, evaluation=evaluate(problem, fixture.genotype, delayed)
    )
    assert encode_schedule(problem, candidate) is None


def test_hybrid_entries_do_not_modify_frozen_14_method_matrix():
    assert len(ALGORITHMS) == 14
    assert not set(HYBRID).intersection(ALGORITHMS)
    assert set(ALGORITHMS) | set(HYBRID) <= set(AVAILABLE_ALGORITHMS)


def test_first_hybrid_initial_population_matches_existing_baselines_exactly():
    problem, _ = synthetic(4)
    config = ext_hybrid.arm_config("EXTH_MOEAD", 1, 60)
    evidence = {}
    gateway = EvaluationGateway(problem, Archive())
    expected = ext_hybrid.hybrid_initializer(problem, config, evidence)(gateway, RNGManager(1))
    ctx = HybridSearchContext(problem, seed=1, seconds=60)
    actual = ctx.initialize(100)
    assert [x.candidate for x in actual] == expected
    assert (
        ctx.hybrid_evidence[0]["initial_population_sha256"] == evidence["initial_population_sha256"]
    )
    assert ctx.hybrid_evidence[0]["initial_archive_sha256"] == evidence["initial_archive_sha256"]


@pytest.mark.parametrize("algorithm", HYBRID)
def test_hybrid_methods_use_charged_initializer_and_generation_polish(tmp_path, algorithm):
    problem, _ = synthetic(4)
    source = tmp_path / "instance.json"
    save_instance(problem, source)
    out = tmp_path / "out"
    summary = run_case(
        algorithm, source, out, seed=1, seconds=60, population_size=6, max_generations=1
    )
    assert summary["independent_exact_audit"] == "passed"
    info = summary["run_evidence"]
    expected = 1 if algorithm == "CMOEA_CD_HYBRID_PD" else 2
    assert len(info["initializations"]) == expected
    assert all(
        (e["heuristic"], e["bill"], e["random"]) == (20, 1, 79) for e in info["initializations"]
    )
    assert info["generations"] == 1
    assert info["polish"]["calls"] >= expected
    assert (
        info["polish"]["reencoded"] + info["polish"]["unrepresentable"]
        == info["polish"]["accepted"]
    )
    config = json.loads((out / "config.json").read_text(encoding="utf8"))
    assert config["enhancement"] and config["seconds_including_initialization"] == 60
