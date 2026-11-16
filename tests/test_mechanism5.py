"""Five-arm isolation, balanced scheduling and observation counter counterexamples."""

import importlib.util
import random
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.engine.trigger import triggered
from geo_llm_scheduler.experiments.comparison13 import arm_config, case_problem
from geo_llm_scheduler.experiments.mechanism5 import (
    ARMS,
    StructuralObserver,
    balanced_order,
    mechanism_config,
    study_specs,
)
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.moead.core import NormalizationContext, weights


def test_arms_change_only_declared_gate_and_polish():
    expected = [
        ("preference", 0.1, True),
        ("strict", 0, True),
        ("preference", 0.1, False),
        ("strict", 0, False),
        ("preference", 0.1, False),
    ]
    for arm, (mode, delta, polish) in zip(ARMS, expected):
        c = mechanism_config(arm, 19101, 300)
        assert (c.trigger_mode, c.trigger_delta, c.polish) == (mode, delta, polish)
        source = arm_config("MOEAD" if arm == "E_MOEAD" else "M11_I4", 19101, 300)
        assert asdict(
            replace(
                c,
                trigger_mode=source.trigger_mode,
                trigger_delta=source.trigger_delta,
                polish=source.polish,
            )
        ) == asdict(source)
    assert mechanism_config(ARMS[-1], 1, 300).method == "plain"
    with pytest.raises(ValueError):
        mechanism_config("bad", 1, 300)


def test_all_formal_cells_and_rotation_are_fresh_and_paired():
    specs = study_specs()
    assert len(specs) == len({s["key"] for s in specs}) == 64
    assert len({s["base_seed"] for s in specs}) == 16
    assert [s["n"] for s in specs] == [20, 50, 100, 200] * 16
    assert not {s["base_seed"] for s in specs} & {s["base_seed"] for s in study_specs(True)}
    for base in {s["base_seed"] for s in specs}:
        rows = [s for s in specs if s["base_seed"] == base]
        assert {(s["variant"], s["seed"]) for s in rows} == {
            (v, a) for v in (0, 1) for a in (19101, 19201)
        }
    for n in (20, 50, 100, 200):
        assert {s["pressure"] for s in specs if s["n"] == n} == {
            "arrival",
            "kv",
            "cheap_capacity",
            "demand",
        }


def test_williams_all_positions_and_predecessors():
    rows = [balanced_order(i) for i in range(10)]
    assert all(set(r) == set(ARMS) for r in rows)
    assert all(Counter(r[i] for r in rows) == dict.fromkeys(ARMS, 2) for i in range(5))
    assert set(Counter((a, b) for r in rows for a, b in zip(r, r[1:])).values()) == {2}


def point(flow, bill, tag=0):
    return Candidate(
        Genotype((tag, tag), (0, 0)),
        Schedule((0.0, 0.0)),
        EvaluationResult(True, flow, bill, (), ()),
    )


def test_strict_index_and_zero_quality_tolerance_and_zero_vector():
    context = NormalizationContext((0, 0), (10, 10))
    w = weights(100)
    incumbent = point(10, 0)
    broad = mechanism_config("A_CURRENT", 1, 300)
    strict = mechanism_config("B_STRICT", 1, 300)
    # Both low-flow-weight endpoints belong to the same preference, but strict
    # requires the exact associated direction.
    assert triggered(point(10, 0), incumbent, 1, w, context, broad, random.Random(0))
    assert not triggered(point(10, 0), incumbent, 1, w, context, strict, random.Random(0))
    assert triggered(point(0, 0), incumbent, 99, w, context, strict, random.Random(0))
    assert not triggered(
        point(10.5, 0.0105), point(10, 0.01), 0, w, context, strict, random.Random(0)
    )
    assert triggered(point(10.5, 0.0105), point(10, 0.01), 0, w, context, broad, random.Random(0))


def test_structural_observer_retains_repeats_and_distinguishes_timing():
    observer = StructuralObserver()
    a, b = point(1, 2), point(2, 1, tag=1)
    observer.candidate(a, 1)
    observer.initialized = True
    observer.candidate(a, 2)
    observer.candidate(b, 3)
    observer.candidate(b, 3)
    observer.candidate(b, 4)
    summary = observer.summary([a, b, b], [a, b, b])
    assert summary["counts"]["search:structural_evaluations"] == 3
    assert summary["counts"]["search:novel_structure_in_run"] == 1
    assert summary["counts"]["search:repeated_structure_in_run"] == 2
    assert summary["counts"]["search:repeated_structure_in_search"] == 1
    assert summary["counts"]["search:timing_evaluations"] == 1
    assert summary["archive_unique_genotypes"] == 2 and summary["archive_members"] == 3


def test_observer_does_not_change_fixed_work_candidates_q_or_rng():
    problem = case_problem(6, 2, 4, 19, "arrival", 0)
    cfg = Config(
        population=6, neighborhood=2, generations=1, method="full", seed=19, trigger_mode="always"
    )
    seen = StructuralObserver()
    snapshots = []

    def observe(event, data):
        if event == "candidate":
            seen.candidate(data["candidate"], data["gateway"].counts["ssgs"])
        if event == "initialization":
            seen.initialized = True
        if event == "offspring":
            snapshots.append(data["streams"].snapshot())

    a = run(problem, cfg)
    b = run(problem, cfg, observer=observe)
    assert [solution(c) for c in a.population] == [solution(c) for c in b.population]
    assert [solution(c) for c in a.archive.members] == [solution(c) for c in b.archive.members]
    assert a.gateway.counts == b.gateway.counts and a.controller.q == b.controller.q
    assert snapshots and seen.counts["search:structural_evaluations"] > 0


def test_prepare_and_hash_validation_reject_mutated_input(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_mechanism5.py"
    spec = importlib.util.spec_from_file_location("mechanism5_runner_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    manifest = module.prepare(tmp_path / "formal")
    assert manifest["runs"] == 320 and manifest["arms"] == list(ARMS)
    assert module.validate_inputs(tmp_path / "formal") == manifest
    name = next(n for n in manifest["files"] if n.startswith("inputs/"))
    (tmp_path / "formal" / name).write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError):
        module.validate_inputs(tmp_path / "formal")
