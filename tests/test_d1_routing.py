"""Counterexamples for actual-vector gates, paired fallback and exact route results."""

import random
from dataclasses import replace

import pytest

from geo_llm_scheduler.domain.models import Candidate, Genotype, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.d1_routing import (
    GROUPS,
    gate_transfer,
    matched_random_routes,
    routing_probe,
)
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.scheduling.ssgs import decode


def delayed_source(problem):
    genotype = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    schedule = Schedule((300, 500, 600, 800))
    return Candidate(genotype, schedule, evaluate(problem, genotype, schedule))


def test_gate_checks_actual_assignments_order_and_positive_wait():
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    os = replace(g, os=(1, 0, 1, 0))
    ms = replace(os, ms=(1, 0, 0, 0))
    assert gate_transfer(g, os, (1, 0, 0, 0))
    assert not gate_transfer(g, ms, (1, 0, 0, 0))
    assert not gate_transfer(g, g, (1, 0, 0, 0))
    assert not gate_transfer(g, os, (0, 0, 0, 0))
    for intent in ((1,), (-1, 0, 0, 0), (float("nan"), 0, 0, 0)):
        with pytest.raises(ValueError):
            gate_transfer(g, os, intent)
    with pytest.raises(ValueError):
        gate_transfer(g, Genotype((0, 0), (0, 0)), (1, 0, 0, 0))


def test_random_matched_count_does_not_restrict_to_os_and_replays():
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    targets = [replace(g, os=(1, 0, 1, 0)), replace(g, ms=(1, 0, 0, 0))]
    intent = (1, 0, 0, 0)
    observed = {matched_random_routes(g, targets, intent, random.Random(s)) for s in range(20)}
    assert observed == {(True, False), (False, True)}
    a = matched_random_routes(g, targets, intent, random.Random(13))
    assert a == matched_random_routes(g, targets, intent, random.Random(13))
    assert sum(a) == 1
    assert matched_random_routes(g, targets, (0, 0, 0, 0), random.Random(13)) == (False, False)
    assert matched_random_routes(g, [], intent, random.Random(13)) == ()
    with pytest.raises(ValueError):
        matched_random_routes(g, [g], intent, random.Random(1))


def test_routing_exact_candidates_and_preproposal_context_are_replayable(problem):
    source = delayed_source(problem)
    target = replace(source.genotype, os=(1, 0, 1, 0))
    context = NormalizationContext((0, 0), (1e5, 1e5))
    first = routing_probe(problem, source, target, (0.2, 0.8), context, 91, False, 5)
    second = routing_probe(problem, source, target, (0.2, 0.8), context, 91, False, 5)
    assert first["groups"]["CONDITIONAL"]["route"] == "TRUE"
    assert first["groups"]["RANDOM"]["route"] == "A7"
    assert first["context"] == {"ideal": (0, 0), "maximum": (1e5, 1e5)}
    assert first["policy_order"] == second["policy_order"]
    assert first["shared_exact"] == 2 and first["shared_ssgs"] == 1
    for group in GROUPS:
        a, b = first["groups"][group], second["groups"][group]
        assert a["starts"] == b["starts"] and a["scalar"] == b["scalar"]
        assert a["exact"] <= 3 and a["feasible"] == a["exact"] == a["on_time_exact"]
        assert a["relative_gain"] >= 0
        assert evaluate(problem, target, Schedule(a["starts"])).objectives == a["objectives"]
        for proposal in a["proposals"]:
            e = evaluate(problem, target, Schedule(proposal["starts"]))
            assert e.feasible and e.objectives == proposal["objectives"]
    assert first["groups"]["ALWAYS_A7"]["starts"] == first["groups"]["RANDOM"]["starts"]


def test_empty_intent_and_assignment_changes_use_paired_a7(problem):
    genotype = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    schedule = decode(problem, genotype)
    source = Candidate(genotype, schedule, evaluate(problem, genotype, schedule))
    target = replace(genotype, ms=(1, 1, 0, 0))
    result = routing_probe(
        problem, source, target, (0.5, 0.5), NormalizationContext((0, 0), (1e4, 1e4)), 5, True, 5
    )
    assert {r["route"] for r in result["groups"].values()} == {"A7"}
    assert len({r["starts"] for r in result["groups"].values()}) == 1
    assert result["groups"]["ALWAYS_A7"]["extraction_seconds"] == 0
    assert result["groups"]["CONDITIONAL"]["extraction_seconds"] > 0


def test_tiny_quota_rollback_and_invalid_source_are_not_hidden(problem):
    source = delayed_source(problem)
    target = replace(source.genotype, os=(1, 0, 1, 0))
    context = NormalizationContext((0, 0), (1e4, 1e4))
    result = routing_probe(problem, source, target, (0.5, 0.5), context, 19, True, 1e-12)
    assert all(r["base_is_best"] and r["exact"] == 0 for r in result["groups"].values())
    assert result["groups"]["CONDITIONAL"]["overshoot_seconds"] > 0
    for quota in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            routing_probe(problem, source, target, (0.5, 0.5), context, 19, True, quota)
    with pytest.raises(ValueError):
        routing_probe(problem, source, source.genotype, (0.5, 0.5), context, 19, True, 1)
    with pytest.raises(ValueError):
        routing_probe(
            problem,
            replace(source, evaluation=replace(source.evaluation, flow=0)),
            target,
            (0.5, 0.5),
            context,
            19,
            True,
            1,
        )
