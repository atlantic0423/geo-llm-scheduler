"""Actual parent provenance, bounded sampling, paired routing and hash failures."""

import json
import random
from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, Schedule
from geo_llm_scheduler.engine import offspring
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import d1_realpath as campaign
from geo_llm_scheduler.experiments.d1_realpath_analysis import analyse_realpath, weighted_run_values
from geo_llm_scheduler.experiments.d1_realpath_observation import (
    RealPathObserver,
    allocate_routes,
    retained_gains,
)
from geo_llm_scheduler.experiments.d1_routing import GROUPS, routing_probe
from geo_llm_scheduler.experiments.p1_observation import SemanticRecorder
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext, weights
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.utils.rng import RNGManager


def delayed_source(problem):
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    schedule = Schedule((300, 500, 600, 800))
    return Candidate(g, schedule, evaluate(problem, g, schedule))


def test_variation_first_parent_is_prespecified_and_has_caller_weight(problem, monkeypatch):
    population = [
        delayed_source(problem),
        replace(delayed_source(problem), genotype=Genotype((1, 1, 1, 1), (1, 0, 1, 0))),
    ]
    observed = []
    monkeypatch.setattr(offspring, "reproduce", lambda p, a, b, c, r: b)
    gateway = EvaluationGateway(problem, Archive())
    offspring.create_offspring(
        population,
        0,
        (0, 1),
        Config(),
        gateway,
        RNGManager(4),
        observer=lambda e, v: observed.append(v),
    )
    event = observed[0]
    a, b = event["parents"]
    assert event["source"] is population[a]
    assert event["target"] == population[b].genotype != population[a].genotype


def test_macro_observer_sees_unique_actually_evaluated_structure_only(problem):
    gateway = EvaluationGateway(problem, Archive())
    x = gateway.evaluate(Genotype((0, 0, 0, 0), (0, 1, 0, 1)))
    a, b = replace(x.genotype, ms=(1, 0, 0, 0)), replace(x.genotype, ms=(0, 1, 0, 0))
    observed = []
    result = execute(
        ProposalBatch([Proposal(a), Proposal(a), Proposal(b)], 3),
        x,
        1,
        (0.5, 0.5),
        NormalizationContext(gateway.ideal, (1000, 1000)),
        gateway,
        "A1",
        structural_observer=observed.append,
    )
    assert observed == [a] and result.effective == 1


def test_observation_preserves_archive_population_rng_and_q(problem):
    config = Config(
        population=6,
        neighborhood=3,
        generations=2,
        seed=31,
        method="full",
        trigger_mode="always",
        rl_steps=2,
        enabled_operators=(1,),
        fixed_budget=6,
    )
    signatures = []
    for collect in (False, True):
        recorder = SemanticRecorder()
        observer = RealPathObserver(problem, 100, 31, cap=2)

        def callback(event, data):
            recorder(event, data)
            if collect:
                observer(event, data)

        result = run(problem, config, retain_trace=False, observer=callback)
        assert result.observer_seconds > 0
        signatures.append(recorder.signature(result))
        if collect:
            data = observer.export()
            assert observer.events > 12
            assert "path:variation" in data["counts"] and "path:A1" in data["counts"]
            assert len(data["rows"]) <= 18
            assert all(tuple(r["weight"]) == weights(6)[r["subproblem"]] for r in data["rows"])
    assert signatures[0] == signatures[1]
    assert run(problem, replace(config, generations=1)).observer_seconds == 0


def test_reservoir_bound_gate_cache_probabilities_and_unchanged(problem):
    source = delayed_source(problem)
    target = replace(source.genotype, os=(1, 0, 1, 0))
    observer = RealPathObserver(problem, 10, 8, cap=3)
    values = {
        "source": source,
        "target": target,
        "path": "variation",
        "elapsed": 1,
        "subproblem": 0,
        "population": [source] * 6,
        "generation": 0,
        "weight": (0, 1),
        "context": NormalizationContext((0, 0), (10000, 10000)),
    }
    for i in range(100):
        observer("structure", {**values, "target": source.genotype if i == 0 else target})
    data = observer.export()
    assert data["counts"]["all"] == {
        "events": 100,
        "unchanged": 1,
        "os_only": 99,
        "ms_changed": 0,
        "gate_true": 99,
    }
    assert len(data["rows"]) == 3 and all(r["probability"] == 0.03 for r in data["rows"])
    assert len(observer.cache) <= 32
    with pytest.raises(ValueError):
        RealPathObserver(problem, 0, 1)


def test_matched_random_cross_source_assignment_pool_and_degenerate_blocks(problem):
    source = delayed_source(problem)
    rows = [
        {
            "source": source,
            "target": replace(source.genotype, os=(1, 0, 1, 0)),
            "stage": 0,
            "preference": 0,
        },
        {
            "source": source,
            "target": replace(source.genotype, ms=(1, 0, 0, 0)),
            "stage": 0,
            "preference": 0,
        },
        {"source": source, "target": source.genotype, "stage": 0, "preference": 0},
    ]
    intents = [(1, 0, 0, 0)] * 3
    observed = {allocate_routes(rows, intents, random.Random(s))["routes"] for s in range(20)}
    assert observed == {(True, False, False), (False, True, False)}
    result = allocate_routes(rows, intents, random.Random(7))
    assert result["blocks"][0]["true"] == 1 and not result["blocks"][0]["degenerate"]
    assert allocate_routes(rows[:1], intents[:1], random.Random(1))["blocks"][0]["degenerate"]
    with pytest.raises(ValueError):
        allocate_routes(rows, [], random.Random(1))


def test_updated_selection_unchanged_route_and_independent_exact_rejects_tamper(problem):
    source = delayed_source(problem)
    row = routing_probe(
        problem,
        source,
        source.genotype,
        (0.2, 0.8),
        NormalizationContext((0, 0), (10000, 10000)),
        9,
        True,
        5,
        allow_unchanged=True,
        normalization="updated",
    )
    assert {r["route"] for r in row["groups"].values()} == {"A7"}
    assert len({r["starts"] for r in row["groups"].values()}) == 1
    assert campaign.audit_case(problem, row) >= 4
    assert set(retained_gains(row)) == set(GROUPS)
    row["base_starts"] = tuple(t + 1 for t in row["base_starts"])
    with pytest.raises(ValueError, match="SSGS"):
        campaign.audit_case(problem, row)
    with pytest.raises(ValueError, match="normalization"):
        routing_probe(
            problem,
            source,
            replace(source.genotype, os=(1, 0, 1, 0)),
            (0.2, 0.8),
            NormalizationContext((0, 0), (1e4, 1e4)),
            9,
            True,
            5,
            normalization="invalid",
        )


def test_weighting_respects_natural_preference_frequency_and_missing_phase():
    buckets = {f"{s}:{p}": 0 for s in range(3) for p in range(3)}
    buckets.update({"0:0": 100, "0:1": 10})
    rows = [
        {
            "stage": 0,
            "preference": 0,
            "bucket_events": 100,
            "bucket_sampled": 1,
            "gain": dict.fromkeys(GROUPS, 0),
        },
        {
            "stage": 0,
            "preference": 1,
            "bucket_events": 10,
            "bucket_sampled": 1,
            "gain": dict.fromkeys(GROUPS, 1),
        },
    ]
    result = weighted_run_values(rows, buckets, "gain")
    assert result["CONDITIONAL"] == pytest.approx(10 / 110 / 3)
    with pytest.raises(ValueError):
        weighted_run_values(rows[1:], buckets, "gain")


@pytest.fixture
def small_campaign(tmp_path, problem, monkeypatch):
    repository = campaign.Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        campaign, "git_metadata", lambda *a: {"git_commit": "unit", "git_dirty": False}
    )
    monkeypatch.setattr(campaign, "generate_e15_pair", lambda *a: (problem, problem))
    monkeypatch.setattr(
        campaign,
        "load_config",
        lambda *a: Config(
            population=6, neighborhood=3, method="full", trigger_mode="always", rl_steps=2
        ),
    )
    root = tmp_path / "pilot"
    manifest = campaign.prepare_realpath(root, repository, pilot_seconds=0.02)
    return root, manifest


def test_campaign_hashes_worker_exact_and_completed_pilot_queue(small_campaign):
    root, manifest = small_campaign
    assert len(manifest["keys"]) == 8
    for key in manifest["keys"]:
        source = campaign.execute_realpath_job(root, "samples", key)
        assert source["cases"] <= 144
        probe = campaign.execute_realpath_job(root, "probes", key)
        assert probe["true_counts"]["CONDITIONAL"] == probe["true_counts"]["RANDOM"]
        assert campaign.validate_realpath_job(root, "probes", key) == (True, "complete")
    report = analyse_realpath(root, 100)
    assert report["verification_status"] == "PILOT_VALIDATED" and "primary" not in report
    assert campaign.run_realpath(root, 1, 1)["state"] == "complete"
    key = manifest["keys"][0]
    with pytest.raises(FileExistsError):
        campaign.execute_realpath_job(root, "samples", key)
    (root / "samples" / key / "data.json").write_text("{}")
    assert not campaign.validate_realpath_job(root, "samples", key)[0]
    with pytest.raises(ValueError):
        campaign.run_realpath(root, 1, 1)


def test_prepare_freeze_drift_stop_and_failure_preservation(small_campaign):
    root, manifest = small_campaign
    with pytest.raises(FileExistsError):
        campaign.prepare_realpath(root, root, 1)
    for workers in (0, 7):
        with pytest.raises(ValueError):
            campaign.run_realpath(root, workers, 1)
    (root / "stop.request").write_text("stop before dispatch")
    assert campaign.run_realpath(root, 1, 1)["state"] == "paused"
    failure = root / "ops/failures/preserved.json"
    failure.parent.mkdir()
    failure.write_text("{}")
    with pytest.raises(ValueError):
        campaign.run_realpath(root, 1, 1)
    (root / "manifest.json").write_text(json.dumps({**manifest, "source_hash": "drift"}))
    with pytest.raises(ValueError, match="checksum"):
        campaign.frozen_manifest(root)
