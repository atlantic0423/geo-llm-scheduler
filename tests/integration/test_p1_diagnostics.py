"""P1 observational invariance, checkpoint recovery and counterfactual boundaries."""

import gzip
import json
from dataclasses import asdict, replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import EvaluationResult
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.nsga2 import run_nsga2
from geo_llm_scheduler.experiments.p1_diagnostics import diagnose_checkpoint, prefix_audit
from geo_llm_scheduler.experiments.p1_observation import (
    P1Observer,
    SemanticRecorder,
    replacement_audit,
    restore_candidate,
    sample_population,
)
from geo_llm_scheduler.experiments.p1_worker import execute_spec, validate_spec
from geo_llm_scheduler.experiments.runner import solution, source_digest
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.moead.core import NormalizationContext, maximum
from geo_llm_scheduler.operators.active_pack import ActivePack, select_packing_moves
from geo_llm_scheduler.rl.controller import Controller
from geo_llm_scheduler.utils.rng import RNGManager


@pytest.mark.parametrize("method", ["full", "plain", "nsga2", "nsga2_memetic"])
def test_compact_history_preserves_candidates_decisions_q_and_archive(problem, method):
    config = Config(
        population=6,
        neighborhood=3,
        generations=4,
        seed=81,
        method=method,
        trigger_mode="always",
        rl_steps=2,
    )
    runner = run_nsga2 if method.startswith("nsga2") else run
    full, compact = SemanticRecorder(), SemanticRecorder()
    a = runner(problem, config, retain_trace=True, observer=full)
    b = runner(problem, config, retain_trace=False, observer=compact)
    assert full.signature(a) == compact.signature(b)
    assert a.offspring_count == b.offspring_count == 24
    assert len(a.trace) == 24 and b.trace == []


def test_rng_snapshot_is_observational():
    streams = RNGManager(17)
    a = streams.stream("A7")
    before = a.getstate()
    assert streams.snapshot() == {"A7": before}
    assert a.getstate() == before
    assert RNGManager(17).stream("A7").random() == a.random()


def test_sampling_duplicates_are_missing_not_resampled(problem):
    result = run(problem, Config(population=3, neighborhood=2, generations=1))
    population = [result.population[0]] * 9
    rows = sample_population(population, RNGManager(1).stream("sampling"))
    assert rows[0]["probability"] == 1
    assert sum(row["index"] is not None for row in rows) == 1
    assert [r.get("missing") for r in rows[1:]] == ["duplicate", "duplicate", "empty"]


def test_checkpoint_recovers_population_archive_q_and_rng(problem, tmp_path):
    config = Config(population=6, neighborhood=3, seconds=10, seed=5)
    gateway = EvaluationGateway(problem, Archive())
    streams = RNGManager(config.seed)
    initial = initial_genotypes(problem, config, streams.stream("initialization"))
    population = [gateway.evaluate(g) for g in initial]
    observer = P1Observer(tmp_path / "sample", config)
    payload = {
        "population": population,
        "gateway": gateway,
        "controller": Controller(config),
        "streams": streams,
        "stagnation": [0] * 6,
        "elapsed": 9.1,
        "processed": 1,
        "row": {"steps": [], "generation": 0},
    }
    rng_before = streams.snapshot()
    observer("initialization", payload)
    observer("offspring", payload)
    observer.close()
    assert len(observer.checkpoints) == 3
    assert streams.snapshot() == rng_before
    with gzip.open(tmp_path / "sample" / observer.checkpoints[0], "rt") as h:
        data = json.load(h)
    assert [restore_candidate(c) for c in data["population"]] == population
    assert data["q"] == Controller(config).q
    assert data["ideal"] == list(gateway.ideal)
    assert data["maximum"] == list(maximum(population))


def test_prefix_ideal_cannot_see_later_candidates(problem):
    result = run(problem, Config(population=3, neighborhood=2, generations=1))
    c = result.population[0]

    def point(flow, bill):
        return replace(c, evaluation=EvaluationResult(True, flow, bill, (0,), ((),)))

    incumbent = point(8, 8)
    candidates = [point(7, 7)] * 3 + [point(0, 0)]
    rows = prefix_audit(
        incumbent, candidates, NormalizationContext((5, 5), (10, 10)), (0.5, 0.5), [incumbent]
    )
    assert rows[0]["ideal"] == (5, 5)
    assert rows[1]["ideal"] == (0, 0)
    assert rows[0]["actual_exact"] == 3 and rows[2]["actual_exact"] == 4


def test_h2_separates_outside_birth_and_cap(problem):
    base = run(problem, Config(population=3, neighborhood=2)).population[0]
    population = [
        replace(base, evaluation=EvaluationResult(True, 10, 10, (0,), ((),))) for _ in range(9)
    ]
    child = replace(base, evaluation=EvaluationResult(True, 1, 1, (0,), ((),)))
    result = replacement_audit(
        population,
        child,
        NormalizationContext((0, 0), (10, 10)),
        4,
        Config(population=9, neighborhood=3, replacement_cap=1),
        RNGManager(9).stream("audit"),
    )
    assert len(result["improving"]) == 9
    assert len(result["outside_birth"]) == 6
    assert result["cap_truncated"] == 2
    assert result["original"] == [4]


def test_a7_filter_audit_matches_feasible_selected_moves(problem):
    c = run(problem, Config(population=3, neighborhood=2)).population[0]
    audit = select_packing_moves(problem, c, 6, RNGManager(7).stream("A7"))
    batch = ActivePack().propose(problem, c, 6, Config(), RNGManager(7).stream("A7"))
    assert len(batch.proposals) == len(audit.selected)
    assert set(audit.selected) <= set(audit.pool) <= set(audit.representatives) <= set(audit.raw)
    assert [p.schedule.starts[m[0]] for p, m in zip(batch.proposals, audit.selected)] == [
        m[1] for m in audit.selected
    ]


def test_all_four_diagnostics_use_restorable_feasible_candidates(problem):
    config = Config(population=6, neighborhood=3, generations=2, seed=12)
    result = run(problem, config)
    data = {
        "population": [solution(c) for c in result.population],
        "archive": [solution(c) for c in result.archive.members],
        "ideal": result.gateway.ideal,
        "maximum": maximum(result.population),
        "fraction": 0.1,
        "sample": [{"index": 0, "stratum": "Flow"}],
    }
    rows = diagnose_checkpoint(problem, data, config)
    assert rows[0]["h1_exact"] == 1
    assert [a["action"] for a in rows[0]["actions"]] == list(range(1, 9))
    assert all(a["exact"] <= 10 for a in rows[0]["actions"])
    assert rows[0]["actions"][6]["raw_audit"]["extra_exact"] <= 40
    full_rows = diagnose_checkpoint(problem, data, replace(config, method="full"))
    for key in ("continuation_retained", "continuation_redecoded"):
        assert len(full_rows[0][key]["steps"]) == 5
        assert restore_candidate(full_rows[0][key]["final"]).evaluation.feasible


@pytest.mark.parametrize("stage", ["P0R1", "P1"])
def test_frozen_worker_rejects_tampered_input_and_checksum(problem, tmp_path, stage):
    instance, initial_path = tmp_path / "input.json", tmp_path / "initial.json"
    save_instance(problem, instance)
    config = Config(population=3, neighborhood=2, generations=2)
    if stage == "P1":
        config = replace(config, method="full", generations=1000000, seconds=0.15)
    initial = initial_genotypes(problem, config, RNGManager(1).stream("initial"))
    atomic_json(initial_path, [asdict(g) for g in initial])
    spec = JobSpec(
        stage,
        "M0",
        1,
        1,
        asdict(config),
        str(instance),
        file_hash(instance),
        str(initial_path),
        file_hash(initial_path),
        source_digest(),
        "test",
    )
    path = tmp_path / "spec.json"
    atomic_json(path, asdict(spec))
    output = tmp_path / "run"
    execute_spec(path, output)
    assert validate_spec(output, spec)
    (output / "semantic_signature.json").write_text("tampered")
    assert not validate_spec(output, spec)
    instance.write_text("tampered")
    with pytest.raises(ValueError, match="input changed"):
        execute_spec(path, tmp_path / "bad")
