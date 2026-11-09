"""Numeric counterexamples, crossed configurations and same-host pairing for D8."""

import importlib.util
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import d8_worker, runner
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d8_design import ARMS, FACTORS, factor_config, factor_settings
from geo_llm_scheduler.experiments.e15_instances import generate_e15_pair
from geo_llm_scheduler.experiments.runner import deterministic_trace
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.operators.structural import (
    StructuralOperator,
    relocation_tou_proxy,
    relocation_tou_rank,
)
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.numeric import TOL
from geo_llm_scheduler.utils.rng import RNGManager


def test_stable_cost_rank_zero_boundary_transitive_and_sign():
    # Realistic decimal exposures: the historical left-associative form is nonzero.
    a, b = 0.1, 0.2
    assert a + b - a - b != 0
    assert relocation_tou_rank(a - a, b - b) == 0
    assert relocation_tou_rank(0.75 * TOL.cost, 0.25 * TOL.cost) == 0
    assert relocation_tou_rank(-TOL.cost, 0) == 0
    values = (-5, -2.5, -1.1, -1, -0.5, 0, 0.5, 1, 1.1, 2.5, 5)
    ranks = [relocation_tou_rank(v * TOL.cost, 0) for v in values]
    assert ranks == sorted(ranks)
    assert ranks[0] < 0 < ranks[-1]
    assert relocation_tou_rank(4 * TOL.cost, -4 * TOL.cost) == 0


def test_homogeneous_tariffs_preserve_full_pool_sample_attempts_and_rng():
    p, _ = generate_e15_pair(10, 1300001)
    c = Config(method="full", population=6, neighborhood=3)
    initial = initial_genotypes(p, c, RNGManager(101).stream("initialization"))
    op = StructuralOperator(3)
    for g in initial:
        schedule = decode(p, g)
        x = Candidate(g, schedule, evaluate(p, g, schedule))
        proxy = relocation_tou_proxy(p, x)
        for o in range(p.operation_count):
            assert len({proxy[o, m] for m in range(len(p.instances))}) == 1
        corrected = replace(c, a3_region_policy="tariff")
        assert op._assignments(p, x, c) == op._assignments(p, x, corrected)
        for budget in (1, 6, 10):
            a, b = random.Random(51), random.Random(51)
            assert op.propose(p, x, budget, c, a) == op.propose(p, x, budget, corrected, b)
            assert a.getstate() == b.getstate()


def test_homogeneous_fixed_generation_default_replay_matches_corrected_tariff():
    p, _ = generate_e15_pair(4, 1300002)
    c = Config(method="full", population=4, neighborhood=2, generations=2, trigger_mode="always")
    old = run(p, c)
    corrected = run(p, replace(c, a3_region_policy="tariff"))
    assert deterministic_trace(old.trace) == deterministic_trace(corrected.trace)
    assert old.population == corrected.population
    assert old.archive.members == corrected.archive.members


@pytest.mark.parametrize("mask", range(16))
def test_four_factor_config_exact_changes_and_labels(mask):
    baseline = Config(method="full")
    changed = factor_config(baseline, mask)
    actual = {k: v for k, v in asdict(changed).items() if v != asdict(baseline)[k]}
    assert actual == factor_settings(mask)
    assert len(actual) == mask.bit_count()
    assert ARMS[mask] == (
        "+".join(name for bit, name in enumerate(FACTORS) if mask & (1 << bit)) or "BASE"
    )


@pytest.mark.parametrize("mask", [-1, 16, True, 1.5])
def test_invalid_factor_masks_rejected(mask):
    with pytest.raises(ValueError):
        factor_settings(mask)


def test_matrix_four_shards_disjoint_pairing_and_williams_carryover(tmp_path, monkeypatch):
    module_spec = importlib.util.spec_from_file_location(
        "d8_campaign", "scripts/run_d8_campaign.py"
    )
    assert module_spec and module_spec.loader
    campaign = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(campaign)
    monkeypatch.setattr(
        campaign, "git_metadata", lambda *a: {"git_dirty": False, "git_commit": "fixture"}
    )
    # This test checks the full matrix's metadata and carryover arithmetic. Use
    # small problems for its initialization; actual 50/100-job frozen inputs are
    # independently checked before deployment and all target workers use those.
    monkeypatch.setattr(
        campaign, "generate_e15_pair", lambda _jobs, seed: generate_e15_pair(4, seed)
    )
    all_bases = set()
    total = 0
    for node in range(4):
        root = tmp_path / str(node)
        matrix = campaign.prepare(root, Path.cwd(), node=node)
        assert matrix["run_count"] == 1024 and len(matrix["blocks"]) == 64
        assert matrix["nominal_worker_hours"] == 384
        positions = {arm: [0] * 16 for arm in ARMS}
        precedes = {(a, b): 0 for a in ARMS for b in ARMS if a != b}
        bases = set()
        for block in matrix["blocks"]:
            specs = [json.loads((root / "specs" / f"{k}.json").read_text()) for k in block["runs"]]
            assert (
                len({s["instance_hash"] for s in specs})
                == len({s["initial_hash"] for s in specs})
                == 1
            )
            assert {s["arm"] for s in specs} == set(ARMS)
            bases.add(specs[0]["base_seed"])
            baseline = next(s["config"] for s in specs if s["arm"] == "BASE")
            for position, s in enumerate(specs):
                positions[s["arm"]][position] += 1
                actual = {
                    k: v for k, v in s["config"].items() if v != baseline[k] and k != "output"
                }
                assert actual == factor_settings(s["factor_mask"])
                if position:
                    precedes[specs[position - 1]["arm"], s["arm"]] += 1
        assert all(counts == [4] * 16 for counts in positions.values())
        assert set(precedes.values()) == {4}
        assert not bases & all_bases
        all_bases.update(bases)
        total += matrix["run_count"]
    assert total == 4096 and len(all_bases) == 64


@pytest.mark.parametrize("mask", [0, 3, 15])
def test_combination_engine_keeps_exact_ground_truth(problem, mask):
    c = Config(method="full", population=4, neighborhood=2, generations=2, trigger_mode="always")
    result = run(problem, factor_config(c, mask))
    assert len(result.controller.q) == (84 if mask & 4 else 42)
    for candidate in result.population + result.archive.members:
        assert candidate.evaluation == evaluate(problem, candidate.genotype, candidate.schedule)


@pytest.mark.parametrize("mask", [0, 3, 15])
def test_crossed_worker_transaction_exact_and_tamper(tmp_path, problem, monkeypatch, mask):
    provenance = {"git_dirty": False, "git_commit": "fixture", "git_branch": "test"}
    monkeypatch.setattr(d8_worker, "git_metadata", lambda *a: provenance)
    monkeypatch.setattr(runner, "git_metadata", lambda *a: provenance)
    c = factor_config(
        Config(method="full", population=4, neighborhood=2, generations=2, trigger_mode="always"),
        mask,
    )
    (tmp_path / "inputs").mkdir()
    save_instance(problem, tmp_path / "inputs/problem.json")
    initial = initial_genotypes(problem, c, RNGManager(1).stream("initialization"))
    atomic_json(tmp_path / "inputs/initial.json", [asdict(g) for g in initial])
    spec = {
        "key": "key",
        "arm": ARMS[mask],
        "factor_mask": mask,
        "instance": "inputs/problem.json",
        "initial": "inputs/initial.json",
        "instance_hash": file_hash(tmp_path / "inputs/problem.json"),
        "initial_hash": file_hash(tmp_path / "inputs/initial.json"),
        "source_hash": d8_worker.canonical_source_hash(),
        "source_commit": "fixture",
        "config": asdict(c),
    }
    atomic_json(tmp_path / "specs/key.json", spec)
    d8_worker.execute(tmp_path, "key")
    output = tmp_path / "runs/key"
    assert d8_worker.validate_complete(output, spec) == (True, "complete")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["light_budget_histogram"] and summary["offspring_count"] > 0
    (output / "qtable.json").write_text("{}")
    assert d8_worker.validate_complete(output, spec)[0] is False
