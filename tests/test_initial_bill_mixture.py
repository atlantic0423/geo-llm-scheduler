"""Independent exact, prefix overlap/tail, provenance and resource guard checks."""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.initial_bill_mixture import (
    ARMS,
    PrefixPower,
    assemble,
    guided_genotype,
    retain_original,
    seed_bank,
    tariff_integral,
)
from geo_llm_scheduler.experiments.initial_frontier import diagnostic_problem, sample_population
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.scheduling.ssgs import decode


def test_overlap_idle_and_fixed_tail_match_independent_exact(problem):
    """Machine zero has overlapping operations; every server's idle is charged."""
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    s = Schedule((0, 100, 50, 150))
    reference = evaluate(problem, g, s)
    assert reference.feasible
    state = PrefixPower(problem)
    state.commit(state.trial({0: [(0, 100), (100, 300)]}, 300))
    trial = state.trial({0: [(50, 150), (150, 350)]}, 350)
    assert trial[0] == pytest.approx(reference.tou)
    assert trial[1] == pytest.approx(sum(reference.demand))
    assert state.horizon == 300
    state.commit(trial)
    assert state.unions[0] == [(0, 350)]
    assert state.trial({}, 350)[:2] == pytest.approx((reference.tou, sum(reference.demand)))


def test_integral_rejects_uncovered_tariff(problem):
    assert tariff_integral(problem, 0, 900, 1000) == pytest.approx(20)
    with pytest.raises(ValueError, match="outside"):
        tariff_integral(problem, 0, 0, 10001)


@pytest.mark.parametrize(
    "scenario", ("mixed", "burst", "kv_delay", "compute_tight", "vram_tight", "demand_high")
)
@pytest.mark.parametrize("family", ("TOU", "DEMAND", "BILL"))
def test_guided_ssgs_replay_and_full_exact(scenario, family):
    p = diagnostic_problem(12, 1472001, "T", scenario)
    g, info = guided_genotype(p, family, random.Random(93), 0.25)
    repeated, again = guided_genotype(p, family, random.Random(93), 0.25)
    assert g == repeated and info == again
    e = evaluate(p, g, decode(p, g))
    assert e.feasible
    assert info["prefix_tou"] == pytest.approx(e.tou, abs=1e-8)
    assert info["prefix_demand"] == pytest.approx(sum(e.demand), abs=1e-8)


def original(p):
    """Create ten exact, original source quotas from the unmodified tracing path."""
    trace = sample_population(p, Config(seed=11001), "MIXED")
    from geo_llm_scheduler.domain.models import Candidate

    return [
        Candidate(g, decode(p, g), evaluate(p, g, decode(p, g)), f"initial_mode_{m}")
        for g, m in zip(trace.genotypes, trace.modes, strict=True)
    ]


def test_shared_bank_unique_nested_quotas_and_accounting():
    p = diagnostic_problem(12, 1472002, "T", "mixed")
    parents = original(p)
    bank, stats = seed_bank(p, parents, random.Random(1234), 1)
    assert stats["exact"] == stats["independent_exact_verified"] == 80
    assert len(stats["observations"]) == 80
    assert len({c.genotype for members in bank.values() for c in members}) == 80
    for arm in ARMS:
        members = assemble(parents, bank, arm)
        assert len(members) == len({c.genotype for c in members}) == 100
        quota = 9 if arm.endswith("10") else 8
        counts = Counter(c.origin for c in members if c.origin.startswith("initial_mode_"))
        assert counts == {f"initial_mode_{i}": quota for i in range(10)}
    assert set(c.genotype for c in retain_original(parents, 20)) <= set(
        c.genotype for c in retain_original(parents, 10)
    )
    assert stats["production_seconds"] > 0 and stats["verification_seconds"] > 0
    with pytest.raises(ValueError):
        seed_bank(p, parents, random.Random(1), 2)
    with pytest.raises(ValueError):
        guided_genotype(p, "unknown", random.Random(1), 0.1)
    with pytest.raises(ValueError):
        retain_original(parents[:-1], 10)
    with pytest.raises(ValueError):
        assemble(parents, bank, "bad")


def test_exact_failure_never_falls_back(monkeypatch):
    from dataclasses import replace

    p = diagnostic_problem(6, 1472003, "T", "mixed")
    parents = original(p)
    monkeypatch.setattr(
        "geo_llm_scheduler.experiments.initial_bill_mixture.evaluate",
        lambda *args: replace(parents[0].evaluation, feasible=False),
    )
    with pytest.raises(AssertionError, match="independent exact"):
        seed_bank(p, parents, random.Random(1), 1)


def test_runner_transaction_resume_tamper_and_resource_guards(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_initial_bill_mixture as runner

    from geo_llm_scheduler.experiments.campaign.support import atomic_json
    from geo_llm_scheduler.experiments.campaign.validation import file_hash
    from geo_llm_scheduler.io.loaders import save_instance

    p = diagnostic_problem(12, 1472004, "T", "mixed")
    parents = original(p)
    (tmp_path / "instances").mkdir()
    save_instance(p, tmp_path / "instances/input.json")
    atomic_json(tmp_path / "parents/case.json", [solution(c) for c in parents])
    spec = {
        "key": "case",
        "seed": 11001,
        "base_seed": 1,
        "tariff": "T",
        "rng_version": "test",
        "instance": "instances/input.json",
        "instance_sha256": file_hash(tmp_path / "instances/input.json"),
        "parent_population": "parents/case.json",
        "parent_population_sha256": file_hash(tmp_path / "parents/case.json"),
        "source_commit": "test",
        "source_hash": "test",
        "expected_host": runner.platform.node(),
        "runner_sha256": file_hash(Path(runner.__file__)),
        "trials_per_slot": 1,
    }
    atomic_json(tmp_path / "specs/case.json", spec)
    monkeypatch.setattr(runner, "canonical_source_hash", lambda: "test")
    monkeypatch.setattr(runner, "git_metadata", lambda: {"git_commit": "test", "git_dirty": False})
    runner.worker(tmp_path, "case")
    assert runner.complete(tmp_path, "case", spec)
    marker = file_hash(tmp_path / "runs/case/complete.json")
    runner.worker(tmp_path, "case")
    assert marker == file_hash(tmp_path / "runs/case/complete.json")
    summary = json.loads((tmp_path / "runs/case/summary.json").read_text())
    assert summary["baseline_replay_verified"] == 100
    assert summary["selected_exact_verified"] == 1000
    (tmp_path / "runs/case/TOU10/population.json").write_text("corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.complete(tmp_path, "case", spec)
    assert runner.dispatch_allowed({"available_memory_gib": 3, "free_disk_gib": 15}, 0, 4000, 3600)
    assert not runner.dispatch_allowed(
        {"available_memory_gib": 2.1, "free_disk_gib": 15}, 0, 4000, 3600
    )
    assert not runner.dispatch_allowed(
        {"available_memory_gib": 3, "free_disk_gib": 15}, 500, 4000, 3600
    )
