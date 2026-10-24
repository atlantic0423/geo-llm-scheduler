"""Bit-exact fast scores, matched RNG, guarded combination and default invariance."""

import importlib.util
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_d4_peak import peak_fixture

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Schedule
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments import d5_worker, runner
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d4_prototypes import ResearchPeak, partial_windows
from geo_llm_scheduler.experiments.d5_peak import OnlinePeak, position_repair_fast
from geo_llm_scheduler.experiments.d5_windows import InsertionWindows
from geo_llm_scheduler.experiments.p2_worker import canonical_source_hash, validate_complete
from geo_llm_scheduler.experiments.runner import deterministic_trace
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.utils.rng import RNGManager


@pytest.mark.parametrize("seed", range(24))
def test_bit_identical_insertion_overlap_idle_tail_and_missing_decode(seed):
    p, x = peak_fixture()
    rng = random.Random(seed)
    for o in range(p.operation_count):
        starts = [rng.choice((-1.0, 0.0, 0.1, 899.9, 900.0, 1800.0)) for _ in x.schedule.starts]
        starts[o] = -1
        partial = Schedule(tuple(starts))
        cache = InsertionWindows(p, x.genotype, partial, 0, 1801.0)
        for t in (-1.0, 0.0, 0.1, 899.9, 900.0, 1800.0):
            trial = list(starts)
            trial[o] = t
            assert cache.windows(o, t) == partial_windows(
                p, x.genotype, Schedule(tuple(trial)), 0, 1801.0
            )
        # Regions with no instances have zero fixed windows, including the tail.
        assert InsertionWindows(p, x.genotype, partial, 1, 1801).windows(o, 0) == (0, 0, 0)


@pytest.mark.parametrize("seed", range(24))
def test_fast_w_preserves_complete_pool_attempts_and_rng_and_gw_guard(seed):
    p, x = peak_fixture()
    reference_rng, rng, combined_rng = (random.Random(seed) for _ in range(3))
    slow = ResearchPeak(positions="peak_rank").propose(p, x, 6, Config(), reference_rng)
    fast = OnlinePeak(positions="peak_rank", diagnose=True).propose(p, x, 6, Config(), rng)
    combined = OnlinePeak("all", "peak_rank").propose(p, x, 6, Config(), combined_rng)
    assert slow.proposals == fast.proposals == combined.proposals
    assert slow.attempts == fast.attempts == combined.attempts
    assert reference_rng.getstate() == rng.getstate() == combined_rng.getstate()
    assert slow.diagnostics == fast.diagnostics
    for proposal in fast.proposals:
        assert proposal.schedule is not None
        assert evaluate(p, proposal.genotype, proposal.schedule).feasible


def test_fast_boundaries_and_default_engine_unchanged(problem):
    p, x = peak_fixture()
    with pytest.raises(ValueError):
        OnlinePeak("invalid")
    with pytest.raises(ValueError):
        position_repair_fast(p, x, frozenset({0}), 0, 6, random.Random(1), "invalid", {})
    record = {}
    assert (
        position_repair_fast(p, x, frozenset({0}), 0, 0, random.Random(1), "peak_rank", record)
        is None
    )
    assert record["stage"] == "resource_positions"
    config = Config(
        population=6, neighborhood=3, generations=3, method="full", trigger_mode="always"
    )
    baseline = run(problem, config)
    injected = run(problem, config, operator_overrides={8: PeakCoalition()})
    assert deterministic_trace(baseline.trace) == deterministic_trace(injected.trace)
    assert baseline.population == injected.population
    assert baseline.archive.members == injected.archive.members
    with pytest.raises(ValueError, match="limited to A8"):
        run(problem, config, operator_overrides={7: PeakCoalition()})
    reference = run(problem, config, operator_overrides={8: ResearchPeak(diagnose=False)})
    guarded = run(problem, config, operator_overrides={8: OnlinePeak("all")})
    assert reference.population == guarded.population
    assert reference.archive.members == guarded.archive.members
    assert run(problem, replace(config, method="plain"), operator_overrides={8: OnlinePeak()})


def test_five_arm_freeze_host_budget_balance_and_resume(tmp_path, monkeypatch):
    path = Path("scripts/run_d5_campaign.py").resolve()
    spec = importlib.util.spec_from_file_location("d5_campaign", path)
    assert spec is not None and spec.loader is not None
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    monkeypatch.setattr(
        campaign, "git_metadata", lambda *a: {"git_dirty": False, "git_commit": "fixture"}
    )
    root = tmp_path / "formal"
    matrix = campaign.prepare(root, Path.cwd(), node=0)
    assert matrix["run_count"] == 240
    assert len(matrix["blocks"]) == 48
    assert matrix["nominal_worker_hours"] == 120
    assert [b["jobs"] for b in matrix["blocks"]] == [100] * 24 + [50] * 24
    for block in matrix["blocks"]:
        assert len(block["runs"]) == 5
        specs = [json.loads((root / "specs" / f"{key}.json").read_text()) for key in block["runs"]]
        assert len({s["instance_hash"] for s in specs}) == 1
        assert len({s["initial_hash"] for s in specs}) == 1
        assert {s["arm"] for s in specs} == set(campaign.ARMS)
        assert campaign.recover_block(root, block, "fixture-host") is False
    with pytest.raises(FileExistsError):
        campaign.prepare(root, Path.cwd(), node=0)
    with pytest.raises(ValueError):
        campaign.prepare(tmp_path / "bad", Path.cwd(), node=2)


@pytest.mark.parametrize("arm", ("LEGACY", "REFERENCE", "G", "W", "GW"))
def test_transactional_worker_all_arms_final_exact_and_tamper(tmp_path, problem, monkeypatch, arm):
    provenance = {"git_dirty": False, "git_commit": "fixture", "git_branch": "test"}
    monkeypatch.setattr(d5_worker, "git_metadata", lambda *a: provenance)
    monkeypatch.setattr(runner, "git_metadata", lambda *a: provenance)
    config = Config(
        population=6,
        neighborhood=3,
        generations=1000000,
        seconds=0.5,
        method="full",
        trigger_mode="always",
    )
    instance, initial = "inputs/problem.json", "inputs/initial.json"
    (tmp_path / "inputs").mkdir()
    save_instance(problem, tmp_path / instance)
    genotypes = initial_genotypes(problem, config, RNGManager(1).stream("initialization"))
    atomic_json(tmp_path / initial, [asdict(g) for g in genotypes])
    spec = {
        "key": "key",
        "arm": arm,
        "instance": instance,
        "initial": initial,
        "instance_hash": file_hash(tmp_path / instance),
        "initial_hash": file_hash(tmp_path / initial),
        "source_hash": canonical_source_hash(),
        "source_commit": "fixture",
        "config": asdict(config),
    }
    atomic_json(tmp_path / "specs/key.json", spec)
    d5_worker.execute(tmp_path, "key")
    assert validate_complete(tmp_path / "runs/key", spec) == (True, "complete")
    with pytest.raises(FileExistsError):
        d5_worker.execute(tmp_path, "key")
    (tmp_path / "runs/key/config.json").write_text("{}")
    assert validate_complete(tmp_path / "runs/key", spec)[0] is False
