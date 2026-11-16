"""Counterfactual gate equivalence, matched position quota and fixed-tail regression."""

import json
import random
from dataclasses import replace

import pytest
from test_d2_opportunity import captured_panel
from test_d4_peak import peak_fixture

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Schedule
from geo_llm_scheduler.experiments import d4_replay_campaign as campaign
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d4_peak import audit_residual, residual_certificate
from geo_llm_scheduler.experiments.d4_prototypes import (
    ResearchPeak,
    cached_certificate,
    certificate_cache,
    partial_windows,
    position_repair,
    timed_batch,
)
from geo_llm_scheduler.io.loaders import save_instance


def test_fixed_tail_partial_integrator_and_active_union():
    p, x = peak_fixture()
    assert partial_windows(p, x.genotype, x.schedule, 0, 1801) == pytest.approx(
        x.evaluation.windows[0]
    )
    missing = Schedule((0, -1, 0, -1))
    assert partial_windows(p, x.genotype, missing, 0, 1801) == pytest.approx((110, 20, 20 / 900))
    all_missing = Schedule((-1, -1, -1, -1))
    assert partial_windows(p, x.genotype, all_missing, 0, 1801) == pytest.approx((20, 20, 20 / 900))
    cache = certificate_cache(p, x, 0)
    for members in (frozenset({0}), frozenset({0, 2}), frozenset({1, 3}), frozenset({1})):
        cached = cached_certificate(cache, members)
        original = residual_certificate(p, x, members, 0)
        assert cached["denied"] == original["denied"]
        assert cached["applicable"] == original["applicable"]
        assert cached["lower_windows"] == pytest.approx(audit_residual(p, x, members, 0))


@pytest.mark.parametrize("seed", range(24))
def test_gate_preserves_stream_and_exact_proposal_pool(seed):
    p, x = peak_fixture()
    reference_rng = random.Random(seed)
    reference = ResearchPeak().propose(p, x, 3, Config(), reference_rng)
    for guard in ("all", "single"):
        rng = random.Random(seed)
        batch = ResearchPeak(guard).propose(p, x, 3, Config(), rng)
        assert (batch.proposals, batch.attempts, rng.getstate()) == (
            reference.proposals,
            reference.attempts,
            reference_rng.getstate(),
        )
        assert [r["members"] for r in batch.diagnostics["trace"]] == [
            r["members"] for r in reference.diagnostics["trace"]
        ]
        for guarded, ref in zip(batch.diagnostics["trace"], reference.diagnostics["trace"]):
            if guarded["stage"] == "certificate":
                assert ref["stage"] not in ("proposal", "duplicate")


def test_position_scoring_control_and_empty_invalid_boundaries():
    p, x = peak_fixture()
    record = {}
    assert (
        position_repair(p, x, frozenset({0, 2}), 0, 0, random.Random(3), "peak_rank", record)
        is None
    )
    assert record["stage"] == "resource_positions"
    with pytest.raises(ValueError):
        position_repair(p, x, frozenset({0}), 0, 6, random.Random(3), "invalid", {})
    with pytest.raises(ValueError):
        ResearchPeak("all", "peak_rank")
    with pytest.raises(ValueError):
        ResearchPeak("invalid")
    assert not ResearchPeak().propose(p, x, 0, Config(), random.Random(3)).proposals
    no_demand = replace(p, regions=(replace(p.regions[0], demand_rate=0),))
    assert not ResearchPeak().propose(no_demand, x, 3, Config(), random.Random(3)).proposals
    _, timing = timed_batch(ResearchPeak("all"), p, x, Config(), 3)
    assert timing["wall_seconds"] >= 0 and timing["cpu_seconds"] >= 0
    for seed in range(6):
        rng1, rng2 = random.Random(seed), random.Random(seed)
        baseline = ResearchPeak().propose(p, x, 3, Config(), rng1)
        control = ResearchPeak(positions="random_scored").propose(p, x, 3, Config(), rng2)
        assert baseline.proposals == control.proposals and rng1.getstate() == rng2.getstate()


@pytest.mark.parametrize("role", ("guard", "positions"))
def test_real_unit_exact_audit_and_both_controls(problem, role):
    result = campaign.replay_unit(problem, captured_panel(problem), Config(), role, 311, 2)
    assert result["exact_audits"] >= len(result["rows"])
    for row in result["rows"]:
        arms = row["arms"]
        assert len(arms["REFERENCE"]["timing"]) == 2
        if role == "guard":
            assert arms["REFERENCE"]["candidates"] == [
                dict(c, origin="REFERENCE") for c in arms["GUARD_ALL"]["candidates"]
            ]


def test_transactional_freeze_resume_hash_and_stop(tmp_path, problem, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    instance = parent / "inputs/problem.json"
    instance.parent.mkdir()
    save_instance(problem, instance)
    spec = {"instance": "inputs/problem.json", "instance_hash": file_hash(instance), "config": {}}
    atomic_json(parent / "specs/k.json", spec)
    atomic_json(parent / "samples/k/data.json", {"panels": [captured_panel(problem)]})
    atomic_json(
        parent / "samples/k/complete.json",
        {"files": {"data.json": file_hash(parent / "samples/k/data.json")}},
    )
    atomic_json(
        parent / "manifest.json",
        {
            "keys": ["k"],
            "source_commit": "parent",
            "input_files": {"specs/k.json": file_hash(parent / "specs/k.json")},
        },
    )
    monkeypatch.setattr(
        campaign, "git_metadata", lambda *a: {"git_dirty": False, "git_commit": "fixture"}
    )
    root = tmp_path / "campaign"
    manifest = campaign.prepare_replay(parent, root, "guard", 1, 1)
    assert campaign.frozen_replay(root) == manifest
    with pytest.raises(ValueError, match="overwriting"):
        campaign.prepare_replay(parent, root, "guard")
    (root / "stop.request").write_text("user stop")
    campaign.execute_replay(root, "k")
    assert not campaign.validate_replay_job(root, manifest, "k")
    (root / "stop.request").unlink()
    campaign.execute_replay(root, "k")
    assert campaign.validate_replay_job(root, manifest, "k")
    campaign.execute_replay(root, "k")
    campaign.supervise_replay(root, 1)
    assert json.loads((root / "ops/exit.json").read_text())["state"] == "complete"
    (root / "results/k/p0_s0.json").write_text("tamper")
    with pytest.raises(ValueError, match="completion"):
        campaign.validate_replay_job(root, manifest, "k")
    with pytest.raises(ValueError):
        campaign.supervise_replay(root, 0)
