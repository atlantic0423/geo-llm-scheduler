"""A8 observer equivalence and active-union/tail counterexamples."""

import json
import random
from dataclasses import replace

import pytest
from test_d2_opportunity import captured_panel, source_for

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import (
    Genotype,
    Job,
    ProblemInstance,
    Profile,
    Region,
    Schedule,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments import d2_campaign
from geo_llm_scheduler.experiments.d4_peak import (
    analyse_peak,
    audit_residual,
    probe_peak_panel,
    residual_certificate,
)
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition, repair


@pytest.mark.parametrize("seed", range(12))
def test_diagnostics_preserve_original_rng_proposals_and_budget(problem, seed):
    x = source_for(problem)
    rng1, rng2 = random.Random(seed), random.Random(seed)
    reference = PeakCoalition().propose(problem, x, 3, Config(), rng1)
    observed = PeakCoalition(True).propose(problem, x, 3, Config(), rng2)
    assert reference.proposals == observed.proposals
    assert reference.attempts == observed.attempts
    assert reference.instrumentation == observed.instrumentation
    assert rng1.getstate() == rng2.getstate()
    assert len(observed.diagnostics.get("trace", [])) == observed.attempts


def peak_fixture():
    p = ProblemInstance(
        (
            Job("a", 0, Profile(900, 0.4, 1), Profile(1, 0.4, 1)),
            Job("b", 0, Profile(900, 0.4, 1), Profile(1, 0.4, 1)),
        ),
        (Region("r", (Tariff(0, 5000, 0.1),), 1),),
        (ServingInstance("active", 0, 4, 10, 100), ServingInstance("idle", 0, 4, 10, 10)),
    )
    g = Genotype((0, 1, 0, 1), (0, 1, 0, 1))
    x = EvaluationGateway(p, Archive()).evaluate(g, Schedule((0, 1800, 0, 1800)))
    assert x.evaluation.feasible
    return p, x


def test_overlapping_instance_activity_cannot_be_removed_individually():
    p, x = peak_fixture()
    cert = residual_certificate(p, x, frozenset({0}), 0)
    assert cert["applicable"] and cert["denied"]
    assert cert["lower_windows"] == pytest.approx(x.evaluation.windows[0])
    both = residual_certificate(p, x, frozenset({0, 2}), 0)
    assert not both["denied"]
    assert both["lower_windows"] == pytest.approx(audit_residual(p, x, frozenset({0, 2}), 0))
    # Idle survives all removals, including the fixed-divisor partial tail.
    assert both["lower_windows"] == pytest.approx((20, 20, 20 / 900))


def test_all_tail_operations_selected_disables_the_certificate():
    p, x = peak_fixture()
    cert = residual_certificate(p, x, frozenset({1, 3}), 0)
    assert not cert["applicable"] and not cert["denied"]
    cert = residual_certificate(p, x, frozenset({1}), 0)
    assert cert["applicable"]


def test_failed_repair_stage_and_success_are_observable(problem):
    x = source_for(problem)
    record = {}
    assert repair(problem, x, frozenset({0}), 0, 0, random.Random(1), record) is None
    assert record["stage"] == "empty_positions"
    p, x = peak_fixture()
    record = {}
    schedule = repair(p, x, frozenset({0, 2}), 0, 30, random.Random(1), record)
    if schedule:
        assert schedule.horizon(p) == x.schedule.horizon(p)
        assert all(
            a <= b + 1e-8
            for a, b in zip(
                audit_residual(p, x, frozenset({0, 2}), 0),
                EvaluationGateway(p, Archive())
                .evaluate(x.genotype, schedule)
                .evaluation.windows[0],
            )
        )


def test_panel_exact_audit_and_empty_sources_are_retained(problem):
    result = probe_peak_panel(problem, captured_panel(problem), Config(), 31)
    assert len(result["rows"]) == 3 and result["audit_checks"] >= 3
    for row in result["rows"]:
        assert row["exact"] <= 3
        assert len(row["recipes"]) == row["attempts"]
    no_demand = replace(problem, regions=tuple(replace(r, demand_rate=0) for r in problem.regions))
    result = probe_peak_panel(no_demand, captured_panel(no_demand), Config(), 31)
    assert all(r["empty"] and not r["recipes"] for r in result["rows"])


def test_d4_transactional_pair_analysis_and_hash_tamper(tmp_path, problem, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "docs/experiments").mkdir(parents=True)
    (repo / d2_campaign.PROTOCOL).write_text("sampling protocol")
    (repo / "docs/experiments/d4_peak_diagnosis.md").write_text("D4 fixture protocol")
    monkeypatch.setattr(
        d2_campaign, "git_metadata", lambda *a: {"git_commit": "fixture", "git_dirty": False}
    )
    monkeypatch.setattr(
        d2_campaign, "load_config", lambda p: Config(population=6, neighborhood=3, method="plain")
    )
    monkeypatch.setattr(d2_campaign, "generate_e15_pair", lambda n, b: (problem, problem))
    root = tmp_path / "campaign"
    manifest = d2_campaign.prepare_opportunity(root, repo, 0.1, "D4")
    assert manifest["kind"] == "D4_peak_diagnosis_v1"
    assert "inputs/d4_peak_diagnosis.md" in manifest["input_files"]
    key = manifest["keys"][0]
    manifest["keys"] = [key]
    monkeypatch.setattr(d2_campaign, "frozen_manifest", lambda p: manifest)
    d2_campaign.execute_opportunity_job(root, "samples", key)
    d2_campaign.execute_opportunity_job(root, "probes", key)
    result = analyse_peak(root)
    assert result["false_rejections"] == 0 and result["total"]["calls"] == 27
    assert result["online_benefit"] == "UNVERIFIED"
    data = json.loads((root / "probes" / key / "data.json").read_text())
    assert all(p["d1"]["rows"] == [] for p in data["panels"])
    (root / "probes" / key / "data.json").write_text("tampered")
    with pytest.raises(ValueError, match="artifact hash"):
        analyse_peak(root)
